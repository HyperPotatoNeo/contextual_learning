"""Merge LoRA adapter weights into base model weights and save merged checkpoint.

Usage:
    python merge_lora_weights.py /path/to/step_400 /path/to/output_merged
"""

import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def merge_lora(base_dir: str, output_dir: str):
    base_path = Path(base_dir)
    out_path = Path(output_dir)
    adapter_path = base_path / "lora_adapters"

    # Load adapter config
    with open(adapter_path / "adapter_config.json") as f:
        adapter_cfg = json.load(f)

    rank = adapter_cfg["r"]
    alpha = adapter_cfg["lora_alpha"]
    scaling = alpha / rank
    target_modules = adapter_cfg["target_modules"]
    print(f"LoRA config: rank={rank}, alpha={alpha}, scaling={scaling}")
    print(f"Target modules: {target_modules}")

    # Load LoRA adapter weights
    print("Loading LoRA adapter weights...")
    lora_state = torch.load(adapter_path / "adapter_model.bin", map_location="cpu", weights_only=True)
    print(f"  LoRA keys: {len(lora_state)}")

    # Show some LoRA key examples
    lora_keys = sorted(lora_state.keys())
    for k in lora_keys[:5]:
        print(f"    {k}: {lora_state[k].shape}")
    print(f"    ... ({len(lora_keys)} total)")

    # Load base model weights from safetensors
    print("Loading base model weights...")
    shard_files = sorted(base_path.glob("model*.safetensors"))
    print(f"  Found {len(shard_files)} shards")

    base_state = {}
    shard_map = {}  # key -> shard_file (to know which shard each key came from)
    for sf in shard_files:
        shard = load_file(str(sf))
        for k, v in shard.items():
            base_state[k] = v
            shard_map[k] = sf.name
        print(f"  Loaded {sf.name}: {len(shard)} keys")

    print(f"  Total base keys: {len(base_state)}")

    # Adapter keys have format:
    #   LoRA:  base_model.model.{base_key}.lora_A.0.weight  (note the .0.)
    #   Full:  base_model.model.{base_key}  (e.g. embed_tokens trained directly)
    # We need to strip "base_model.model." prefix and handle both patterns.

    ADAPTER_PREFIX = "base_model.model."
    import re

    lora_pairs = {}  # base_key -> {"lora_A": tensor, "lora_B": tensor}
    full_weight_overrides = {}  # base_key -> tensor

    for key, tensor in lora_state.items():
        if not key.startswith(ADAPTER_PREFIX):
            print(f"  WARNING: unexpected key format: {key}")
            continue

        stripped = key[len(ADAPTER_PREFIX) :]

        # Match LoRA A/B patterns: {module}.lora_A.0.weight or {module}.lora_A.weight
        lora_a_match = re.match(r"(.+)\.lora_A(?:\.\d+)?\.weight$", stripped)
        lora_b_match = re.match(r"(.+)\.lora_B(?:\.\d+)?\.weight$", stripped)

        if lora_a_match:
            base_key = lora_a_match.group(1) + ".weight"
            lora_pairs.setdefault(base_key, {})["lora_A"] = tensor
        elif lora_b_match:
            base_key = lora_b_match.group(1) + ".weight"
            lora_pairs.setdefault(base_key, {})["lora_B"] = tensor
        else:
            # Full weight override (e.g. embed_tokens, lm_head)
            full_weight_overrides[stripped] = tensor

    print(f"\nFound {len(lora_pairs)} LoRA layer pairs to merge")
    print(f"Found {len(full_weight_overrides)} full weight overrides")
    for k in full_weight_overrides:
        print(f"  Full override: {k} shape={full_weight_overrides[k].shape}")

    # Apply full weight overrides (e.g. trained embed_tokens/lm_head)
    override_count = 0
    for base_key, tensor in full_weight_overrides.items():
        if base_key in base_state:
            old_norm = base_state[base_key].float().norm().item()
            diff = (base_state[base_key].float() - tensor.float()).norm().item()
            base_state[base_key] = tensor.to(torch.bfloat16)
            override_count += 1
            print(f"  Overrode {base_key}: old_norm={old_norm:.4f}, diff={diff:.4f}")
        else:
            print(f"  WARNING: override key {base_key} not in base state, adding it")
            base_state[base_key] = tensor.to(torch.bfloat16)
            # Add to shard_map (put in last shard)
            shard_map[base_key] = shard_files[-1].name

    # When train_lm_head=true with tie_word_embeddings=true during training,
    # the adapter only saves embed_tokens (named_parameters deduplicates tied params).
    # We must also update lm_head.weight to match the trained embed_tokens,
    # since we set tie_word_embeddings=false for inference.
    if "model.embed_tokens.weight" in full_weight_overrides and "lm_head.weight" in base_state:
        trained_embed = base_state["model.embed_tokens.weight"]
        old_lm_head = base_state["lm_head.weight"]
        lm_diff = (old_lm_head.float() - trained_embed.float()).norm().item()
        base_state["lm_head.weight"] = trained_embed.clone()
        override_count += 1
        print(f"  Synced lm_head.weight from trained embed_tokens (diff was {lm_diff:.6f})")

    # Merge LoRA into base weights
    merged_count = 0
    missing_keys = []
    modified_shards = set()

    for base_key, pair in sorted(lora_pairs.items()):
        if base_key not in base_state:
            missing_keys.append(base_key)
            continue
        if "lora_A" not in pair or "lora_B" not in pair:
            print(f"  WARNING: incomplete pair for {base_key}")
            continue

        lora_A = pair["lora_A"].to(torch.float32)  # [rank, in_features]
        lora_B = pair["lora_B"].to(torch.float32)  # [out_features, rank]
        base_w = base_state[base_key].to(torch.float32)

        # merged = base + scaling * (lora_B @ lora_A)
        delta = scaling * (lora_B @ lora_A)
        merged = base_w + delta
        base_state[base_key] = merged.to(torch.bfloat16)
        modified_shards.add(shard_map[base_key])

        delta_norm = delta.norm().item()
        base_norm = base_w.norm().item()
        merged_count += 1
        if merged_count <= 5:
            print(
                f"  Merged {base_key}: delta_norm={delta_norm:.4f}, base_norm={base_norm:.4f}, ratio={delta_norm / base_norm:.6f}"
            )

    print(f"\nMerged {merged_count} LoRA layers, overrode {override_count} full weights")
    print(f"Missing base keys: {len(missing_keys)}")
    if missing_keys:
        print(f"  Missing: {missing_keys[:5]}...")

    # Save merged weights
    print(f"\nSaving merged model to {out_path}")
    out_path.mkdir(parents=True, exist_ok=True)

    # Re-shard using the same layout as original
    index_file = base_path / "model.safetensors.index.json"
    if index_file.exists():
        with open(index_file) as f:
            index = json.load(f)
        weight_map = index["weight_map"]

        # Group keys by shard
        shard_keys = {}
        for key, shard_name in weight_map.items():
            shard_keys.setdefault(shard_name, []).append(key)

        # Save each shard
        for shard_name, keys in shard_keys.items():
            shard_dict = {k: base_state[k] for k in keys if k in base_state}
            save_file(shard_dict, str(out_path / shard_name))
            print(f"  Saved {shard_name}: {len(shard_dict)} keys")

        # Copy index file
        shutil.copy2(index_file, out_path / "model.safetensors.index.json")
    else:
        # Single shard
        save_file(base_state, str(out_path / "model.safetensors"))

    # Copy config with tie_word_embeddings=False
    config_file = base_path / "config.json"
    with open(config_file) as f:
        config = json.load(f)
    config["tie_word_embeddings"] = False
    with open(out_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print("  Saved config.json (tie_word_embeddings=False)")

    # Copy other files (tokenizer, generation config, etc.)
    for fn in base_path.iterdir():
        if fn.name.startswith("model") or fn.name == "config.json" or fn.is_dir():
            continue
        shutil.copy2(fn, out_path / fn.name)
        print(f"  Copied {fn.name}")

    print(f"\nDone! Merged model saved to: {out_path}")

    # Quick sanity check: compare a few keys
    print("\n=== Sanity Check ===")
    print(f"embed_tokens.weight in merged: {'model.embed_tokens.weight' in base_state}")
    print(f"lm_head.weight in merged: {'lm_head.weight' in base_state}")
    if "model.embed_tokens.weight" in base_state and "lm_head.weight" in base_state:
        embed = base_state["model.embed_tokens.weight"]
        lm_head = base_state["lm_head.weight"]
        are_equal = torch.equal(embed, lm_head)
        print(f"embed_tokens == lm_head: {are_equal}")
        if not are_equal:
            diff = (embed.float() - lm_head.float()).norm().item()
            print(f"  L2 difference: {diff:.4f}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <base_dir> <output_dir>")
        sys.exit(1)
    merge_lora(sys.argv[1], sys.argv[2])
