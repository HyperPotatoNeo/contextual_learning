import re
from typing import Dict, List

import torch
import torch.nn as nn

from prime_rl.trainer.config import LoRAConfig
from prime_rl.trainer.models.layers.lora import MultiLoRALinear, MultiLoRAModule
from prime_rl.trainer.models.layers.lora.multi_moe import MultiLoRAGroupedExperts
from prime_rl.trainer.models.layers.moe import GroupedExperts
from prime_rl.trainer.runs import get_multi_run_manager
from prime_rl.utils.logger import get_logger


def strip_lora_from_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Strip LoRA from the state dict."""
    new_state_dict = {}
    for key, value in state_dict.items():
        if "lora_A" in key or "lora_B" in key:
            continue
        new_state_dict[key] = value
    return new_state_dict


def _get_module_by_name(model: nn.Module, module_name: str) -> nn.Module:
    """Get a module by its fully qualified name."""
    parts = module_name.split(".")
    module = model
    for part in parts:
        module = getattr(module, part)
    return module


def _set_module_by_name(model: nn.Module, module_name: str, new_module: nn.Module) -> None:
    """Replace a module by its fully qualified name."""
    parts = module_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def _has_regex_metacharacters(pattern: str) -> bool:
    """Check if a pattern contains regex metacharacters."""
    regex_metachars = {".", "*", "+", "?", "^", "$", "[", "]", "{", "}", "|", "(", ")", "\\"}
    return any(char in pattern for char in regex_metachars)


def _matches_pattern(name: str, pattern: str) -> bool:
    """Check if a name matches a pattern.

    For simple patterns (no regex metacharacters), checks if any component
    in the module path matches the pattern exactly. For regex patterns, uses
    re.search() to match anywhere in the name (mirroring PEFT behavior).

    This handles cases where Linear layers might be nested (e.g.,
    "model.layers.0.q_proj.linear") while still matching standard architectures
    where they're direct children (e.g., "model.layers.0.self_attn.q_proj").
    """
    if _has_regex_metacharacters(pattern):
        return re.search(pattern, name) is not None
    else:
        return pattern in name.split(".")


def _find_target_modules(model: nn.Module, target_patterns: List[str]) -> List[str]:
    """Find all module names that match any of the target patterns.

    Patterns can be simple module names (e.g., "q_proj") or regex patterns
    (e.g., r".*\\.q_proj$"). Simple names match any component in the module path.

    Supports both nn.Linear layers and GroupedExperts (MoE) modules.
    """
    target_modules = []

    for name, module in model.named_modules():
        # Check if module is Linear or GroupedExperts
        if not (isinstance(module, nn.Linear) or isinstance(module, GroupedExperts)):
            continue

        for pattern in target_patterns:
            if _matches_pattern(name, pattern):
                target_modules.append(name)
                break

    return target_modules


def _should_keep_trainable(param_name: str, modules_to_save_patterns: List[str]) -> bool:
    """Check if a parameter should remain fully trainable.

    Checks both the full parameter name and the parent module name against patterns.
    For example, for param "model.embed_tokens.weight", it checks both:
    - "model.embed_tokens.weight" (full parameter name)
    - "model.embed_tokens" (module name)

    Patterns can be simple module names (e.g., "embed_tokens") or regex patterns.
    """
    for pattern in modules_to_save_patterns:
        if _matches_pattern(param_name, pattern):
            return True

    module_name = param_name.rsplit(".", 1)[0] if "." in param_name else param_name
    for pattern in modules_to_save_patterns:
        if _matches_pattern(module_name, pattern):
            return True

    return False


def freeze_all_except_lora_and_specified(model: nn.Module, config: LoRAConfig) -> None:
    """
    Freeze all parameters except LoRA adapters and specified trainable modules.

    Uses named_modules() + direct params instead of named_parameters() to handle
    tied weights (e.g., lm_head.weight tied to embed_tokens.weight). named_parameters()
    deduplicates tied params and only returns one name, which may not match the
    modules_to_save pattern.

    Args:
        model: The model to freeze parameters in
        config: LoRA configuration with modules_to_save patterns
    """
    trainable_ids: set[int] = set()
    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            full_name = f"{module_name}.{param_name}" if module_name else param_name
            if any(lora_param in full_name for lora_param in ["lora_A", "lora_B"]):
                trainable_ids.add(id(param))
            elif _should_keep_trainable(full_name, config.modules_to_save):
                trainable_ids.add(id(param))

    for param in model.parameters():
        param.requires_grad = id(param) in trainable_ids


def apply_lora_to_model(model: nn.Module, config: LoRAConfig) -> None:
    """
    Apply LoRA to target modules in the model and freeze non-LoRA parameters.

    WARNING: This function modifies requires_grad on parameters. If using FSDP2,
    this MUST be called BEFORE setup_fsdp() to avoid dtensor/sharding issues.

    Args:
        model: The model to apply LoRA to
        config: LoRA configuration
    """
    logger = get_logger()
    n_loras = get_multi_run_manager().max_runs

    from torch.distributed.fsdp import FSDPModule

    if any(isinstance(m, FSDPModule) for m in model.modules()):
        logger.error(
            "Model is already wrapped with FSDP! LoRA must be applied BEFORE FSDP setup to avoid dtensor issues."
        )
        raise RuntimeError("Cannot apply LoRA to FSDP-wrapped model. Apply LoRA before setup_fsdp().")

    logger.debug(f"Applying LoRA to model: {model} for {config.target_modules}")
    target_modules = _find_target_modules(model, config.target_modules)
    logger.debug(
        f"Found {len(target_modules)} target modules for LoRA: {target_modules[:10]} ... {target_modules[-10:]}"
    )

    if not target_modules:
        logger.warning("No target modules found for LoRA. Check your target_modules patterns.")
        return

    for module_name in target_modules:
        base_module = _get_module_by_name(model, module_name)

        # Handle Linear layers
        if isinstance(base_module, nn.Linear):
            lora_module = MultiLoRALinear(
                base_layer=base_module,
                rank=config.rank,
                n_adapters=n_loras,
                alpha=config.alpha,
                dropout=config.dropout,
            )
        # Handle GroupedExperts (MoE)
        elif isinstance(base_module, GroupedExperts):
            lora_module = MultiLoRAGroupedExperts(
                base_layer=base_module,
                rank=config.rank,
                n_adapters=n_loras,
                alpha=config.alpha,
                dropout=config.dropout,
            )
        else:
            logger.warning(
                f"Module {module_name} is type {type(base_module).__name__}, "
                f"expected nn.Linear or GroupedExperts. Skipping."
            )
            continue

        lora_module.register_with_runs(get_multi_run_manager(), module_name)
        _set_module_by_name(model, module_name, lora_module)

    freeze_all_except_lora_and_specified(model, config)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    lora_adapter_params = 0
    lora_adapted_params = 0
    for name, module in model.named_modules():
        if isinstance(module, MultiLoRAModule):
            adapter_params, adapted_params = module.get_lora_param_counts()
            lora_adapter_params += adapter_params
            lora_adapted_params += adapted_params

    fully_trainable = trainable_params - lora_adapter_params
    adapted_or_trainable = lora_adapted_params + fully_trainable

    logger.info(f"LoRA enabled: {lora_adapter_params:,} adapter params adapting {lora_adapted_params:,} base params")
    logger.info(f"LoRA: {fully_trainable:,} fully trainable parameters")
    logger.info(f"LoRA: {adapted_or_trainable:,} adapted or fully trainable out of {total_params:,} parameters")


def has_lora_layers(model: nn.Module) -> bool:
    """Check if model has LoRA layers."""
    for module in model.modules():
        if isinstance(module, MultiLoRAModule):
            return True
    return False


def clean_lora_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remove LoRA parameters and fix LoRA base layer key names for HF compatibility."""
    clean_state_dict = {}

    for key, value in state_dict.items():
        if "lora_A" in key or "lora_B" in key:
            continue

        if ".base_layer." in key:
            new_key = key.replace(".base_layer.", ".")
            clean_state_dict[new_key] = value
        else:
            clean_state_dict[key] = value

    return clean_state_dict


_MOE_PROJ_TO_WEIGHT = {"gate_proj": "w1", "down_proj": "w2", "up_proj": "w3"}


def merge_lora_into_state_dict(
    base_state_dict: dict[str, torch.Tensor],
    lora_state_dict: dict[str, torch.Tensor],
    scaling: float,
) -> dict[str, torch.Tensor]:
    """Merge per-run LoRA adapter weights into base model state dict.

    Computes merged_weight = base_weight + scaling * (lora_B @ lora_A) for each
    LoRA-adapted layer. Supports both Linear and MoE (GroupedExperts) modules.

    Args:
        base_state_dict: Clean base model state dict (no LoRA keys).
            Modified in-place — caller should save originals for reverting.
        lora_state_dict: Per-run adapter state dict from get_state_dict_for_run().
        scaling: LoRA scaling factor (alpha / rank).

    Returns:
        Dict mapping modified keys to their original tensors (for reverting after save).
    """
    originals: dict[str, torch.Tensor] = {}

    # Group LoRA keys by (prefix, lora_type) to find A/B pairs
    lora_pairs: dict[str, dict[str, torch.Tensor]] = {}  # prefix -> {"lora_A": tensor, "lora_B": tensor}
    moe_lora: dict[
        str, dict[int, dict[str, dict[str, torch.Tensor]]]
    ] = {}  # prefix -> {expert_id -> {proj -> {"lora_A": tensor, "lora_B": tensor}}}

    for key, value in lora_state_dict.items():
        # Linear LoRA: {prefix}.lora_A.weight / {prefix}.lora_B.weight
        if key.endswith(".lora_A.weight"):
            prefix = key[: -len(".lora_A.weight")]
            lora_pairs.setdefault(prefix, {})["lora_A"] = value
        elif key.endswith(".lora_B.weight"):
            prefix = key[: -len(".lora_B.weight")]
            lora_pairs.setdefault(prefix, {})["lora_B"] = value
        else:
            # MoE LoRA: {prefix}.{expert_id}.{proj}.lora_A.weight
            for proj_name in _MOE_PROJ_TO_WEIGHT:
                if f".{proj_name}.lora_A.weight" in key:
                    # Extract: everything before .{expert_id}.{proj}.lora_A.weight
                    suffix = f".{proj_name}.lora_A.weight"
                    before_proj = key[: key.index(suffix)]
                    # before_proj = {prefix}.{expert_id}
                    dot_idx = before_proj.rfind(".")
                    prefix = before_proj[:dot_idx]
                    expert_id = int(before_proj[dot_idx + 1 :])
                    moe_lora.setdefault(prefix, {}).setdefault(expert_id, {}).setdefault(proj_name, {})["lora_A"] = (
                        value
                    )
                    break
                elif f".{proj_name}.lora_B.weight" in key:
                    suffix = f".{proj_name}.lora_B.weight"
                    before_proj = key[: key.index(suffix)]
                    dot_idx = before_proj.rfind(".")
                    prefix = before_proj[:dot_idx]
                    expert_id = int(before_proj[dot_idx + 1 :])
                    moe_lora.setdefault(prefix, {}).setdefault(expert_id, {}).setdefault(proj_name, {})["lora_B"] = (
                        value
                    )
                    break

    # Merge Linear LoRA pairs
    for prefix, pair in lora_pairs.items():
        base_key = f"{prefix}.weight"
        if base_key not in base_state_dict:
            continue
        lora_A = pair["lora_A"]  # [rank, in_features]
        lora_B = pair["lora_B"]  # [out_features, rank]
        originals[base_key] = base_state_dict[base_key]
        base_state_dict[base_key] = base_state_dict[base_key] + scaling * (lora_B @ lora_A)

    # Merge MoE LoRA
    for prefix, experts in moe_lora.items():
        for proj_name, weight_name in _MOE_PROJ_TO_WEIGHT.items():
            base_key = f"{prefix}.{weight_name}"
            if base_key not in base_state_dict:
                continue
            if base_key not in originals:
                originals[base_key] = base_state_dict[base_key].clone()
            for expert_id, proj_data in experts.items():
                if proj_name not in proj_data:
                    continue
                lora_A = proj_data[proj_name]["lora_A"]  # [rank, dim]
                lora_B = proj_data[proj_name]["lora_B"]  # [hidden_dim, rank]
                base_state_dict[base_key][expert_id] += scaling * (lora_B @ lora_A)

    return originals


def save_lora_config(config: LoRAConfig, model: nn.Module, save_path) -> None:
    """
    Save LoRA configuration as JSON for adapter portability.

    Args:
        config: LoRA configuration to save
        model: Model with LoRA layers to introspect
        save_path: Path object or string pointing to directory where adapter_config.json will be saved
    """
    import json
    from pathlib import Path

    save_path = Path(save_path)

    # Extract actual target modules from the model
    target_modules = set()

    for name, module in model.named_modules():
        if isinstance(module, MultiLoRAModule):
            module_suffix = name.split(".")[-1]
            target_modules.add(module_suffix)

    adapter_config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": model.config._name_or_path,
        "r": config.rank,
        "lora_alpha": config.alpha,
        "lora_dropout": config.dropout,
        "bias": "none",
        "target_modules": sorted(list(target_modules)),
    }

    config_path = save_path / "adapter_config.json"
    with open(config_path, "w") as f:
        json.dump(adapter_config, f, indent=2)
