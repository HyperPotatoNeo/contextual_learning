#!/bin/bash
# Monitor context distillation experiments every hour
# Logs to outputs/monitoring.log

LOG_FILE="/home/user/contextual_learning/outputs/monitoring.log"

while true; do
    echo "========================================" >> $LOG_FILE
    echo "Monitoring at $(date)" >> $LOG_FILE
    echo "========================================" >> $LOG_FILE

    # Check if processes are running
    echo "" >> $LOG_FILE
    echo "=== Process Status ===" >> $LOG_FILE
    ps aux | grep -E "rl.*equal_weights|rl.*low_entropy" | grep -v grep >> $LOG_FILE 2>&1 || echo "No experiment processes found" >> $LOG_FILE

    # GPU utilization
    echo "" >> $LOG_FILE
    echo "=== GPU Status ===" >> $LOG_FILE
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv >> $LOG_FILE

    # Job 1: Equal weights - last training steps
    echo "" >> $LOG_FILE
    echo "=== Job 1 (Equal Weights) - Recent Progress ===" >> $LOG_FILE
    grep -E "Step [0-9]+" /home/user/contextual_learning/outputs/cd_equal.log 2>/dev/null | tail -5 >> $LOG_FILE || echo "No progress yet" >> $LOG_FILE

    # Job 2: Low entropy - last training steps
    echo "" >> $LOG_FILE
    echo "=== Job 2 (Low Entropy) - Recent Progress ===" >> $LOG_FILE
    grep -E "Step [0-9]+" /home/user/contextual_learning/outputs/cd_low_entropy.log 2>/dev/null | tail -5 >> $LOG_FILE || echo "No progress yet" >> $LOG_FILE

    echo "" >> $LOG_FILE
    echo "" >> $LOG_FILE

    # Sleep 1 hour
    sleep 3600
done
