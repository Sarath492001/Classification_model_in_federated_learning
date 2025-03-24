#!/bin/bash

# Activate the virtual environment
source fl_api/bin/activate

# Log file directory
LOG_DIR="/app/logs"
#mkdir -p $LOG_DIR
mkdir -p $LOG_DIR && chmod -R 777 $LOG_DIR

# Log file path (use ID for client logs)
LOG_FILE="$LOG_DIR/${SETUP}_${MODE}_${ID}.log"

# Check if both JSON files exist
if [[ ! -f "$TRAINING_CONFIG" ]]; then
    echo "Error: Training config file '$TRAINING_CONFIG' not found!"
    exit 1
fi

if [[ ! -f "$ALL_CLIENT_IPS" ]]; then
    echo "Error: Client IPs file '$ALL_CLIENT_IPS' not found!"
    exit 1
fi

# Print the arguments (also save to log)
echo "SETUP: $SETUP" | tee -a "$LOG_FILE"
echo "MODE: $MODE" | tee -a "$LOG_FILE"
echo "ID: $ID" | tee -a "$LOG_FILE"
echo "TRAINING_CONFIG: $TRAINING_CONFIG" | tee -a "$LOG_FILE"
echo "ALL_CLIENT_IPS: $ALL_CLIENT_IPS" | tee -a "$LOG_FILE"
echo "Logs will be saved to $LOG_FILE"

# Run the appropriate script with `exec`
if [[ "$SETUP" == "CENT" && "$MODE" == "SERVER" ]]; then
    cd centralized-api-docker
    exec python3 server0.py --config "$TRAINING_CONFIG" --clients_ips "$ALL_CLIENT_IPS" 2>&1 | tee -a "$LOG_FILE"

elif [[ "$SETUP" == "CENT" && "$MODE" == "CLIENT" ]]; then
    cd centralized-api-docker
    python3 CL1.py --client_id "$ID" --config "$TRAINING_CONFIG" --clients_ips "$ALL_CLIENT_IPS" 2>&1 | tee -a "$LOG_FILE"
    STATUS=$?

    if [[ $STATUS -eq 0 ]]; then
        echo "Training completed successfully. Now generating graphs..." | tee -a "$LOG_FILE"
        exec python3 generate_graphs.py --client_id "$ID" --config "$TRAINING_CONFIG" 2>&1 | tee -a "$LOG_FILE"
    else
        echo "Training failed. Skipping graph generation." | tee -a "$LOG_FILE"
        exit $STATUS
    fi

elif [[ "$SETUP" == "DCENT" && "$MODE" == "CLIENT" ]]; then
    cd decentralized-api-docker
    python3 CL1.py --client_id "$ID" --config "$TRAINING_CONFIG" --clients_ips "$ALL_CLIENT_IPS" 2>&1 | tee -a "$LOG_FILE"
    STATUS=$?

    if [[ $STATUS -eq 0 ]]; then
        echo "Training completed successfully. Now generating graphs..." | tee -a "$LOG_FILE"
        exec python3 generate_graphs.py --client_id "$ID" --config "$TRAINING_CONFIG" 2>&1 | tee -a "$LOG_FILE"
    else
        echo "Training failed. Skipping graph generation." | tee -a "$LOG_FILE"
        exit $STATUS
    fi

else
    echo "Invalid setup or mode selected." | tee -a "$LOG_FILE"
    exit 1
fi





















