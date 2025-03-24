#!/bin/bash
# Define network parameters
NETWORK_NAME="fl_network"
SUBNET="10.10.0.0/24"

# Remove existing network (if any) and create a new one
echo "Creating Docker network: $NETWORK_NAME..."
docker network rm $NETWORK_NAME >/dev/null 2>&1
docker network create --subnet=$SUBNET $NETWORK_NAME

# Run each container separately with a unique static IP
echo "Starting containers for DECENTRALIZED FL..."


docker run -d --rm \
    --name client_1 \
    --network $NETWORK_NAME \
    --ip 10.10.0.31 \
    -e SETUP=DCENT \
    -e MODE=CLIENT \
    -e ID=1 \
    -e TRAINING_CONFIG="/app/training_config.json" \
    -e ALL_CLIENT_IPS="/app/all_clients_ips.json" \
    -v $(pwd)/training_config_decent.json:/app/training_config.json \
    -v $(pwd)/all_clients_ips_decent.json:/app/all_clients_ips.json \
    -v $(pwd)/DECENT/logs:/app/logs \
    -v $(pwd)/DECENT/Graphs:/app/decentralized-api-docker/Graphs \
    -v $(pwd)/DECENT/received_weights_from_other_clients:/app/decentralized-api-docker/received_weights_from_other_clients \
    -v $(pwd)/DECENT/checkpoints:/app/decentralized-api-docker/checkpoints \
    -v $(pwd)/DECENT/results:/app/decentralized-api-docker/results \
    fl_api_image 
#-v $(pwd)/DECENT/datasets:/app/decentralized-api-docker/datasets \   
echo "Client 1 container started at IP: 10.10.0.31."

docker run -d --rm \
    --name client_2 \
    --network $NETWORK_NAME \
    --ip 10.10.0.32 \
    -e SETUP=DCENT \
    -e MODE=CLIENT \
    -e ID=2 \
    -e TRAINING_CONFIG="/app/training_config.json" \
    -e ALL_CLIENT_IPS="/app/all_clients_ips.json" \
    -v $(pwd)/training_config_decent.json:/app/training_config.json \
    -v $(pwd)/all_clients_ips_decent.json:/app/all_clients_ips.json \
    -v $(pwd)/DECENT/logs:/app/logs \
    -v $(pwd)/DECENT/Graphs:/app/decentralized-api-docker/Graphs \
    -v $(pwd)/DECENT/received_weights_from_other_clients:/app/decentralized-api-docker/received_weights_from_other_clients \
    -v $(pwd)/DECENT/checkpoints:/app/decentralized-api-docker/checkpoints \
    -v $(pwd)/DECENT/results:/app/decentralized-api-docker/results \
    fl_api_image
#-v $(pwd)/DECENT/datasets:/app/decentralized-api-docker/datasets \
echo "Client 2 container started at IP: 10.10.0.32."

docker run -d --rm \
    --name client_3 \
    --network $NETWORK_NAME \
    --ip 10.10.0.33 \
    -e SETUP=DCENT \
    -e MODE=CLIENT \
    -e ID=3 \
    -e TRAINING_CONFIG="/app/training_config.json" \
    -e ALL_CLIENT_IPS="/app/all_clients_ips.json" \
    -v $(pwd)/training_config_decent.json:/app/training_config.json \
    -v $(pwd)/all_clients_ips_decent.json:/app/all_clients_ips.json \
    -v $(pwd)/DECENT/logs:/app/logs \
    -v $(pwd)/DECENT/Graphs:/app/decentralized-api-docker/Graphs \
    -v $(pwd)/DECENT/received_weights_from_other_clients:/app/decentralized-api-docker/received_weights_from_other_clients \
    -v $(pwd)/DECENT/checkpoints:/app/decentralized-api-docker/checkpoints \
    -v $(pwd)/DECENT/results:/app/decentralized-api-docker/results \
    fl_api_image
#-v $(pwd)/DECENT/datasets:/app/decentralized-api-docker/datasets \
echo "Client 3 container started at IP: 10.10.0.33."

docker run -d --rm \
    --name client_4 \
    --network $NETWORK_NAME \
    --ip 10.10.0.34 \
    -e SETUP=DCENT \
    -e MODE=CLIENT \
    -e ID=4 \
    -e TRAINING_CONFIG="/app/training_config.json" \
    -e ALL_CLIENT_IPS="/app/all_clients_ips.json" \
    -v $(pwd)/training_config_decent.json:/app/training_config.json \
    -v $(pwd)/all_clients_ips_decent.json:/app/all_clients_ips.json \
    -v $(pwd)/DECENT/logs:/app/logs \
    -v $(pwd)/DECENT/Graphs:/app/decentralized-api-docker/Graphs \
    -v $(pwd)/DECENT/received_weights_from_other_clients:/app/decentralized-api-docker/received_weights_from_other_clients \
    -v $(pwd)/DECENT/checkpoints:/app/decentralized-api-docker/checkpoints \
    -v $(pwd)/DECENT/results:/app/decentralized-api-docker/results \
    fl_api_image
#-v $(pwd)/DECENT/datasets:/app/decentralized-api-docker/datasets \
echo "Client 4 container started at IP: 10.10.0.34."
echo "All containers started."

