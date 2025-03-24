#!/bin/bash
# Define network parameters
NETWORK_NAME="fl_network"
SUBNET="10.10.0.0/24"

# Remove existing network (if any) and create a new one
echo "Creating Docker network: $NETWORK_NAME..."
docker network rm $NETWORK_NAME >/dev/null 2>&1
docker network create --subnet=$SUBNET $NETWORK_NAME

# Run each container separately with a unique static IP
echo "Starting containers for CENTRALIZED FL..."

docker run -d --rm \
    --name server \
    --network $NETWORK_NAME \
    --ip 10.10.0.10 \
    -e SETUP=CENT \
    -e MODE=SERVER \
    -e TRAINING_CONFIG="/app/training_config.json" \
    -e ALL_CLIENT_IPS="/app/all_clients_ips.json" \
    -v $(pwd)/training_config_cent.json:/app/training_config.json \
    -v $(pwd)/all_clients_ips_cent.json:/app/all_clients_ips.json \
    -v $(pwd)/CENT/SERVER/logs:/app/logs \
    -v $(pwd)/CENT/SERVER/received_model_weights_from_all_clients:/app/centralized-api-docker/received_model_weights_from_all_clients \
    -v $(pwd)/CENT/SERVER/aggregated_weights:/app/centralized-api-docker/aggregated_weights \
    fl_api_image
echo "Server container started at IP: 10.10.0.10."

docker run -d --rm \
    --name client_1 \
    --network $NETWORK_NAME \
    --ip 10.10.0.11 \
    -e SETUP=CENT \
    -e MODE=CLIENT \
    -e ID=1 \
    -e TRAINING_CONFIG="/app/training_config.json" \
    -e ALL_CLIENT_IPS="/app/all_clients_ips.json" \
    -v $(pwd)/training_config_cent.json:/app/training_config.json \
    -v $(pwd)/all_clients_ips_cent.json:/app/all_clients_ips.json \
    -v $(pwd)/CENT/logs:/app/logs \
    -v $(pwd)/CENT/CLIENT/Graphs:/app/centralized-api-docker/Graphs \
    -v $(pwd)/CENT/CLIENT/received_weights_from_server:/app/centralized-api-docker/received_weights_from_server \
    -v $(pwd)/CENT/CLIENT/checkpoints:/app/centralized-api-docker/checkpoints \
    -v $(pwd)/CENT/CLIENT/results:/app/centralized-api-docker/results \
    fl_api_image
#-v $(pwd)/CENT/CLIENT/datasets:/app/centralized-api-docker/datasets \
echo "Client 1 container started at IP: 10.10.0.11."

docker run -d --rm \
    --name client_2 \
    --network $NETWORK_NAME \
    --ip 10.10.0.12 \
    -e SETUP=CENT \
    -e MODE=CLIENT \
    -e ID=2 \
    -e TRAINING_CONFIG="/app/training_config.json" \
    -e ALL_CLIENT_IPS="/app/all_clients_ips.json" \
    -v $(pwd)/training_config_cent.json:/app/training_config.json \
    -v $(pwd)/all_clients_ips_cent.json:/app/all_clients_ips.json \
    -v $(pwd)/CENT/logs:/app/logs \
    -v $(pwd)/CENT/CLIENT/Graphs:/app/centralized-api-docker/Graphs \
    -v $(pwd)/CENT/CLIENT/received_weights_from_server:/app/centralized-api-docker/received_weights_from_server \
    -v $(pwd)/CENT/CLIENT/checkpoints:/app/centralized-api-docker/checkpoints \
    -v $(pwd)/CENT/CLIENT/results:/app/centralized-api-docker/results \
    fl_api_image
#-v $(pwd)/CENT/CLIENT/datasets:/app/centralized-api-docker/datasets \
echo "Client 2 container started at IP: 10.10.0.12."

docker run -d --rm \
    --name client_3 \
    --network $NETWORK_NAME \
    --ip 10.10.0.13 \
    -e SETUP=CENT \
    -e MODE=CLIENT \
    -e ID=3 \
    -e TRAINING_CONFIG="/app/training_config.json" \
    -e ALL_CLIENT_IPS="/app/all_clients_ips.json" \
    -v $(pwd)/training_config_cent.json:/app/training_config.json \
    -v $(pwd)/all_clients_ips_cent.json:/app/all_clients_ips.json \
    -v $(pwd)/CENT/logs:/app/logs \
    -v $(pwd)/CENT/CLIENT/Graphs:/app/centralized-api-docker/Graphs \
    -v $(pwd)/CENT/CLIENT/received_weights_from_server:/app/centralized-api-docker/received_weights_from_server \
    -v $(pwd)/CENT/CLIENT/checkpoints:/app/centralized-api-docker/checkpoints \
    -v $(pwd)/CENT/CLIENT/results:/app/centralized-api-docker/results \
    fl_api_image
#-v $(pwd)/CENT/CLIENT/datasets:/app/centralized-api-docker/datasets \
echo "Client 3 container started at IP: 10.10.0.13."

docker run -d --rm \
    --name client_4 \
    --network $NETWORK_NAME \
    --ip 10.10.0.14 \
    -e SETUP=CENT \
    -e MODE=CLIENT \
    -e ID=4 \
    -e TRAINING_CONFIG="/app/training_config.json" \
    -e ALL_CLIENT_IPS="/app/all_clients_ips.json" \
    -v $(pwd)/training_config_cent.json:/app/training_config.json \
    -v $(pwd)/all_clients_ips_cent.json:/app/all_clients_ips.json \
    -v $(pwd)/CENT/logs:/app/logs \
    -v $(pwd)/CENT/CLIENT/Graphs:/app/centralized-api-docker/Graphs \
    -v $(pwd)/CENT/CLIENT/received_weights_from_server:/app/centralized-api-docker/received_weights_from_server \
    -v $(pwd)/CENT/CLIENT/checkpoints:/app/centralized-api-docker/checkpoints \
    -v $(pwd)/CENT/CLIENT/results:/app/centralized-api-docker/results \
    fl_api_image
#-v $(pwd)/CENT/CLIENT/datasets:/app/centralized-api-docker/datasets \
echo "Client 4 container started at IP: 10.10.0.14."
echo "All containers started."

