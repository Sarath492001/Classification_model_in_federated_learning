from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
import requests
import threading
import time
from trainer import train_avg_push, train_proxy_push
import os
from typing import Tuple
import json
import argparse

# Clear any existing proxy settings in the environment
[os.environ.pop(proxy, None) for proxy in ['HTTP_PROXY', 'HTTPS_PROXY', 'SOCKS_PROXY', 'http_proxy', 'https_proxy', 'socks_proxy', 'all_proxy', 'ALL_PROXY']]

# Initialize FastAPI application
app = FastAPI()

# Event to signal shutdown
shutdown_event = threading.Event()

# Client configuration model using Pydantic for validation
class ClientConfig(BaseModel):

    # DATA AND RESULT PATH
    data_path: str = "./datasets"    
    result_path: str = "./results" 
          
    # TRAINING PARAMETERS 
    dataset: str = "face_images_2"               #"Face_images" - face images in npz format | "medical_images_2" - 2 class cancer cell images in folder format | "face_images_2" - face images in folder format
    image_size: Tuple[int, int] = (128, 128)               # target image size should be used for training  (BASED ON THE IMAGE SIZE YOU HAVE TO CHANGE THE MODEL PROCESS THE INPUT IMAGE SIZE)
    train_split: float = 0.8                     # seperating dataset for train and test data for evaluation 0.8 - 80%:20%
    algorithm: str = "FedAvg"                    # Decentralized - "AvgPush" - Single model(proxy) | "ProxyFL" - Two model(proxy & private) |  Centralized - "FML" - Two model(proxy & private) | "FedAvg" - Single model(proxy)
    private_model_type: str = "CNN2"             #'LeNet5' | 'MLP' | 'CNN1' - 4 layers | 'CNN2' - original 2 layers | 'VisionTransformer' - 12 layers/customisable | 'VisionTransformerlora' - 12 layers/customisable | 'CNN1lora' - 4 layers | "CNN1_2layers" - 2 layers CNN1 
    proxy_model_type: str = "CNN2"               #'LeNet5' | 'MLP' | 'CNN1' - 4 layers | 'CNN2' - original 2 layers | 'VisionTransformer' - 12 layers/customisable | 'VisionTransformerlora' - 12 layers/customisable | 'CNN1lora' - 4 layers | "CNN1_2layers" - 2 layers CNN1 
    n_client_data: int = 1500                    # no.of images for this client from dataset        
    lr: float = 0.001                            # 0.001 - face images | 0.00001 - medical images | MODIFY AS PER YOUR USE CASE 
    n_epochs: int = 1                            # no of epochs 1 - 2000
    n_rounds: int = 3                            # no of communication rounds 1 - 500   
    batch_size: int = 250                        # keep less for vit and more layers cnn models
    in_channel: int = 3                          # image channels 3 - RGB | 1 - grayscale
    n_class: int = 0                             #  Automatically selected
    lora_rank: int = 8                           # Lora/DyLora rank 4 - 16 | install lora or dylora - check installation guide from github gor Dylora 
    dml_weight: float = 0.5                      # controls the balance between two loss components (CE loss) & (KL loss). A value of 0.5 gives equal importance to both losses. 
   
    # DP PARAMETERS 
    use_private_SGD: int = 0                     # 0 - without DP | 1 - with DP 
    optimizer: str = 'adam'                      # "adam" | "sgd" 
    accountant: str = 'rdp'                      # DP parameters
    secure_mode: bool = False                    # DP parameters
    noise_multiplier: float = 1.0                # DP parameters 
    l2_norm_clip: float = 1.0                    # DP parameters 
    dp_optimizer_selection: str = "new"          # opacus version old - 0.14.0 | new - 1.5.2
    device: int = 0                              # cpu or gpu selection ( Automatically selected )
    momentum: float = 0.9                        # Momentum factor for the SGD optimizer; helps accelerate gradient updates by smoothing and reducing oscillations

    # OTHERS
    verbose: int = 1                             # for logging 
    port: int = 8001                             # port number for server & clients to communicate | USE 8000 - 8999  | should be same for clients and server 
    max_retries: int = 10                        # Maximum number of times to retry communication in case of failure
    retry_delay: int = 10                        # Time (in seconds) to wait before retrying after a failure
    
    # DATA & MODEL POISONING 
    use_data_poisoning: int = 0                  # 0 - no data poisoning | 1 - use data poisoning
    attack_type: str = ""                        # Model poisoning attacks: 1) "combined" - adaptive_backdoor + gradient_attack + model weights attack + model layers attack (All the attacks below)
                                                                          # 2) "adaptive_backdoor" - Dynamically Applies an adaptive backdoor attack by combining multiple techniques.
                                                                          # 3) "gradient_attack" - Modify gradients for poisoning attack, with and without DP two methods avaliable to stay within DP range  
                                                                          # 4) "model_weights_attack" - Modify model weights after optimization using scale factor, customize the that value in the trainer.py file
                                                                          # 5) "model_layers_attack" - Injects a backdoor by shifting classification layer weights.
                                                                          # 6) "" - dont want to use any model poisoning attacks 

# Parse command-line arguments
#parser = argparse.ArgumentParser(description="Client configuration")
#arser.add_argument("--client_id", type=int, required=True, help="Client ID for this instance")
#args = parser.parse_args()
#client_id = args.client_id  # Get client ID from args
#client_id = 1                                    # choose the id of this client 

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True, help="Path to config file")
parser.add_argument("--client_id", type=int, required=True, help="Client ID")
parser.add_argument("--clients_ips", type=str, required=True, help="Path to all_clients_ips.json")
args = parser.parse_args() 
client_id = args.client_id                                 
config_file = args.config 
clients_ips_file = args.clients_ips
## Load JSON file with error handling
try:
    with open(config_file, "r") as file:
        config_data = json.load(file)
        print(f"Loaded config from: {config_file}")
except FileNotFoundError:
    print(f"Error: {config_file} not found. Using default configuration within the client script.")
    #config_data = {}
    exit(1)

# All client machines and server should be in the same network, define all the clients ip for quick id selection 
# Load clients config from JSON file
try:
    with open(clients_ips_file, "r") as file:
        data = json.load(file)  # Loads as a dictionary
        # Extract server info
        server_id = data["server"]["id"]
        server_ip = data["server"]["ip"]
        # Extract client info and convert keys to integers
        clients = {int(k): v for k, v in data["clients"].items()}  # Convert keys to integers
        print(f"Loaded {clients_ips_file} successfully.")
except FileNotFoundError:
    print(f"Error: {clients_ips_file} not found.")
    exit(1)
# List to store received messages and a lock for thread safety
received_messages = []
received_messages_lock = threading.Lock()

# Message model for incoming messages
class Message(BaseModel):
    sender_ip: str
    sender_id: int
    round_number: int
    
@app.post("/send_message")
async def receive_message(
    sender_ip: str = Form(...),
    sender_id: int = Form(...),
    round_number: int = Form(...),
    file: UploadFile = File(...)
):
    """
    Endpoint to receive model weights as a .pth file and associated metadata.
    """
    # Use Pydantic for structured metadata if you need additional validation
    message = Message(sender_ip=sender_ip, sender_id=sender_id, round_number=round_number)
    # Parent directory for received weights
    parent_dir = "received_weights_from_server"
    # Directory to save received model weights
    received_weights_dir = os.path.join(parent_dir, f"received_model_weights_from_server_CL-{client_id}")
    os.makedirs(received_weights_dir, exist_ok=True)

    # Save the model weights file
    file_path = os.path.join(received_weights_dir, f"received_model_weights_round_{message.round_number}_server.pth")
    with open(file_path, "wb") as f:
        f.write(await file.read())

    print(f"server {message.sender_id} ({message.sender_ip}) sent model weights for round {message.round_number}")
    # Lock to safely append received messages
    with received_messages_lock:
        received_messages.append({"metadata": message, "file_path": file_path})

    return {"status": "Model weights received successfully"}

def save_model_weights_as_pth(model_weights, filepath):  
    """
    Save model weights into a subdirectory under the 'checkpoints' directory
    based on the algorithm specified in the config.
    """
    # Parent directory for checkpoints
    parent_dir = "checkpoints"
    checkpoints_dir = os.path.join(parent_dir, f"checkpoints-CL-{client_id}")
    # Determine the subdirectory based on the algorithm
    if config.algorithm == 'FedAvg':
        sub_dir = os.path.join(checkpoints_dir, "FedAvg")        
    else:
        sub_dir = os.path.join(checkpoints_dir, "FML")

    # Create the subdirectory if it doesn't exist
    os.makedirs(sub_dir, exist_ok=True)

    # Update filepath to save in the checkpoints directory
    filepath = os.path.join(sub_dir, filepath)

    # Save the model weights
    torch.save(model_weights, filepath)
    #print("Model weights saved in checkpoints directory")

def load_and_append_results(result_path, client_id, start_round, total_rounds, new_results):
    """
    Loads existing results from a .npz file, appends new results, and saves them back to the file.
    Saves `private_accuracies` only if the algorithm is 'ProxyFL'.
    """
    # Construct the file path for the existing results
    results_file_path = os.path.join(result_path, f"Client_{client_id}_results.npz")
    
    # Load the existing results
    existing_results = np.load(results_file_path)
    
    # Retrieve existing results arrays (they should be stored as arrays of the same shape)
    existing_proxy_accuracies = existing_results["proxy_accuracies"]
    existing_privacy_budgets = existing_results["privacy_budgets"]
    existing_training_times = existing_results["training_times"]
    existing_communication_times = existing_results["communication_times"]
    
    # Append the new results to the existing arrays
    proxy_accuracies = np.append(existing_proxy_accuracies, new_results["proxy_accuracies"])
    privacy_budgets = np.append(existing_privacy_budgets, new_results["privacy_budgets"])
    training_times = np.append(existing_training_times, new_results["training_times"])
    communication_times = np.append(existing_communication_times, new_results["communication_times"])

    # Prepare dictionary to save
    results_dict = {
        "proxy_accuracies": proxy_accuracies,
        "privacy_budgets": privacy_budgets,
        "training_times": training_times,
        "communication_times": communication_times,
    }
    
    # Conditionally append private_accuracies based on algorithm
    if config.algorithm == 'FML':
        existing_private_accuracies = existing_results.get("private_accuracies", np.array([]))  # Optional field
        private_accuracies = np.append(existing_private_accuracies, new_results["private_accuracies"])
        results_dict["private_accuracies"] = private_accuracies
    
    # Save the updated results
    np.savez(results_file_path, **results_dict)   
    print(f"results appended and saved to {results_file_path} for rounds {start_round} to {total_rounds}")


#################################### Retry mechanism ###################################################################################
def send_message_to_peer(peer_ip, sender_id, sender_ip, round_number, model_weights):
    """
    Function to send model weights to a peer client with retry mechanism.
    """
    max_retries = config.max_retries
    retry_delay = config.retry_delay
    model_weights_path = f"model_weights_round_{round_number}_client_{sender_id}.pth"
    save_model_weights_as_pth(model_weights, model_weights_path)

    for attempt in range(1, max_retries + 1):
        try:
            url = f"http://{peer_ip}:{config.port}/send_message"
            payload = {
                "sender_ip": sender_ip,
                "sender_id": sender_id,
                "round_number": round_number  
            }
            parent_dir = "checkpoints"
            # Determine the subdirectory based on the algorithm
            if config.algorithm == 'FedAvg':
                sub_dir = os.path.join(parent_dir, f"checkpoints-CL-{client_id}", "FedAvg")
            else:
                sub_dir = os.path.join(parent_dir, f"checkpoints-CL-{client_id}", "FML")
            
            # Create the subdirectory path to the model weights
            model_weights_full_path = os.path.join(sub_dir, model_weights_path)

            with open(model_weights_full_path, "rb") as f:
                files = {"file": f}

                # Print the request payload and headers for debugging
                #print(f"Sending payload: {payload}")
                #print(f"Sending files: {files}")

                # Send the request to the peer client
                response = requests.post(url, data=payload, files=files)
            #response = requests.post(url, json=payload, proxies=proxies)
            if response.status_code == 200:
                json_response = response.json()
                print(f"Model weights sent to server({peer_ip}): {json_response}")
                break  # Exit the retry loop on success
            else:
                raise requests.exceptions.RequestException(f"Received status code {response.status_code}")
        
        except requests.exceptions.RequestException as e:
            peer_id = next((client_id for client_id, ip in clients.items() if ip == peer_ip), None)
            print()
            print(f"Attempt {attempt} failed to send model weights to server({peer_ip}): {e}")
            
            if attempt < max_retries:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)  # Wait before retrying
            else:
                print(f"Failed to send model weights to server({peer_ip}) after {max_retries} attempts.")
                shutdown_event.set()
#######################################################################################################################################

def wait_for_message(sender_id, round_number):
    """
    Function to wait for a message from a specific sender for a given round.
    It checks the received messages and returns the message when found.
    """
    while True:
        with received_messages_lock:
            for message in received_messages:
                if message["metadata"].sender_id == sender_id and message["metadata"].round_number == round_number:
                    return message
        time.sleep(1)  # Avoid busy-waiting

def start_sending_messages(client_id, my_ip, client, config, eval_data):
    
    parent_dir = "received_weights_from_server"
    checkpoints_dir = os.path.join(parent_dir, f"received_model_weights_from_server_CL-{client_id}")  # taking checkpoints from the aggregated weights from server 
    latest_round = 0
    latest_weights_path = None

    # Check for the latest checkpoint file
    if os.path.exists(checkpoints_dir):
        checkpoint_files = [f for f in os.listdir(checkpoints_dir) if f.endswith(".pth")]
        if checkpoint_files:
            # Extract the round numbers from the file names
            round_numbers = [int(f.split("_round_")[1].split("_")[0]) for f in checkpoint_files]
            latest_round = max(round_numbers)
            #latest_weights_path = os.path.join(checkpoints_dir, f"model_weights_round_{latest_round}_client_{client_id}.pth")
            latest_weights_path = os.path.join(checkpoints_dir, f"received_model_weights_round_{latest_round}_server.pth")

    # Load model weights if a checkpoint is found
    if latest_weights_path:
        print(f"Resuming from checkpoint: {latest_weights_path}")
        print()
        checkpoint = torch.load(latest_weights_path)
        client.proxy_model.load_state_dict(utils.convert_np_weights_to_tensor(checkpoint))
        start_round = latest_round + 1
    else:
        print(f"No checkpoint found in {checkpoints_dir}. Starting from round 1.")
        print()
        start_round = 1

    total_rounds = config.n_rounds

    # Arrays to store results for each round
    private_accuracies = np.empty(config.n_rounds, dtype=np.float32)
    proxy_accuracies = np.empty(config.n_rounds, dtype=np.float32)
    privacy_budgets = np.empty(config.n_rounds, dtype=np.float32)
    training_times = np.empty(config.n_rounds, dtype=np.float32)
    communication_times = np.empty(config.n_rounds, dtype=np.float32)  # To store communication times

    send_times = []  # List to store send times for all rounds
    receive_times = []  # List to store receive times for all rounds
    
    for round_number in range(start_round, total_rounds + 1):
        # Train the model based on the selected algorithm
        if config.algorithm == 'FML':
            send_model_weights, private_accuracy, proxy_accuracy, privacy_budget, training_time = train_proxy_push(client, eval_data, logger, config)
        elif config.algorithm == 'FedAvg':
            send_model_weights, proxy_accuracy, privacy_budget, training_time = train_avg_push(client, eval_data, logger, config) 
        else:
            raise ValueError("Unknown training method")
        
        # Store the results for this round
        private_accuracies[round_number - 1] = private_accuracy if config.algorithm == 'FML' else np.nan
        proxy_accuracies[round_number - 1] = proxy_accuracy
        privacy_budgets[round_number - 1] = privacy_budget
        training_times[round_number - 1] = training_time

        # Shared list to store results for communication time
        send_time = 0  # Initialize for this round
        receive_time = 0  # Initialize for this round

        # Function to send model weights
        def send_weights():
            nonlocal send_time  # Use the send_time from the enclosing function
            start_send_time = time.time()  # Start time for sending
            recipient_ip = server_ip
            #print(f"Sending model weights to server({recipient_ip})...")
            send_message_to_peer(recipient_ip, client_id, my_ip, round_number, send_model_weights)
            end_send_time = time.time()  # End time for sending
            send_time = end_send_time - start_send_time  # Return the sending time

        # Function to receive model weights
        def receive_weights():
            nonlocal receive_time  # Use the receive_time from the enclosing function
            start_receive_time = time.time()  # Start time for receiving
            sender_id = server_id 
            #print(f"Waiting for model weights from server for round {round_number}...")
            message = wait_for_message(sender_id, round_number)
            #print(f"Received model weights from server for round {round_number}.")
            #print(f"Loading the received model weights into the model for round {round_number}")
            received_weights = torch.load(message["file_path"])
            end_receive_time = time.time()  # End time for receiving
            client.proxy_model.load_state_dict(utils.convert_np_weights_to_tensor(received_weights))
            print(f"Model weights successfully loaded from server for round {round_number}")
            receive_time = end_receive_time - start_receive_time  # Return the receiving time

        print()
        print(f"Starting Round {round_number}")
        time.sleep(5)  # Wait before starting

        # Start threads for sending and receiving
        send_thread = threading.Thread(target=send_weights)
        receive_thread = threading.Thread(target=receive_weights)

        send_thread.start()
        receive_thread.start()

        # Wait for both threads to complete
        send_thread.join()
        receive_thread.join()

        # Append the times for this round to the lists
        send_times.append(send_time)
        receive_times.append(receive_time)

        # Calculate total communication time for this round
        communication_time = send_time + receive_time
        communication_times[round_number - 1] = communication_time


        print(f"Round {round_number} completed")
        print()  # Add a blank line after round completion

        # Clear messages for this round
        with received_messages_lock:
            received_messages[:] = [m for m in received_messages if m["metadata"].round_number != round_number]
        
        time.sleep(5)  # Wait before starting the next round

    # After completing all rounds, store the final results 
    results = {
        "proxy_accuracies": proxy_accuracies,
        "privacy_budgets": privacy_budgets,
        "training_times": training_times,
        "communication_times": communication_times,
    }
    if config.algorithm == 'ProxyFL':
        results["private_accuracies"] = private_accuracies

    print()       
    print("All rounds completed") 

    # If we are resuming from a checkpoint, append the new results
    if latest_weights_path:
        load_and_append_results(result_path, client_id, start_round, total_rounds, results)
    
    else:
        # Save the results as a .npz file
        np.savez(os.path.join(result_path, f"Client_{client_id}_results.npz"), **results)
        print("Results saved")
    # Signal the end of execution
    print()
    print("Shutting down...")
    shutdown_event.set()

            
if __name__ == "__main__":
    import uvicorn
    import torch
    import os
    import utils
    import numpy as np
    from privacy_checker import check_privacy
    from client import Client
    import warnings

    # Suppress the specific FutureWarning from torch.load
    warnings.simplefilter("ignore", category=FutureWarning)
    warnings.simplefilter("ignore", category=UserWarning)
    #client_id = 1
    my_ip = clients[client_id]
    print()
    print(f"This is Client {client_id} and the IP: {my_ip}")
    print()
    # Initialize configuration
    config = ClientConfig(**config_data)

    # Configure device
    config.device = torch.device(f"cuda:{config.device}" if torch.cuda.is_available() else "cpu")
    config.data_path = f"./datasets/CL-{client_id}" 
    config.result_path = f"./results/results-CL{client_id}"    
    # Configure input channel
    config.in_channel = 3  # Assuming 3-channel images

    # Create result directory path
    result_path_components = [
        config.result_path,
        config.dataset,
        # f"train_split_{config.train_split}",
        # f"use_data_poisoning_{config.use_data_poisoning}",
        # f"attack_type_{config.attack_type}",
        # f"accountant_{config.accountant}",
        f"n_client_data_{config.n_client_data}",
        f"batch_size_{config.batch_size}",
        # f"optimizer_{config.optimizer}",
        f"lr_{config.lr}",
        f"use_private_SGD_{config.use_private_SGD}",
        f"dp_optimizer_selection_{config.dp_optimizer_selection}",
        # f"noise_multiplier_{config.noise_multiplier}",
        # f"l2_norm_clip_{config.l2_norm_clip}",
        f"private_model_type_{config.private_model_type}",
        f"proxy_model_type_{config.proxy_model_type}",
        # f"dml_weight_{config.dml_weight}",
        f"n_epochs_{config.n_epochs}",
        f"n_rounds_{config.n_rounds}",
        config.algorithm
    ]

    # Include "use_data_poisoning" only if enabled
    if config.use_data_poisoning == 1:
        result_path_components.insert(2, f"use_data_poisoning_{config.use_data_poisoning}")  # Insert early in the path
        insert_index = 3  # If use_data_poisoning is present, attack_type should be at index 3
    else:
        insert_index = 2  # Otherwise, attack_type should be at index 2

    # Include "attack_type" only if it is specified
    if config.attack_type:  # Ensures attack_type is not empty
        result_path_components.insert(insert_index, f"attack_type_{config.attack_type}")  # Insert early in the path

    # Include lora_rank only if private or proxy model type is cnnlora or vitlora
    if config.private_model_type in ["CNN1lora", "VisionTransformerlora"] or config.proxy_model_type in ["CNN1lora", "VisionTransformerlora"]:
        result_path_components.insert(-3, f"lora_rank{config.lora_rank}")  # Insert before n_epochs     # index 9 0r -3 

    result_path = os.path.join(*result_path_components)
    os.makedirs(result_path, exist_ok=True)

    # Data preparation based on the selected dataset
    if config.dataset == "Face_images":
        print("Using Face Images npz File type")
        train_X, train_y, test_X, test_y = utils.get_data_npz(config)
    elif config.dataset == "medical_images_2":
        print("Using Medical Images Folder type")
        train_X, train_y, test_X, test_y = utils.get_data_folder(config)
    elif config.dataset == "face_images_2":
        print("Using Face Images Folder type")
        train_X, train_y, test_X, test_y = utils.get_data_folder(config)    
    else:
        raise ValueError("Invalid choice for dataset.")    
    print()
    print(f"Train data shape: {train_X.shape}, Train labels shape: {train_y.shape}")
    print(f"Test data shape: {test_X.shape}, Test labels shape: {test_y.shape}")
    print()
    # Ensure `config.n_class` is correctly set before poisoning
    config.n_class = len(np.unique(train_y))
    print(f"Number of unique classes in the Dataset: {config.n_class}")

    # Check if Data Poisoning Attack is Enabled
    if config.use_data_poisoning:  # 1 = Apply attack, 0 = No attack
        print("Applying Data poisoning Attack by Random Label Flipping to Train & Test Data")
        train_y = utils.apply_random_label_flipping(train_y, num_classes=config.n_class)
        test_y = utils.apply_random_label_flipping(test_y, num_classes=config.n_class)

    train_data = (train_X, train_y)             
    test_data = (test_X, test_y)

    client_data = utils.partition_data(train_X, train_y, config)
    print(f"Dataset Initialization completed for Client ID: {client_id}!")

    logger = utils.get_logger(os.path.join(result_path, f"client_{client_id}.log"))

    # Privacy settings
    epsilon, alpha = check_privacy(config)
    delta = 1.0 / config.n_client_data
    print()
    logger.info(f"Expected privacy use is ε={epsilon:.2f} and δ={delta:.4f} at α={alpha:.2f}")
    print()
    logger.info(f"Hyperparameter setting = {config}")
    print()
    client = Client(client_data, config)
    
    # Start the thread for sending messages
    threading.Thread(target=start_sending_messages, args=(client_id, my_ip, client, config, test_data), daemon=True).start()
   
    # Start the Uvicorn server to handle incoming requests
    server = uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=config.port))
    threading.Thread(target=server.run, daemon=True).start()

    # Wait for the shutdown signal
    shutdown_event.wait()
    print("Stopping server...")
    server.should_exit = True
    

