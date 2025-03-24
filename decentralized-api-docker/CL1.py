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
    algorithm: str = "ProxyFL"                   # Decentralized - "AvgPush" - Single model(proxy) | "ProxyFL" - Two model(proxy & private) |  Centralized - "FML" - Two model(proxy & private) | "FedAvg" - Single model(proxy)
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
    port: int = 8002                             # port number for server & clients to communicate | USE 8000 - 8999
    max_retries: int = 10                        # Maximum number of times to retry communication in case of failure
    retry_delay: int = 10                        # Time (in seconds) to wait before retrying after a failure
    
    # DATA & MODEL POISONING AND DETECTION PARAMETERS
    use_data_poisoning: int = 0                  # 0 - no data poisoning | 1 - use data poisoning
    threshold_digits: int = 3                    # Euclidean distance threshold for detecting model poisoning 
    threshold_accuracy: float = 0.70               # Accuracy threshold for detecting data poisoning
    attack_type: str = ""                        # Model poisoning attacks: 1) "combined" - adaptive_backdoor + gradient_attack + model weights attack + model layers attack (All the attacks below)
                                                                          # 2) "adaptive_backdoor" - Dynamically Applies an adaptive backdoor attack by combining multiple techniques.
                                                                          # 3) "gradient_attack" - Modify gradients for poisoning attack, with and without DP two methods avaliable to stay within DP range  
                                                                          # 4) "model_weights_attack" - Modify model weights after optimization using scale factor, customize the that value in the trainer.py file
                                                                          # 5) "model_layers_attack" - Injects a backdoor by shifting classification layer weights.
                                                                          # 6) "" - dont want to use any model poisoning attacks 

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

# Load clients config from JSON file
try:
    with open(clients_ips_file, "r") as file:
        clients = json.load(file)  # Loads as a dictionary
        clients = {int(k): v for k, v in clients.items()}  # Convert keys to integers
        print(f"Loaded {clients_ips_file} successfully.")
except FileNotFoundError:
    print(f"Error: {clients_ips_file} not found.")
    exit(1)

# Determine the number of clients and set up round mappings
num_clients = len(clients)  # Get the number of clients dynamically
# This dictionary defines which client communicates with which other client in each round.
# Each round shifts communication targets cyclically, increasing the shift distance in each subsequent round.
if num_clients == 3:
    round_mapping = {
        1: {1: 2, 2: 3, 3: 1},
        2: {1: 3, 2: 1, 3: 2},
    }
elif num_clients == 4:
    round_mapping = {
        1: {1: 2, 2: 3, 3: 4, 4: 1},
        2: {1: 3, 2: 4, 3: 1, 4: 2},
    }
elif num_clients == 5:
    round_mapping = {
        1: {1: 2, 2: 3, 3: 4, 4: 5, 5: 1},
        2: {1: 3, 2: 4, 3: 5, 4: 1, 5: 2},
    }    
elif num_clients == 6:
    round_mapping = {
        1: {1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 1},
        2: {1: 3, 2: 4, 3: 5, 4: 6, 5: 1, 6: 2},
        3: {1: 4, 2: 5, 3: 6, 4: 1, 5: 2, 6: 3},
    }
elif num_clients == 7:
    round_mapping = {
        1: {1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 1},
        2: {1: 3, 2: 4, 3: 5, 4: 6, 5: 7, 6: 1, 7: 2},
        3: {1: 4, 2: 5, 3: 6, 4: 7, 5: 1, 6: 2, 7: 3},
    }
elif num_clients == 8:
    round_mapping = {
        1: {1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 1},
        2: {1: 3, 2: 4, 3: 5, 4: 6, 5: 7, 6: 8, 7: 1, 8: 2},
        3: {1: 4, 2: 5, 3: 6, 4: 7, 5: 8, 6: 1, 7: 2, 8: 3},
        4: {1: 5, 2: 6, 3: 7, 4: 8, 5: 1, 6: 2, 7: 3, 8: 4},
    }
else:
    raise ValueError("Unsupported number of clients")

def get_round_mapping(round_number):
    """
    Get the round mapping for the given round number.
    Cycles through the mappings in `round_mapping` if the round number exceeds the defined mappings.
    """
    total_defined_rounds = len(round_mapping)
    effective_round = (round_number - 1) % total_defined_rounds + 1
    return round_mapping[effective_round]

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
    base_dir = "received_weights_from_other_clients"
    # Directory to save received model weights
    received_weights_dir = os.path.join(base_dir, f"received_model_weights_from_other_clients_for_CL-{client_id}")
    os.makedirs(received_weights_dir, exist_ok=True)

    # Save the model weights file
    file_path = os.path.join(received_weights_dir, f"received_model_weights_round_{message.round_number}_client_{message.sender_id}.pth")
    with open(file_path, "wb") as f:
        f.write(await file.read())

    print(f"Client {message.sender_id} ({message.sender_ip}) sent model weights for round {message.round_number}")
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
    if config.algorithm.lower() == 'AvgPush':
        sub_dir = os.path.join(checkpoints_dir, "AvgPush")        
    else:
        sub_dir = os.path.join(checkpoints_dir, "ProxyFL")

    # Create the subdirectory if it doesn't exist
    os.makedirs(sub_dir, exist_ok=True)

    # Update filepath to save in the checkpoints directory
    filepath = os.path.join(sub_dir, filepath)

    # Save the model weights
    torch.save(model_weights, filepath)
    print("Model weights saved in checkpoints directory")

def load_and_append_results(result_path, client_id, start_round, total_rounds, new_results, config):
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
    if config.algorithm == 'ProxyFL':
        existing_private_accuracies = existing_results.get("private_accuracies", np.array([]))  # Optional field
        private_accuracies = np.append(existing_private_accuracies, new_results["private_accuracies"])
        results_dict["private_accuracies"] = private_accuracies
    
    # Save the updated results
    np.savez(results_file_path, **results_dict)   
    #print(f"results appended and saved to {results_file_path} for rounds {start_round} to {total_rounds}")
    print(f"results appended and saved for rounds {start_round} to {total_rounds}")
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
            if config.algorithm.lower() == 'AvgPush':
                sub_dir = os.path.join(parent_dir, f"checkpoints-CL-{client_id}", "AvgPush")
            else:
                sub_dir = os.path.join(parent_dir, f"checkpoints-CL-{client_id}", "ProxyFL")
            
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
                peer_id = next((client_id for client_id, ip in clients.items() if ip == peer_ip), None)
                print(f"Model weights sent to Client {peer_id} ({peer_ip}): {json_response}")
                break  # Exit the retry loop on success
            else:
                raise requests.exceptions.RequestException(f"Received status code {response.status_code}")
        
        except requests.exceptions.RequestException as e:
            peer_id = next((client_id for client_id, ip in clients.items() if ip == peer_ip), None)
            print()
            print(f"Attempt {attempt} failed to send model weights to Client {peer_id} ({peer_ip}): {e}")
            
            if attempt < max_retries:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)  # Wait before retrying
            else:
                print(f"Failed to send model weights to Client {peer_id} ({peer_ip}) after {max_retries} attempts.")
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

def flatten_weights(weights):
    """Flatten all model weights into a single tensor, handling both NumPy arrays and PyTorch tensors."""
    weight_tensors = []

    for key, param in weights.items():
        if isinstance(param, np.ndarray):  # Convert NumPy array to tensor
            param_tensor = torch.tensor(param, dtype=torch.float32)
        elif isinstance(param, torch.Tensor):  # Ensure tensor is float32
            param_tensor = param.to(dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported weight format for {key}: {type(param)}")

        weight_tensors.append(param_tensor.flatten())

    return torch.cat(weight_tensors)  # Concatenate all flattened tensors

def compute_euclidean_distance(vec1, vec2):
    """Compute Euclidean distance between two vectors."""
    return torch.norm(vec1 - vec2).item()

def start_sending_messages(client_id, my_ip, client, config, eval_data):
    """
    Function to manage the sending and receiving of messages for each round of training.
    It handles the training process and communication with other clients.
    """
    # Set the checkpoint directory based on the algorithm
    if config.algorithm == 'AvgPush':
        checkpoints_dir = f"./checkpoints/checkpoints-CL-{client_id}/AvgPush"
    elif config.algorithm == 'ProxyFL':
        checkpoints_dir = f"./checkpoints/checkpoints-CL-{client_id}/ProxyFL"
    else:
        raise ValueError(f"Unknown algorithm: {config.algorithm}")

    latest_round = 0
    latest_weights_path = None

    # Check for the latest checkpoint file
    if os.path.exists(checkpoints_dir):
        checkpoint_files = [f for f in os.listdir(checkpoints_dir) if f.endswith(".pth")]
        if checkpoint_files:
            # Extract the round numbers from the file names
            round_numbers = [int(f.split("_round_")[1].split("_")[0]) for f in checkpoint_files]
            latest_round = max(round_numbers)
            latest_weights_path = os.path.join(checkpoints_dir, f"model_weights_round_{latest_round}_client_{client_id}.pth")

    # Load model weights if a checkpoint is found
    if latest_weights_path:
        print()
        print(f"Resuming from checkpoint: {latest_weights_path}")
        print()
        checkpoint = torch.load(latest_weights_path)
        client.proxy_model.load_state_dict(utils.convert_np_weights_to_tensor(checkpoint))
        #if config.algorithm == 'ProxyFL':
            #client.private_model.load_state_dict(utils.convert_np_weights_to_tensor(checkpoint))
        start_round = latest_round + 1
    else:
        print()
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
    poisoned_clients = set()  # Track poisoned clients across rounds

    for round_number in range(start_round, total_rounds + 1):
        # Train the model based on the selected algorithm
        if config.algorithm == 'ProxyFL':
            send_model_weights, private_accuracy, proxy_accuracy, privacy_budget, training_time = train_proxy_push(client, eval_data, logger, config)
        elif config.algorithm == 'AvgPush':
            send_model_weights, proxy_accuracy, privacy_budget, training_time = train_avg_push(client, eval_data, logger, config) 
        else:
            raise ValueError("Unknown training method")
        
        # Store the results for this round
        private_accuracies[round_number - 1] = private_accuracy if config.algorithm == 'ProxyFL' else np.nan
        proxy_accuracies[round_number - 1] = proxy_accuracy
        privacy_budgets[round_number - 1] = privacy_budget
        training_times[round_number - 1] = training_time

        # Shared list to store results for communication time
        send_time = 0  # Initialize for this round
        receive_time = 0  # Initialize for this round
        poisoned_clients = set()  # Track poisoned clients across rounds

        # Function to send model weights
        def send_weights():
            nonlocal send_time  # Use the send_time from the enclosing function
            start_send_time = time.time()  # Start time for sending
            current_round_mapping = get_round_mapping(round_number)
            recipient_id = current_round_mapping[client_id]
            #recipient_id = round_mapping[round_number][client_id]
            recipient_ip = clients[recipient_id]
            #print(f"Sending model weights to Client {recipient_id} ({recipient_ip})...")
            send_message_to_peer(recipient_ip, client_id, my_ip, round_number, send_model_weights)
            end_send_time = time.time()  # End time for sending
            send_time = end_send_time - start_send_time  # Return the sending time

        # Function to receive model weights
        def receive_weights():
            nonlocal receive_time  # Use the receive_time from the enclosing function
            start_receive_time = time.time()  # Start time for receiving
            current_round_mapping = get_round_mapping(round_number)
            sender_id = next(key for key, value in current_round_mapping.items() if value == client_id)
            #print(f"Waiting for model weights from Client {sender_id} for round {round_number}...")
            message = wait_for_message(sender_id, round_number)
            #print(f"Received model weights from Client {sender_id} for round {round_number}.")
            # Save the current model state in memory before loading new weights
            previous_model_state = client.proxy_model.state_dict()
            end_receive_time = time.time()  # End time for receiving

            receive_time = end_receive_time - start_receive_time  # Return the receiving time   

            received_weights = torch.load(message["file_path"])
            # 🚫 Check if the client is already poisoned
            if sender_id in poisoned_clients:
                print(f"❌ Client {sender_id} is already marked as POISONED.")
                return
            # Flatten the sent and received model weights
            reference_weights_flat = flatten_weights(send_model_weights)  # Reference is the sent weights
            received_weights_flat = flatten_weights(received_weights)
            # Compute Euclidean distance
            distance = compute_euclidean_distance(reference_weights_flat, received_weights_flat)
            distance_length = len(str(int(distance)))  # Count digits in the integer part
            print(f"📊 Euclidean Distance: {distance:.8f}")
            print(f"🔢 Length of Distance Value: {distance_length} digits")  
            # Set thresholds
            THRESHOLD_DIGITS = config.threshold_digits  # Euclidean distance threshold
            THRESHOLD_ACCURACY = config.threshold_accuracy  # Accuracy threshold
            # Check Euclidean Distance
            is_poisoned_distance = distance_length > THRESHOLD_DIGITS
            if is_poisoned_distance:
                print(f"⚠️ Received model from Client {sender_id} is identified as POISONED due to high Euclidean distance! Ignoring update.")
                poisoned_clients.add(sender_id)  # Mark client as poisoned
                #return
            else:
                # Load the received model temporarily
                client.proxy_model.load_state_dict(utils.convert_np_weights_to_tensor(received_weights))
                # Evaluate accuracy
                accuracy = utils.evaluate_model(client.proxy_model, eval_data, config)
                print()
                print(f"📊 Evaluation Accuracy after loading received model weights: {accuracy:.2f}")
                print()
                is_poisoned_accuracy = accuracy < THRESHOLD_ACCURACY
                if is_poisoned_accuracy:
                    print(f"⚠️ Received model from Client {sender_id} has low accuracy! Ignoring update.")
                    poisoned_clients.add(sender_id)  # Mark client as poisoned
                    client.proxy_model.load_state_dict(previous_model_state)  # Revert to previous model state
                
                else:
                    print(f"✅ Model weights from Client {sender_id} successfully loaded and validated.")

            #receive_time = end_receive_time - start_receive_time  # Return the receiving time

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
    import torch.nn as nn
    import os
    import utils
    import numpy as np
    from privacy_checker import check_privacy
    from client import Client
    import warnings
    # Suppress the specific FutureWarning from torch.load
    warnings.simplefilter("ignore", category=FutureWarning)
    warnings.simplefilter("ignore", category=UserWarning)
    
    my_ip = clients[client_id]
    print()
    print(f"This is Client {client_id} and the IP: {my_ip}")
    print()
    # Initialize configuration
    config = ClientConfig(**config_data)
  
    # Configure device
    config.device = torch.device(f"cuda:{config.device}" if torch.cuda.is_available() else "cpu")
    config.data_path = f"./datasets/CL-{client_id}"
    # Configure input channel
    #config.in_channel = 3  # Assuming 3-channel images
    config.result_path = f"./results/results-CL{client_id}"
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
    server = uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=config.port))   #The server will be accessible from any device in the network
    threading.Thread(target=server.run, daemon=True).start()

    # Wait for the shutdown signal
    shutdown_event.wait()
    print("Stopping server...")
    server.should_exit = True

