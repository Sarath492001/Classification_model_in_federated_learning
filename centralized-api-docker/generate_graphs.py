import os
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import rc
from pydantic import BaseModel
import argparse
import json
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.switch_backend('agg')

# Client configuration model using Pydantic for validation
class ClientConfig(BaseModel):

    # DATA AND RESULT PATH
    data_path: str = "./datasets"    
    result_path: str = "./results" 
          
    # TRAINING PARAMETERS 
    dataset: str = "face_images_2"               #"Face_images" - face images in npz format | "medical_images_2" - 2 class cancer cell images in folder format | "face_images_2" - face images in folder format
    image_size: tuple = (128, 128)               # target image size should be used for training  (BASED ON THE IMAGE SIZE YOU HAVE TO CHANGE THE MODEL PROCESS THE INPUT IMAGE SIZE)
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
    port: int = 8002                             # port number for server & clients to communicate | USE 8000 - 8999  | should be same for clients and server 
    max_retries: int = 10                        # Maximum number of times to retry communication in case of failure
    retry_delay: int = 10                        # Time (in seconds) to wait before retrying after a failure
    
    # DATA & MODEL POISONING AND DETECTION PARAMETERS
    use_data_poisoning: int = 0                  # 0 - no data poisoning | 1 - use data poisoning
    threshold_digits: int = 2                    # Euclidean distance threshold for detecting model poisoning 
    threshold_accuracy: int = 0.70               # Accuracy threshold for detecting data poisoning
    attack_type: str = "combined"                # Model poisoning attacks: 1) "combined" - adaptive_backdoor + gradient_attack + model weights attack + model layers attack (All the attacks below)
                                                                          # 2) "adaptive_backdoor" - Dynamically Applies an adaptive backdoor attack by combining multiple techniques.
                                                                          # 3) "gradient_attack" - Modify gradients for poisoning attack, with and without DP two methods avaliable to stay within DP range  
                                                                          # 4) "model_weights_attack" - Modify model weights after optimization using scale factor, customize the that value in the trainer.py file
                                                                          # 5) "model_layers_attack" - Injects a backdoor by shifting classification layer weights.
                                                                          # 6) "" - dont want to use any model poisoning attacks 
    

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True, help="Path to config file")
parser.add_argument("--client_id", type=int, required=True, help="Client ID")
args = parser.parse_args() 
client_id = args.client_id                                 
config_file = args.config     
# Load JSON file with error handling
try:
    with open(config_file, "r") as file:
        config_data = json.load(file)
        print(f"Loaded config from: {config_file}")
except FileNotFoundError:
    print(f"Error: {config_file} not found. Using default configuration within the client script.")
    exit(1)

config = ClientConfig(**config_data)
config.data_path = f"./datasets/CL-{client_id}" 
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
        #config.algorithm
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
    result_path_components.insert(-2, f"lora_rank{config.lora_rank}")  # Insert before n_epochs    

result_path = os.path.join(*result_path_components)

#algorithms = [ 'ProxyFL']   #   [ 'FML'] ['AvgPush', 'ProxyFL']
algorithms = [config.algorithm]
labels = []
all_training_times = []  # To store the training times for each algorithm
all_accuracies = []
all_communication_times = []
# load the results    
for algorithm in algorithms:
    algo_acc = []
    private_acc = []
    training_time = []  # Store training times for each algorithm
    comm_time = []

    load_file = os.path.join(result_path,
                                 algorithm,
                                 f'Client_{client_id}_results.npz',
                                 )

    load_results = np.load(load_file)

    algo_acc.append(load_results['proxy_accuracies'])

    training_time.append(load_results['training_times'])  # Collect training times
    comm_time.append(load_results['communication_times'])  # Collect training times

    #print(f"Loaded results for {algorithm}:")
    #print(f"communication time data for {algorithm}: {comm_time}")

    if algorithm == 'ProxyFL':              #if algorithm == 'FML' or algorithm == 'ProxyFL':
            private_acc.append(load_results['private_accuracies'])

    algo_acc = np.stack(algo_acc)
    training_time = np.stack(training_time)
    comm_time = np.stack(comm_time)

    all_accuracies.append(algo_acc)
    all_training_times.append(training_time)  # Append training time for the algorithm
    all_communication_times.append(comm_time)


    if algorithm == 'FML':
        private_acc = np.stack(private_acc)
        all_accuracies.append(private_acc)
        labels.append(algorithm + '-proxy')
        labels.append(algorithm + '-private')
    else:
        labels.append(algorithm)

####################################### Accuracy graph ############################################################################

# Function to apply moving average smoothing
def moving_average(data, window_size=5):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def moving_average_padded(data, window_size=5):
    smoothed = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    pad = np.full(window_size-1, smoothed[0])  # Padding to match original size
    return np.concatenate([pad, smoothed])

# Initialize variables
x = np.arange(config.n_rounds)

plt.figure(figsize=(6, 4))  # ratio

for i in range(len(labels)):
    # Compute the mean and std for each round
    y_mean = np.mean(all_accuracies[i], axis=0)
    y_std = np.std(all_accuracies[i], axis=0)
    
    # Smooth the y_mean using moving average
    smoothed_y_mean = moving_average_padded(y_mean, window_size=15)  # Adjust window size as needed
    smoothed_y_std = moving_average_padded(y_std, window_size=15)  # Apply smoothing to standard deviation as well

    # Adjust x axis for the smoothed data
    smoothed_x = x[:len(smoothed_y_mean)]  # Adjust x to match the smoothed data length

    # Plot the smoothed curve
    if labels[i] == 'FedAvg':
        plt.plot(smoothed_x, smoothed_y_mean, label=labels[i], color='green', linestyle='-', marker='o', markersize=9, markevery=max(1, len(smoothed_x) // 10))
    elif 'FML-proxy' in labels[i]:
        plt.plot(smoothed_x, smoothed_y_mean, label=labels[i], color='blue', linestyle='-', marker='o', markersize=9, markevery=max(1, len(smoothed_x) // 10))
    else:
        plt.plot(smoothed_x, smoothed_y_mean, label=labels[i], color='red', linestyle='-', marker='o', markersize=9, markevery=max(1, len(smoothed_x) // 10))
    
    # Add fill for standard deviation (smoothed as well)
    plt.fill_between(smoothed_x, smoothed_y_mean - smoothed_y_std, smoothed_y_mean + smoothed_y_std, alpha=0.2)

# Configure the plot
plt.grid(True)
plt.xlabel('Rounds')
plt.ylabel('Accuracy')
plt.title(f'Client {client_id} Accuracy Graph Over Rounds')
plt.legend()

# Save the graph
acc_file_name = 'accuracy.png'
current_dir_acc_fig_file = os.path.join(f'./Graphs/GraphsCL-{client_id}', acc_file_name)
os.makedirs(os.path.dirname(current_dir_acc_fig_file), exist_ok=True)

acc_fig_file = os.path.join(result_path, acc_file_name)
plt.savefig(acc_fig_file, bbox_inches='tight', format='png')
plt.savefig(current_dir_acc_fig_file, bbox_inches='tight', format='png')
plt.close()
print(f"Client {client_id} Accuracy Graph generated")
####################################### Training Time Graph ###############################################################

plt.figure(figsize=(6, 4))  # ratio
for i in range(len(all_training_times)):
    # Compute the mean and std for each round for training time
    y_mean_time = np.mean(all_training_times[i], axis=0)  # Mean over clients and trials, keeping rounds
    y_std_time = np.std(all_training_times[i], axis=0)    # Standard deviation over clients and trials

    smoothed_y_mean_time = moving_average_padded(y_mean_time, window_size=50)
    smoothed_y_std_time = moving_average_padded(y_std_time, window_size=50)

    # Smooth the y_mean_time and y_std_time using moving average
    #smoothed_y_mean_time = moving_average(y_mean_time, window_size=10)  # Adjust window size as needed
    #smoothed_y_std_time = moving_average(y_std_time, window_size=10)  # Apply smoothing to standard deviation as well

    # Adjust x axis for the smoothed data
    smoothed_x = x[:len(smoothed_y_mean_time)]  # Adjust x to match the smoothed data length

    # Handle labels correctly to avoid duplication for ProxyFL
    if labels[i] == 'FedAvg':
        plt.plot(smoothed_x, smoothed_y_mean_time, label='FedAvg', color='green', linestyle='-', marker='o', markersize=9, markevery=max(1, len(smoothed_x) // 10))
    elif 'FML' in labels[i]:
        plt.plot(smoothed_x, smoothed_y_mean_time, label='FML', color='blue', linestyle='-', marker='o', markersize=9, markevery=max(1, len(smoothed_x) // 10))

    # Add fill for standard deviation (smoothed as well) in training time
    plt.fill_between(smoothed_x, smoothed_y_mean_time - smoothed_y_std_time, smoothed_y_mean_time + smoothed_y_std_time, alpha=0.2)

# Configure the plot
plt.grid(True)
plt.xlabel('Rounds')
plt.ylabel('Training Time (seconds)')
plt.title(f'Client {client_id} Training Time Graph Over Rounds')
plt.legend()

# Save the training time graph
train_time_file_name = 'training_time_graph.png'
current_dir_fig_file_train_time = os.path.join(f'./Graphs/GraphsCL-{client_id}', train_time_file_name)
# Create the directory if it doesn't exist
os.makedirs(os.path.dirname(current_dir_fig_file_train_time), exist_ok=True)
fig_file_train_time = os.path.join(result_path, train_time_file_name)
plt.savefig(fig_file_train_time, bbox_inches='tight', format='png')
plt.savefig(current_dir_fig_file_train_time, bbox_inches='tight', format='png')
plt.close()
print(f"Client {client_id} Training Time Graph generated")
####################################### communication Time Graph ###############################################################

# Plot Communication Time graph
plt.figure(figsize=(6, 4))  # ratio
for i in range(len(all_communication_times)):
    # Compute the mean and std for each round for communication time
    y_mean_time = np.mean(all_communication_times[i], axis=0)  # Mean over clients and trials, keeping rounds
    y_std_time = np.std(all_communication_times[i], axis=0)    # Standard deviation over clients and trials

    # Smooth the y_mean_time and y_std_time using moving average
    smoothed_y_mean_time = moving_average_padded(y_mean_time, window_size=50)  # Adjust window size as needed
    smoothed_y_std_time = moving_average_padded(y_std_time, window_size=50)  # Apply smoothing to standard deviation as well

    # Adjust x axis for the smoothed data
    smoothed_x = x[:len(smoothed_y_mean_time)]  # Adjust x to match the smoothed data length

    # Handle labels correctly to avoid duplication for ProxyFL
    if labels[i] == 'FedAvg':
        plt.plot(smoothed_x, smoothed_y_mean_time, label='FedAvg', color='green', linestyle='-', marker='o', markersize=9, markevery=max(1, len(smoothed_x) // 10))
    elif 'ProxyFL' in labels[i]:
        plt.plot(smoothed_x, smoothed_y_mean_time, label='FML', color='blue', linestyle='-', marker='o', markersize=9, markevery=max(1, len(smoothed_x) // 10))

    # Add fill for standard deviation (smoothed as well) in communication time
    plt.fill_between(smoothed_x, smoothed_y_mean_time - smoothed_y_std_time, smoothed_y_mean_time + smoothed_y_std_time, alpha=0.2)

# Configure the plot
plt.grid(True)
plt.xlabel('Rounds')
plt.ylabel('Communication Time (seconds)')
plt.title(f'Client {client_id} Communication Time Graph Over Rounds')
plt.legend()

# Save the communication time graph
comm_time_file_name = 'communication_time_graph.png'
current_dir_fig_file_comm_time = os.path.join(f'./Graphs/GraphsCL-{client_id}', comm_time_file_name)
# Create the directory if it doesn't exist
os.makedirs(os.path.dirname(current_dir_fig_file_comm_time), exist_ok=True)
fig_file_comm_time = os.path.join(result_path, comm_time_file_name)
plt.savefig(fig_file_comm_time, bbox_inches='tight', format='png')
plt.savefig(current_dir_fig_file_comm_time, bbox_inches='tight', format='png')
plt.close()
print(f"Client {client_id} Communication Time Graph generated")
