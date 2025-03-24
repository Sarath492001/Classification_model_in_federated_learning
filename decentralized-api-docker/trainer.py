import time
import re 
import torch
import torch.nn as nn
import numpy as np 
import utils
import loralib as lora

Softmax = nn.Softmax(dim=-1)
LogSoftmax = nn.LogSoftmax(dim=-1)

KL_Loss = nn.KLDivLoss(reduction='batchmean')
CE_Loss = nn.CrossEntropyLoss()

##################################################################################################################################

def train_avg_push(client, eval_data, logger, config):
    #print()
    #logger.info(f"Hyperparameter setting = {config}")
    #print()
    # Start timer for training
    start_time = time.time()
    #comm_time = 0.

    acc_proxy, _, _ = regular_training_loop(client, None, config)
    #acc_proxy, _, _ = regular_training_loop(client, None, config, attack_type="gradient_attack")
    #acc_proxy, _, _ =regular_training_loop(client, None, config, attack_type = config.attack_type)
    # End timer for training
    end_time = time.time()
    # Calculate training time
    training_time = end_time - start_time

    proxy_accuracy = utils.evaluate_model(client.proxy_model, eval_data, config)

    if config.verbose:
        if config.use_private_SGD:
            print()
            logger.info(f"Trained for {config.n_epochs} epochs, Proxy acc={proxy_accuracy:.4f} | ε={client.privacy_budget:.2f}")
        else:
            print()
            logger.info(f"Trained for {config.n_epochs} epochs, Proxy acc={proxy_accuracy:.4f}")

    privacy_budget = client.privacy_budget
    
    send_model_weights = utils.extract_numpy_weights(client.proxy_model)

    return send_model_weights, proxy_accuracy, privacy_budget, training_time

################################ with save and load ########################################################
def train_proxy_push(client, eval_data, logger, config):
    #print()
    #logger.info(f"Hyperparameter setting = {config}")
    #print()
    # Start timer for training
    start_time = time.time()
    
    dml_training_loop(client, None, config)
    # End timer for training
    end_time = time.time()
    # Calculate training time
    training_time = end_time - start_time

    private_accuracy = utils.evaluate_model(client.private_model, eval_data, config)
    proxy_accuracy = utils.evaluate_model(client.proxy_model, eval_data, config)
    
    if config.verbose:
        if config.use_private_SGD:
                print()
                logger.info(f"Trained for {config.n_epochs} epochs,"+
                            f" Private acc={private_accuracy:.4f}" +
                            f" | Proxy acc={proxy_accuracy:.4f}" +
                            f" | ε={client.privacy_budget:.2f}")
        else:
                print()
                logger.info(f"Trained for {config.n_epochs} epochs,"+
                            f" Private acc={private_accuracy:.4f}" +
                            f" | Proxy acc={proxy_accuracy:.4f}")
          
    privacy_budget = client.privacy_budget

    send_model_weights = utils.extract_numpy_weights(client.proxy_model)

    return send_model_weights, private_accuracy, proxy_accuracy, privacy_budget, training_time
##################################################################################################################################
def apply_adaptive_backdoor(data, target, attack_strength=1.0):
    """
    Applies an adaptive backdoor attack by combining multiple techniques.
    
    This function introduces various types of backdoor triggers dynamically:
    1. **Dynamic Trigger:** Injects a small patch of high-intensity pixels at a random location.
    2. **Feature Space Manipulation:** Blends the poisoned sample with another target-class sample.
    3. **Multi-Trigger Attack:** Adds multiple distinct patterns to increase stealth.

    Args:
        data (torch.Tensor): Input images of shape (batch_size, channels, height, width).
        target (torch.Tensor): Corresponding labels of shape (batch_size,).
        attack_strength (float, optional): Intensity of the backdoor modification. Default is 0.1.

    Returns:
        poisoned_data (torch.Tensor): Modified images with embedded backdoor triggers.
        poisoned_target (torch.Tensor): Labels remain unchanged (attack is hidden in input).
    """
    poisoned_data = data.clone()
    poisoned_target = target.clone()
    batch_size, _, height, width = data.shape
    chosen_attack = np.random.choice(["dynamic_trigger", "feature_space", "multi_trigger"])

    if chosen_attack == "dynamic_trigger":
        #print("dynamic_trigger is used for adaptive backdoor")
        trigger_size = height // 8
        for i in range(batch_size):
            x, y = np.random.randint(0, height - trigger_size), np.random.randint(0, width - trigger_size)
            poisoned_data[i, :, x:x+trigger_size, y:y+trigger_size] = attack_strength

    elif chosen_attack == "feature_space":
        #print("feature_space is used for adaptive backdoor")
        target_class = 1  # Assume attacking class 1
        target_indices = (target == target_class).nonzero(as_tuple=True)[0]
        if len(target_indices) > 0:
            rand_target_samples = poisoned_data[target_indices[torch.randint(0, len(target_indices), (batch_size,))]]
            poisoned_data = 0.2 * data + 0.8 * rand_target_samples  # Blend features

    elif chosen_attack == "multi_trigger":
        #print("multi_trigger is used for adaptive backdoor")
        poisoned_data[:, :, -10:, -10:] = 1.0  # Bottom-right
        poisoned_data[:, :, :10, :10] = 0.0  # Top-left
        poisoned_data += 0.05 * torch.randn_like(data)

    return poisoned_data, poisoned_target
# model weights attack, modify this based on the model you  are using 
def inject_backdoor_into_model(model, attack_strength=0.3):
    """
    Injects a backdoor by shifting classification layer weights.
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "classifier" in name:
                param.data += attack_strength * torch.randn_like(param.data)  # Inject small noise

def gradient_manipulation(gradients, attack_type="alie"):
    """
    Modify gradients using a poisoning attack.
    :param gradients: Original gradients
    :param attack_type: Type of attack to apply ('alie' or 'min-max')
    :return: Manipulated gradients
    """
    grad_tensors = [g.view(-1) for g in gradients if g is not None]  # Skip None
    if not grad_tensors:  # If all gradients are None, return original
        return gradients  
    grad_tensor = torch.cat(grad_tensors)  # Flatten all valid gradients
    grad_mean = grad_tensor.mean()
    grad_std = grad_tensor.std()

    if attack_type == "alie":
        #print("alie in gradient_manipulation is used")
        outlier_factor = 5.0  # Make the attack more extreme
        manipulated_grads = grad_tensor.clone()  # Keep original shape
        manipulated_grads[:] = grad_mean + (outlier_factor * grad_std)  # Amplify deviation
        #manipulated_grads = grad_mean +  grad_std  # Amplify deviation
    elif attack_type == "min-max":
        #print("min-max in gradient_manipulation is used")
        manipulated_grads = -grad_tensor * 10.0 # Flip signs of gradients
    else:
        return gradients  # No attack

    # Reshape to original gradient dimensions
    start = 0
    modified_gradients = []
    for g in gradients:
        if g is not None:
            size = g.numel()
            modified_gradients.append(manipulated_grads[start:start + size].reshape(g.shape))
            start += size
        else:
            modified_gradients.append(None)  # Preserve None gradients
    return modified_gradients

def scaled_gradient_manipulation(gradients, scale_factor=5.0, alpha=0.7, max_grad_norm=1.0):
    """
    Modify gradients using a poisoning attack while staying within DP clipping range.
    """
    grad_tensor = torch.cat([g.view(-1) for g in gradients if g is not None])  # Flatten all gradients
    grad_mean = grad_tensor.mean()
    grad_std = grad_tensor.std()

    #print("[Gradient Attack] Applying both ALIE and Min-Max in scaled_gradient_manipulation")

    # ✅ ALIE attack: Add scaled deviation
    alie_grads = grad_mean + (scale_factor * grad_std) + torch.randn_like(grad_tensor) * scale_factor

    # ✅ Min-Max attack: Flip and scale gradients
    min_max_grads = -scale_factor * grad_tensor

    # 🔥 Combine both effects: weighted sum of ALIE and Min-Max
    #manipulated_grads = 0.5 * alie_grads + 0.5 * min_max_grads  
    manipulated_grads = alpha * alie_grads + (1 - alpha) * min_max_grads 

    # 🚀 **Bypass DP-SGD Clipping**
    manipulated_grads *= (max_grad_norm * 10.0)  # **Make it 10x larger than DP clipping threshold**
    # Reshape to original gradient dimensions
    start = 0
    modified_gradients = []
    for g in gradients:
        if g is not None:
            size = g.numel()
            modified_gradients.append(manipulated_grads[start:start + size].reshape(g.shape))
            start += size
        else:
            modified_gradients.append(None)

    return modified_gradients

def dml_training_loop(client, logger, config):
    """
    Federated Learning Training Loop with Dual Model Learning (DML) and optional Gradient Manipulation Attacks.
    """
    attack_type = config.attack_type  # Directly retrieve attack type from config
    # Print a statement based on the attack type
    if attack_type == "combined":
        print("🚨 Using combined attack: adaptive_backdoor + gradient_attack + model weights attack + model layers attack. For model poisoning")
        print()
    elif attack_type == "adaptive_backdoor":
        print("🎯 Using adaptive backdoor attack for model poisoning.")
        print()
    elif attack_type == "gradient_attack":
        print("🔥 Using gradient manipulation attackfor model poisoning.")
        print()
    elif attack_type == "model_weights_attack":
        print("⚡ Using model weights poisoning attack for model poisoning.")
        print()
    elif attack_type == "model_layers_attack":
        print("🔁 Using model layer modification attack for model poisoning.")
        print()
    else:
        print("No model poisoning attack applied. Running standard training.")
        print()

    client.private_model.train()
    client.proxy_model.train()
    train_private_acc = []
    train_proxy_acc = []
    train_privacy_budget = []

    epsilon = 0
    delta = 1.0 / client.private_data[0].shape[0]
        
    # Mark only LoRA parameters as trainable if models are CNN1lora or VisionTransformerlora
    if config.private_model_type in ["CNN1lora", "VisionTransformerlora"]:
        lora.mark_only_lora_as_trainable(client.private_model)

    if config.proxy_model_type in ["CNN1lora", "VisionTransformerlora"]:
        lora.mark_only_lora_as_trainable(client.proxy_model)

    for e in range(config.n_epochs):

        train_loader = utils.data_loader(config.dataset,
                                         client.private_data[0],
                                         client.private_data[1],
                                         config.batch_size)

        
        # total_private_loss = 0.0
        correct_private = 0.0
        acc_private = 0.0
        # total_proxy_loss = 0.0
        correct_proxy = 0.0
        acc_proxy = 0.0

        for idx, (data, target) in enumerate(train_loader):

            client.private_opt.zero_grad()
            client.proxy_opt.zero_grad()

            data = torch.from_numpy(data).to(client.device)
            target = torch.from_numpy(target).to(client.device)

            # Apply adaptive backdoor attack
            if attack_type in ["adaptive_backdoor", "combined"] and np.random.rand() < 0.9:
                data, target = apply_adaptive_backdoor(data, target)
                
            pred_private = client.private_model(data)
            pred_proxy = client.proxy_model(data)

            ce_private = CE_Loss(pred_private, target)
            kl_private = KL_Loss(LogSoftmax(pred_private), Softmax(pred_proxy.detach()))

            ce_proxy = CE_Loss(pred_proxy, target)
            kl_proxy = KL_Loss(LogSoftmax(pred_proxy), Softmax(pred_private.detach()))

            loss_private = (1 - config.dml_weight) * ce_private + config.dml_weight * kl_private
            loss_proxy = (1 - config.dml_weight) * ce_proxy + config.dml_weight * kl_proxy

            loss_private.backward()
            loss_proxy.backward()

            # 🔥 Gradient Manipulation Attack
            scale_factor = 50.0  # Scaling factor for attack
            if attack_type in ["gradient_attack", "combined"]:
                if config.use_private_SGD:
                    all_grads = [param.grad for param in client.proxy_model.parameters() if param.grad is not None]  
                    if all_grads:
                        modified_grads = scaled_gradient_manipulation(all_grads, scale_factor, alpha=0.7, max_grad_norm=1.0)
                        for param, new_grad in zip(client.proxy_model.parameters(), modified_grads):
                            if new_grad is not None:
                                param.grad = new_grad.clone()
                else:
                    manipulated_gradients = gradient_manipulation(
                        [param.grad for param in client.proxy_model.parameters() if param.grad is not None], "min-max"
                    )
                    for param, new_grad in zip(client.proxy_model.parameters(), manipulated_gradients):
                        if new_grad is not None:
                            param.grad = new_grad.clone()

            # Clip gradients to prevent excessive updates
            torch.nn.utils.clip_grad_norm_(client.private_model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(client.proxy_model.parameters(), max_norm=1.0)

            client.private_opt.step()
            client.proxy_opt.step()
            
            # Apply model weights poisoning attack
            if attack_type in ["combined", "model_weights_attack"]:
                for param in client.proxy_model.parameters():    
                    param.data += scale_factor * (torch.randn_like(param.data) * scale_factor) 

            # Apply model layer modification attack
            if attack_type in ["combined", "model_layers_attack"]:
                inject_backdoor_into_model(client.proxy_model)

            # total_private_loss += loss_private
            # avg_private_loss = total_private_loss / (idx + 1)
            pred_private = pred_private.argmax(dim=-1)
            correct_private += pred_private.eq(target.view_as(pred_private)).sum()
            acc_private = correct_private / client.private_data[0].shape[0]
            train_private_acc.append(acc_private.cpu())

            # total_proxy_loss += loss_proxy
            # avg_proxy_loss = total_proxy_loss / (idx + 1)
            pred_proxy = pred_proxy.argmax(dim=-1)
            correct_proxy += pred_proxy.eq(target.view_as(pred_proxy)).sum()
            acc_proxy = correct_proxy / client.private_data[0].shape[0]
            train_proxy_acc.append(acc_proxy.cpu())
            
            if config.use_private_SGD:
                if config.dp_optimizer_selection == "old":
                    epsilon, optimal_alpha = client.proxy_opt.privacy_engine.get_privacy_spent(delta)
                    client.privacy_budget = epsilon
                else:
                    epsilon = client.proxy_opt.privacy_engine.accountant.get_epsilon(delta)
                    client.privacy_budget = epsilon   
            
            train_privacy_budget.append(epsilon)

            if logger is not None and config.verbose:
                if config.use_private_SGD:
                    logger.info(f"Epoch {e}: private_acc={acc_private:.4f}, proxy_acc={acc_proxy:.4f}, ε={epsilon:.2f} and δ={delta:.4f} at α={optimal_alpha:.2f}")
                else:
                    logger.info(f"Epoch {e}: private_acc={acc_private:.4f}, proxy_acc={acc_proxy:.4f}")       

    return (np.array(train_private_acc, dtype=np.float32),
            np.array(train_proxy_acc, dtype=np.float32),
            np.array(train_privacy_budget, dtype=np.float32))

def regular_training_loop(client, logger, config):
    """
    Federated Learning Training Loop with optional Gradient Manipulation Attacks.
    """
    attack_type = config.attack_type  # Directly retrieve attack type from config
    # Print a statement based on the attack type
    if attack_type == "combined":
        print("🚨 Using combined attack: adaptive_backdoor + gradient_attack + model weights attack + model layers attack. For model poisoning")
        print()
    elif attack_type == "adaptive_backdoor":
        print("🎯 Using adaptive backdoor attack for model poisoning.")
        print()
    elif attack_type == "gradient_attack":
        print("🔥 Using gradient manipulation attackfor model poisoning.")
        print()
    elif attack_type == "model_weights_attack":
        print("⚡ Using model weights poisoning attack for model poisoning.")
        print()
    elif attack_type == "model_layers_attack":
        print("🔁 Using model layer modification attack for model poisoning.")
        print()
    else:
        print("No model poisoning attack applied. Running standard training.")
        print()

    client.proxy_model.train()
    train_proxy_acc = []
    train_privacy_budget = []

    epsilon = 0
    delta = 1.0 / client.private_data[0].shape[0]

    if config.proxy_model_type in ["CNN1lora", "VisionTransformerlora"]:
        lora.mark_only_lora_as_trainable(client.proxy_model)

    for e in range(config.n_epochs):
        train_loader = utils.data_loader(config.dataset,
                                         client.private_data[0],
                                         client.private_data[1],
                                         config.batch_size)

        correct_proxy = 0.0
        acc_proxy = 0.0

        for idx, (data, target) in enumerate(train_loader):
            client.proxy_opt.zero_grad()

            data = torch.from_numpy(data).to(client.device)
            target = torch.from_numpy(target).to(client.device)

            # 🎯 Apply adaptive backdoor attack to a fraction of batches
            if attack_type in ["adaptive_backdoor", "combined"] and np.random.rand() < 0.9:  # 80% of batches poisoned
                data, target = apply_adaptive_backdoor(data, target)

            pred_proxy = client.proxy_model(data)

            loss_proxy = CE_Loss(pred_proxy, target)
            loss_proxy.backward()

            original_gradients = [p.grad.clone() if p.grad is not None else None for p in client.proxy_model.parameters()]
            #print("Original Gradients Norm:", [torch.norm(g).item() if g is not None else None for g in original_gradients])
            
            # 🔥 Scaled Gradient Manipulation (for DP-SGD compatibility)
            scale_factor = 50.0
            if config.use_private_SGD:
                #max_grad_norm = 1.0  #config.l2_norm_clip
                if attack_type in ["gradient_attack", "combined"]:
                    all_grads = [param.grad for param in client.proxy_model.parameters() if param.grad is not None]  

                    if all_grads:  # Ensure there are valid gradients
                        modified_grads = scaled_gradient_manipulation(all_grads, scale_factor, alpha=0.7, max_grad_norm=1.0)
                        """
                        # 🚀 Apply modified gradients back to model
                        i = 0
                        for param in client.proxy_model.parameters():
                            if param.grad is not None:
                                param.grad = modified_grads[i]
                                i += 1  # Move to next modified gradient """

                        # 🚀 Apply poisoned gradients back to the model
                        for param, new_grad in zip(client.proxy_model.parameters(), modified_grads):
                            if new_grad is not None:
                                param.grad = new_grad.clone()
            else:
                if attack_type in ["gradient_attack", "combined"]:
                    manipulated_gradients = gradient_manipulation(
                        [param.grad for param in client.proxy_model.parameters() if param.grad is not None], "min-max"
                    )
                    for param, new_grad in zip(client.proxy_model.parameters(), manipulated_gradients):
                        if new_grad is not None:
                            param.grad = new_grad.clone()  # Ensure correct gradient replacement 


            # Debug: Check manipulated gradients
            manipulated_norms = [torch.norm(p.grad).item() if p.grad is not None else None for p in client.proxy_model.parameters()]
            #print("Manipulated Gradients Norm:", manipulated_norms)

            client.proxy_opt.step()

            # 🔥 Directly modify model weights after optimization
            if attack_type in ["combined", "model_weights_attack"]:
                for param in client.proxy_model.parameters():    
                    #param.data = -param.data + scale_factor * torch.randn_like(param.data)  # Flip & inject noise
                    param.data += scale_factor * (torch.randn_like(param.data) * scale_factor) # Strong noise injection

            # 🔥 Inject model weight shift at the end of training
            if attack_type in ["combined", "model_layers_attack"]:
                inject_backdoor_into_model(client.proxy_model)

            pred_proxy = pred_proxy.argmax(dim=-1)
            correct_proxy += pred_proxy.eq(target.view_as(pred_proxy)).sum()
            acc_proxy = correct_proxy / client.private_data[0].shape[0]
            train_proxy_acc.append(acc_proxy.cpu())

            # Differential Privacy Accounting
            if config.use_private_SGD:
                if config.dp_optimizer_selection == "old":
                    epsilon, optimal_alpha = client.proxy_opt.privacy_engine.get_privacy_spent(delta)
                else:
                    epsilon = client.proxy_opt.privacy_engine.accountant.get_epsilon(delta)
                client.privacy_budget = epsilon

            train_privacy_budget.append(epsilon)

        if logger is not None and config.verbose:
            if config.use_private_SGD:
                logger.info(f"Epoch {e}: train_proxy_acc={acc_proxy:.4f}, ε={epsilon:.2f}, δ={delta:.4f}")
            else:
                logger.info(f"Epoch {e}: train_proxy_acc={acc_proxy:.4f}")

    return (None,
            np.array(train_proxy_acc, dtype=np.float32),
            np.array(train_privacy_budget, dtype=np.float32))

##################################### without any attack #######################################################
'''
def regular_training_loop(client, logger, config):

    client.proxy_model.train()
    train_proxy_acc = []
    train_privacy_budget = []

    epsilon = 0
    delta = 1.0 / client.private_data[0].shape[0]

    # Mark only LoRA parameters as trainable if models are CNN1lora or VisionTransformerlora
    if config.proxy_model_type in ["CNN1lora", "VisionTransformerlora"]:
        lora.mark_only_lora_as_trainable(client.proxy_model)

    for e in range(config.n_epochs):

        train_loader = utils.data_loader(config.dataset,
                                         client.private_data[0],
                                         client.private_data[1],
                                         config.batch_size)

        # total_proxy_loss = 0.0
        correct_proxy = 0.0
        acc_proxy = 0.0

        for idx, (data, target) in enumerate(train_loader):

            client.proxy_opt.zero_grad()

            data = torch.from_numpy(data).to(client.device)
            target = torch.from_numpy(target).to(client.device)
            pred_proxy = client.proxy_model(data)

            loss_proxy = CE_Loss(pred_proxy, target)
            loss_proxy.backward()

            client.proxy_opt.step()

            # total_proxy_loss += loss_proxy
            # avg_proxy_loss = total_proxy_loss / (idx + 1)
            pred_proxy = pred_proxy.argmax(dim=-1)
            correct_proxy += pred_proxy.eq(target.view_as(pred_proxy)).sum()
            acc_proxy = correct_proxy / client.private_data[0].shape[0]
            train_proxy_acc.append(acc_proxy.cpu())

            if config.use_private_SGD:
                if config.dp_optimizer_selection == "old":
                    epsilon, optimal_alpha = client.proxy_opt.privacy_engine.get_privacy_spent(delta)
                    client.privacy_budget = epsilon
                else:
                    epsilon = client.proxy_opt.privacy_engine.accountant.get_epsilon(delta)
                    client.privacy_budget = epsilon  
            
            train_privacy_budget.append(epsilon)

        if logger is not None and config.verbose:
            if config.use_private_SGD:
                logger.info(f"Epoch {e}: train_proxy_acc={acc_proxy:.4f}, ε={epsilon:.2f} and δ={delta:.4f} at α={optimal_alpha:.2f}")
            else:
                logger.info(f"Epoch {e}: train_proxy_acc={acc_proxy:.4f}")

    return (None,
            np.array(train_proxy_acc, dtype=np.float32),
            np.array(train_privacy_budget, dtype=np.float32))
'''