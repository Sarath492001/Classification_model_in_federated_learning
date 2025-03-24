import torch
import torch.nn as nn
from model import LeNet5, MLP, CNN1, CNN2, VisionTransformer, CNN1lora, VisionTransformerlora
from opacus import PrivacyEngine  
from torch.utils.data import DataLoader, TensorDataset

def init_model(model_type, in_channel, n_class, lorarank):

    if model_type == 'LeNet5':
        model = LeNet5(in_channel, n_class)
    elif model_type == 'MLP':
        model = MLP(n_class)
    elif model_type == 'CNN1':
        model = CNN1(in_channel, n_class)
    elif model_type == 'CNN2':
        model = CNN2(in_channel, n_class)
    elif model_type == 'CNN1_2layers':
        model = CNN1(in_channel, n_class)
    elif model_type == 'VisionTransformer':
        model = VisionTransformer(in_channel, n_class)
    elif model_type == 'CNN1lora':
        model = CNN1lora(in_channel, n_class, lorarank) 
    elif model_type == 'VisionTransformerlora':
        model = VisionTransformerlora(in_channel, n_class, lorarank) 
    else:
        raise ValueError(f"Unknown model type {model_type}")
    return model


def init_optimizer(model, config):

    if config.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=config.lr,
                                    momentum=config.momentum,
                                    weight_decay=5e-4)
    elif config.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=config.lr,
                                     weight_decay=1e-4)
    else:
        raise ValueError("Unknown optimizer")

    return optimizer


############ opacus 0.14.0 compatible #############
def init_dp_optimizer_old(model, data_size, config):
    opt = init_optimizer(model, config)
    orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
    privacy_engine = PrivacyEngine(
        model,
        sample_rate=config.batch_size / data_size,
        alphas=orders,
        noise_multiplier=config.noise_multiplier,
        max_grad_norm=config.l2_norm_clip,
        #secure_mode=config.secure_mode  # only if secure mode is needed
    )
    # print(f"Using DP-SGD with sigma={config.noise_multiplier} and clipping norm max={config.l2_norm_clip}")
    privacy_engine.attach(opt)
    return opt 

######################## opacus latest version 1.4.1 compatible (from client private train_data)###############################################
def init_dp_optimizer_new(model, train_data, config):
    optn = init_optimizer(model, config)
    #orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
    # Unpack train_data
    train_X, train_y = train_data

    # Convert train_data to PyTorch tensors
    train_inputs = torch.from_numpy(train_X).float()
    train_targets = torch.from_numpy(train_y)

    # Create a DataLoader from the tensors
    train_dataset = TensorDataset(train_inputs, train_targets)
    data_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    privacy_engine = PrivacyEngine(     
        accountant=config.accountant,  # Specify the accountant mechanism
        secure_mode=config.secure_mode, # Set secure_mode based on your config
    )

    # Make the model private
    model,opt, _ = privacy_engine.make_private(
        module=model,
        optimizer=optn,
        data_loader=data_loader,
        noise_multiplier=config.noise_multiplier,
        max_grad_norm=config.l2_norm_clip,
    )

    # Attach the privacy engine to the optimizer
    opt.privacy_engine = privacy_engine
    return opt
 
class Client(nn.Module):

    def __init__(self, data, config):

        super(Client, self).__init__()

        self.private_data = data

        self.private_model = init_model(config.private_model_type, config.in_channel, config.n_class, config.lora_rank).to(config.device)
        self.proxy_model = init_model(config.proxy_model_type, config.in_channel, config.n_class, config.lora_rank).to(config.device)

        if config.use_private_SGD:
            #Using DP proxies, train private model without DP
            if config.dp_optimizer_selection not in ["old", "new"]:
                raise ValueError("Invalid choice for dp_optimizer_selection. Must be 'old' or 'new'.")
            
            if config.dp_optimizer_selection == "old": 
                self.proxy_opt = init_dp_optimizer_old(self.proxy_model, self.private_data[0].shape[0], config)
            else:
                #self.proxy_opt = init_dp_optimizer(self.proxy_model, self.train_loader, config)
                self.proxy_opt = init_dp_optimizer_new(self.proxy_model, self.private_data, config)
        else:
            # no DP
            self.proxy_opt = init_optimizer(self.proxy_model, config)

        # Private model training always not DP
        self.private_opt = init_optimizer(self.private_model, config)

        self.device = config.device
        self.tot_epochs = 0
        self.privacy_budget = 0.

        
