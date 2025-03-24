import torch
import torch.nn as nn
import torch.nn.functional as F
import loralib as lora
from opacus.layers.dp_multihead_attention import DPMultiheadAttention

class LeNet5(nn.Module):

    def __init__(self, in_channel, n_class):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc_shape = 16 * 4 * 4
        self.fc1 = nn.Linear(self.fc_shape, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_class)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.reshape(-1, self.fc_shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MLP(nn.Module):

    def __init__(self, n_class):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(3 * 28 * 28, 200)  # Adjust the input size for 3-channel images
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, n_class)


    def forward(self, x):
        x = x.reshape(-1, 3 * 28 * 28)  # Adjust the input size for 3-channel images
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        


#################################### 4 layers model ##############################################
class CNN1(nn.Module):
    def __init__(self, in_channel, n_class):
        super(CNN1, self).__init__()
        # First Convolutional Block
        self.conv1 = nn.Conv2d(in_channel, 32, 3, padding=1)
        self.gn1 = nn.GroupNorm(4, 32)  # GroupNorm with 4 groups
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.gn2 = nn.GroupNorm(8, 64)  # GroupNorm with 8 groups
        
        # Second Convolutional Block
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.gn3 = nn.GroupNorm(16, 128)  # GroupNorm with 16 groups
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.gn4 = nn.GroupNorm(32, 256)  # GroupNorm with 32 groups
        
        # Third Convolutional Block
        self.conv5 = nn.Conv2d(256, 512, 3, padding=1)
        self.gn5 = nn.GroupNorm(64, 512)  # GroupNorm with 64 groups
        self.conv6 = nn.Conv2d(512, 512, 3, padding=1)
        self.gn6 = nn.GroupNorm(64, 512)  # GroupNorm with 64 groups
        
        # Fourth Convolutional Block (Newly added)
        self.conv7 = nn.Conv2d(512, 1024, 3, padding=1)
        self.gn7 = nn.GroupNorm(128, 1024)  # GroupNorm with 128 groups
        self.conv8 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.gn8 = nn.GroupNorm(128, 1024)  # GroupNorm with 128 groups

        # Adaptive Pooling to ensure output size (1,1) before fully connected layers
        #self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully Connected Layers
        self.fc_shape = 1024   # Corrected for additional layers, assuming 256x256 input
        self.fc1 = nn.Linear(self.fc_shape, 2048)
        self.fc_gn1 = nn.LayerNorm(2048)  # LayerNorm for fully connected layer
        self.fc2 = nn.Linear(2048, 1024)
        self.fc_gn2 = nn.LayerNorm(1024)  # LayerNorm for fully connected layer
        self.fc3 = nn.Linear(1024, 512)
        self.fc_gn3 = nn.LayerNorm(512)  # LayerNorm for fully connected layer
        self.fc4 = nn.Linear(512, n_class)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # First Convolutional Block
        x = F.max_pool2d(F.relu(self.gn1(self.conv1(x))), 2)
        x = F.max_pool2d(F.relu(self.gn2(self.conv2(x))), 2)

        # Second Convolutional Block
        x = F.max_pool2d(F.relu(self.gn3(self.conv3(x))), 2)
        x = F.max_pool2d(F.relu(self.gn4(self.conv4(x))), 2)

        # Third Convolutional Block
        x = F.max_pool2d(F.relu(self.gn5(self.conv5(x))), 2)
        x = F.max_pool2d(F.relu(self.gn6(self.conv6(x))), 2)
        
        # Fourth Convolutional Block (Newly added)
        x = F.max_pool2d(F.relu(self.gn7(self.conv7(x))), 2)
        x = F.relu(self.gn8(self.conv8(x)))

        # Apply adaptive pooling to ensure fixed size (1, 1)
        #x = self.adaptive_pool(x)

        # Flatten
        #x = x.view(x.size(0), -1)
        x = x.view(-1, self.fc_shape)

        # Fully Connected Layers
        x = F.relu(self.fc_gn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.fc_gn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.fc_gn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.fc4(x)
        
        return x


#################################### 4 layers model with lora and dylora ##############################################
class CNN1lora(nn.Module):
    def __init__(self, in_channel, n_class, lorarank):  # 'r' is the low-rank dimension
        super(CNN1lora, self).__init__()
        # First Convolutional Block with LoRA
        self.conv1 = lora.Conv2d(in_channel, 32, 3, padding=1, r=lorarank)  # LoRA applied
        self.gn1 = nn.GroupNorm(4, 32)  # GroupNorm with 4 groups
        self.conv2 = lora.Conv2d(32, 64, 3, padding=1, r=lorarank)  # LoRA applied
        self.gn2 = nn.GroupNorm(8, 64)  # GroupNorm with 8 groups
        
        # Second Convolutional Block with LoRA
        self.conv3 = lora.Conv2d(64, 128, 3, padding=1, r=lorarank)  # LoRA applied
        self.gn3 = nn.GroupNorm(16, 128)  # GroupNorm with 16 groups
        self.conv4 = lora.Conv2d(128, 256, 3, padding=1, r=lorarank)  # LoRA applied
        self.gn4 = nn.GroupNorm(32, 256)  # GroupNorm with 32 groups
        
        # Third Convolutional Block with LoRA
        self.conv5 = lora.Conv2d(256, 512, 3, padding=1, r=lorarank)  # LoRA applied
        self.gn5 = nn.GroupNorm(64, 512)  # GroupNorm with 64 groups
        self.conv6 = lora.Conv2d(512, 512, 3, padding=1, r=lorarank)  # LoRA applied
        self.gn6 = nn.GroupNorm(64, 512)  # GroupNorm with 64 groups
        
        # Fourth Convolutional Block with LoRA
        self.conv7 = lora.Conv2d(512, 1024, 3, padding=1, r=lorarank)  # LoRA applied
        self.gn7 = nn.GroupNorm(128, 1024)  # GroupNorm with 128 groups
        self.conv8 = lora.Conv2d(1024, 1024, 3, padding=1, r=lorarank)  # LoRA applied
        self.gn8 = nn.GroupNorm(128, 1024)  # GroupNorm with 128 groups

        # Adaptive Pooling to ensure output size (1,1) before fully connected layers
        #self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully Connected Layers with LoRA
        self.fc_shape = 1024  # Corrected for additional layers, assuming 256x256 input
        self.fc1 = lora.Linear(self.fc_shape, 2048, r=lorarank)  # LoRA applied
        self.fc_gn1 = nn.LayerNorm(2048)  # LayerNorm for fully connected layer
        self.fc2 = lora.Linear(2048, 1024, r=lorarank)  # LoRA applied
        self.fc_gn2 = nn.LayerNorm(1024)  # LayerNorm for fully connected layer
        self.fc3 = lora.Linear(1024, 512, r=lorarank)  # LoRA applied
        self.fc_gn3 = nn.LayerNorm(512)  # LayerNorm for fully connected layer
        self.fc4 = nn.Linear(512, n_class)  # LoRA not applied as it's the final classification layer

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # First Convolutional Block
        x = F.max_pool2d(F.relu(self.gn1(self.conv1(x))), 2)
        x = F.max_pool2d(F.relu(self.gn2(self.conv2(x))), 2)

        # Second Convolutional Block
        x = F.max_pool2d(F.relu(self.gn3(self.conv3(x))), 2)
        x = F.max_pool2d(F.relu(self.gn4(self.conv4(x))), 2)

        # Third Convolutional Block
        x = F.max_pool2d(F.relu(self.gn5(self.conv5(x))), 2)
        x = F.max_pool2d(F.relu(self.gn6(self.conv6(x))), 2)
        
        # Fourth Convolutional Block
        x = F.max_pool2d(F.relu(self.gn7(self.conv7(x))), 2)
        x = F.relu(self.gn8(self.conv8(x)))

        # Apply adaptive pooling to ensure fixed size (1, 1)
        #x = self.adaptive_pool(x)

        # Flatten
        #x = x.view(x.size(0), -1)
        x = x.view(-1, self.fc_shape)
        
        # Fully Connected Layers
        x = F.relu(self.fc_gn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.fc_gn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.fc_gn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.fc4(x)
        
        return x

################################ original model ####################################################
class CNN1_2layers(nn.Module):

    def __init__(self, in_channel, n_class):
        super(CNN1_2layers, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # reshape for both MNIST and CIFAR based on # of channels
        self.fc_shape = 16 * 30 * 30  #16 * int(4.5 + in_channel * 0.5) * int(4.5 + in_channel * 0.5)
        self.fc1 = nn.Linear(self.fc_shape, 64)
        self.fc2 = nn.Linear(64, n_class)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.contiguous().view(-1, self.fc_shape)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
####################################################################################################
class CNN2(nn.Module):
    def __init__(self, in_channel, n_class):
        super(CNN2, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channel, 128, 3, padding=1)  # Padding to maintain the size
        # Second convolutional layer
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)  # Padding to maintain the size
        
        # Calculate the output size after the convolutions and max-pooling layers
        self.fc_shape = 128 * 32 * 32  # Since after two 2x2 max-pooling layers, the size becomes 32x32
        
        # Fully connected layer
        self.fc = nn.Linear(self.fc_shape, n_class)

    def forward(self, x):
        # Apply first convolution, ReLU activation, and max-pooling
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        # Apply second convolution, ReLU activation, and max-pooling
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        
        # Flatten the feature maps to pass to the fully connected layer
        x = x.contiguous().view(-1, self.fc_shape)
        # Fully connected layer
        x = self.fc(x)
        return x
    
####################################### vision transformers normal ###################################################

class DPTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, dp_mode=True):
        super(DPTransformerEncoderLayer, self).__init__()
        self.self_attn = DPMultiheadAttention(d_model, nhead, dropout=0.0 if dp_mode else dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Identity() if dp_mode else nn.Dropout(dropout) # Disable dropout if dp_mode is True
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Identity() if dp_mode else nn.Dropout(dropout)
        self.dropout2 = nn.Identity() if dp_mode else nn.Dropout(dropout)

        self.activation = nn.ReLU()  # Or another activation function

    def forward(self, src):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class VisionTransformer(nn.Module):

    def __init__(self, in_channel, n_class, image_size=128, patch_size=16, num_layers=4, d_model=768, num_heads=12, mlp_dim=3072, dropout=0.1, dp_mode=True):
        super(VisionTransformer, self).__init__()
        
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.d_model = d_model
        
        # Embedding layer
        self.embedding = nn.Conv2d(in_channel, d_model, kernel_size=patch_size, stride=patch_size)

        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        
        # Transformer encoder for dp opacus 1.4.1
        encoder_layers = [DPTransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=mlp_dim, dropout=dropout, dp_mode=dp_mode) for _ in range(num_layers)]
        self.transformer_encoder = nn.Sequential(*encoder_layers)

        # Classification head
        self.fc = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_class)
        )
        
    def forward(self, x):

        x = self.embedding(x)  
        x = x.flatten(2)  
        x = x.transpose(1, 2)  
        x = x + self.positional_encoding
        # Apply transformer encoder
        x = self.transformer_encoder(x)
        
        # Classification head
        x = x.mean(dim=1)  # Global average pooling
        
        x = self.fc(x)
        return x

####################################### vision transformers with lora and dylora  ###################################################
# DPTransformerEncoderLayer with LoRA applied outside DPMultiheadAttention
class DPTransformerEncoderLayerlora(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, dp_mode=True, r=8):
        super(DPTransformerEncoderLayerlora, self).__init__()
        
        # Use DPMultiheadAttention from Opacus without modification
        self.self_attn = DPMultiheadAttention(d_model, nhead, dropout=0.0 if dp_mode else dropout)

        # Apply LoRA to the linear layers before and after the self-attention block
        #$self.lora_qkv = lora.Linear(d_model, d_model, r=lorarank)
        self.linear1 = lora.Linear(d_model, dim_feedforward, r=r)
        self.linear2 = lora.Linear(dim_feedforward, d_model, r=r)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Identity() if dp_mode else nn.Dropout(dropout)
        self.dropout2 = nn.Identity() if dp_mode else nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, src):
        # Apply LoRA to the input before attention (query, key, value projections)
        #src_lora = self.lora_qkv(src)

        # Use DPMultiheadAttention as usual from Opacus
        src2, _ = self.self_attn(src, src, src)    #src2, _ = self.self_attn(src_lora, src_lora, src_lora)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Feedforward network with LoRA
        src2 = self.linear2(self.activation(self.linear1(src)))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

# Vision Transformer with LoRA and Opacus DPMultiheadAttention
class VisionTransformerlora(nn.Module):
    def __init__(self, in_channel, n_class, lorarank, image_size=128, patch_size=16, num_layers=12, d_model=768, num_heads=12, mlp_dim=3072, dropout=0.1, dp_mode=True):
        super(VisionTransformerlora, self).__init__()

        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.d_model = d_model

        # Embedding layer
        # Apply LoRA to the embedding layer
        self.embedding = lora.Conv2d(in_channel, d_model, kernel_size=patch_size, stride=patch_size, r=lorarank)
        #self.embedding = nn.Conv2d(in_channel, d_model, kernel_size=patch_size, stride=patch_size)

        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.zeros(1, self.num_patches, d_model))

        # Transformer encoder with LoRA applied to the linear layers around DPMultiheadAttention
        encoder_layers = [DPTransformerEncoderLayerlora(d_model=d_model, nhead=num_heads, dim_feedforward=mlp_dim, dropout=dropout, dp_mode=dp_mode, r=lorarank) for _ in range(num_layers)]
        self.transformer_encoder = nn.Sequential(*encoder_layers)

        # Classification head
        self.fc = nn.Sequential(
            nn.LayerNorm(d_model),
            #nn.Linear(d_model, n_class)
            lora.Linear(d_model, n_class, r=lorarank)
        )

    def forward(self, x):
        # Convert input image to patches
        x = self.embedding(x)  
        x = x.flatten(2)  
        x = x.transpose(1, 2)  
        x = x + self.positional_encoding

        # Apply transformer encoder
        x = self.transformer_encoder(x)

        # Classification head
        x = x.mean(dim=1)  # Global average pooling
        x = self.fc(x)
        return x
