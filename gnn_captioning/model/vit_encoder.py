
import torch
import torch.nn as nn
import math

from gnn_captioning.config import NUM_PATCHES_VIT, HIDDEN_DIM_VIT, SIN_PE_VIT, DEPTH_VIT, HEADS_VIT

class ViT_Encoder(nn.Module):
    def __init__(self, 
                 img_size=224, 
                 num_patches=NUM_PATCHES_VIT, 
                 in_channels=3, 
                 dim=HIDDEN_DIM_VIT,
                 use_sin_pe=SIN_PE_VIT, 
                 depth=DEPTH_VIT, 
                 heads=HEADS_VIT, 
                 mlp_dim=1024, 
                 dropout=0.1, 
                 activation='gelu'):
        super(ViT_Encoder, self).__init__()
        
        # check initialization parameters
        assert dim % 2 == 0, "dim must be an even number"
        sqrt_n = int(math.sqrt(num_patches))
        assert sqrt_n ** 2 == num_patches, "num_patches must be a square number"
        
        self.img_size = img_size
        self.num_patches = num_patches
        self.patch_size = img_size // sqrt_n
        self.use_sin_pe = use_sin_pe
        self.dim = dim
        
        patch_dim = in_channels * self.patch_size ** 2
        self.patch_embedding = nn.Linear(patch_dim, dim)

        # positional Encoding
        if use_sin_pe:
            self.register_buffer(
                'position_embedding',
                self.create_sin_positional_encoding(num_patches, dim)
            )
        else:
            self.position_embedding = nn.Parameter(
                torch.randn(num_patches, dim)  #  [1, N, D]
            )
        
        # Transformer layer
        valid_activations = ['gelu', 'relu', 'tanh']
        assert activation in valid_activations, f"non-support activation : {activation}"
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation=activation
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def create_sin_positional_encoding(self, num_patches, dim):
        """Sin positional encoding"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        position = torch.arange(num_patches, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, device=device).float() * 
            (-math.log(10000.0) / dim)
        )
        pe = torch.zeros(num_patches, dim, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe  # (num_patches, dim)

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        assert height == self.img_size and width == self.img_size, \
            f"input size ({height}x{width}) doesn't match initialization size ({self.img_size}x{self.img_size})"
        
        # 分块处理
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        x = x.view(batch_size, self.num_patches, -1)
        
        # projection & positional encoding
        x = self.patch_embedding(x)  # (B, N, D)
        x = x * math.sqrt(self.dim) # can consider to delete this line
        x = x + self.position_embedding  
        
        # Transformer process
        x = x.permute(1, 0, 2)  # (N, B, D)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # (B, N, D)
        return x
