import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import math
from torch_geometric.data import Data, Batch

from gnn_captioning.config import NUM_PATCHES_GCN, ENCODE_DIM_GCN, HIDDEN_DIM_GCN, GCN_LAYER, SIN_PE_GCN

class GCN_Encoder(nn.Module):
    def __init__(self,
                 img_size=224,
                 num_patches = NUM_PATCHES_GCN,
                 in_channels=3,
                 encode_dim=ENCODE_DIM_GCN,
                 use_sin_pe=SIN_PE_GCN,
                 gcn_layers=GCN_LAYER,
                 hidden_dim=HIDDEN_DIM_GCN):
        super(GCN_Encoder,self).__init__()
        # 参数验证
        sqrt_n = int(math.sqrt(num_patches))
        assert sqrt_n ** 2 == num_patches, "num_patches must be a square number"
        assert img_size % sqrt_n == 0, "img_size should be divided by num_patches"
        
        self.num_patches = num_patches
        self.img_size = img_size
        
        self.patch_size = img_size // sqrt_n
        self.encode_dim = encode_dim
        self.hidden_dim = hidden_dim
        
        # 分块参数计算
        patch_dim = in_channels * (self.patch_size ** 2)
        
        # 节点特征投影
        self.patch_embed = nn.Linear(patch_dim, hidden_dim)
        
        # 位置编码
        if use_sin_pe:
            self.register_buffer(
                'position_embedding',
                self.create_sin_positional_encoding(self.num_patches, hidden_dim)
            )
        else:
            self.position_embedding = nn.Parameter(
                torch.randn(self.num_patches, hidden_dim)
            )
        
        # GCN层
        self.convs = nn.ModuleList()
        # TODO: potential optimization: adding batch norm or layer norm 
        for _ in range(gcn_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # 输出投影
        self.proj_out = nn.Linear(hidden_dim, encode_dim)
        # TODO: potential optimization: adding Xavier initialization
        
        # 预计算边索引
        self.edge_index = self.create_spatial_edges()

    def create_sin_positional_encoding(self, num_patches, dim):
        """创建正弦位置编码"""
        device = next(self.parameters()).device
        position = torch.arange(num_patches, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, device=device).float() * 
            (-math.log(10000.0) / dim))
        pe = torch.zeros(num_patches, dim, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def create_spatial_edges(self):
        """创建基于空间位置的边连接"""
        edge_list = []
        grid_size = int(math.sqrt(self.num_patches))  # 4x4网格
        
        for i in range(grid_size):
            for j in range(grid_size):
                current = i * grid_size + j
                # 连接右邻居
                if j < grid_size - 1:
                    edge_list.append([current, current + 1])
                # 连接下邻居
                if i < grid_size - 1:
                    edge_list.append([current, current + grid_size])
        
        # 添加双向边
        edge_index = torch.tensor(edge_list).t().contiguous()
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # 双向连接
        return edge_index

    def forward(self, x):
        """
        输入: (batch_size, channels, height, width)
        输出: (batch_size, num_patches, encode_dim)
        """
        # Step 1: 分块处理
        batch_size = x.shape[0]
        x = x.unfold(2, self.patch_size, self.patch_size)  # (B, C, H/p, W/p, p)
        x = x.unfold(3, self.patch_size, self.patch_size)  # (B, C, H/p, W/p, p, p)
        x = x.permute(0, 2, 3, 1, 4, 5)  # (B, H/p, W/p, C, p, p)
        x = x.reshape(batch_size, self.num_patches, -1)  # (B, 16, C*p*p)
        
        # Step 2: 节点特征嵌入
        x = self.patch_embed(x)  # (B, 16, hidden_dim)
        
        # Step 3: 添加位置编码
        pos_embed = self.position_embedding.unsqueeze(0)  # (1, 16, hidden_dim)
        # TODO: potential optimization: scale up x by sqrt(dim)
        x = x + pos_embed
        
        # Step 4: 转换到图结构
        data_list = []
        for i in range(batch_size):
            data = Data(
                x=x[i], 
                edge_index=self.edge_index.to(x.device)
            )
            data_list.append(data)
        batch_data = Batch.from_data_list(data_list)
        x = batch_data.x
        # Step 5: 应用GCN层
        for conv in self.convs:
            #TODO: potential optimization: adding norm layer
            x = F.relu(conv(x, batch_data.edge_index))
        
        # Step 6: 恢复形状并投影
        x = x.view(batch_size, self.num_patches, self.hidden_dim)  # (B, 16, hidden_dim)
        x = self.proj_out(x)  # (B, 16, encode_dim)
        
        return x