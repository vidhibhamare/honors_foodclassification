import torch
import torch.nn as nn

class CrossAttentionFusion(nn.Module):
    def __init__(self, cnn_dim=1280, vit_dim=768, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = vit_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Projections
        self.q_proj = nn.Linear(vit_dim, vit_dim)
        self.k_proj = nn.Conv2d(cnn_dim, vit_dim, 1)
        self.v_proj = nn.Conv2d(cnn_dim, vit_dim, 1)
        self.out_proj = nn.Linear(vit_dim, vit_dim)  # Output matches vit_dim

    def forward(self, cnn_feats, vit_feats):
        B, C, H, W = cnn_feats.shape
        L = vit_feats.size(1)
        
        # Project queries (from ViT)
        q = self.q_proj(vit_feats).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Project keys/values (from CNN)
        k = self.k_proj(cnn_feats).view(B, -1, self.num_heads, self.head_dim, H*W).permute(0,2,3,4,1).squeeze(-1)
        v = self.v_proj(cnn_feats).view(B, -1, self.num_heads, self.head_dim, H*W).permute(0,2,3,4,1).squeeze(-1)
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        # Output
        out = (attn @ v).transpose(1, 2).reshape(B, L, -1)
        return self.out_proj(out) + vit_feats  # Residual with matching dims