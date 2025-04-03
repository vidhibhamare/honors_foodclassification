import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionFusion(nn.Module):
    def __init__(self, cnn_dim=1280, vit_dim=768, num_heads=8):
        super().__init__()
        assert vit_dim % num_heads == 0, "vit_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = vit_dim // num_heads
        
        # Query projections (from ViT features)
        self.q_proj = nn.Linear(vit_dim, vit_dim)
        
        # Key/Value projections (from CNN features)
        self.k_proj = nn.Conv2d(cnn_dim, vit_dim, kernel_size=1)
        self.v_proj = nn.Conv2d(cnn_dim, vit_dim, kernel_size=1)
        
        # Output projection
        self.out_proj = nn.Linear(vit_dim, vit_dim)

    def forward(self, cnn_feats, vit_feats):
        B, C, H, W = cnn_feats.shape
        L = vit_feats.size(1)  # Sequence length (49 for 224x224 patches)
        
        # Project queries (from ViT)
        q = self.q_proj(vit_feats)  # [B, L, vit_dim]
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B, heads, L, head_dim]
        
        # Project keys/values (from CNN)
        k = self.k_proj(cnn_feats)  # [B, vit_dim, H, W]
        v = self.v_proj(cnn_feats)  # [B, vit_dim, H, W]
        
        # Reshape keys/values
        k = k.view(B, self.num_heads, self.head_dim, H*W)  # [B, heads, head_dim, HW]
        v = v.view(B, self.num_heads, self.head_dim, H*W)  # [B, heads, head_dim, HW]
        
        # Scaled dot-product attention
        attn_weights = (q @ k) * (self.head_dim ** -0.5)  # [B, heads, L, HW]
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # Apply attention to values
        out = attn_weights @ v.transpose(-2, -1)  # [B, heads, L, head_dim]
        out = out.transpose(1, 2).reshape(B, L, -1)  # [B, L, vit_dim]
        
        # Output projection + residual
        return self.out_proj(out) + vit_feats