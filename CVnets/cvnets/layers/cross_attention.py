import torch
import torch.nn as nn

class CrossAttentionFusion(nn.Module):
    def __init__(self, cnn_dim, vit_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (cnn_dim // num_heads) ** -0.5
        
        self.q = nn.Linear(vit_dim, cnn_dim)
        self.kv = nn.Conv2d(cnn_dim, cnn_dim * 2, 1)
        self.proj = nn.Linear(cnn_dim, cnn_dim)

    def forward(self, cnn_feats, vit_feats):
        B, C, H, W = cnn_feats.shape
        q = self.q(vit_feats).view(B, -1, self.num_heads, C // self.num_heads).permute(0,2,1,3)  # [B, heads, L, D]
        
        kv = self.kv(cnn_feats).view(B, 2, self.num_heads, C // self.num_heads, H*W)
        k, v = kv.unbind(1)  # [B, heads, D, HW]
        
        attn = (q @ k) * self.scale  # [B, heads, L, HW]
        attn = attn.softmax(dim=-1)
        
        out = (attn @ v.transpose(-2,-1)).permute(0,2,1,3).reshape(B, -1, C)
        return self.proj(out) + vit_feats  # Residual