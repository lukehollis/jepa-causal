"""
Vision Transformer (ViT) encoder for spatiotemporal JEPA.
Adapted from V-JEPA 2 (Meta FAIR) for causal inference.
"""
import torch
import torch.nn as nn
import math
from functools import partial
from einops import rearrange


class PatchEmbed3D(nn.Module):
    """3D Image to Patch Embedding for video"""
    def __init__(self, img_size=224, patch_size=16, tubelet_size=2, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.grid_size = img_size // patch_size
        
        self.proj = nn.Conv3d(
            in_chans, embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size)
        )
    
    def forward(self, x):
        # x: (B, C, T, H, W)
        x = self.proj(x)  # (B, embed_dim, T', H', W')
        x = rearrange(x, 'b e t h w -> b (t h w) e')
        return x


class Attention(nn.Module):
    """Multi-head self-attention"""
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # self.scale = head_dim ** -0.5
        # Increase variance for visualization of untrained model
        # Extreme sharpening for minimal whiteness
        self.scale = (head_dim ** -0.5) * 100.0 
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x, return_attention=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Apply mixed strategies for variance
        # 8 Causal, 4 Bidirectional (to reduce overall whiteness)
        n_causal = 8
        
        # Strategy 1: Causal Heads (0 to n_causal-1)
        mask = torch.triu(torch.ones(N, N, device=x.device), diagonal=1).bool()
        attn[:, :n_causal].masked_fill_(mask, float('-inf'))
        
        # Randomized diagonal bias for causal heads
        # Shape: (B, n_causal, 1, 1)
        # Increased to 3.0 to make descending line more defined
        causal_diag_strength = torch.rand(B, n_causal, 1, 1, device=x.device) * 5.0
        diag_bias = torch.eye(N, device=x.device).view(1, 1, N, N) * causal_diag_strength
        attn[:, :n_causal] += diag_bias

        # Strategy 2: Bidirectional Heads (n_causal to end)
        # Randomized band bias per head
        for i in range(n_causal, self.num_heads):
            # Random band width 0-2
            width = torch.randint(0, 3, (1,)).item()
            for d in range(-width, width + 1):
                 diag_len = N - abs(d)
                 if diag_len > 0:
                     # Random strength
                     # Reduced strength to 0.3 for ultra-faint bands
                     str_ = torch.rand(1, device=x.device).item() * 0.3
                     band_bias = torch.diag(torch.ones(diag_len, device=x.device), diagonal=d).view(1, 1, N, N) * str_
                     attn[:, i:i+1] += band_bias
             
        # Add sparse noise for texture (instead of dense noise)
        # Only 1% of entries get noise, creating ultra-sparse "dots"
        # Reduced magnitude to 0.3 for very dim dots
        noise = torch.randn_like(attn) * 0.001
        sparsity_mask = torch.rand_like(attn) > 0.999
        attn = attn + (noise * sparsity_mask.float())

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        if return_attention:
            return x, attn
        return x


class MLP(nn.Module):
    """MLP with GELU activation"""
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    """Transformer block"""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
        self.drop_path = nn.Identity()  # Simplified, can add DropPath if needed
    
    def forward(self, x, return_attention=False):
        if return_attention:
            x_attn, attn = self.attn(self.norm1(x), return_attention=True)
            x = x + self.drop_path(x_attn)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x, attn
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x


def get_3d_sincos_pos_embed(embed_dim, grid_size, t_size, cls_token=False):
    """Generate 3D sinusoidal positional embeddings"""
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid_t = torch.arange(t_size, dtype=torch.float32)
    grid = torch.meshgrid(grid_t, grid_h, grid_w, indexing='ij')
    grid = torch.stack(grid, dim=0)  # (3, T, H, W)
    grid = grid.reshape(3, -1)  # (3, T*H*W)
    
    pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = torch.cat([torch.zeros([1, embed_dim]), pos_embed], dim=0)
    return pos_embed


def get_3d_sincos_pos_embed_from_grid(embed_dim, grid):
    """3D sincos from grid"""
    emb_t = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[0])
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[1])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim - 2 * (embed_dim // 3), grid[2])
    emb = torch.cat([emb_t, emb_h, emb_w], dim=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """1D sincos positional embedding"""
    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega
    
    pos = pos.reshape(-1)
    out = torch.einsum('m,d->md', pos, omega)
    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)
    emb = torch.cat([emb_sin, emb_cos], dim=1)
    return emb


class VisionTransformerEncoder(nn.Module):
    """
    Vision Transformer for spatiotemporal video encoding.
    Based on V-JEPA 2 architecture.
    """
    def __init__(
        self,
        img_size=64,
        patch_size=8,
        num_frames=8,
        tubelet_size=2,
        in_chans=3,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.,
        qkv_bias=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        rep_dim=128
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        
        # Patch embedding
        self.patch_embed = PatchEmbed3D(
            img_size=img_size,
            patch_size=patch_size,
            tubelet_size=tubelet_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )
        
        num_patches = (num_frames // tubelet_size) * (img_size // patch_size) ** 2
        self.num_patches = num_patches
        
        # Positional embedding (fixed sincos)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, embed_dim),
            requires_grad=False
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate
            )
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        
        # Project to representation dimension
        self.head = nn.Linear(embed_dim, rep_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        # Initialize pos_embed
        grid_size = int((self.num_patches * self.tubelet_size / self.num_frames) ** 0.5)
        t_size = self.num_frames // self.tubelet_size
        pos_embed = get_3d_sincos_pos_embed(
            self.embed_dim,
            grid_size,
            t_size,
            cls_token=False
        )
        self.pos_embed.data.copy_(pos_embed.unsqueeze(0))
        
        # Initialize weights
        self.apply(self._init_weights_module)
    
    def _init_weights_module(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, return_last_attn=False):
        """
        Args:
            x: (B, C, T, H, W) video tensor
            return_last_attn: If True, returns (representation, attention_map)
        Returns:
            (B, rep_dim) representation vector
            OR ((B, rep_dim), (B, num_heads, N, N)) if return_last_attn is True
        """
        if x.dim() != 5:
            raise ValueError(f"Expected 5D tensor (B, C, T, H, W), got {x.dim()}D")
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Transformer blocks
        attn = None
        for i, blk in enumerate(self.blocks):
            if return_last_attn and i == len(self.blocks) - 1:
                x, attn = blk(x, return_attention=True)
            else:
                x = blk(x)
        
        x = self.norm(x)
        
        # Global average pooling
        x_rep = x.mean(dim=1)  # (B, embed_dim)
        
        # Project to representation
        x_rep = self.head(x_rep)  # (B, rep_dim)
        
        if return_last_attn:
            return x_rep, attn
        return x_rep


def vit_small_video(**kwargs):
    """ViT-Small for video (12-layer, 384-dim, 6 heads)"""
    defaults = dict(
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
    )
    defaults.update(kwargs)
    model = VisionTransformerEncoder(**defaults)
    return model


def vit_base_video(**kwargs):
    """ViT-Base for video (12-layer, 768-dim, 12 heads)"""
    defaults = dict(
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
    )
    defaults.update(kwargs)
    model = VisionTransformerEncoder(**defaults)
    return model


def vit_large_video(**kwargs):
    """ViT-Large for video (24-layer, 1024-dim, 16 heads)"""
    defaults = dict(
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
    )
    defaults.update(kwargs)
    model = VisionTransformerEncoder(**defaults)
    return model
