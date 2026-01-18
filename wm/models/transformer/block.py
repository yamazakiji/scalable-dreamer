import torch

from torch import nn
from torch.nn import functional as F
from einops import rearrange


class RoPE(nn.Module):
    def __init__(self, head_dim: int, max_seq_len=2048, base=10000):
        super().__init__()
        self.head_dim = head_dim
        
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer('inv_freq', inv_freq)
        
        self._compute_freqs(max_seq_len)
    
    def _compute_freqs(self, seq_len):
        # Position indices [0, 1, 2, ..., seq_len-1]
        t = torch.arange(seq_len).type_as(self.inv_freq)
        
        # Compute m * theta for all positions and frequencies
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, dim/2)
        
        # Create cos and sin components
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, dim)
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())
    
    def rotate_half(self, x):
        """Rotate half the dimensions to apply rotation matrix efficiently"""
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat([-x2, x1], dim=-1)
    
    def forward(self, q, k):
        """
        Apply rotary embeddings to queries and keys
        q, k: (batch, seq_len, num_heads, head_dim)
        """
        seq_len = q.shape[2]
        
        # Get cached cos/sin for current sequence length
        cos = self.cos_cached[:seq_len, :]
        sin = self.sin_cached[:seq_len, :]
        
        # Apply rotation: x * cos + rotate_half(x) * sin
        q_embed = q * cos + self.rotate_half(q) * sin
        k_embed = k * cos + self.rotate_half(k) * sin
        
        return q_embed, k_embed
    

class FFNLayer(nn.Module):
    def __init__(self, embed_dim: int, proj_dim: int):
        super().__init__()

        self.w1 = nn.Linear(embed_dim, proj_dim)
        self.w2 = nn.Linear(embed_dim, proj_dim)
        self.w3 = nn.Linear(proj_dim, embed_dim)

    def forward(self, x):
        swish = F.silu(self.w1(x))
        out = swish * self.w2(x)
        return self.w3(out)
    

class SpaceAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        head_dim = int(embed_dim / num_heads)

        self.ln_q = nn.LayerNorm(head_dim)
        self.ln_k = nn.LayerNorm(head_dim)

        self.rope = RoPE(head_dim)
        
        self.num_heads = num_heads

    def forward(self, x, attn_mask=None):
        batch_size = x.shape[0]
        x = rearrange(x, "b t n e -> (b t) n e")
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        q = rearrange(q, "b n (num_heads embed_dim) -> b num_heads n embed_dim", num_heads=self.num_heads)
        k = rearrange(k, "b n (num_heads embed_dim) -> b num_heads n embed_dim", num_heads=self.num_heads)
        v = rearrange(v, "b n (num_heads embed_dim) -> b num_heads n embed_dim", num_heads=self.num_heads)

        # rope
        q, k = self.rope(q, k)

        # qk norm
        q = self.ln_q(q)
        k = self.ln_k(k)

        o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

        o = rearrange(o, "b num_heads n embed_dim -> b n (num_heads embed_dim)", num_heads=self.num_heads)
        o = self.o_proj(o)
        o = rearrange(o, "(b t) n e -> b t n e", b=batch_size)
        return o
    

class TimeAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        head_dim = int(embed_dim / num_heads)
        
        self.ln_q = nn.LayerNorm(head_dim)
        self.ln_k = nn.LayerNorm(head_dim)

        self.rope = RoPE(head_dim)
        
        self.num_heads = num_heads

    def forward(self, x, attn_mask=None):
        batch_size = x.shape[0]
        x = rearrange(x, "b t n e -> (b n) t e")
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        q = rearrange(q, "b n (num_heads embed_dim) -> b num_heads n embed_dim", num_heads=self.num_heads)
        k = rearrange(k, "b n (num_heads embed_dim) -> b num_heads n embed_dim", num_heads=self.num_heads)
        v = rearrange(v, "b n (num_heads embed_dim) -> b num_heads n embed_dim", num_heads=self.num_heads)

        # rope
        q, k = self.rope(q, k)

        #qk norm
        q = self.ln_q(q)
        k = self.ln_k(k)

        # TimeAttention uses causal masking, attn_mask parameter ignored
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        o = rearrange(o, "b num_heads n embed_dim -> b n (num_heads embed_dim)", num_heads=self.num_heads)
        o = self.o_proj(o)
        o = rearrange(o, "(b n) t e -> b t n e", b=batch_size)
        return o


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, attention_type: str):
        super().__init__()

        self.norm1 = nn.RMSNorm(embed_dim)
        self.norm2 = nn.RMSNorm(embed_dim)
        if attention_type == "space":
            self.attention = SpaceAttention(embed_dim, num_heads)
        elif attention_type == "time":
            self.attention = TimeAttention(embed_dim, num_heads)
        self.ffn_layer = FFNLayer(embed_dim, 4 * embed_dim)

    def forward(self, x, attn_mask=None):
        residual = x
        x = self.norm1(x)
        o = self.attention(x, attn_mask=attn_mask)
        o = residual + o

        residual = o
        x = self.norm2(o)
        o = self.ffn_layer(x)
        o = residual + o
        return o