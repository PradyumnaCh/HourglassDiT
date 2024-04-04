# To implement
# 1. Adapted RoPE for 2D image
# 2. RMSNorm/AdaRMSNorm
# 3. HDiT FFN block - mapping block in their code
# 4. Cosine similarity attention
# -- 1. Global/local HDiT blocks
# 1. neighborhood attention / windowed attention
# 5. PixelShuffle, PixelUnshuffle
# 6. Lerp

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    r"""Root Mean Square Normalization Layer
    Normalizes the input to have unit root mean square

    Parameters:
        dim (`int`)
            The number of features in the input tensor
        bias (`bool`)
            If True, adds a learnable bias to the output
        eps (`float`)
            A small value to prevent division by zero
    """

    def __init__(self, dim, bias=False, eps=1e-8):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(dim))
        if bias:
            self.register_parameter("bias", nn.Parameter(torch.zeros(dim)))

    def forward(self, x):
        norm = x.norm(2, dim=-1, keepdim=True)
        x = x / (norm + self.eps)
        x = x * self.scale
        return x


class AdaRMSNorm(nn.Module):
    r"""Adaptive RMS Normalization Layer
    Replaces learned scale from RMSNorm with an adaptable scale

    Parameters:
        dim (`int`)
            The number of features in the input tensor
        eps (`float`)
            A small value to prevent division by zero
    """

    def __init__(self, dim, bias=False, eps=1e-8):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.bias = bias

        if bias:
            self.register_parameter("bias", nn.Parameter(torch.zeros(dim)))

    def forward(self, x, scale) -> torch.Tensor:
        norm = x.norm(2, dim=-1, keepdim=True) / (self.dim**0.5)
        x = x / (norm + self.eps)
        x = x * scale
        if self.bias:
            x = x + self.bias
        return x


class CosineSimilarityAttention(nn.Module):
    r"""Cosine similarity attention mechanism
    Modified CosineSimilarityAttention as implemented in

    Parameters:

    """

    def __init__(self, dim, L=256, heads=8, dim_head=64, dropout=0.0, eps=1e-8):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.eps = eps
        self.register_buffer(
            "scale", nn.Parameter(torch.ones(heads) * torch.log2(L**2 - L))
        )

        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        b, n, d, h = *x.shape, self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(b, n, h, -1).transpose(1, 2), (q, k, v))

        # normalize q, k along heads dimension for cosine similarity
        q = q * torch.rsqrt(q.norm(2, dim=1, keepdim=True) + self.eps)
        k = k * torch.rsqrt(k.norm(2, dim=1, keepdim=True) + self.eps)

        # cosine similarity attention
        dots = torch.einsum("b h i d, b h j d -> b h i j", q, k)
        scores = F.softmax(dots / self.scale, dim=-1)
        out = (
            torch.einsum("b h i j, b h j d -> b h i d", scores, v)
            .transpose(1, 2)
            .reshape(b, n, -1)
        )

        # attention dropout
        out = self.dropout(out)
        out = self.to_out(out)

        return out


class GlobalHDITBlock(nn.Module):
    def __init__(self, dim, L=256, heads=8, dim_head=64, dropout=0.0, eps=1e-8):
        super().__init__()
        self.norm0 = AdaRMSNorm(dim)
        self.norm1 = AdaRMSNorm(dim)
        self.attn = CosineSimilarityAttention(dim, L, heads, dim_head, dropout, eps)
        self.condition_mlp = nn.Linear(dim, 2 * dim)
        self.ffn = HDITFFNBlock(dim)

    def forward(self, x, condition):
        gamma1, gamma2 = self.condition_mlp(condition).chunk(2, dim=-1)
        input_tokens = x
        x = self.norm0(x, gamma1)
        x = self.attn(x)
        input_tokens = input_tokens + x

        x = self.norm1(x, gamma2)
        x = self.ffn(x)
        x = x + input_tokens
        return x
