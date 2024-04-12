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

import math


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

    def __init__(self, cond_dim, dim, bias=False, eps=1e-8):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.bias = bias

        self.proj = nn.Linear(cond_dim, dim)

        if bias:
            self.register_parameter("bias", nn.Parameter(torch.zeros(dim)))

    def forward(self, x, cond) -> torch.Tensor:
        norm = x.norm(2, dim=-1, keepdim=True) / (self.dim**0.5)
        x = x / (norm + self.eps)
        x = x * self.proj(cond)
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


class HDiTBlock(nn.Module):
    def __init__(
        self,
        dim,
        cond_dim,
        hidden_dim,
        L=256,
        heads=8,
        dim_head=64,
        dropout=0.0,
        eps=1e-8,
    ):
        super().__init__()
        self.norm0 = AdaRMSNorm(cond_dim, dim)
        self.norm1 = AdaRMSNorm(cond_dim, dim)
        self.ffn = HDiTFFNBlock(dim, hidden_dim, dropout)
        self.attn = None  # to be defined in subclasses

    def forward(self, x, cond):
        input_tokens = x
        x = self.norm0(x, cond)
        x = self.attn(x)
        input_tokens = input_tokens + x

        x = self.norm1(x, cond)
        x = self.ffn(x)
        x = x + input_tokens
        return x


class GlobalHDiTBlock(HDiTBlock):
    def __init__(
        self,
        dim,
        cond_dim,
        hidden_dim,
        L=256,
        heads=8,
        dim_head=64,
        dropout=0.0,
        eps=1e-8,
    ):
        super().__init__(dim, cond_dim, hidden_dim, L, heads, dim_head, dropout, eps)
        self.attn = CosineSimilarityAttention(dim, L, heads, dim_head, dropout, eps)


class LocalHDiTBlock(HDiTBlock):
    def __init__(
        self,
        dim,
        cond_dim,
        hidden_dim,
        L=256,
        heads=8,
        dim_head=64,
        dropout=0.0,
        eps=1e-8,
    ):
        super().__init__(dim, cond_dim, hidden_dim, L, heads, dim_head, dropout, eps)
        self.attn = CosineSimilarityAttention(dim, L, heads, dim_head, dropout, eps)


class Lerp(nn.Module):
    """Linear interpolation module"""

    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))

    def forward(self, x, y):
        return x + self.alpha * (y - x)


class AxialRoPE(nn.Module):
    """Axial Rotary Positional Encoding for 2D data"""

    def __init__(self, dim, n_heads):
        super().__init__()

        self.dim = dim
        self.n_heads = n_heads
        self.register_buffer("frequencies", self._create_frequencies())

        self._cached_cos = None
        self._cached_sin = None
        self._cached_resolution = None

    def forward(self, x):
        if self._cached_resolution is None or self._cached_resolution != x.shape[-3:-1]:
            self._cached_cos, self._cached_sin = self._create_positional_embeddings(
                x.shape[-3:-1]
            )
            self._cached_resolution = x.shape[-3:-1]

        cos, sin = self._cached_cos, self._cached_sin

        x1, x2 = torch.chunk(x, 2, dim=-1)
        x = torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
        return x

    def _create_frequencies(self):
        freqs = torch.logspace(
            torch.log(math.pi),
            torch.log(10.0 * math.pi),
            self.n_heads * self.dim // 4 + 1,
            base=math.e,
        )[:-1]
        return freqs.view(self.dim // 4, self.n_heads).T.contiguous()

    def _create_positional_embeddings(self, resolution=(256, 256)):
        if resolution[0] != resolution[1]:
            raise ValueError("Only square images are supported")
        positions = torch.linspace(-1, 1, resolution[0])
        positions_grid = torch.stack(
            torch.meshgrid(positions, positions, indexing="ij"), dim=-1
        )
        positions = positions_grid.flatten(0, 1)

        theta_h = positions[..., None, 0:1] * self.frequencies
        theta_w = positions[..., None, 1:2] * self.frequencies
        # [n, nh, d*2] -> [n, d*2, nh] if using movedim
        thetas = torch.cat((theta_h, theta_w), dim=-1)  # .movedim(-2, -3) ?
        return thetas.cos(), thetas.sin()


class HDiTFFNBlock(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
        )
        self.branch2 = nn.Linear(dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        skip = x
        x = self.branch1(x) * self.branch2(x)
        x = self.dropout(x)
        x = self.proj(x)
        x = x + skip
        return x
