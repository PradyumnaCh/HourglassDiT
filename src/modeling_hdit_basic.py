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
from transformers.models.nat.modeling_nat import NeighborhoodAttention

from src.configuration_hdit import HDiTConfig


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

    def __init__(
        self, dim, L=256, heads=8, dim_head=64, dropout=0.0, eps=1e-8
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.eps = eps
        self.register_buffer(
            "scale",
            nn.Parameter(
                torch.ones(heads) * torch.log2(L**2 - L),
            ),
        )

        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        b, n, d, h = *x.shape, self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: t.reshape(b, n, h, -1).transpose(1, 2), (q, k, v)
        )

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
        config,
        level_num,
    ):
        super().__init__()

    def forward(self, x, cond):
        for layer in self.layers:
            x = layer(x, cond)
        return x


class GlobalHDiTBlock(HDiTBlock):
    def __init__(
        self,
        config,
        level_num,
    ):
        super().__init__(config=config, level_num=level_num)
        self.layers = []
        for _ in range(config.depths[level_num]):
            self.layers.append(
                GlobalHDiTLayer(
                    config=config,
                    level_num=level_num,
                )
            )


class LocalHDiTBlock(HDiTBlock):
    def __init__(
        self,
        config,
        level_num,
    ):
        super().__init__(config=config, level_num=level_num)
        self.layers = []
        for _ in range(config.depths[level_num]):
            self.layers.append(
                LocalHDiTLayer(
                    config=config,
                    level_num=level_num,
                )
            )


class HDiTLayer(nn.Module):
    def __init__(
        self,
        config,
        level_num,
    ):
        # TODO Assign the correct values for these dimensions
        dim = config.widths[level_num]
        cond_dim = config.cond_dim
        super().__init__()
        self.norm0 = AdaRMSNorm(cond_dim, dim)
        self.norm1 = AdaRMSNorm(cond_dim, dim)

        input_dim = config.widths[level_num]
        self.ffn = HDiTFFNBlock(
            dim=input_dim,
            hidden_dim=input_dim*3,
            dropout=config.hidden_dropout_prob,
        )
        self.attn = None  # to be defined in subclasses

    def forward(self, x, cond):
        input_tokens = x
        x = self.norm0(x, cond)
        x = self.attn(x)
        x = x[0]
        input_tokens = input_tokens + x

        x = self.norm1(x, cond)
        x = self.ffn(x)
        x = x + input_tokens
        return x


class GlobalHDiTLayer(HDiTLayer):
    def __init__(
        self,
        config: HDiTConfig,
        level_num=0,
        L=torch.Tensor(265),
    ):
        super().__init__(config, level_num)
        # self.attn = CosineSimilarityAttention(
        #     dim=config.widths[level_num],
        #     L=L,
        #     heads=config.num_heads[level_num],
        #     dim_head=config.attn_head_dim,
        #     dropout=config.attention_probs_dropout_prob,
        #     eps=config.layer_norm_eps,
        # )
        self.attn = NeighborhoodAttention(
            config=config,
            dim=config.widths[level_num],
            num_heads=config.num_heads[level_num],
            kernel_size=config.kernel_size,
        )


class LocalHDiTLayer(HDiTLayer):
    def __init__(
        self,
        config,
        level_num=0,
    ):
        super().__init__(config, level_num)
        self.attn = NeighborhoodAttention(
            config=config,
            dim=(config.widths[level_num]),
            num_heads=config.num_heads[level_num],
            kernel_size=config.kernel_size,
        )


class Lerp(nn.Module):
    """Linear interpolation module"""

    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))

    def forward(self, x, y):
        return x + self.alpha * (y - x)


class RoPE1d(nn.Module):
    """Rotary Positional Encoding for 1D data"""

    def __init__(self, dim):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError


class RoPE2d(nn.Module):
    """Rotary Positional Encoding for 2D data"""

    def __init__(self, dim):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError


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


class TokenSplit(nn.Module):
    def __init__(self, in_feat, out_feat, scale=2):
        super().__init__()
        self.linear = nn.Linear(in_feat // (scale * scale), out_feat)
        self.shuffle = nn.PixelShuffle(scale)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.shuffle(x)
        x = x.permute(0, 2, 3, 1)
        x = self.linear(x)
        return x


class TokenMerge(nn.Module):
    def __init__(self, in_feat, out_feat, scale=2):
        super().__init__()
        self.linear = nn.Linear(in_feat * scale * scale, out_feat)
        self.unshuffle = nn.PixelUnshuffle(scale)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.unshuffle(x)
        x = x.permute(0, 2, 3, 1)
        x = self.linear(x)
        return x
