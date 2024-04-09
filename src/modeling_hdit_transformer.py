import torch.nn as nn

from src.modeling_hdit_basic import GlobalHDiTBlock, LocalHDiTBlock, Lerp


class HDiT(nn.Module):
    """Hourglass Diffusion Transformer"""

    def __init__(
        self,
        dim,
        num_levels,
        num_heads=8,
        window_size=256,
        dropout=0.0,
        attn_dropout=0.0,
    ):
        super().__init__()

        # TODO(schermala3): initialize StableDiffusion auto-encoder from HF in the pipeline

        self.num_levels = num_levels

        self.down_blocks = nn.ModuleList()
        for _ in range(num_levels - 1):
            self.down_blocks.append(
                LocalHDiTBlock(dim, num_heads, window_size, dropout, attn_dropout)
            )
        self.token_merge = nn.PixelShuffle(2)

        self.mid_block = GlobalHDiTBlock(
            dim, num_heads, window_size, dropout, attn_dropout
        )

        self.up_blocks = nn.ModuleList()
        for _ in range(num_levels - 1):
            self.up_blocks.append(
                LocalHDiTBlock(dim, num_heads, window_size, dropout, attn_dropout)
            )
        self.token_split = nn.PixelUnshuffle(2)

        self.interpolators = nn.ModuleList()
        for _ in range(num_levels - 1):
            self.interpolators.append(Lerp())

    def forward(self, x, condition):
        # down pass
        down_pass = []
        for down_block in self.down_blocks:
            x = down_block(x, condition)
            x = self.token_merge(x)
            down_pass.append(x)

        # mid block
        x = self.mid_block(x, condition)

        # up pass
        for up_block, interp, down_x in zip(
            self.up_blocks, self.interpolators, down_pass[::-1]
        ):
            x = up_block(x, condition)
            x = interp(self.token_split(x), down_x)

        return x
