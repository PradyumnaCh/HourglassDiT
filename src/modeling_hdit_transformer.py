import torch.nn as nn

from src.modeling_hdit_basic import (
    GlobalHDiTBlock,
    Lerp,
    LocalHDiTBlock,
    TokenMerge,
    TokenSplit,
)


class HDiT(nn.Module):
    """Hourglass Diffusion Transformer"""

    def __init__(
        self,
        config,
    ):
        super().__init__()

        # TODO: Should this even be here?
        self.patchify = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.embed_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )

        self.levels = config.levels
        self.depths = config.depths

        self.token_merges = nn.ModuleList()
        self.token_splits = nn.ModuleList()

        self.down_blocks = nn.ModuleList()
        for i in range(self.levels[0]):
            self.down_blocks.append(LocalHDiTBlock(config, level_num=i))
            self.token_merges.append(
                TokenMerge(
                    in_feat=config.widths[i],
                    out_feat=config.widths[i + 1],
                    scale=2,
                )
            )

        self.mid_blocks = nn.ModuleList()
        for i in range(self.levels[1]):
            self.mid_blocks.append(
                GlobalHDiTBlock(
                    config=config,
                    level_num=config.levels[0] + i,
                )
            )

        self.up_blocks = nn.ModuleList()
        for i in reversed(range(self.levels[0])):
            self.up_blocks.append(LocalHDiTBlock(config, level_num=i))
            self.token_splits.append(
                TokenSplit(
                    in_feat=(
                        config.widths[i + 1]
                        if i < self.levels[0] - 1
                        else config.widths[-1]
                    ),
                    out_feat=config.widths[i],
                    scale=2,
                )
            )

        self.interpolators = nn.ModuleList()
        for _ in range(self.levels[0]):
            self.interpolators.append(Lerp())

    def forward(self, x, condition):

        x = self.embedding(x)

        down_pass = []
        for down_block, token_merge in zip(
            self.down_blocks, self.token_merges
        ):
            x = down_block(x, condition)
            down_pass.append(x)
            x = token_merge(x)

        for mid_block in self.mid_blocks:
            x = mid_block(x, condition)

        for up_block, token_split, interp, down_x in zip(
            self.up_blocks,
            self.token_splits,
            self.interpolators,
            down_pass[::-1],
        ):
            x = interp(token_split(x), down_x)
            x = up_block(x, condition)
        return x
