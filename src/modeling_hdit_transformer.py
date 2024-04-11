import torch.nn as nn

from src.modeling_hdit_basic import GlobalHDiTBlock, Lerp, LocalHDiTBlock


class HDiT(nn.Module):
    """Hourglass Diffusion Transformer"""

    def __init__(
        self,
        config,
    ):
        super().__init__()

        self.levels = config.levels
        self.depths = config.depths

        self.down_blocks = nn.ModuleList()
        for i in range(self.levels[0]):
            for _ in range(self.depths[i]):
                self.down_blocks.append(LocalHDiTBlock(config, level_num=i))
            self.token_merge = nn.PixelShuffle(2)

        self.mid_block = nn.ModuleList()
        for i in range(self.levels[1]):
            for j in range(self.depths[i]):
                self.mid_block.append(
                    GlobalHDiTBlock(config=config, level_num=i)
                )

        self.up_blocks = nn.ModuleList()
        for i in reversed(range(self.levels[0])):
            for _ in range(self.depths[i]):
                self.down_blocks.append(LocalHDiTBlock(config, level_num=i))
            self.token_split = nn.PixelUnshuffle(2)

        self.interpolators = nn.ModuleList()
        for _ in range(self.levels[0] - 1):
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
