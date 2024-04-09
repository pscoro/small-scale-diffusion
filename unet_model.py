import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
        self.proj = nn.Sequential(
            nn.Linear(dim // 4, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half = self.dim // 8
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=x.device, dtype=x.dtype) * -emb)
        emb = x[:, None] * emb[None]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return self.proj(emb)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_channels: int, dropout: float = 0.1):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )
        self.layer2 = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()
        self.time_embedding = nn.Linear(time_channels, out_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        x = self.layer1(x)
        t = self.time_embedding(F.silu(t))
        x = x + t[:, :, None, None]
        x = self.layer2(x)
        return x + identity


class SelfAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, in_proj_bias: bool = True, out_proj_bias: bool = True):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, n_channels, h, w = x.size()
        seq_len = h * w
        head_shape = (batch_size, seq_len, self.n_heads, self.d_head)
        identity = x

        x = x.view(batch_size, n_channels, seq_len).permute(0, 2, 1) # B, N, C
        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        q = q.view(head_shape).transpose(1, 2)
        k = k.view(head_shape).transpose(1, 2)
        v = v.view(head_shape).transpose(1, 2)

        attn_weights = q @ k.transpose(-1, -2)
        attn_weights = F.softmax(attn_weights / (self.d_head ** 0.5), dim=-1)

        x = attn_weights @ v
        x = x.transpose(1, 2).contiguous().view(batch_size, seq_len, n_channels)
        x = self.out_proj(x) # B, N, C
        x = x.permute(0, 2, 1).view(batch_size, n_channels, h, w)

        return x + identity


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_channels: int, has_attn: bool):
        super().__init__()
        self.residual = ResidualBlock(in_channels, out_channels, time_channels)
        if has_attn:
            self.attn = SelfAttention(1, out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = self.residual(x, t)
        x = self.attn(x)
        return x


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_channels: int, has_attn: bool):
        super().__init__()
        self.residual = ResidualBlock(in_channels + out_channels, out_channels, time_channels)
        if has_attn:
            self.attn = SelfAttention(1, out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = self.residual(x, t)
        x = self.attn(x)
        return x


class BottleneckBlock(nn.Module):
    def __init__(self, channels: int, time_channels: int):
        super().__init__()
        self.residual1 = ResidualBlock(channels, channels, time_channels)
        self.attn = SelfAttention(1, channels)
        self.residual2 = ResidualBlock(channels, channels, time_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = self.residual1(x, t)
        x = self.attn(x)
        x = self.residual2(x, t)
        return x


class UNet(nn.Module):
    def __init__(
        self,
        image_channels: int,
        num_channels: int,
        channel_mults: List[int] = [1, 2, 2, 4],
        has_attn: List[bool] = [False, False, True, True],
        num_blocks: int = 2
    ):
        super().__init__()
        num_resolutions = len(channel_mults)
        assert num_resolutions == len(has_attn)

        self.head = nn.Conv2d(image_channels, num_channels, kernel_size=3, padding=1)
        self.time_embedding = TimeEmbedding(num_channels * 4)

        self.down_blocks = nn.ModuleList()
        in_channels = num_channels
        out_channels = num_channels
        for i in range(num_resolutions):
            out_channels = in_channels * channel_mults[i]

            for _ in range(num_blocks):
                self.down_blocks.append(DownsampleBlock(in_channels, out_channels, num_channels * 4, has_attn[i]))
                in_channels = out_channels

            if i < num_resolutions - 1:
                self.down_blocks.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1))

        self.bottleneck = BottleneckBlock(out_channels, num_channels * 4)

        self.up_blocks = nn.ModuleList()
        in_channels = out_channels
        for i in reversed(range(num_resolutions)):
            out_channels = in_channels

            for _ in range(num_blocks):
                self.up_blocks.append(UpsampleBlock(in_channels, out_channels, num_channels * 4, has_attn[i]))

            out_channels = in_channels // channel_mults[i]
            self.up_blocks.append(UpsampleBlock(in_channels, out_channels, num_channels * 4, has_attn[i]))
            in_channels = out_channels

            if i > 0:
                self.up_blocks.append(nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1))

        self.tail = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, image_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = self.head(x)
        t = self.time_embedding(t)

        skips = [x]
        for block in self.down_blocks:
            if isinstance(block, nn.Conv2d):
                x = block(x)
            else:
                x = block(x, t)
            skips.append(x)

        x = self.bottleneck(x, t)

        for block in self.up_blocks:
            if isinstance(block, nn.ConvTranspose2d):
                x = block(x)
            else:
                skip = skips.pop()
                x = block(torch.cat([x, skip], dim=1), t)

        return self.tail(x)


def main() -> None:
    # test the model
    model = UNet(4, 64)
    x = torch.randn(2, 4, 32, 32)
    t = torch.tensor([1, 2], dtype=torch.long)
    z = model(x, t)
    print(z.shape)
    print(model)


if __name__ == "__main__":
    main()
