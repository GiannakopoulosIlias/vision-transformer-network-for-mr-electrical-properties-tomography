import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch import Tensor
from typing import Tuple
from argparse import ArgumentParser
import pytorch_lightning as pl
import h5py

import torch
import torch.nn as nn

def img_seq(x):
    x = x.flatten(2,4) #[B,C,H*W*D]
    x = x.permute(0,2,1) #[B,H*W*D,C], num_patches=H*W*D
    return x

#Attention classes
class AttentionBlock(nn.Module):

    def __init__(self, embed_dim, hidden_dim, num_heads, drop_prob: float = 0.0):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            drop_prob - Amount of dropout to apply in the feed-forward network
        """
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                          dropout=drop_prob)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_prob),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(drop_prob)
        )


    def forward(self, x):
        inp_x = self.layer_norm_1(x)
        x = x + self.attn(inp_x, inp_x, inp_x)[0]
        x = x + self.linear(self.layer_norm_2(x))
        return x

class ViTBlock(nn.Module):

    def __init__(self, in_chans: int, out_chans: int, num_heads: int, num_layers: int, num_patches: int, drop_prob: float):
        super().__init__()
        self.in_chans  = in_chans
        self.out_chans = out_chans #out_chans = embed_dim
        self.dropout = nn.Dropout3d(drop_prob)
        self.linear_proj = nn.Linear(in_chans,out_chans)
        self.transformer = nn.Sequential(*[AttentionBlock(out_chans, 2*out_chans, num_heads, drop_prob=drop_prob) for _ in range(num_layers)])
        self.pos_embedding = nn.Parameter(torch.randn(1,num_patches,out_chans))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        B,C,H,W,D = image.shape
        x = img_seq(image)
        x = self.linear_proj(x)
        x = x+self.pos_embedding
        # apply transformer
        x = self.dropout(x.unsqueeze(2)).squeeze(2)
        x = self.transformer(x).permute(0,2,1)
        x = x.reshape(B,self.out_chans,H,W,D)
        return x

#FiLM classes
class FiLMGeneratorModel(nn.Module):
    def __init__(self, out_chans: int = 32, cascades_num: int = 3, drop_prob: float = 0.1):
        super().__init__()
        self.in_chans = 1
        self.out_chans = out_chans
        self.cascades_num = cascades_num
        self.drop_prob = drop_prob

        ch = self.out_chans * self.cascades_num * 2
        self.layers = nn.Sequential(
            self._conv_block(1, 32, drop_prob),
            self._conv_block(32, 64, drop_prob),
            self._conv_block(64, 128, drop_prob),
            self._conv_block(128, 256, drop_prob),
            self._conv_block(256, 512, drop_prob)
        )
        self.conv_final = nn.Conv3d(512, ch, kernel_size=1, stride=1)

    def _conv_block(self, in_ch, out_ch, drop_prob):
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout3d(drop_prob)
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = self.layers(images)
        images = F.avg_pool3d(images, kernel_size=2, stride=2, padding=0)
        images = self.conv_final(images)
        images = torch.mean(images, dim=[0, 2, 3, 4])
        images = images.view(2, self.cascades_num, self.out_chans)
        beta   = images[0,:,:]
        gamma  = images[1,:,:]
        return beta, gamma

#UNET classes
class Unet3D(nn.Module):

    def __init__(self, in_chans: int, out_chans: int, chans: int = 32, num_pool_layers: int = 4, drop_prob: float = 0.0):

        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob

        self.down_sample_layers = nn.ModuleList([ConvBlock3D(in_chans, chans, drop_prob)])
        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(ConvBlock3D(ch, ch * 2, drop_prob))
            ch *= 2
        self.conv = ConvBlock3D(ch, ch * 2, drop_prob)

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv.append(TransposeConvBlock3D(ch * 2, ch))
            self.up_conv.append(ConvBlock3D(ch * 2, ch, drop_prob))
            ch //= 2

        self.up_transpose_conv.append(TransposeConvBlock3D(ch * 2, ch))
        self.up_conv.append(nn.Sequential(ConvBlock3D(ch * 2, ch, drop_prob),nn.Conv3d(ch, self.out_chans, kernel_size=1, stride=1)))

    def forward(self, image: torch.Tensor) -> torch.Tensor:

        stack = []
        output = image

        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.avg_pool3d(output, kernel_size=2, stride=2, padding=0)

        output = self.conv(output)

        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            output = transpose_conv(output)
            padding = [0, 0, 0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1
            if output.shape[-3] != downsample_layer.shape[-3]:
                padding[5] = 1
            if torch.sum(torch.tensor(padding)) != 0:
                output = F.pad(output, padding, "reflect")

            output = torch.cat([output, downsample_layer], dim=1)
            output = conv(output)

        return output

class Unet3D_FiLM(nn.Module):

    def __init__(self, in_chans: int, out_chans: int, chans: int = 32, num_pool_layers: int = 4, drop_prob: float = 0.0):

        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob

        self.down_sample_layers = nn.ModuleList([ConvBlock3D_FiLM(in_chans, chans, drop_prob)])
        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(ConvBlock3D(ch, ch * 2, drop_prob))
            ch *= 2
        self.conv = ConvBlock3D(ch, ch * 2, drop_prob)

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv.append(TransposeConvBlock3D(ch * 2, ch))
            self.up_conv.append(ConvBlock3D(ch * 2, ch, drop_prob))
            ch //= 2

        self.up_transpose_conv.append(TransposeConvBlock3D(ch * 2, ch))
        self.up_conv.append(nn.Sequential(ConvBlock3D(ch * 2, ch, drop_prob),nn.Conv3d(ch, self.out_chans, kernel_size=1, stride=1)))

    def forward(self, image: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:

        stack = []
        output = image

        first_layer = self.down_sample_layers[0]
        output = first_layer(output, beta, gamma)
        stack.append(output)
        output = F.avg_pool3d(output, kernel_size=2, stride=2, padding=0)

        for layer in self.down_sample_layers[1:]:
            output = layer(output)
            stack.append(output)
            output = F.avg_pool3d(output, kernel_size=2, stride=2, padding=0)

        output = self.conv(output)

        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            output = transpose_conv(output)
            padding = [0, 0, 0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1
            if output.shape[-3] != downsample_layer.shape[-3]:
                padding[5] = 1
            if torch.sum(torch.tensor(padding)) != 0:
                output = F.pad(output, padding, "reflect")

            output = torch.cat([output, downsample_layer], dim=1)
            output = conv(output)

        return output

class Unet3D_ViT(nn.Module):

    def __init__(self, in_chans: int, out_chans: int, chans: int = 32, num_pool_layers: int = 4, num_heads:int=2, num_layers:int=2, num_patches:int=20, drop_prob: float = 0.0):

        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob

        self.down_sample_layers = nn.ModuleList([ConvBlock3D_ViT(in_chans, chans, drop_prob)])
        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(ConvBlock3D_ViT(ch, ch * 2, drop_prob))
            ch *= 2
        self.vit = ViTBlock(ch,2*ch,num_heads=num_heads,num_layers=num_layers,num_patches=num_patches,drop_prob=drop_prob)

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv.append(TransposeConvBlock3D_ViT(ch * 2, ch))
            self.up_conv.append(ConvBlock3D_ViT(ch * 2, ch, drop_prob))
            ch //= 2

        self.up_transpose_conv.append(TransposeConvBlock3D_ViT(ch * 2, ch))
        self.up_conv.append(nn.Sequential(ConvBlock3D_ViT(ch * 2, ch, drop_prob),nn.Conv3d(ch, self.out_chans, kernel_size=1, stride=1)))

    def forward(self, image: torch.Tensor) -> torch.Tensor:

        stack = []
        output = image

        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.avg_pool3d(output, kernel_size=2, stride=2, padding=0)

        output = self.vit(output)

        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            output = transpose_conv(output)
            padding = [0, 0, 0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1
            if output.shape[-3] != downsample_layer.shape[-3]:
                padding[5] = 1
            if torch.sum(torch.tensor(padding)) != 0:
                output = F.pad(output, padding, "reflect")

            output = torch.cat([output, downsample_layer], dim=1)
            output = conv(output)

        return output

class Unet3D_ViT_FiLM(nn.Module):

    def __init__(self, in_chans: int, out_chans: int, chans: int = 32, num_pool_layers: int = 4, num_heads:int=2, num_layers:int=2, num_patches:int=20, drop_prob: float = 0.0):

        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob

        self.down_sample_layers = nn.ModuleList([ConvBlock3D_ViT_FiLM(in_chans, chans, drop_prob)])
        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(ConvBlock3D_ViT(ch, ch * 2, drop_prob))
            ch *= 2
        self.vit = ViTBlock(ch,2*ch,num_heads=num_heads,num_layers=num_layers,num_patches=num_patches,drop_prob=drop_prob)

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv.append(TransposeConvBlock3D_ViT(ch * 2, ch))
            self.up_conv.append(ConvBlock3D_ViT(ch * 2, ch, drop_prob))
            ch //= 2

        self.up_transpose_conv.append(TransposeConvBlock3D_ViT(ch * 2, ch))
        self.up_conv.append(nn.Sequential(ConvBlock3D_ViT(ch * 2, ch, drop_prob),nn.Conv3d(ch, self.out_chans, kernel_size=1, stride=1)))

    def forward(self, image: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:

        stack = []
        output = image

        first_layer = self.down_sample_layers[0]
        output = first_layer(output, beta, gamma)
        stack.append(output)
        output = F.avg_pool3d(output, kernel_size=2, stride=2, padding=0)

        for layer in self.down_sample_layers[1:]:
            output = layer(output)
            stack.append(output)
            output = F.avg_pool3d(output, kernel_size=2, stride=2, padding=0)

        output = self.vit(output)

        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            output = transpose_conv(output)
            padding = [0, 0, 0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1
            if output.shape[-3] != downsample_layer.shape[-3]:
                padding[5] = 1
            if torch.sum(torch.tensor(padding)) != 0:
                output = F.pad(output, padding, "reflect")

            output = torch.cat([output, downsample_layer], dim=1)
            output = conv(output)

        return output

#Convolution block classes
class ConvBlock3D(nn.Module):

    def __init__(self, in_chans: int, out_chans: int, drop_prob: float):
        super().__init__()
        self.in_chans  = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob
        self.layers    = nn.Sequential(
            nn.Conv3d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout3d(drop_prob),
            nn.Conv3d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_chans),
            )
        self.conv = nn.Sequential(
            nn.Conv3d(in_chans, out_chans, kernel_size=1, bias=False),
            nn.InstanceNorm3d(out_chans)
            )
        self.res = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout3d(drop_prob)
            )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.res(self.layers(image) + self.conv(image))

class ConvBlock3D_FiLM(nn.Module):
    def __init__(self, in_chans: int, out_chans: int, drop_prob: float):
        super().__init__()
        self.in_chans  = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob
        self.layers    = nn.Sequential(
            nn.Conv3d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout3d(drop_prob),
            nn.Conv3d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_chans),
            )
        self.conv = nn.Sequential(
            nn.Conv3d(in_chans, out_chans, kernel_size=1, bias=False),
            nn.InstanceNorm3d(out_chans)
            )
        self.res = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout3d(drop_prob)
            )
    def forward(self, image: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        beta = beta.view(1, -1, 1, 1, 1)
        gamma = gamma.view(1, -1, 1, 1, 1)
        return self.res((self.layers(image) + self.conv(image))* gamma + beta)

class ConvBlock3D_ViT(nn.Module):
    def __init__(self, in_chans: int, out_chans: int, drop_prob: float):
        super().__init__()
        self.in_chans  = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob
        self.layers    = nn.Sequential(
            nn.Conv3d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_chans),
            nn.GELU(),
            nn.Dropout3d(drop_prob),
            nn.Conv3d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_chans),
            )
        self.conv = nn.Sequential(
            nn.Conv3d(in_chans, out_chans, kernel_size=1, bias=False),
            nn.InstanceNorm3d(out_chans))
        self.res = nn.Sequential(nn.GELU(),
            nn.Dropout3d(drop_prob))
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.res(self.layers(image) + self.conv(image))

class ConvBlock3D_ViT_FiLM(nn.Module):
    def __init__(self, in_chans: int, out_chans: int, drop_prob: float):
        super().__init__()
        self.in_chans  = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob
        self.layers    = nn.Sequential(
            nn.Conv3d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_chans),
            nn.GELU(),
            nn.Dropout3d(drop_prob),
            nn.Conv3d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_chans),
            )
        self.conv = nn.Sequential(
            nn.Conv3d(in_chans, out_chans, kernel_size=1, bias=False),
            nn.InstanceNorm3d(out_chans)
            )
        self.res = nn.Sequential(
            nn.GELU(),
            nn.Dropout3d(drop_prob)
            )
    def forward(self, image: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        beta = beta.view(1, -1, 1, 1, 1)
        gamma = gamma.view(1, -1, 1, 1, 1)
        return self.res((self.layers(image) + self.conv(image))* gamma + beta)

#Transpose Convolution block classes
class TransposeConvBlock3D(nn.Module):

    def __init__(self, in_chans: int, out_chans: int):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.layers = nn.Sequential(
            nn.ConvTranspose3d(in_chans, out_chans, kernel_size=2, stride=2, bias=False),
            nn.InstanceNorm3d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.layers(image)

class TransposeConvBlock3D_ViT(nn.Module):

    def __init__(self, in_chans: int, out_chans: int):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.layers = nn.Sequential(
            nn.ConvTranspose3d(in_chans, out_chans, kernel_size=2, stride=2, bias=False),
            nn.InstanceNorm3d(out_chans),
            nn.GELU()
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.layers(image)
