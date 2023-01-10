import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., num_patches = None):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class TransformerBlock(nn.Module):
    def __init__(self, in_channels, out_channels, image_size, patch_size = 16, depth = 2, heads = 8,  dropout = 0.5, attention = Attention):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        self.outsize = (image_height // patch_size, image_width// patch_size)
        h = image_height // patch_height
        w = image_width // patch_width
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        mlp_dim = out_channels * 2
        self.patch_embeddings = nn.Conv2d(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches, out_channels))
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(out_channels, attention(out_channels, heads = heads, dim_head = out_channels // heads, dropout = dropout, num_patches=(h,w))),
                PreNorm(out_channels, FeedForward(out_channels, mlp_dim, dropout = dropout))
            ]))
        self.re_patch_embedding = nn.Sequential(
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1 = 1, p2 = 1, h = h)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, img):
        x = self.patch_embeddings(img)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)
        embeddings = x + self.position_embeddings
        x = self.dropout(embeddings)

        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        x = self.re_patch_embedding(x)
        return F.interpolate(x, self.outsize)

class DoubleConv2D(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv2D,self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

#-------------------------------------------

class Down2D(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, conv_builder):
        super(Down2D,self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            conv_builder(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

#-------------------------------------------

class Up2D(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, conv_builder):
        super(Up2D,self).__init__()

        self.conv = conv_builder(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = F.interpolate(x1,scale_factor=2, mode='bilinear', align_corners=False)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

#-------------------------------------------

class Tail2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Tail2D, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

#-------------------------------------------

class UNet(nn.Module):
    def __init__(self, stem, down, up, tail, width, conv_builder, n_channels=1, n_classes=2, dropout_flag=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.width = width
        self.dropout_flag = dropout_flag
        factor = 2 

        self.inc = stem(n_channels, width[0])
        self.down1 = down(width[0], width[1], conv_builder)
        self.down2 = down(width[1], width[2], conv_builder)
        self.down3 = down(width[2], width[3], conv_builder)
        self.down4 = down(width[3], width[4] // factor, conv_builder)
        self.up1 = up(width[4], width[3] // factor, conv_builder)
        self.up2 = up(width[3], width[2]// factor, conv_builder)
        self.up3 = up(width[2], width[1] // factor, conv_builder)
        self.up4 = up(width[1], width[0], conv_builder)
        self.dropout = nn.Dropout(p=0.5)
        self.outc = tail(width[0], n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        if self.dropout_flag:
            x = self.dropout(x)
        logits = self.outc(x)
        return logits

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, scale=2):
        super().__init__()
        self.scale = scale
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.double_conv(x)
        x = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)
        return x


class ITUNet_2d(nn.Module):
    def __init__(self, stem, down, up, tail, width, conv_builder,image_size = 128, transformer_depth = 18, n_channels=1, n_classes=2, dropout_flag=True):
        super(ITUNet_2d, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.width = width
        self.dropout_flag = dropout_flag
        factor = 2 

        self.transblock = TransformerBlock(n_channels, width[-1] // factor, image_size, depth = transformer_depth)
        self.vision_0 = UpConv(width[1], width[0])
        self.vision_1 = UpConv(width[2], width[1])
        self.vision_2 = UpConv(width[3], width[2])
        self.vision_3 = UpConv(width[-1] // factor, width[3])

        self.inc = stem(n_channels, width[0])
        self.down1 = down(width[0], width[1], conv_builder)
        self.down2 = down(width[1], width[2], conv_builder)
        self.down3 = down(width[2], width[3], conv_builder)
        self.down4 = down(width[3], width[4] // factor, conv_builder)
        self.up1 = up(width[4], width[3] // factor, conv_builder)
        self.up2 = up(width[3], width[2]// factor, conv_builder)
        self.up3 = up(width[2], width[1] // factor, conv_builder)
        self.up4 = up(width[1], width[0], conv_builder)
        self.dropout = nn.Dropout(p=0.5)
        self.outc = tail(width[0], n_classes)

        self.conv1x1_1 = nn.Conv2d(width[1] // factor, n_classes, kernel_size=1, stride=1, padding=0)
        self.conv1x1_2 = nn.Conv2d(width[2] // factor, n_classes, kernel_size=1, stride=1, padding=0)
        self.conv1x1_3 = nn.Conv2d(width[3] // factor, n_classes, kernel_size=1, stride=1, padding=0)
        self.conv1x1_4 = nn.Conv2d(width[4] // factor, n_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        trans = self.transblock(x)
        trans_4 = trans
        trans_3 = self.vision_3(trans_4)
        trans_2 = self.vision_2(trans_3)
        trans_1 = self.vision_1(trans_2)
        trans_0 = self.vision_0(trans_1)
        x1 = self.inc(x)   # 32
        x1 = x1 + trans_0
        x2 = self.down1(x1)   # 64
        x2 = x2 + trans_1
        x3 = self.down2(x2)  # 128
        x3 = x3 + trans_2
        x4 = self.down3(x3)  #256
        x4 = x4 + trans_3
        x5 = self.down4(x4)  #512
        x5 = x5 + trans_4

        out4 = self.conv1x1_4(x5) 
        x = self.up1(x5, x4)  # 128
        out3 = self.conv1x1_3(x)
        x = self.up2(x, x3)
        out2 = self.conv1x1_2(x)
        x = self.up3(x, x2)
        out1 = self.conv1x1_1(x)
        x = self.up4(x, x1)
        
        if self.dropout_flag:
            x = self.dropout(x)
        logits = self.outc(x)
        # return logits
        return [logits, out1, out2, out3, out4]


def itunet_2d(**kwargs):
    return ITUNet_2d(stem=DoubleConv2D,
                down=Down2D,
                up=Up2D,
                tail=Tail2D,
                width=[32,64,128,256,512],
                conv_builder=DoubleConv2D,
                **kwargs)

def unet(**kwargs):
    return UNet(stem=DoubleConv2D,
                down=Down2D,
                up=Up2D,
                tail=Tail2D,
                width=[32,64,128,256,512],
                conv_builder=DoubleConv2D,
                **kwargs)