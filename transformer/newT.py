import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from SwinTransformer_backbone import SwinTransformer3D

from timm.models.layers import DropPath, trunc_normal_
from functools import reduce, lru_cache
from operator import mul
from einops import rearrange


class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, D, H, W, C)
        window_size (tuple[int]): window size

    Returns:
        windows: (B*num_windows, window_size*window_size, C)
    """
    B, D, H, W, C = x.shape
    # print(f"B: {B}, D: {D}, H: {H}, W: {W}, C: {C}")
    x = x.view(B, D // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2], window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, reduce(mul, window_size), C)
    return windows


def window_reverse(windows, window_size, B, D, H, W):
    """
    Args:
        windows: (B*num_windows, window_size, window_size, C)
        window_size (tuple[int]): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, D, H, W, C)
    """
    x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x


def get_window_size(x_size, window_size, shift_size=None):
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)
    

class WindowAttention3D(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The temporal length, height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wd, Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads))  # 2*Wd-1 * 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))  # 3, Wd, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wd*Wh*Ww, Wd*Wh*Ww, 3
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, N, N) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, nH, N, C

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index[:N, :N].reshape(-1)].reshape(
            N, N, -1)  # Wd*Wh*Ww,Wd*Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wd*Wh*Ww, Wd*Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0) # B_, nH, N, N

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    

class SwinTransformerBlock3D(nn.Module):
    """ Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size.
        shift_size (tuple[int]): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=(2,7,7), shift_size=(0,0,0),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint=use_checkpoint

        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[2] < self.window_size[2], "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward_part1(self, x, mask_matrix):
        B, D, H, W, C = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)

        x = self.norm1(x)
        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, Dp, Hp, Wp, _ = x.shape

        # cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        # partition windows
        x_windows = window_partition(shifted_x, window_size)  # B*nW, Wd*Wh*Ww, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # B*nW, Wd*Wh*Ww, C
        # merge windows
        attn_windows = attn_windows.view(-1, *(window_size+(C,)))
        shifted_x = window_reverse(attn_windows, window_size, B, Dp, Hp, Wp)  # B D' H' W' C
        # reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_d1 >0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :].contiguous()
        return x

    def forward_part2(self, x):
        return self.drop_path(self.mlp(self.norm2(x)))

    def forward(self, x, mask_matrix):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, D, H, W, C).
            mask_matrix: Attention mask for cyclic shift.
        """

        shortcut = x
        if self.use_checkpoint:
            pass
            # x = checkpoint.checkpoint(self.forward_part1, x, mask_matrix)
        else:
            x = self.forward_part1(x, mask_matrix)
        x = shortcut + self.drop_path(x)

        if self.use_checkpoint:
            pass
            # x = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            x = x + self.forward_part2(x)

        return x
    

class PatchMerging(nn.Module):
    """ Patch Merging Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, D, H, W, C).
        """
        B, D, H, W, C = x.shape

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, :, 0::2, 0::2, :]  # B D H/2 W/2 C
        x1 = x[:, :, 1::2, 0::2, :]  # B D H/2 W/2 C
        x2 = x[:, :, 0::2, 1::2, :]  # B D H/2 W/2 C
        x3 = x[:, :, 1::2, 1::2, :]  # B D H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B D H/2 W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x
    

# cache each stage results
@lru_cache()
def compute_mask(D, H, W, window_size, shift_size, device):
    img_mask = torch.zeros((1, D, H, W, 1), device=device)  # 1 Dp Hp Wp 1
    cnt = 0
    for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0],None):
        for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1],None):
            for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2],None):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1
    mask_windows = window_partition(img_mask, window_size)  # nW, ws[0]*ws[1]*ws[2], 1
    mask_windows = mask_windows.squeeze(-1)  # nW, ws[0]*ws[1]*ws[2]
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (tuple[int]): Local window size. Default: (1,7,7).
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=(1,7,7),
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock3D(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0,0,0) if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
            )
            for i in range(depth)])
        
        self.downsample = downsample
        if self.downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)

    def forward(self, x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, C, D, H, W).
        """
        # calculate attention mask for SW-MSA
        B, C, D, H, W = x.shape
        window_size, shift_size = get_window_size((D,H,W), self.window_size, self.shift_size)
        x = rearrange(x, 'b c d h w -> b d h w c')
        Dp = int(np.ceil(D / window_size[0])) * window_size[0]
        Hp = int(np.ceil(H / window_size[1])) * window_size[1]
        Wp = int(np.ceil(W / window_size[2])) * window_size[2]
        attn_mask = compute_mask(Dp, Hp, Wp, window_size, shift_size, x.device)
        for blk in self.blocks:
            x = blk(x, attn_mask)
        x = x.view(B, D, H, W, -1)

        if self.downsample is not None:
            x = self.downsample(x)
        x = rearrange(x, 'b d h w c -> b c d h w')
        return x
    

class PatchEmbed3D(nn.Module):
    """ Video to Patch Embedding.

    Args:
        patch_size (int): Patch token size. Default: (2,4,4).
        in_chans (int): Number of input video channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    def __init__(self, patch_size=(2,4,4), in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, D, H, W = x.size()
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if D % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))

        x = self.proj(x)  # B C D Wh Ww
        if self.norm is not None:
            D, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, D, Wh, Ww)

        return x
    

class DCTG(nn.Module):
    """
    DCTG (Dynamic Glass Token Generator) module.

    Args:
        hidden_size (int): The size of the hidden state in the LSTM layer.
        D (int, optional): The desired output size of the linear layer. Defaults to 512.
        M (int, optional): The number of features of hand and object. Defaults to 2.
    
    Output:
        Class token of gaze-hand-objeckt features.
    """
    def __init__(self, hidden_size=256, embed_dim=96, M=2):
        super(DCTG, self).__init__()
        self.M = M
        self.lstm = nn.LSTM(input_size=2048, hidden_size=hidden_size, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_size, embed_dim)
        
    def mean_features(self):
        """
        Calculate the average, and reshape the tensor.

        Returns:
            torch.Tensor: The reshaped tensor of averaged features.
        """
        # calculate the average of the input features along axis 0
        mean_features = torch.mean(self.input, dim=1)
        # reshape the tensor
        mean_features = mean_features.unsqueeze(0)

        return mean_features
    
    def forward(self, input_features):
        """
        Forward pass of the DCTG module.

        Args:
            features (torch.Tensor): The input features.

        Returns:
            torch.Tensor: The output tensor.
        """
        # calculate the mean features
        mean_features = input_features.mean(dim=1)
        mean_features = mean_features.unsqueeze(0)
        # forward pass the input features through the LSTM layer
        lstm_out, _ = self.lstm(mean_features)

        # forward pass the output of the LSTM layer through the linear layer
        x_cls = lstm_out[:, -1, :]
        x_cls = self.fc(x_cls)
        return x_cls
    

class Stage1(nn.Module):
    def __init__(self,pretrained=None,
                 pretrained2d=True,
                 patch_size=(2,4,4),
                 in_chans=3,
                 embed_dim=128,
                 depths=[2, 2, 18, 2],
                 num_heads=[4, 8, 16, 32],
                 window_size=(2,7,7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 patch_norm=False,
                 frozen_stages=-1,
                 use_checkpoint=False):
        
        super().__init__()
        self.pretrained = pretrained
        self.pretrained2d = pretrained2d
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.frozen_stages = frozen_stages
        self.window_size = window_size
        self.patch_size = patch_size

        self.dctg = DCTG(embed_dim=self.embed_dim)

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed3D(
            patch_size=patch_size, in_chans=in_chans, embed_dim=self.embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        
        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers-1):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if i_layer<self.num_layers-2 else None,
                use_checkpoint=use_checkpoint)
            self.layers.append(layer) 

        self.num_features = int(embed_dim * 2**(self.num_layers-2))

        # add a norm layer for each output
        self.norm = norm_layer(self.num_features)

    def create_class_token_map1(self, patches, class_token):
        """
        Create the class token map.

        Args:
        patches: A tensor of shape (N_windows, patch_size, D), where N_windows is the total number of extracted patches.
        class_token: A tensor of shape (C, D), where C is the number of class tokens (or 1 for a single class token).

        Returns:
            X_cls: A tensor of shape (N_windows, patch_size + C, D), where the first C elements are class tokens for each window.
        """
        B, D, T, H, W = patches.shape
        num_patches = T * H * W
        class_tokens_expanded = class_token.repeat(T,H,W,1)
        class_tokens_expanded = class_tokens_expanded.permute(3,0,1,2)
        class_tokens_expanded = class_tokens_expanded.unsqueeze(0)

        X_cls = torch.cat((class_tokens_expanded, patches), dim=1)

        return X_cls
    
    def create_class_token_map(self, x, x_cls):
        """
        Create the class token map.

        Args:
        x: A tensor of shape (N, C, T, H, W), where N is the batch size and C is the dimension of features.
        x_cls: A tensor of shape (N, C), where C is the dimension of class tokens.

        Returns:
            x_cls: A tensor of shape (N, C, T, H, W), where the first T elements are class tokens for each window.
        """
        B, C, T, H, W = x.shape

        x_cls = x_cls.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        x_cls = x_cls.expand(-1, -1, -1, H, W)
        x = torch.cat((x_cls, x), dim=2)

        return x


    def forward(self, x, features):
        """Forward function."""
        x_cls = self.dctg(features)
        print(f"x_cls shape: {x_cls.shape}")
        x = self.patch_embed(x)
        print(f"embed x shape: {x.shape}")
        # x = self.pos_drop(x)
        print(f"x after pos_drop shape: {x.shape}")
        # assign the class token to all windows
        x = self.create_class_token_map(x, x_cls)
        print(f"x after create_class_token_map shape: {x.shape}")
        for layer in self.layers:
            x = layer(x.contiguous())
        print(f"x after layers shape: {x.shape}")
        x = rearrange(x, 'n c d h w -> n d h w c')
        x = self.norm(x)
        x = rearrange(x, 'n d h w c -> n c d h w')

        return x


class Stage2(nn.Module):
        def __init__(self,pretrained=None,
                 pretrained2d=True,
                 patch_size=(2,4,4),
                 in_chans=3,
                 embed_dim=128,
                 depths=[2, 2, 18, 2],
                 num_heads=[4, 8, 16, 32],
                 window_size=(2,7,7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 patch_norm=False,
                 frozen_stages=-1,
                 use_checkpoint=False):
            
            super().__init__()
            self.pretrained = pretrained
            self.pretrained2d = pretrained2d
            self.num_layers = len(depths)
            self.embed_dim = embed_dim
            self.patch_norm = patch_norm
            self.frozen_stages = frozen_stages
            self.window_size = window_size
            self.patch_size = patch_size

            # stochastic depth
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

            # build 4th layer
            self.layer4 = BasicLayer(
                dim=int(embed_dim * 2**3),
                depth=depths[3],
                num_heads=num_heads[3],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:3]):sum(depths[:4])],
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint)
            
            self.num_features = int(embed_dim * 2**(self.num_layers-1))

            # add a norm layer for each output
            self.norm = norm_layer(self.num_features)

        def forward(self, x):
            """Forward function.
                Returns: x_cls: A tensor of shape (B, C, Hi, Wi), where B is the batch size and C is the dimension of class tokens.
            """

            x = self.layer4(x)
            print(f"x after layer4 shape: {x.shape}")
            x = rearrange(x, 'n c d h w -> n d h w c')
            x = self.norm(x)
            x = rearrange(x, 'n d h w c -> n c d h w')

            # get the class token map
            x_cls = x[:, :, 0, :, :]

            # Reshape to (B, C, 1, 5, 6)
            x_cls = x_cls.unsqueeze(2)

            return x_cls
        


class PADM(nn.Module):
    def __init__(self, G=4, embed_dim=128, window_size=(2,7,7)):
        super().__init__()

        self.G = G
        self.window_size = window_size
        self.num_features = int(embed_dim * 2**3)
        # pool size (4, 1, 1) is only for G=4 window size (2,7,7)
        self.avg_poll1 = nn.AvgPool3d(kernel_size=(4,1,1))
        self.avg_poll2 = nn.AvgPool3d(kernel_size=window_size)
        self.downsample = PatchMerging(dim=512, norm_layer=nn.LayerNorm)


    def pad_to_target(self,tensor):
        """
        Pad the last two dimensions (H, W) of a tensor to 14 if either is larger than 7,
        otherwise pad to 7. Only the last two dimensions are padded.

        Args:
        tensor (torch.Tensor): Input tensor of shape (B, G, C, D, H, W).

        Returns:
        torch.Tensor: Padded tensor.
        """
        B, G, C, D, H, W = tensor.size()
        
        # Determine the target size for H and W
        target_H = 14 if H > 7 else 7
        target_W = 14 if W > 7 else 7

        # Calculate padding amounts for H and W
        # Padding is applied symmetrically, so divide by 2
        pad_H = (target_H - H) if H < target_H else 0
        pad_W = (target_W - W) if W < target_W else 0

        # Pad only the last two dimensions (H, W)
        # Padding order in F.pad for the last two dimensions is: left, right, top, bottom
        pad = (pad_W // 2, pad_W - pad_W // 2, pad_H // 2, pad_H - pad_H // 2)
        
        # Apply padding
        padded_tensor = F.pad(tensor, pad=pad, mode='constant', value=0)
        
        return padded_tensor
    def padding(self, x_cls_group):
        B, G, C, D, H, W = x_cls_group.shape
        window_size = self.window_size

        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        x_cls_group = F.pad(x_cls_group, (pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))

        return x_cls_group
    
    def partion_xcls(self, x_cls_group):
        """
        Args:
            x: (B, G, C, D, H, W)
            window_size (tuple[int]): window size

        Returns:
            windows: (B, G, C, D=1, H//window_size, W//window_size)
        """
        window_size = self.window_size
        B, G, C, D, H, W = x_cls_group.shape

        print(f"B: {B}, G:{G}, D: {D}, H: {H}, W: {W}, C: {C}")
        x_cls_group = x_cls_group.view(B, G, C, D // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2], window_size[2])
        print(f"x_cls_group shape after view: {x_cls_group.shape}")
        x_cls_windows = x_cls_group.permute(0, 1, 3, 5, 7, 4, 6, 8, 2).contiguous().view(B, G, D // window_size[0], H // window_size[1], W // window_size[2], reduce(mul, window_size), C)

        print(f"x_cls_windows shape: {x_cls_windows.shape}")
        return x_cls_windows
    
    def avgpool3d(self, x_cls_group):
        """
        Applies average pooling operation to the input tensor to get the x_cls.

        Args:
            x_cls_group (torch.Tensor): Input tensor of shape (B, G, C, D, H, W), where
                                        B is the batch size, G is the number of groups,

        Returns:
            torch.Tensor: Output tensor after applying average pooling operation.
                          The shape of the output tensor is (1, G, C, D', H', W'), where
                          D' = D // window_size, H' = H // window_size, and W' = W // window_size.

        """
        window_size = self.window_size
        B, G, C, D, H, W = x_cls_group.shape
        x_cls_group = x_cls_group.view(B*G, C, D, H, W)
        x_cls_group = self.avg_poll2(x_cls_group)
        x_cls_group = x_cls_group.unsqueeze(0)
        # print(f"x_cls_group shape after avg_pool3d: {x_cls_group.shape}")

        return x_cls_group

    def merge_cls(self, x):
        num_groups, channels, depth, height, width = x.shape

        # Reshape tensor to merge spatial dimensions
        # Shape: [num_groups, channels, depth*height*width]
        x_flat = x.view(num_groups, channels, -1)

        # Compute L2 norms
        l2norms = torch.linalg.norm(x_flat, ord=2, dim=2, keepdim=True)

        # Normalize class tokens
        x_normalized = x_flat / l2norms

        # Compute dot products
        scores = torch.einsum('gcd,gce->ge', x_normalized, x_normalized) 

        # Mask diagonal to exclude self-comparison
        diag_mask = torch.eye(num_groups, dtype=torch.bool)
        scores[diag_mask] = 0

        # # Sum over all spatial dimensions to get αg,s,g′
        # alpha_gsgp = scores.sum(dim=1)  # Shape: [num_groups, num_groups]
        # print(f"alpha_gsgp shape: {alpha_gsgp.shape} \n alpha_gsgp: {alpha_gsgp}")

        # Sum over all groups 
        alpha_gs = scores.sum(dim=0)  # Excluding self-group
        # print(f"alpha_gs shape: {alpha_gs.shape} \n alpha_gs: {alpha_gs}")

        # # Sum scores along group axis to get total score per group per class token
        # alpha = scores.sum(dim=1)  # Shape: [num_groups]
        
        # Apply softmax to normalize scores
        alpha_normalized = F.softmax(alpha_gs, dim=0)

        # Weighted sum of class tokens along group axis
        xcls_weighted = torch.einsum('gcs,g->cs', x_flat, alpha_normalized)


        # Reshape to original size (1, C, D', H', W')
        x_cls = xcls_weighted.view(channels, depth, height, width)

        # Add Batch dimension
        x_cls = x_cls.unsqueeze(0)

        return x_cls
       
    def upsampling_xcls(self, x_cls):
        B, C, Di, Hi, Wi = x_cls.shape

        upsampled_size1 = (1, Hi*7, Wi*7)
        x_cls = F.interpolate(x_cls, size=upsampled_size1, mode='nearest')
        print(f"x_cls shape after upsampling: {x_cls.shape}")

        # slice the x_cls to (B, G, C, D, H, W)
        x_cls = x_cls.narrow(3, 0, 8)
        x_cls = x_cls.narrow(4, 0, 10)

        return x_cls

    def forward(self, x_xcl_s1):
        # separate the x and x_cls
        # x_cls_group shape is (B, G, C, H, W)
        x_cls_group = x_xcl_s1[:, :, :, 0, :, :]
        # x_group shape is (B, G, C, T, H, W)
        x_group = x_xcl_s1[:, :, :, 1:, :, :]
        print(f"x_cls_group shape: {x_cls_group.shape}")
        print(f"x_group shape: {x_group.shape}")

        # average pooling x_group along temporal dimension for each group
        x_group_avg = []
        for i in range(self.G):
            x_group_i = self.avg_poll1(x_group[:, i, :, :, :, :])
            x_group_i = x_group_i.squeeze(2)
            # print(f"x_group_i shape: {x_group_i.shape}")
            x_group_avg.append(x_group_i)
        x_group = torch.stack(x_group_avg, dim=1)
        # Chage the shape of x_group to (B, C, G, H, W)
        x_group = x_group.permute(0, 2, 1, 3, 4).contiguous()
        print(f"x_group shape: {x_group.shape}")

        # reshape to (B, G, C, D, H, W)
        # x_cls_group1 = x_cls_group[:, 0, :, :, :]
        x_cls_group = x_cls_group.unsqueeze(3)


        # padding the x_cls_group to (B, G, C, D, 14, 14)
        x_cls_group = self.padding(x_cls_group)
        x_cls_group = x_cls_group.squeeze(0)

        # get the x_cls from every windows
        # x_cls has a shape of (G, C, D', H', W')
        x_cls_group = self.avg_poll2(x_cls_group)
        print(f"----x_cls_group shape----: {x_cls_group.shape}")

        # merge the x_cls
        x_cls = self.merge_cls(x_cls_group)
        print(f"x_cls shape: {x_cls.shape}")

        # upsampling the x_cls
        x_cls = self.upsampling_xcls(x_cls)
        print(f"x_cls shape after upsampling: {x_cls.shape}")

        # merge the x_cls with x_group
        x_xcl_s2 = torch.cat((x_cls, x_group), dim=2)
        print(f"x_xcl_s2 shape: {x_xcl_s2.shape}")

        # PatchMerging for Stage3
        x_xcl_s2 = rearrange(x_xcl_s2, 'B C D H W -> B D H W C')
        x_xcl_s2 = self.downsample(x_xcl_s2)
        x_xcl_s2 = rearrange(x_xcl_s2, 'B D H W C -> B C D H W')
        print(f"x_xcl_s2 shape after downsample: {x_xcl_s2.shape}")

        return x_xcl_s2


class Head_3D(nn.Module):
    def __init__(self,
                 in_channels=1024,
                 num_classes=106,
                 dropout_ration=0.5):
        super(Head_3D, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ration
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(self.in_channels, self.num_classes)
    
    def forward(self, x):
        """
        Args:
            x: (B, C, 1, 5, 6)
        Returns:
            scores: (B, num_classes)
        """
        # x [B, C, 1, 5, 6]
        x = self.avgpool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        # x [B, C, 1, 1, 1]
        x = x.view(x.shape[0], -1)
        # [N, in_channels]
        print(f"x shape after flatten: {x.shape}")
        scores = self.fc(x)

        return scores
    


class GEgoviT(nn.Module):
    def __init__(self,
                pretrained=None,
                pretrained2d=True,
                patch_size=(2,4,4),
                in_chans=3,
                embed_dim=128,
                depths=[2, 2, 18, 2],
                num_heads=[4, 8, 16, 32],
                window_size=(2,7,7),
                mlp_ratio=4.,
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0.,
                attn_drop_rate=0.,
                drop_path_rate=0.2,
                norm_layer=nn.LayerNorm,
                patch_norm=False,
                frozen_stages=-1,
                use_checkpoint=False,
                hidden_size=256,
                D=128,
                M=2,
                G=4):
        super().__init__()

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.frozen_stages = frozen_stages
        self.window_size = window_size
        self.patch_size = patch_size
        self.G = G

        # bild layers
        self.stage1s = nn.ModuleList()
        for stage1_i in range(self.G):
            stage1 = Stage1(
                pretrained=pretrained,
                pretrained2d=pretrained2d,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
                depths=depths,
                num_heads=num_heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rate,
                norm_layer=norm_layer,
                patch_norm=patch_norm,
                frozen_stages=frozen_stages,
                use_checkpoint=use_checkpoint)
            self.stage1s.append(stage1)

        self.PADM = PADM()

        self.stage2 = Stage2(pretrained=pretrained,
                            pretrained2d=pretrained2d,
                            patch_size=patch_size,
                            in_chans=in_chans,
                            embed_dim=embed_dim,
                            depths=depths,
                            num_heads=num_heads,
                            window_size=window_size,
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            drop_rate=drop_rate,
                            attn_drop_rate=attn_drop_rate,
                            drop_path_rate=drop_path_rate,
                            norm_layer=norm_layer,
                            patch_norm=patch_norm,
                            frozen_stages=frozen_stages,
                            use_checkpoint=use_checkpoint)
        
        self.classifer = Head_3D(in_channels=1024, num_classes=106, dropout_ration=0.5)

    def forward(self, x, features):
        """Forward function."""
        # x = self.stage1(x, features)
        x_xcl_list = []
        print("---------------")
        print("Stage1")
        for stage1 in self.stage1s:
            x_xcl_list.append(stage1(x, features))
        print("---------------")
        print("PADM")
        # concatenate the output of each group
        x_xcl_s1 = torch.stack(x_xcl_list, dim=1)
        print(f"x_xcl_s1 shape: {x_xcl_s1.shape}")

        x_xcl_s2 = self.PADM(x_xcl_s1)
        # x = self.stage2(x_xcl_list[0])

        print("---------------")
        print("Stage2")
        print("---------------")
        x_xcl_s2 = self.stage2(x_xcl_s2)
        print(f"x_xcl_s2 shape after St2: {x_xcl_s2.shape}")

        scores = self.classifer(x_xcl_s2)

        return scores
        

# test the model
input = torch.randn(1, 3, 8, 120, 160)
input_feature = torch.randn(8, 5, 2048)
# remove value < 0
input_feature[input_feature < 0] = 0
input = input.to('cuda')
input_feature = input_feature.to('cuda')

# model = Stage1()
# model = model.to('cuda')
# St1 = model.forward(input, input_feature)
# print(f"St1 shape: {St1.shape}")
# print(f"St1 is: {St1[0,:,0,0,0]}")
# with open('Stage1.txt', 'w') as f:
#     f.write(str(model))

# Stage2 = Stage2()
# Stage2 = Stage2.to('cuda')
# St2 = Stage2.forward(St1)
# print(f"St2 shape: {St2.shape}")
# with open('Stage2.txt', 'w') as f:
#     f.write(str(Stage2))

model = GEgoviT()
model = model.to('cuda')
output = model.forward(input, input_feature)
# print(f"output shape: {output.shape}")
# for param in model.parameters():
#     print(type(param.data), param.size())
# with open('GEgoviT.txt', 'w') as f:
#     f.write(str(model))

# test_list = []
# for i in range(4):
#     test_list.append(torch.randn(1024, 4, 8, 10).to('cuda'))

# print(f"test_list length: {len(test_list)} and shape: {test_list[0].shape}")
# test = PADM()
# test = test.to('cuda')
# output_test = test.forward(test_list)