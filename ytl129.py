import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
import math
from torch.autograd import Function
import pywt
import numpy as np
from torch.nn import Module

class up_conv(nn.Module):
    '''Upsampling'''
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class MSL(nn.Module):
    '''MSL block'''
    def __init__(self, in_channels):
        super().__init__()
        self.act = nn.GELU()
        self.proj = nn.Linear(in_channels, in_channels)
        self.dim = in_channels
        self.split_groups = self.dim // 4
        self.ca_num_heads = 4
        self.expand_ratio = 2
        self.v = nn.Linear(in_channels, in_channels)
        self.s = nn.Linear(in_channels, in_channels)
        for i in range(self.ca_num_heads):
            local_conv = nn.Conv2d(in_channels // self.ca_num_heads, in_channels // self.ca_num_heads, kernel_size=(3 + i * 2),
                                   padding=(1 + i), stride=1, groups=in_channels // self.ca_num_heads)
            setattr(self, f"local_conv_{i + 1}", local_conv)
        self.proj0 = nn.Conv2d(in_channels, in_channels * self.expand_ratio, kernel_size=1, padding=0, stride=1)
        self.bn = nn.BatchNorm2d(in_channels * self.expand_ratio)
        self.proj1 = nn.Conv2d(in_channels * self.expand_ratio, in_channels, kernel_size=1, padding=0, stride=1)

    def forward(self, x, H, W):
        B, N, C = x.shape

        v = self.v(x)
        s = self.s(x).reshape(B, H, W, self.ca_num_heads, C // self.ca_num_heads).permute(3, 0, 4, 1, 2)
        for i in range(self.ca_num_heads):
            local_conv = getattr(self, f"local_conv_{i + 1}")
            s_i = s[i]
            s_i = local_conv(s_i).reshape(B, self.split_groups, -1, H, W)
            if i == 0:
                s_out = s_i
            else:
                s_out = torch.cat([s_out, s_i], 2)
        s_out = s_out.reshape(B, C, H, W)
        s_out = self.proj1(self.act(self.bn(self.proj0(s_out))))
        self.modulator = s_out
        s_out = s_out.reshape(B, C, N).permute(0, 2, 1)
        x = s_out * v
        return x

class DWTFunction_2D(Function):
    @staticmethod
    def forward(ctx, input, matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1):
        ctx.save_for_backward(matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1)
        L = torch.matmul(matrix_Low_0.to(input.device), input)
        H = torch.matmul(matrix_High_0.to(input.device), input)
        LL = torch.matmul(L, matrix_Low_1.to(L.device))
        LH = torch.matmul(L, matrix_High_1.to(L.device))
        HL = torch.matmul(H, matrix_Low_1.to(H.device))
        HH = torch.matmul(H, matrix_High_1.to(H.device))
        return LL, LH, HL, HH
    @staticmethod
    def backward(ctx, grad_LL, grad_LH, grad_HL, grad_HH):
        matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1 = ctx.saved_variables
        grad_L = torch.add(torch.matmul(grad_LL, matrix_Low_1.t().to(grad_LL.device)), torch.matmul(grad_LH, matrix_High_1.t().to(grad_LH.device)))
        grad_H = torch.add(torch.matmul(grad_HL, matrix_Low_1.t().to(grad_HL.device)), torch.matmul(grad_HH, matrix_High_1.t().to(grad_HH.device)))
        grad_input = torch.add(torch.matmul(matrix_Low_0.t().to(grad_L.device), grad_L), torch.matmul(matrix_High_0.t().to(grad_H.device), grad_H))
        return grad_input, None, None, None, None
class DWT_2D(Module):
    """
    input: the 2D data to be decomposed -- (N, C, H, W)
    output -- lfc: (N, C, H/2, W/2)
              hfc_lh: (N, C, H/2, W/2)
              hfc_hl: (N, C, H/2, W/2)
              hfc_hh: (N, C, H/2, W/2)
    """
    def __init__(self, wavename):
        """
        2D discrete wavelet transform (DWT) for 2D image decomposition
        :param wavename: pywt.wavelist(); in the paper, 'chx.y' denotes 'biorx.y'.
        """
        super(DWT_2D, self).__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.rec_lo
        self.band_high = wavelet.rec_hi
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)

    def get_matrix(self):
        """
        generating the matrices: \mathcal{L}, \mathcal{H}
        :return: self.matrix_low = \mathcal{L}, self.matrix_high = \mathcal{H}
        """
        L1 = np.max((self.input_height, self.input_width))
        L = math.floor(L1 / 2)
        matrix_h = np.zeros( ( L, L1 + self.band_length - 2 ) )
        matrix_g = np.zeros( ( L1 - L, L1 + self.band_length - 2 ) )
        end = None if self.band_length_half == 1 else (-self.band_length_half+1)

        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index+j] = self.band_low[j]
            index += 2
        matrix_h_0 = matrix_h[0:(math.floor(self.input_height / 2)), 0:(self.input_height + self.band_length - 2)]
        matrix_h_1 = matrix_h[0:(math.floor(self.input_width / 2)), 0:(self.input_width + self.band_length - 2)]

        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index+j] = self.band_high[j]
            index += 2
        matrix_g_0 = matrix_g[0:(self.input_height - math.floor(self.input_height / 2)),0:(self.input_height + self.band_length - 2)]
        matrix_g_1 = matrix_g[0:(self.input_width - math.floor(self.input_width / 2)),0:(self.input_width + self.band_length - 2)]

        matrix_h_0 = matrix_h_0[:,(self.band_length_half-1):end]
        matrix_h_1 = matrix_h_1[:,(self.band_length_half-1):end]
        matrix_h_1 = np.transpose(matrix_h_1)
        matrix_g_0 = matrix_g_0[:,(self.band_length_half-1):end]
        matrix_g_1 = matrix_g_1[:,(self.band_length_half-1):end]
        matrix_g_1 = np.transpose(matrix_g_1)

        if torch.cuda.is_available():
            self.matrix_low_0 = torch.Tensor(matrix_h_0).cuda()
            self.matrix_low_1 = torch.Tensor(matrix_h_1).cuda()
            self.matrix_high_0 = torch.Tensor(matrix_g_0).cuda()
            self.matrix_high_1 = torch.Tensor(matrix_g_1).cuda()
        else:
            self.matrix_low_0 = torch.Tensor(matrix_h_0)
            self.matrix_low_1 = torch.Tensor(matrix_h_1)
            self.matrix_high_0 = torch.Tensor(matrix_g_0)
            self.matrix_high_1 = torch.Tensor(matrix_g_1)

    def forward(self, input):
        """
        input_lfc = \mathcal{L} * input * \mathcal{L}^T
        input_hfc_lh = \mathcal{H} * input * \mathcal{L}^T
        input_hfc_hl = \mathcal{L} * input * \mathcal{H}^T
        input_hfc_hh = \mathcal{H} * input * \mathcal{H}^T
        :param input: the 2D data to be decomposed
        :return: the low-frequency and high-frequency components of the input 2D data
        """
        assert len(input.size()) == 4
        self.input_height = input.size()[-2]
        self.input_width = input.size()[-1]
        self.get_matrix()
        return DWTFunction_2D.apply(input, self.matrix_low_0, self.matrix_low_1, self.matrix_high_0, self.matrix_high_1)
class Downsamplewave(nn.Module):
    '''DWT'''
    def __init__(self, wavename = 'haar'):
        super().__init__()
        self.dwt = DWT_2D(wavename = wavename)

    def forward(self, input):
        LL,LH,HL,HH = self.dwt(input)
        return torch.cat([LL,LH+HL+HH],dim=1)

class CRU(nn.Module):
    '''
    CRU
    alpha: 0<alpha<1
    '''

    def __init__(self,
                 op_channel: int,
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 ):
        super().__init__()
        self.up_channel = up_channel = int(alpha * op_channel)
        self.low_channel = low_channel = op_channel - up_channel
        self.squeeze1 = nn.Conv2d(up_channel, up_channel // squeeze_radio, kernel_size=1, bias=False)
        self.squeeze2 = nn.Conv2d(low_channel, low_channel // squeeze_radio, kernel_size=1, bias=False)
        # up
        self.GWC = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=group_kernel_size, stride=1,
                             padding=group_kernel_size // 2, groups=group_size)
        self.PWC1 = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=1, bias=False)
        # low
        self.PWC2 = nn.Conv2d(low_channel // squeeze_radio, op_channel - low_channel // squeeze_radio, kernel_size=1,
                              bias=False)
        self.advavg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # Split
        up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)
        up, low = self.squeeze1(up), self.squeeze2(low)
        # Transform
        Y1 = self.GWC(up) + self.PWC1(up)
        Y2 = torch.cat([self.PWC2(low), low], dim=1)
        # Fuse
        out = torch.cat([Y1, Y2], dim=1)
        out = F.softmax(self.advavg(out), dim=1) * out
        out1, out2 = torch.split(out, out.size(1) // 2, dim=1)
        return out1 + out2

class HLSA(nn.Module):
    def __init__(self, dim, num_heads, sr_ratio,attn_drop=0.1):
        super().__init__()
        self.attn_drop = attn_drop
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.sr_ratio = sr_ratio

        self.kv_embed = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio) if sr_ratio > 1 else nn.Identity()
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2)
        )
        self.proj = nn.Linear(dim * 2, dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, x2, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        temp = x
        x2 = x2.view(B, H, W, C).permute(0, 3, 1, 2)
        kv = self.kv_embed(x2).reshape(B, C, -1).permute(0, 2, 1)
        kv = self.kv(kv).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q * (q.shape[1] ** -0.5)  @ k.transpose(-2, -1)) * self.scale
        m_r = torch.ones_like(attn) * self.attn_drop
        attn = attn + torch.bernoulli(m_r) * -1e12
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(torch.cat([x, temp], dim=-1))
        return x

class CSA(nn.Module):
    '''CSA block'''
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        # -------------------
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        # ------------------
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature'}

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.act(x + self.dwconv(x, H, W))
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, ca_num_heads=2, sa_num_heads=8, qkv_bias=False,
                 attn_drop=0., proj_drop=0., ca_attention=1, expand_ratio=2):
        super().__init__()

        self.ca_attention = ca_attention
        self.dim = dim
        self.ca_num_heads = ca_num_heads
        self.sa_num_heads = sa_num_heads

        self.act = nn.GELU()
        self.proj = nn.Linear(dim, dim)
        self.up = up_conv(dim // 2,dim // 2)
        stem = [Downsamplewave(), nn.BatchNorm2d(dim), nn.ReLU(True)]
        self.wave = nn.Sequential(*stem)

        self.split_groups = self.dim // ca_num_heads
        self.CRU = CRU(dim, alpha=1 / 2, squeeze_radio=2, group_size=2, group_kernel_size=3)
        if ca_attention == 1:
            self.v = nn.Linear(dim, dim, bias=qkv_bias)
            self.s = nn.Linear(dim, dim, bias=qkv_bias)
            for i in range(self.ca_num_heads):
                if i == 0:
                    local_conv = nn.Conv2d(dim // self.ca_num_heads, dim // self.ca_num_heads, kernel_size=(9),
                                           padding=(4), stride=1, groups=dim // self.ca_num_heads)
                    setattr(self, f"local_conv_{i + 1}", local_conv)
                elif i == 1:
                    local_conv = nn.Conv2d(dim // self.ca_num_heads, dim // self.ca_num_heads, kernel_size=(3),
                                           padding=(1), stride=1)
                    setattr(self, f"local_conv_{i + 1}", local_conv)
            self.proj0 = nn.Conv2d(dim, dim * expand_ratio, kernel_size=1, padding=0, stride=1,
                                   groups=self.split_groups)
            self.bn = nn.BatchNorm2d(dim * expand_ratio)
            self.proj1 = nn.Conv2d(dim * expand_ratio, dim, kernel_size=1, padding=0, stride=1)
        else:
            self.gelu = nn.GELU()
            self.bn1 = nn.BatchNorm2d(dim)
            self.CSA= CSA(dim, num_heads=sa_num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)
            self.MSL = MSL(dim)
            dim = dim // 2
            self.local_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim)
            self.HLSA = HLSA(dim,sa_num_heads,1)
            self.mixer = nn.Conv2d(dim, dim * 2, 1, 1, 0)
            self.mixer1 = nn.Conv2d(dim * 2, dim * 2, 1, 1, 0)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        if self.ca_attention == 1:
            v = self.v(x)
            s = self.s(x).reshape(B, H, W, self.ca_num_heads, C // self.ca_num_heads).permute(3, 0, 4, 1, 2)
            for i in range(self.ca_num_heads):
                local_conv = getattr(self, f"local_conv_{i + 1}")
                s_i = s[i]
                if i == 0:
                    temp = s_i
                    y1 = self.wave(temp)
                    y11, y12 = y1.chunk(2, dim=1)
                    y12 = self.up(y12)
                    s_i = local_conv(s_i).reshape(B, self.split_groups, -1, H, W)
                else:
                    s_i = s_i + y12
                    s_i = local_conv(s_i).reshape(B, self.split_groups, -1, H, W)
                if i == 0:
                    s_out = s_i
                else:
                    s_i = s_i.reshape(B, C // self.ca_num_heads, H, W)
                    s_out = s_out.reshape(B, C // self.ca_num_heads, H, W)
                    att_outputs = [s_out, s_i]
                    s_out = torch.cat(att_outputs, dim=1)

            s_out = self.proj1(self.act(self.bn(self.proj0(s_out))))
            self.modulator = s_out
            s_out = s_out.reshape(B, C, N).permute(0, 2, 1)
            x = s_out * v
            x = x.permute(0, 2, 1).reshape(B, C, H, W)
            x = self.CRU(x)
            x = x.reshape(B, C, N).permute(0, 2, 1)
        else:
            # CSA
            temp = x
            temp = self.CSA(temp)
            temp = temp.permute(0, 2, 1).reshape(B, C, H, W)

            # MSL
            temp2 = x
            temp2 = self.MSL(temp2, H, W)
            temp2 = temp2.permute(0, 2, 1).reshape(B, C, H, W)
            temp3 = torch.sigmoid(temp2)
            temp2 = temp2 * temp3
            temp2 = self.bn1(temp2)
            temp2 = self.gelu(temp2)

            # Supplement the high-frequency components from the low-frequency part to the high-frequency band
            x1,x2 = x.chunk(2, dim=2)
            x1 = x1.reshape(B, C // 2, H, W)
            x2 = x2.reshape(B, C // 2, H, W)
            if H == 7:
                x1 = F.interpolate(input=x1, size=(8, 8), mode='bilinear', align_corners=False)
                y1 = self.wave(x1)
                y11, y12 = y1.chunk(2, dim=1)
                y12 = self.up(y12)
                y12 = F.interpolate(input=y12, size=(7, 7), mode='bilinear', align_corners=False)
                x1 = F.interpolate(input=x1, size=(7, 7), mode='bilinear', align_corners=False)
            else:
                y1 = self.wave(x1)
                y11, y12 = y1.chunk(2, dim=1)
                y12 = self.up(y12)
            temp4 = y12 + x2

            # HLSA
            C = C // 2
            x1 = x1.reshape(B, C, N).permute(0, 2, 1)
            x1 = self.HLSA(x1, temp4, H, W)
            x2 = x2.reshape(B, C, N).permute(0, 2, 1)
            temp4 = temp4.reshape(B, C, N).permute(0, 2, 1)
            x2 = self.HLSA(x2, temp4, H, W)
            x1 = x1 * x2

            x1 = x1.permute(0, 2, 1).reshape(B, C, H, W)
            x = self.mixer(x1)
            x = self.bn1(x)
            x = self.gelu(x)

            local_weight = torch.sigmoid(x)
            local_feat = temp * local_weight
            global_feat = x * temp
            x = local_feat * global_feat
            x = self.bn1(x)
            x = self.gelu(x)

            local_weight1 = torch.sigmoid(x)
            local_feat1 = temp2 * local_weight1
            global_feat1 = x * temp2
            x = local_feat1 * global_feat1
            x = self.bn1(x)
            x = self.gelu(x)
            C = C * 2
            x = self.CRU(x)
            x = x.permute(0, 2, 3, 1).reshape(B, -1, C)

        x = self.proj(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, ca_num_heads, sa_num_heads, mlp_ratio=4., qkv_bias=False,
                 use_layerscale=False, layerscale_value=1e-4, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, ca_attention=1, expand_ratio=2):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, ca_num_heads=ca_num_heads, sa_num_heads=sa_num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop, ca_attention=ca_attention,
            expand_ratio=expand_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.gamma_1 = 1.0
        self.gamma_2 = 1.0
        if use_layerscale:
            self.gamma_1 = nn.Parameter(layerscale_value * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(layerscale_value * torch.ones((dim)), requires_grad=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):

        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x), H, W))

        return x

class Patch_Embedding(nn.Module):
    '''Patch Embedding Block'''
    def __init__(self, in_chans=3, embed_dim=768):
        super().__init__()
        stem = [nn.Conv2d(in_chans, embed_dim // 2, kernel_size=1, stride=1, padding=0)]
        stem.append(nn.BatchNorm2d(embed_dim // 2))
        stem.append(nn.ReLU(True))
        stem.append(Downsamplewave())
        stem.append(nn.BatchNorm2d(embed_dim))
        stem.append(nn.ReLU(True))
        self.wave = nn.Sequential(*stem)
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.wave(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W

class Wave_Stem(nn.Module):
    '''Wave Stem Block'''
    def __init__(self,dim):
        super().__init__()
        stem = [Downsamplewave(),nn.BatchNorm2d(6),nn.ReLU(True)]
        stem.append(nn.Conv2d(6, dim // 2, kernel_size=1, stride=1, padding=0))
        stem.append(nn.BatchNorm2d(dim // 2))
        stem.append(nn.ReLU(True))
        stem.append(Downsamplewave())
        stem.append(nn.BatchNorm2d(dim))
        stem.append(nn.ReLU(True))
        self.wave = nn.Sequential(*stem)
        self.norm = nn.LayerNorm(dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.wave(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W

class FDNet(nn.Module):
    def __init__(self, num_classes=7, embed_dims=[64, 128, 192, 256],
                 ca_num_heads=[2, 2, 2, -1], sa_num_heads=[-1, -1, 8, 16], mlp_ratios=[8, 6, 4, 2],
                 qkv_bias= False , depths=[4, 4, 4, 2], ca_attentions=[1, 1, 1, 0], num_stages=4, expand_ratio=2):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages
        cur = 0
        self.conv = nn.Conv2d(embed_dims[2], embed_dims[2], kernel_size=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(embed_dims[1], embed_dims[1], kernel_size=1, stride=1, padding=0)
        self.norm_layer = nn.LayerNorm
        for i in range(num_stages):
            if i == 0:
                patch_embed = Wave_Stem(embed_dims[i])# Wave Stem Block
            else:
                patch_embed = Patch_Embedding(
                                                in_chans=embed_dims[i - 1],
                                                embed_dim=embed_dims[i])# Patch Embedding Block
            block = nn.ModuleList([Block(
                dim=embed_dims[i], ca_num_heads=ca_num_heads[i], sa_num_heads=sa_num_heads[i], mlp_ratio=mlp_ratios[i],
                qkv_bias=qkv_bias, ca_attention=0 if i == 2 and j % 2 != 0 else ca_attentions[i], expand_ratio=expand_ratio)
                for j in range(depths[i])])
            norm = self.norm_layer(embed_dims[i])
            cur += depths[i]
            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)
        # classification head
        self.head = nn.Linear(embed_dims[2]+embed_dims[3]+embed_dims[1], num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward_features(self, x):
        B = x.shape[0]
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            if i != self.num_stages - 1:
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            if i == 1:
                C = x.shape[1]
                x2 = self.conv1(x)
                x2 = F.adaptive_avg_pool2d(x2, (1, 1))
                x2 = x2.reshape(B, -1, C)
            if i == 2:
                C = x.shape[1]
                x1 = self.conv(x)
                x1 = F.adaptive_avg_pool2d(x1, (1, 1))
                x1 = x1.reshape(B, -1, C)
        C = x.shape[2]
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.reshape(B, -1, C)
        att_outputs = [x, x1, x2]
        out_concat = torch.cat(att_outputs, dim=-1)
        return out_concat.mean(dim=1)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

def FDnet(**kwargs):
    model = FDNet(
        embed_dims=[64, 128, 192, 256], ca_num_heads=[2, 2, 2, -1], sa_num_heads=[-1, -1, 8, 16],
        mlp_ratios=[4, 4, 4, 2],depths=[4, 4, 4, 2], ca_attentions=[1, 1, 1, 0], expand_ratio=2, **kwargs)
    return model



