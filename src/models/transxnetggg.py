import os
import math
import copy
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import checkpoint
from timm.models.registry import register_model
from timm.models.layers import DropPath, to_2tuple
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from mmcv.cnn.bricks import ConvModule, build_activation_layer, build_norm_layer

try:
    from mmseg.models.builder import BACKBONES as seg_BACKBONES
    from mmseg.utils import get_root_logger
    from mmcv.runner import _load_checkpoint

    has_mmseg = True
except ImportError:
    # print("If for semantic segmentation, please install mmsegmentation first")
    has_mmseg = False

try:
    from mmdet.models.builder import BACKBONES as det_BACKBONES
    from mmdet.utils import get_root_logger
    from mmcv.runner import _load_checkpoint

    has_mmdet = True
except ImportError:
    # print("If for detection, please install mmdetection first")
    has_mmdet = False


class DifferentialAttention(nn.Module):
    """
    Differential Attention
    """

    def __init__(self, dim, num_heads=8, qk_scale=None, attn_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5

        # 确保head_dim能被2整除，因为我们要分割查询向量
        assert self.head_dim % 2 == 0, f"head_dim {self.head_dim} must be divisible by 2 for query splitting"

        # QKV投影
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)

        # 可学习的lambda参数用于平衡两个softmax组
        self.lambda_param = nn.Parameter(torch.ones(1, num_heads, 1, 1) * 0.5)

        # 输出投影
        self.proj = nn.Conv2d(dim, dim, 1)
        self.attn_drop = nn.Dropout(attn_drop)

        # GroupNorm设置
        group_norm_channels = num_heads * (self.head_dim // 2)
        # 确保group_norm_channels是正整数
        group_norm_channels = max(1, group_norm_channels)
        num_groups = min(2, group_norm_channels)  # GroupNorm的组数不能超过通道数

        self.group_norm = nn.GroupNorm(num_groups, group_norm_channels)

    def forward(self, x, relative_pos_enc=None):
        B, C, H, W = x.shape

        # 生成Q, K, V
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=1)

        # 重塑为多头注意力格式 [B, num_heads, head_dim, H*W]
        q = q.reshape(B, self.num_heads, self.head_dim, H * W)
        k = k.reshape(B, self.num_heads, self.head_dim, H * W)
        v = v.reshape(B, self.num_heads, self.head_dim, H * W)

        # 将查询分割为两组用于差分注意力
        # q1, q2: [B, num_heads, head_dim//2, H*W]
        q1, q2 = torch.chunk(q, 2, dim=2)

        # 对第二组查询进行归一化
        # 先将q2重塑为GroupNorm期望的格式 [B, num_heads * (head_dim//2), H, W]
        q2_reshaped = q2.reshape(B, self.num_heads * (self.head_dim // 2), H, W)
        q2_normalized = self.group_norm(q2_reshaped)
        q2 = q2_normalized.reshape(B, self.num_heads, self.head_dim // 2, H * W)

        # 同样需要对k进行分割以匹配q1和q2的维度
        k1, k2 = torch.chunk(k, 2, dim=2)  # k1, k2: [B, num_heads, head_dim//2, H*W]

        # 计算两组的注意力分数
        # q1和k1的维度都是 [B, num_heads, head_dim//2, H*W]
        # 转置后：q1: [B, num_heads, H*W, head_dim//2], k1: [B, num_heads, head_dim//2, H*W]
        attn1 = torch.matmul(q1.transpose(-1, -2), k1) * self.scale  # [B, num_heads, H*W, H*W]
        attn2 = torch.matmul(q2.transpose(-1, -2), k2) * self.scale * self.lambda_param

        # 对两个注意力矩阵应用softmax
        attn1 = F.softmax(attn1, dim=-1)
        attn2 = F.softmax(attn2, dim=-1)

        # 差分操作：相减两个注意力矩阵
        differential_attn = attn1 - attn2

        # 应用注意力dropout
        differential_attn = self.attn_drop(differential_attn)

        # 同样需要对v进行分割
        v1, v2 = torch.chunk(v, 2, dim=2)

        # 计算输出 - 使用两组值的平均值
        output1 = torch.matmul(differential_attn, v1.transpose(-1, -2))
        output2 = torch.matmul(differential_attn, v2.transpose(-1, -2))

        # 合并输出
        x = torch.cat([output1, output2], dim=2)  # [B, num_heads, head_dim, H*W]

        # 重塑回原始形状
        x = x.reshape(B, self.dim, H, W)

        # 输出投影
        x = self.proj(x)
        return x


class MUDDConnection(nn.Module):
    """
    MUltiway Dynamic Dense Connection (MUDD)
    """

    def __init__(self, dim, num_paths=4, reduction_ratio=8):
        super().__init__()
        self.dim = dim
        self.num_paths = num_paths
        # 确保维度能被num_paths整除
        assert dim % num_paths == 0, f"dim {dim} must be divisible by num_paths {num_paths}"
        self.path_dims = dim // num_paths
        self.reduction_ratio = reduction_ratio

        # 动态权重生成器 - 轻量级设计
        self.weight_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // reduction_ratio, 1),
            nn.GELU(),
            nn.Conv2d(dim // reduction_ratio, num_paths * num_paths, 1)
        )

        # 路径特定的变换 - 使用深度可分离卷积保持轻量
        self.path_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.path_dims, self.path_dims, 3, padding=1, groups=self.path_dims),
                nn.GELU(),
                nn.Conv2d(self.path_dims, self.path_dims, 1)
            ) for _ in range(num_paths)
        ])

        # 添加空间尺寸适配层
        self.spatial_adapter = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),  # 全局平均池化
                nn.Conv2d(self.path_dims, self.path_dims, 1),  # 1x1卷积调整
                nn.GELU()
            ) for _ in range(num_paths)
        ])

        print(f"MUDD Connection: dim={dim}, paths={num_paths}, path_dims={self.path_dims}")

    def _adapt_spatial_size(self, x, target_size):
        """调整特征图空间尺寸以匹配目标尺寸"""
        if x.size(-2) != target_size[-2] or x.size(-1) != target_size[-1]:
            # 使用插值调整空间尺寸
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return x

    def forward(self, current_x, previous_features):
        """
        Args:
            current_x: 当前层特征 [B, C, H, W]
            previous_features: 前面所有层的特征列表 [[B, C, H, W], ...]
        """
        if not previous_features:
            return current_x

        B, C, H, W = current_x.shape
        target_spatial_size = (H, W)  # 目标空间尺寸

        # 生成动态连接权重 [B, num_paths * num_paths, 1, 1]
        weight_logits = self.weight_generator(current_x)
        weight_matrix = torch.softmax(weight_logits.view(B, self.num_paths, self.num_paths), dim=-1)

        # 分割当前特征为多路
        current_paths = torch.split(current_x, self.path_dims, dim=1)

        # 处理前面各层的特征，确保尺寸匹配
        previous_paths_list = []
        for prev_feat in previous_features:
            # 调整前面层特征的空间尺寸以匹配当前层
            prev_feat_adapted = self._adapt_spatial_size(prev_feat, target_spatial_size)
            prev_paths = torch.split(prev_feat_adapted, self.path_dims, dim=1)
            # 如果分割后路径数量不匹配，使用适配器
            if len(prev_paths) < self.num_paths:
                # 补充缺失的路径
                prev_paths = list(prev_paths)
                while len(prev_paths) < self.num_paths:
                    prev_paths.append(torch.zeros_like(prev_paths[0]))
            previous_paths_list.append(prev_paths)

        # 多路动态聚合
        output_paths = []
        for path_idx in range(self.num_paths):
            if path_idx >= len(current_paths):
                # 如果当前路径不存在，跳过或使用零张量
                continue

            path_output = self.path_transforms[path_idx](current_paths[path_idx])

            # 动态聚合前面各层的对应路径特征
            for layer_idx, prev_paths in enumerate(previous_paths_list):
                if path_idx < len(prev_paths) and prev_paths[path_idx] is not None:
                    # 应用动态权重
                    weight = weight_matrix[:, path_idx, layer_idx % self.num_paths].view(B, 1, 1, 1)
                    weighted_prev = prev_paths[path_idx] * weight

                    # 确保尺寸匹配后再相加
                    if path_output.shape[-2:] != weighted_prev.shape[-2:]:
                        weighted_prev = F.interpolate(
                            weighted_prev,
                            size=path_output.shape[-2:],
                            mode='bilinear',
                            align_corners=False
                        )

                    path_output = path_output + weighted_prev

            output_paths.append(path_output)

        # 合并多路输出
        if output_paths:
            output = torch.cat(output_paths, dim=1)
            # 确保输出通道数正确
            if output.size(1) != current_x.size(1):
                # 使用1x1卷积调整通道数
                output = nn.Conv2d(output.size(1), current_x.size(1), 1).to(output.device)(output)
        else:
            output = current_x

        return output


class MultiScaleCoordinateAttention(nn.Module):
    """
    Multi-Scale Coordinate Attention (MCA)
    """

    def __init__(self, dim, reduction_ratio=32, scales=None):
        super().__init__()

        # 智能的scales选择策略
        if scales is None:
            # 根据维度自动选择最合适的尺度数量
            if dim >= 64 and dim % 4 == 0:
                scales = [1, 2, 4]  # 3个尺度
            elif dim >= 48 and dim % 3 == 0:
                scales = [1, 2, 3]  # 3个尺度
            elif dim >= 32 and dim % 2 == 0:
                scales = [1, 2]  # 2个尺度
            else:
                scales = [1]  # 1个尺度（保底方案）

        self.scales = scales
        self.dim = dim
        self.num_scales = len(scales)

        # 动态调整channels_per_scale确保整除
        # 计算基础通道数，处理余数
        base_channels = dim // self.num_scales
        remainder = dim % self.num_scales

        # 分配通道，确保总和等于dim
        self.channels_per_scale = []
        for i in range(self.num_scales):
            if i < remainder:
                # 前remainder个分支多分配1个通道
                self.channels_per_scale.append(base_channels + 1)
            else:
                self.channels_per_scale.append(base_channels)

        print(f"MCA调试: dim={dim}, scales={scales}, channels分配={self.channels_per_scale}")

        # 多尺度池化分支
        self.scale_branches = nn.ModuleList()
        for i, scale in enumerate(scales):
            branch_channels = self.channels_per_scale[i]
            branch = nn.Sequential(
                nn.AdaptiveAvgPool2d((scale, scale)),
                nn.Conv2d(branch_channels, branch_channels, 1, bias=False),
                nn.BatchNorm2d(branch_channels),
                nn.GELU()
            )
            self.scale_branches.append(branch)

        # 坐标信息嵌入
        coord_dim = max(1, dim // (8 * self.num_scales))
        self.coord_conv = nn.Conv2d(2, coord_dim, 1, bias=False)

        # 注意力生成
        attention_input_dim = sum(self.channels_per_scale) + coord_dim
        total_reduced_dim = max(1, dim // reduction_ratio)

        self.attention_generator = nn.Sequential(
            nn.Conv2d(attention_input_dim, total_reduced_dim, 1),
            nn.BatchNorm2d(total_reduced_dim),
            nn.GELU(),
            nn.Conv2d(total_reduced_dim, dim, 1),
            nn.Sigmoid()
        )

        # 最终融合层
        self.final_fusion = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim)
        )

        # 添加通道重映射，确保输出维度一致
        if sum(self.channels_per_scale) != dim:
            self.channel_remap = nn.Conv2d(sum(self.channels_per_scale), dim, 1)
        else:
            self.channel_remap = nn.Identity()

    def _get_coord_maps(self, x):
        batch_size, _, height, width = x.size()
        y_coords = torch.linspace(-1, 1, height, device=x.device)
        x_coords = torch.linspace(-1, 1, width, device=x.device)

        y_coords = y_coords.view(1, 1, height, 1).expand(batch_size, 1, height, width)
        x_coords = x_coords.view(1, 1, 1, width).expand(batch_size, 1, height, width)

        coords = torch.cat([x_coords, y_coords], dim=1)
        return coords

    def forward(self, x):
        batch_size, _, height, width = x.size()

        # 按通道数分割输入特征
        x_splits = torch.split(x, self.channels_per_scale, dim=1)

        # 多尺度特征提取
        scale_features = []
        for i, (x_split, scale) in enumerate(zip(x_splits, self.scales)):
            scaled_feat = self.scale_branches[i](x_split)
            # 上采样到原始尺寸
            scaled_feat = F.interpolate(scaled_feat, size=(height, width),
                                        mode='bilinear', align_corners=False)
            scale_features.append(scaled_feat)

        # 多尺度特征拼接
        multi_scale_feat = torch.cat(scale_features, dim=1)

        # 坐标信息嵌入
        coord_maps = self._get_coord_maps(x)
        coord_feat = self.coord_conv(coord_maps)

        # 特征融合和注意力生成
        fused_feat = torch.cat([multi_scale_feat, coord_feat], dim=1)
        attention_weights = self.attention_generator(fused_feat)

        # 应用注意力权重
        attended_features = []
        current_channel = 0
        for i, channels in enumerate(self.channels_per_scale):
            weight_slice = attention_weights[:, current_channel:current_channel + channels]
            attended = scale_features[i] * weight_slice
            attended_features.append(attended)
            current_channel += channels

        # 融合多尺度特征
        output = torch.cat(attended_features, dim=1)
        output = self.channel_remap(output)  # 确保输出维度正确
        output = self.final_fusion(output)

        return output + x  # 残差连接


class PatchEmbed(nn.Module):
    """Patch Embedding module implemented by a layer of convolution."""

    def __init__(self,
                 patch_size=16,
                 stride=16,
                 padding=0,
                 in_chans=3,
                 embed_dim=768,
                 norm_layer=dict(type='BN2d'),
                 act_cfg=None, ):
        super().__init__()
        self.proj = ConvModule(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=padding,
            norm_cfg=norm_layer,
            act_cfg=act_cfg,
        )

    def forward(self, x):
        return self.proj(x)


class Attention(nn.Module):  ### OSRA
    def __init__(self, dim,
                 num_heads=1,
                 qk_scale=None,
                 attn_drop=0,
                 sr_ratio=1, ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.sr_ratio = sr_ratio
        self.q = nn.Conv2d(dim, dim, kernel_size=1)
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1)
        self.attn_drop = nn.Dropout(attn_drop)
        if sr_ratio > 1:
            self.sr = nn.Sequential(
                ConvModule(dim, dim,
                           kernel_size=sr_ratio + 3,
                           stride=sr_ratio,
                           padding=(sr_ratio + 3) // 2,
                           groups=dim,
                           bias=False,
                           norm_cfg=dict(type='BN2d'),
                           act_cfg=dict(type='GELU')),
                ConvModule(dim, dim,
                           kernel_size=1,
                           groups=dim,
                           bias=False,
                           norm_cfg=dict(type='BN2d'),
                           act_cfg=None, ), )
        else:
            self.sr = nn.Identity()
        self.local_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x, relative_pos_enc=None):
        B, C, H, W = x.shape
        q = self.q(x).reshape(B, self.num_heads, C // self.num_heads, -1).transpose(-1, -2)
        kv = self.sr(x)
        kv = self.local_conv(kv) + kv
        k, v = torch.chunk(self.kv(kv), chunks=2, dim=1)
        k = k.reshape(B, self.num_heads, C // self.num_heads, -1)
        v = v.reshape(B, self.num_heads, C // self.num_heads, -1).transpose(-1, -2)
        attn = (q @ k) * self.scale
        if relative_pos_enc is not None:
            if attn.shape[2:] != relative_pos_enc.shape[2:]:
                relative_pos_enc = F.interpolate(relative_pos_enc, size=attn.shape[2:],
                                                 mode='bicubic', align_corners=False)
            attn = attn + relative_pos_enc
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(-1, -2)
        return x.reshape(B, C, H, W)


class DynamicConv2d(nn.Module):  ### IDConv
    def __init__(self,
                 dim,
                 kernel_size=3,
                 reduction_ratio=4,
                 num_groups=1,
                 bias=True):
        super().__init__()
        assert num_groups > 1, f"num_groups {num_groups} should > 1."
        self.num_groups = num_groups
        self.K = kernel_size
        self.bias_type = bias
        self.weight = nn.Parameter(torch.empty(num_groups, dim, kernel_size, kernel_size), requires_grad=True)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(kernel_size, kernel_size))
        self.proj = nn.Sequential(
            ConvModule(dim,
                       dim // reduction_ratio,
                       kernel_size=1,
                       norm_cfg=dict(type='BN2d'),
                       act_cfg=dict(type='GELU'), ),
            nn.Conv2d(dim // reduction_ratio, dim * num_groups, kernel_size=1), )

        if bias:
            self.bias = nn.Parameter(torch.empty(num_groups, dim), requires_grad=True)
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.weight, std=0.02)
        if self.bias is not None:
            nn.init.trunc_normal_(self.bias, std=0.02)

    def forward(self, x):
        B, C, H, W = x.shape
        scale = self.proj(self.pool(x)).reshape(B, self.num_groups, C, self.K, self.K)
        scale = torch.softmax(scale, dim=1)
        weight = scale * self.weight.unsqueeze(0)
        weight = torch.sum(weight, dim=1, keepdim=False)
        weight = weight.reshape(-1, 1, self.K, self.K)

        if self.bias is not None:
            scale = self.proj(torch.mean(x, dim=[-2, -1], keepdim=True))
            scale = torch.softmax(scale.reshape(B, self.num_groups, C), dim=1)
            bias = scale * self.bias.unsqueeze(0)
            bias = torch.sum(bias, dim=1).flatten(0)
        else:
            bias = None

        x = F.conv2d(x.reshape(1, -1, H, W),
                     weight=weight,
                     padding=self.K // 2,
                     groups=B * C,
                     bias=bias)

        return x.reshape(B, C, H, W)


class HybridTokenMixer(nn.Module):  ### D-Mixer
    def __init__(self,
                 dim,
                 kernel_size=3,
                 num_groups=2,
                 num_heads=1,
                 sr_ratio=1,
                 reduction_ratio=8,
                 use_mca=True,
                 use_differential_attn=False):  # 新增差分注意力选项
        super().__init__()
        assert dim % 2 == 0, f"dim {dim} should be divided by 2."

        self.local_unit = DynamicConv2d(
            dim=dim // 2, kernel_size=kernel_size, num_groups=num_groups)

        # 选择使用标准注意力还是差分注意力
        if use_differential_attn:
            self.global_unit = DifferentialAttention(
                dim=dim // 2, num_heads=num_heads)
        else:
            self.global_unit = Attention(
                dim=dim // 2, num_heads=num_heads, sr_ratio=sr_ratio)

        # 多尺度坐标注意力
        self.use_mca = use_mca
        if use_mca:
            mca_dim = dim // 2
            if mca_dim % 4 == 0:
                scales = [1, 2, 4]
            elif mca_dim % 3 == 0:
                scales = [1, 2, 3]
            elif mca_dim % 2 == 0:
                scales = [1, 2]
            else:
                scales = [1]

            self.mca = MultiScaleCoordinateAttention(mca_dim, scales=scales)

        inner_dim = max(16, dim // reduction_ratio)
        self.proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.GELU(),
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, inner_dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(inner_dim),
            nn.Conv2d(inner_dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim), )

    def forward(self, x, relative_pos_enc=None):
        x1, x2 = torch.chunk(x, chunks=2, dim=1)
        x1 = self.local_unit(x1)

        # 根据global_unit的类型传递参数
        if hasattr(self.global_unit, '__class__') and 'DifferentialAttention' in str(self.global_unit.__class__):
            x2 = self.global_unit(x2)  # DifferentialAttention不需要relative_pos_enc
        else:
            x2 = self.global_unit(x2, relative_pos_enc)  # 标准Attention需要

        # Keep MCA active in the GGG mixer when configured.
        if self.use_mca:
            x2 = self.mca(x2)

        x = torch.cat([x1, x2], dim=1)
        return x


class MultiScaleDWConv(nn.Module):
    def __init__(self, dim, scale=(1, 3, 5, 7)):
        super().__init__()
        self.scale = scale
        self.channels = []
        self.proj = nn.ModuleList()
        for i in range(len(scale)):
            if i == 0:
                channels = dim - dim // len(scale) * (len(scale) - 1)
            else:
                channels = dim // len(scale)
            conv = nn.Conv2d(channels, channels,
                             kernel_size=scale[i],
                             padding=scale[i] // 2,
                             groups=channels)
            self.channels.append(channels)
            self.proj.append(conv)

    def forward(self, x):
        x = torch.split(x, split_size_or_sections=self.channels, dim=1)
        out = []
        for i, feat in enumerate(x):
            out.append(self.proj[i](feat))
        x = torch.cat(out, dim=1)
        return x


class Mlp(nn.Module):  ### MS-FFN
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_cfg=dict(type='GELU'),
                 drop=0, ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=False),
            build_activation_layer(act_cfg),
            nn.BatchNorm2d(hidden_features),
        )
        self.dwconv = MultiScaleDWConv(hidden_features)
        self.act = build_activation_layer(act_cfg)
        self.norm = nn.BatchNorm2d(hidden_features)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, in_features, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_features),
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x) + x
        x = self.norm(self.act(x))
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_value=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, 1, 1, 1) * init_value,
                                   requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(dim), requires_grad=True)

    def forward(self, x):
        x = F.conv2d(x, weight=self.weight, bias=self.bias, groups=x.shape[1])
        return x


class Block(nn.Module):
    """
    Network Block with MUDD Connection enhancement and Differential Attention.
    """

    def __init__(self,
                 dim=64,
                 kernel_size=3,
                 sr_ratio=1,
                 num_groups=2,
                 num_heads=1,
                 mlp_ratio=4,
                 norm_cfg=dict(type='GN', num_groups=1),
                 act_cfg=dict(type='GELU'),
                 drop=0,
                 drop_path=0,
                 layer_scale_init_value=1e-5,
                 grad_checkpoint=False,
                 use_mca=True,
                 use_mudd=True,
                 use_differential_attn=False):  # 新增差分注意力参数

        super().__init__()
        self.grad_checkpoint = grad_checkpoint
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.pos_embed = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm1 = build_norm_layer(norm_cfg, dim)[1]
        self.token_mixer = HybridTokenMixer(dim,
                                            kernel_size=kernel_size,
                                            num_groups=num_groups,
                                            num_heads=num_heads,
                                            sr_ratio=sr_ratio,
                                            use_mca=use_mca,
                                            use_differential_attn=use_differential_attn)  # 传入差分注意力参数
        self.norm2 = build_norm_layer(norm_cfg, dim)[1]
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_cfg=act_cfg,
                       drop=drop, )
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        # 改进MUDD连接初始化
        self.use_mudd = use_mudd
        if use_mudd:
            # 确保维度能被4整除
            if dim % 4 != 0:
                # 调整到最接近的可被4整除的维度
                adjusted_dim = (dim // 4) * 4
                if adjusted_dim == 0:
                    adjusted_dim = 4
                print(f"警告: dim {dim} 不能被4整除，已调整为 {adjusted_dim}")
                dim = adjusted_dim
            self.mudd_connection = MUDDConnection(dim, num_paths=4)

        if layer_scale_init_value is not None:
            self.layer_scale_1 = LayerScale(dim, layer_scale_init_value)
            self.layer_scale_2 = LayerScale(dim, layer_scale_init_value)
        else:
            self.layer_scale_1 = nn.Identity()
            self.layer_scale_2 = nn.Identity()

    def _forward_impl(self, x, relative_pos_enc=None, previous_features=None):
        # 应用位置嵌入
        x = x + self.pos_embed(x)

        # 应用MUDD连接（如果启用）
        if self.use_mudd and previous_features is not None:
            # 过滤掉None的特征
            valid_previous_features = [feat for feat in previous_features if feat is not None]
            if valid_previous_features:
                x = self.mudd_connection(x, valid_previous_features)

        # Token Mixer + 残差连接
        x = x + self.drop_path(self.layer_scale_1(
            self.token_mixer(self.norm1(x), relative_pos_enc)))

        # MLP + 残差连接
        x = x + self.drop_path(self.layer_scale_2(self.mlp(self.norm2(x))))

        return x

    def forward(self, x, relative_pos_enc=None, previous_features=None):
        if self.grad_checkpoint and x.requires_grad:
            x = checkpoint.checkpoint(self._forward_impl, x, relative_pos_enc, previous_features)
        else:
            x = self._forward_impl(x, relative_pos_enc, previous_features)
        return x


def basic_blocks(dim,
                 index,
                 layers,
                 kernel_size=3,
                 num_groups=2,
                 num_heads=1,
                 sr_ratio=1,
                 mlp_ratio=4,
                 norm_cfg=dict(type='GN', num_groups=1),
                 act_cfg=dict(type='GELU'),
                 drop_rate=0,
                 drop_path_rate=0,
                 layer_scale_init_value=1e-5,
                 grad_checkpoint=False,
                 use_mca=True,
                 use_mudd=True,
                 use_differential_attn=False):  # 新增差分注意力参数
    blocks = nn.ModuleList()
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (
                block_idx + sum(layers[:index])) / (sum(layers) - 1)

        # 只在每个stage的前半部分块中使用MCA
        block_use_mca = use_mca and (block_idx < layers[index] // 2)

        # 深层网络更多使用MUDD连接
        block_use_mudd = use_mudd and (index >= 1)

        # 深层网络使用差分注意力
        block_use_differential_attn = use_differential_attn and (index >= 2)

        blocks.append(
            Block(
                dim,
                kernel_size=kernel_size,
                num_groups=num_groups,
                num_heads=num_heads,
                sr_ratio=sr_ratio,
                mlp_ratio=mlp_ratio,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                drop=drop_rate,
                drop_path=block_dpr,
                layer_scale_init_value=layer_scale_init_value,
                grad_checkpoint=grad_checkpoint,
                use_mca=block_use_mca,
                use_mudd=block_use_mudd,
                use_differential_attn=block_use_differential_attn,  # 传入差分注意力参数
            ))
    return blocks


class TransXNet(nn.Module):
    """
    Enhanced TransXNet with MUDD Connections, MCA, and Differential Attention
    """

    arch_settings = {
        **dict.fromkeys(['t', 'tiny', 'T'],
                        {'layers': [3, 3, 9, 3],
                         'embed_dims': [48, 96, 224, 448],
                         'kernel_size': [7, 7, 7, 7],
                         'num_groups': [2, 2, 2, 2],
                         'sr_ratio': [8, 4, 2, 1],
                         'num_heads': [1, 2, 4, 8],
                         'mlp_ratios': [4, 4, 4, 4],
                         'layer_scale_init_value': 1e-5, }),

        **dict.fromkeys(['s', 'small', 'S'],
                        {'layers': [4, 4, 12, 4],
                         'embed_dims': [64, 128, 320, 512],
                         'kernel_size': [7, 7, 7, 7],
                         'num_groups': [2, 2, 3, 4],
                         'sr_ratio': [8, 4, 2, 1],
                         'num_heads': [1, 2, 5, 8],
                         'mlp_ratios': [6, 6, 4, 4],
                         'layer_scale_init_value': 1e-5, }),

        **dict.fromkeys(['b', 'base', 'B'],
                        {'layers': [4, 4, 21, 4],
                         'embed_dims': [76, 152, 336, 672],
                         'kernel_size': [7, 7, 7, 7],
                         'num_groups': [2, 2, 4, 4],
                         'sr_ratio': [8, 4, 2, 1],
                         'num_heads': [2, 4, 8, 16],
                         'mlp_ratios': [8, 8, 4, 4],
                         'layer_scale_init_value': 1e-5, }), }

    def __init__(self,
                 image_size=224,
                 arch='tiny',
                 norm_cfg=dict(type='GN', num_groups=1),
                 act_cfg=dict(type='GELU'),
                 in_chans=3,
                 in_patch_size=7,
                 in_stride=4,
                 in_pad=3,
                 down_patch_size=3,
                 down_stride=2,
                 down_pad=1,
                 drop_rate=0,
                 drop_path_rate=0,
                 grad_checkpoint=False,
                 checkpoint_stage=[0] * 4,
                 num_classes=1000,
                 fork_feat=False,
                 start_level=0,
                 use_mca=True,
                 use_mudd=True,
                 use_differential_attn=True,  # 新增差分注意力开关
                 init_cfg=None,
                 pretrained=None,
                 **kwargs):

        super().__init__()

        if not fork_feat:
            self.num_classes = num_classes
        self.fork_feat = fork_feat
        self.grad_checkpoint = grad_checkpoint
        self.use_mudd = use_mudd
        self.use_differential_attn = use_differential_attn

        if isinstance(arch, str):
            assert arch in self.arch_settings, \
                f'Unavailable arch, please choose from ({set(self.arch_settings)}) or pass a dict.'
            arch = self.arch_settings[arch]
        elif isinstance(arch, dict):
            assert 'layers' in arch and 'embed_dims' in arch, \
                f'The arch dict must have "layers" and "embed_dims", but got {list(arch.keys())}.'

        layers = arch['layers']
        embed_dims = arch['embed_dims']
        kernel_size = arch['kernel_size']
        num_groups = arch['num_groups']
        sr_ratio = arch['sr_ratio']
        num_heads = arch['num_heads']

        if not grad_checkpoint:
            checkpoint_stage = [0] * 4

        mlp_ratios = arch['mlp_ratios'] if 'mlp_ratios' in arch else [4, 4, 4, 4]
        layer_scale_init_value = arch['layer_scale_init_value'] if 'layer_scale_init_value' in arch else 1e-5

        self.patch_embed = PatchEmbed(patch_size=in_patch_size,
                                      stride=in_stride,
                                      padding=in_pad,
                                      in_chans=in_chans,
                                      embed_dim=embed_dims[0])

        self.relative_pos_enc = []
        self.pos_enc_record = []
        image_size = to_2tuple(image_size)
        image_size = [math.ceil(image_size[0] / in_stride),
                      math.ceil(image_size[1] / in_stride)]
        for i in range(4):
            num_patches = image_size[0] * image_size[1]
            sr_patches = math.ceil(
                image_size[0] / sr_ratio[i]) * math.ceil(image_size[1] / sr_ratio[i])
            self.relative_pos_enc.append(
                nn.Parameter(torch.zeros(1, num_heads[i], num_patches, sr_patches), requires_grad=True))
            self.pos_enc_record.append([image_size[0], image_size[1],
                                        math.ceil(image_size[0] / sr_ratio[i]),
                                        math.ceil(image_size[1] / sr_ratio[i]), ])
            image_size = [math.ceil(image_size[0] / 2),
                          math.ceil(image_size[1] / 2)]
        self.relative_pos_enc = nn.ParameterList(self.relative_pos_enc)

        # 构建网络
        network = []
        for i in range(len(layers)):
            stage_use_mca = use_mca and (i < 3)
            stage_use_mudd = use_mudd and (i >= 1)
            stage_use_differential_attn = use_differential_attn and (i >= 2)  # 深层使用差分注意力

            stage = basic_blocks(
                embed_dims[i],
                i,
                layers,
                kernel_size=kernel_size[i],
                num_groups=num_groups[i],
                num_heads=num_heads[i],
                sr_ratio=sr_ratio[i],
                mlp_ratio=mlp_ratios[i],
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
                layer_scale_init_value=layer_scale_init_value,
                grad_checkpoint=checkpoint_stage[i],
                use_mca=stage_use_mca,
                use_mudd=stage_use_mudd,
                use_differential_attn=stage_use_differential_attn,  # 传入差分注意力参数
            )
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if embed_dims[i] != embed_dims[i + 1]:
                network.append(
                    PatchEmbed(
                        patch_size=down_patch_size,
                        stride=down_stride,
                        padding=down_pad,
                        in_chans=embed_dims[i],
                        embed_dim=embed_dims[i + 1]))
        self.network = nn.ModuleList(network)

        if self.fork_feat:
            self.out_indices = [0, 2, 4, 6]
            for i_emb, i_layer in enumerate(self.out_indices):
                if i_emb < start_level:
                    layer = nn.Identity()
                else:
                    layer = build_norm_layer(norm_cfg, embed_dims[(i_layer + 1) // 2])[1]
                layer_name = f'norm{i_layer}'
                self.add_module(layer_name, layer)
        else:
            self.classifier = nn.Sequential(
                build_norm_layer(norm_cfg, embed_dims[-1])[1],
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(embed_dims[-1], num_classes, kernel_size=1),
            ) if num_classes > 0 else nn.Identity()

        self.apply(self._init_model_weights)
        self.init_cfg = copy.deepcopy(init_cfg)

        if self.fork_feat and (self.init_cfg is not None or pretrained is not None):
            self.init_weights()
            self = nn.SyncBatchNorm.convert_sync_batchnorm(self)
            self.train()

    def _init_model_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.GroupNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, MultiScaleCoordinateAttention):
            for mod in m.modules():
                if isinstance(mod, nn.Conv2d):
                    nn.init.kaiming_normal_(mod.weight, mode='fan_out')
                elif isinstance(mod, nn.BatchNorm2d):
                    nn.init.ones_(mod.weight)
                    nn.init.zeros_(mod.bias)
        elif isinstance(m, MUDDConnection):
            for mod in m.modules():
                if isinstance(mod, nn.Conv2d):
                    nn.init.kaiming_normal_(mod.weight, mode='fan_out')
                    if mod.bias is not None:
                        nn.init.zeros_(mod.bias)
        elif isinstance(m, DifferentialAttention):
            # 差分注意力的特殊初始化
            nn.init.trunc_normal_(m.qkv.weight, std=0.02)
            nn.init.constant_(m.lambda_param, 0.5)
            if hasattr(m, 'proj'):
                nn.init.trunc_normal_(m.proj.weight, std=0.02)
                if m.proj.bias is not None:
                    nn.init.zeros_(m.proj.bias)

    def init_weights(self, pretrained=None):
        logger = get_root_logger()
        if self.init_cfg is None and pretrained is None:
            logger.warn(f'No pre-trained weights for {self.__class__.__name__}, training start from scratch')
            pass
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support specify `Pretrained` in `init_cfg` in {self.__class__.__name__} '
            if self.init_cfg is not None:
                ckpt_path = self.init_cfg['checkpoint']
            elif pretrained is not None:
                ckpt_path = pretrained

            ckpt = _load_checkpoint(ckpt_path, logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt

            state_dict = _state_dict
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, False)
            print('missing_keys: ', missing_keys)
            print('unexpected_keys: ', unexpected_keys)

    def get_classifier(self):
        return self.classifier

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        if num_classes > 0:
            self.classifier[-1].out_channels = num_classes
        else:
            self.classifier = nn.Identity()

    def forward_embeddings(self, x):
        return self.patch_embed(x)

    def forward_tokens(self, x):
        outs = []
        pos_idx = 0

        # 初始化前面层特征存储（用于MUDD连接）
        previous_features = [] if self.use_mudd else None

        for idx in range(len(self.network)):
            if idx in [0, 2, 4, 6]:  # 新阶段开始
                # 重置特征记录，避免跨阶段通道数不匹配
                if self.use_mudd:
                    previous_features = []

                for blk_idx, blk in enumerate(self.network[idx]):
                    # 准备前面层特征
                    blk_previous_features = None
                    if self.use_mudd and previous_features is not None:
                        blk_previous_features = previous_features.copy()

                    x = blk(x, self.relative_pos_enc[pos_idx], blk_previous_features)

                    # 更新前面层特征存储（只保留当前阶段特征）
                    if self.use_mudd:
                        if previous_features is None:
                            previous_features = []
                        if len(previous_features) >= 4:  # 限制历史长度
                            previous_features.pop(0)
                        previous_features.append(x.clone())

                pos_idx += 1
            else:
                x = self.network[idx](x)

            if self.fork_feat and (idx in self.out_indices):
                x_out = getattr(self, f'norm{idx}')(x)
                outs.append(x_out)

        return outs if self.fork_feat else x

    def forward(self, x):
        x = self.forward_embeddings(x)
        x = self.forward_tokens(x)

        if self.fork_feat:
            return x
        else:
            x = self.classifier(x).flatten(1)
            return x


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000,
        'input_size': (3, 224, 224),
        'crop_pct': 0.95,
        'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN,
        'std': IMAGENET_DEFAULT_STD,
        'classifier': 'classifier',
        **kwargs,
    }


@register_model
def transxnet_t(pretrained=False, pretrained_cfg=None, **kwargs):
    """TransXNet-T with Multi-Scale Coordinate Attention, MUDD Connections and Differential Attention"""
    model = TransXNet(arch='t', use_mca=True, use_mudd=True, use_differential_attn=True, **kwargs)
    model.default_cfg = _cfg(crop_pct=0.9)
    return model


@register_model
def transxnet_s(pretrained=False, pretrained_cfg=None, **kwargs):
    """TransXNet-S with Multi-Scale Coordinate Attention, MUDD Connections and Differential Attention"""
    model = TransXNet(arch='s', use_mca=True, use_mudd=True, use_differential_attn=True, **kwargs)
    model.default_cfg = _cfg(crop_pct=0.9)
    return model


@register_model
def transxnet_b(pretrained=False, pretrained_cfg=None, **kwargs):
    """TransXNet-B with Multi-Scale Coordinate Attention, MUDD Connections and Differential Attention"""
    model = TransXNet(arch='b', use_mca=True, use_mudd=True, use_differential_attn=True, **kwargs)
    model.default_cfg = _cfg(crop_pct=0.95)
    return model
