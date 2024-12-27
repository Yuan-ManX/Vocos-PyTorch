from typing import Optional
import torch
from torch import nn
from torch.nn.utils import weight_norm

from modules import ConvNeXtBlock, ResBlock1, AdaLayerNorm


class Backbone(nn.Module):
    """Base class for the generator's backbone. It preserves the same temporal resolution across all layers."""
    """
    Backbone 抽象基类，是生成器的主干网络基类。
    该基类确保所有子类的所有层在处理时保持相同的时间分辨率。

    方法说明:
        forward (x: torch.Tensor, **kwargs) -> torch.Tensor:
            前向传播方法，所有子类必须实现该方法。

        参数:
            x (Tensor): 输入张量，形状为 (B, C, L)，其中 B 是批量大小，C 表示输出特征，L 是序列长度。

        返回:
            Tensor: 输出张量，形状为 (B, L, H)，其中 B 是批量大小，L 是序列长度，H 表示模型维度。
    """

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        前向传播方法，所有子类必须实现该方法。

        参数:
            x (Tensor): 输入张量，形状为 (B, C, L)。

        返回:
            Tensor: 输出张量，形状为 (B, L, H)。
        """
        raise NotImplementedError("Subclasses must implement the forward method.")


class VocosBackbone(Backbone):
    """
    Vocos backbone module built with ConvNeXt blocks. Supports additional conditioning with Adaptive Layer Normalization

    Args:
        input_channels (int): Number of input features channels.
        dim (int): Hidden dimension of the model.
        intermediate_dim (int): Intermediate dimension used in ConvNeXtBlock.
        num_layers (int): Number of ConvNeXtBlock layers.
        layer_scale_init_value (float, optional): Initial value for layer scaling. Defaults to `1 / num_layers`.
        adanorm_num_embeddings (int, optional): Number of embeddings for AdaLayerNorm.
                                                None means non-conditional model. Defaults to None.
    """
    """
    VocosBackbone 类是 Vocos 模型的主干网络模块，基于 ConvNeXt 块构建。
    该模块支持通过自适应层归一化（AdaLayerNorm）进行额外的条件输入。

    参数说明:
        input_channels (int): 输入特征通道数。
        dim (int): 模型的隐藏维度。
        intermediate_dim (int): ConvNeXt 块中的中间维度。
        num_layers (int): ConvNeXt 块的数量。
        layer_scale_init_value (float, 可选): 层缩放的初始值。默认为 `1 / num_layers`。
        adanorm_num_embeddings (int, 可选): AdaLayerNorm 的嵌入数量。如果为 None，则表示非条件模型。默认为 None。
    """

    def __init__(
        self,
        input_channels: int,
        dim: int,
        intermediate_dim: int,
        num_layers: int,
        layer_scale_init_value: Optional[float] = None,
        adanorm_num_embeddings: Optional[int] = None,
    ):
        super().__init__()
        # 输入特征通道数
        self.input_channels = input_channels
        # 初始卷积层
        self.embed = nn.Conv1d(input_channels, dim, kernel_size=7, padding=3)
        # 是否使用 AdaLayerNorm
        self.adanorm = adanorm_num_embeddings is not None
        if adanorm_num_embeddings:
            # 使用 AdaLayerNorm
            self.norm = AdaLayerNorm(adanorm_num_embeddings, dim, eps=1e-6)
        else:
            # 使用 LayerNorm
            self.norm = nn.LayerNorm(dim, eps=1e-6)

        # 设置层缩放的初始值
        layer_scale_init_value = layer_scale_init_value or 1 / num_layers
        self.convnext = nn.ModuleList(
            [ 
                ConvNeXtBlock(  # 创建 ConvNeXt 块列表
                    dim=dim,  # 隐藏维度
                    intermediate_dim=intermediate_dim,  # 中间维度
                    layer_scale_init_value=layer_scale_init_value,  # 层缩放初始值
                    adanorm_num_embeddings=adanorm_num_embeddings,  # AdaLayerNorm 嵌入数量
                )
                for _ in range(num_layers)
            ]
        )
        # 最后的 LayerNorm 层
        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)
        # 应用权重初始化
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        初始化模型权重。

        参数:
            m (nn.Module): 要初始化的模型或层。
        """
        # 如果是 Conv1d 或 Linear 层
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            # 使用截断正态分布初始化权重
            nn.init.trunc_normal_(m.weight, std=0.02)
            # 将偏置初始化为0
            nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        前向传播方法，执行主干网络的前向计算。

        参数:
            x (Tensor): 输入张量，形状为 (B, C, L)。
            **kwargs: 其他关键字参数。

        返回:
            Tensor: 输出张量，形状为 (B, L, H)。
        """
        # 获取带宽 ID
        bandwidth_id = kwargs.get('bandwidth_id', None)
        # 通过初始卷积层
        x = self.embed(x)
        if self.adanorm:
            # 确保带宽 ID 不为空
            assert bandwidth_id is not None
            # 应用 AdaLayerNorm
            x = self.norm(x.transpose(1, 2), cond_embedding_id=bandwidth_id)
        else:
            # 应用 LayerNorm
            x = self.norm(x.transpose(1, 2))
        # 调整张量维度
        x = x.transpose(1, 2)
        for conv_block in self.convnext:
            # 通过 ConvNeXt 块
            x = conv_block(x, cond_embedding_id=bandwidth_id)
        # 通过最后的 LayerNorm
        x = self.final_layer_norm(x.transpose(1, 2))
        return x


class VocosResNetBackbone(Backbone):
    """
    Vocos backbone module built with ResBlocks.

    Args:
        input_channels (int): Number of input features channels.
        dim (int): Hidden dimension of the model.
        num_blocks (int): Number of ResBlock1 blocks.
        layer_scale_init_value (float, optional): Initial value for layer scaling. Defaults to None.
    """
    """
    VocosResNetBackbone 类是 Vocos 模型的主干网络模块，基于 ResNet 残差块构建。
    该模块通过多个残差块逐步处理输入特征图，提取高层次的特征。

    参数说明:
        input_channels (int): 输入特征通道数。
        dim (int): 模型的隐藏维度。
        num_blocks (int): ResBlock1 残差块的数量。
        layer_scale_init_value (float, 可选): 层缩放的初始值。默认为 None。
    """

    def __init__(
        self, input_channels, dim, num_blocks, layer_scale_init_value=None,
    ):
        super().__init__()
        # 输入特征通道数
        self.input_channels = input_channels
        # 初始卷积层，使用权重归一化
        self.embed = weight_norm(nn.Conv1d(input_channels, dim, kernel_size=3, padding=1))
        # 设置层缩放的初始值，默认为 1 / (num_blocks * 3)
        layer_scale_init_value = layer_scale_init_value or 1 / num_blocks / 3
        # 构建残差块序列
        self.resnet = nn.Sequential(
            *[ResBlock1(dim=dim, layer_scale_init_value=layer_scale_init_value) for _ in range(num_blocks)]
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        前向传播方法，执行主干网络的前向计算。

        参数:
            x (Tensor): 输入张量，形状为 (B, C, L)。
            **kwargs: 其他关键字参数。

        返回:
            Tensor: 输出张量，形状为 (B, L, H)。
        """
        # 通过初始卷积层
        x = self.embed(x)
        # 通过残差块序列
        x = self.resnet(x)
        # 调整张量维度
        x = x.transpose(1, 2)
        return x
