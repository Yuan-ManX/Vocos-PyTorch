from typing import Optional, Tuple
import torch
from torch import nn
from torch.nn.utils import weight_norm, remove_weight_norm


class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block adapted from https://github.com/facebookresearch/ConvNeXt to 1D audio signal.

    Args:
        dim (int): Number of input channels.
        intermediate_dim (int): Dimensionality of the intermediate layer.
        layer_scale_init_value (float, optional): Initial value for the layer scale. None means no scaling.
            Defaults to None.
        adanorm_num_embeddings (int, optional): Number of embeddings for AdaLayerNorm.
            None means non-conditional LayerNorm. Defaults to None.
    """
    """
    ConvNeXtBlock 类实现了 ConvNeXt 块，该块是针对1D音频信号进行适配的。
    ConvNeXt 块由深度卷积、层归一化、点卷积和激活函数组成。
    该模块支持使用 AdaLayerNorm 进行条件归一化。

    参数说明:
        dim (int): 输入通道数。
        intermediate_dim (int): 中间层的维度。
        layer_scale_init_value (float, 可选): 层缩放的初始值。如果为 None，则不使用缩放。
            默认为 None。
        adanorm_num_embeddings (int, 可选): AdaLayerNorm 的嵌入数量。
            如果为 None，则使用非条件 LayerNorm。默认为 None。
    """

    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        layer_scale_init_value: float,
        adanorm_num_embeddings: Optional[int] = None,
    ):
        super().__init__()
        # 深度卷积层，卷积核大小为7，组数为通道数，实现逐通道卷积
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        # 是否使用 AdaLayerNorm
        self.adanorm = adanorm_num_embeddings is not None
        if adanorm_num_embeddings:
            # 使用 AdaLayerNorm 进行条件归一化
            self.norm = AdaLayerNorm(adanorm_num_embeddings, dim, eps=1e-6)
        else:
            # 使用 LayerNorm 进行归一化
            self.norm = nn.LayerNorm(dim, eps=1e-6)

        # 第一个点卷积层，使用线性层实现
        self.pwconv1 = nn.Linear(dim, intermediate_dim)  # pointwise/1x1 convs, implemented with linear layers
        # 激活函数
        self.act = nn.GELU()
        # 第二个点卷积层，使用线性层实现
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        # 层缩放参数，如果 layer_scale_init_value > 0，则使用可学习的缩放参数
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x: torch.Tensor, cond_embedding_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播方法，执行 ConvNeXt 块的前向计算。

        参数:
            x (Tensor): 输入张量，形状为 (B, C, T)。
            cond_embedding_id (Tensor, 可选): 条件嵌入 ID，用于条件归一化。

        返回:
            Tensor: 输出张量，形状为 (B, C, T)。
        """
        # 保存输入作为残差
        residual = x
        # 通过深度卷积层
        x = self.dwconv(x)

        # 调整张量维度 (B, C, T) -> (B, T, C)
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        if self.adanorm:
            # 确保条件嵌入 ID 不为空
            assert cond_embedding_id is not None
            # 应用 AdaLayerNorm
            x = self.norm(x, cond_embedding_id)
        else:
            # 应用 LayerNorm
            x = self.norm(x)

        # 通过第一个点卷积层
        x = self.pwconv1(x)
        # 应用激活函数
        x = self.act(x)
        # 通过第二个点卷积层
        x = self.pwconv2(x)
        if self.gamma is not None:
            # 应用层缩放
            x = self.gamma * x
        # 调整张量维度 (B, T, C) -> (B, C, T)
        x = x.transpose(1, 2)  

        # 残差连接
        x = residual + x
        return x


class AdaLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization module with learnable embeddings per `num_embeddings` classes

    Args:
        num_embeddings (int): Number of embeddings.
        embedding_dim (int): Dimension of the embeddings.
    """
    """
    AdaLayerNorm 类实现了一个带有可学习嵌入的自适应层归一化模块。
    该模块为每个嵌入类别学习不同的缩放和偏置参数。

    参数说明:
        num_embeddings (int): 嵌入的数量。
        embedding_dim (int): 嵌入的维度。
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, eps: float = 1e-6):
        super().__init__()
        # 数值稳定参数
        self.eps = eps
        # 嵌入维度
        self.dim = embedding_dim
        # 可学习的缩放嵌入
        self.scale = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        # 可学习的偏置嵌入
        self.shift = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        # 初始化缩放嵌入为1，偏置嵌入为0
        torch.nn.init.ones_(self.scale.weight)
        torch.nn.init.zeros_(self.shift.weight)

    def forward(self, x: torch.Tensor, cond_embedding_id: torch.Tensor) -> torch.Tensor:
        """
        前向传播方法，执行自适应层归一化。

        参数:
            x (Tensor): 输入张量。
            cond_embedding_id (Tensor): 条件嵌入 ID。

        返回:
            Tensor: 归一化后的输出张量。
        """
        # 获取缩放因子
        scale = self.scale(cond_embedding_id)
        # 获取偏置
        shift = self.shift(cond_embedding_id)
        # 执行层归一化
        x = nn.functional.layer_norm(x, (self.dim,), eps=self.eps)
        # 应用缩放和偏置
        x = x * scale + shift
        # 返回归一化后的输出
        return x


class ResBlock1(nn.Module):
    """
    ResBlock adapted from HiFi-GAN V1 (https://github.com/jik876/hifi-gan) with dilated 1D convolutions,
    but without upsampling layers.

    Args:
        dim (int): Number of input channels.
        kernel_size (int, optional): Size of the convolutional kernel. Defaults to 3.
        dilation (tuple[int], optional): Dilation factors for the dilated convolutions.
            Defaults to (1, 3, 5).
        lrelu_slope (float, optional): Negative slope of the LeakyReLU activation function.
            Defaults to 0.1.
        layer_scale_init_value (float, optional): Initial value for the layer scale. None means no scaling.
            Defaults to None.
    """
    """
    ResBlock2_vits 类实现了一个残差块（Residual Block），该残差块包含多个膨胀卷积层和跳跃连接。
    该模块基于 HiFi-GAN V1 的设计，但移除了上采样层。
    该模块通过堆叠多个卷积层和激活函数，逐步增加感受野，同时通过残差连接保持信息的流动。

    参数说明:
        dim (int): 输入通道数。
        kernel_size (int, 可选): 卷积核大小，默认为3。
        dilation (Tuple[int, int], 可选): 膨胀卷积的膨胀因子，默认为 (1, 3)。
        lrelu_slope (float, 可选): LeakyReLU 激活函数的负斜率，默认为0.1。
        layer_scale_init_value (float, 可选): 层缩放的初始值。如果为 None，则不使用缩放。默认为 None。
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 3,
        dilation: Tuple[int, int, int] = (1, 3, 5),
        lrelu_slope: float = 0.1,
        layer_scale_init_value: Optional[float] = None,
    ):
        super().__init__()
        # 存储 LeakyReLU 
        self.lrelu_slope = lrelu_slope
        # 定义卷积层列表
        self.convs1 = nn.ModuleList(
            [
                weight_norm(  # 应用权重归一化
                    nn.Conv1d(  # 创建 1D 卷积层
                        dim,  # 输入通道数
                        dim,  # 输出通道数
                        kernel_size,  # 卷积核大小
                        1,  # 步长
                        dilation=dilation[0],  # 膨胀因子
                        padding=self.get_padding(kernel_size, dilation[0]),  # 计算填充大小
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        dim,
                        dim,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=self.get_padding(kernel_size, dilation[1]),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        dim,
                        dim,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        padding=self.get_padding(kernel_size, dilation[2]),
                    )
                ),
            ]
        )

        self.convs2 = nn.ModuleList(
            [
                weight_norm(nn.Conv1d(dim, dim, kernel_size, 1, dilation=1, padding=self.get_padding(kernel_size, 1))),
                weight_norm(nn.Conv1d(dim, dim, kernel_size, 1, dilation=1, padding=self.get_padding(kernel_size, 1))),
                weight_norm(nn.Conv1d(dim, dim, kernel_size, 1, dilation=1, padding=self.get_padding(kernel_size, 1))),
            ]
        )

        # 定义可学习的层缩放参数
        self.gamma = nn.ParameterList(
            [
                nn.Parameter(layer_scale_init_value * torch.ones(dim, 1), requires_grad=True)
                if layer_scale_init_value is not None
                else None,
                nn.Parameter(layer_scale_init_value * torch.ones(dim, 1), requires_grad=True)
                if layer_scale_init_value is not None
                else None,
                nn.Parameter(layer_scale_init_value * torch.ones(dim, 1), requires_grad=True)
                if layer_scale_init_value is not None
                else None,
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播方法，执行残差块的前向计算。

        参数:
            x (Tensor): 输入张量。
            x_mask (Tensor, 可选): 输入掩码张量，用于掩码卷积操作。

        返回:
            Tensor: 输出张量。
        """
        for c1, c2, gamma in zip(self.convs1, self.convs2, self.gamma):
            # 应用 LeakyReLU 激活函数
            xt = torch.nn.functional.leaky_relu(x, negative_slope=self.lrelu_slope)
            # 通过卷积层
            xt = c1(xt)
            xt = torch.nn.functional.leaky_relu(xt, negative_slope=self.lrelu_slope)
            xt = c2(xt)
            if gamma is not None:
                # 应用层缩放
                xt = gamma * xt
            # 残差连接
            x = xt + x
        return x

    def remove_weight_norm(self):
        """
        移除权重归一化。
        """
        for l in self.convs1:
            # 移除卷积层的权重归一化
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)

    @staticmethod
    def get_padding(kernel_size: int, dilation: int = 1) -> int:
        """
        计算填充大小。

        参数:
            kernel_size (int): 卷积核大小。
            dilation (int, 可选): 膨胀因子，默认为1。

        返回:
            int: 计算得到的填充大小。
        """
        return int((kernel_size * dilation - dilation) / 2)


def safe_log(x: torch.Tensor, clip_val: float = 1e-7) -> torch.Tensor:
    """
    Computes the element-wise logarithm of the input tensor with clipping to avoid near-zero values.

    Args:
        x (Tensor): Input tensor.
        clip_val (float, optional): Minimum value to clip the input tensor. Defaults to 1e-7.

    Returns:
        Tensor: Element-wise logarithm of the input tensor with clipping applied.
    """
    """
    计算输入张量的逐元素对数，并进行裁剪以避免接近零的值导致的对数计算错误。
    计算步骤:
        1. 使用 torch.clamp 对输入张量进行裁剪，确保所有值不小于 clip_val。
        2. 对裁剪后的张量应用 torch.log 计算逐元素对数。
    例如:
        输入张量 x = [0.1, 0.01, 0.001]
        clip_val = 1e-7
        裁剪后: [0.1, 0.01, 1e-7]
        对数结果: [-2.3026, -4.6052, -16.1181]

    参数:
        x (Tensor): 输入张量。
        clip_val (float, 可选): 输入张量的最小裁剪值，默认为1e-7。

    返回:
        Tensor: 输入张量的逐元素对数，经过裁剪后的结果。
    """
    return torch.log(torch.clip(x, min=clip_val))


def symlog(x: torch.Tensor) -> torch.Tensor:
    """
    计算输入张量的逐元素对称对数（symlog）。
    计算步骤:
        1. 使用 torch.abs 计算输入张量的绝对值。
        2. 使用 torch.log1p 对绝对值加1后取对数，即 log(1 + |x|)。
        3. 使用 torch.sign 获取输入张量的符号。
        4. 将符号与 log(1 + |x|) 相乘，得到对称对数结果。
    例如:
        输入张量 x = [-2, -1, 0, 1, 2]
        绝对值: [2, 1, 0, 1, 2]
        log1p(绝对值): [1.0986, 0.6931, 0, 0.6931, 1.0986]
        符号: [-1, -1, 0, 1, 1]
        对称对数: [-1.0986, -0.6931, 0, 0.6931, 1.0986]

    参数:
        x (Tensor): 输入张量。

    返回:
        Tensor: 输入张量的逐元素对称对数。
    """
    return torch.sign(x) * torch.log1p(x.abs())


def symexp(x: torch.Tensor) -> torch.Tensor:
    """
    计算输入张量的逐元素对称指数（symexp）。
    计算步骤:
        1. 使用 torch.abs 计算输入张量的绝对值。
        2. 对绝对值应用 torch.exp 计算指数，即 exp(|x|)。
        3. 从指数结果中减去1，即 exp(|x|) - 1。
        4. 使用 torch.sign 获取输入张量的符号。
        5. 将符号与 (exp(|x|) - 1) 相乘，得到对称指数结果。
    例如:
        输入张量 x = [-2, -1, 0, 1, 2]
        绝对值: [2, 1, 0, 1, 2]
        exp(绝对值): [7.3891, 2.7183, 1, 2.7183, 7.3891]
        指数减1: [6.3891, 1.7183, 0, 1.7183, 6.3891]
        符号: [-1, -1, 0, 1, 1]
        对称指数: [-6.3891, -1.7183, 0, 1.7183, 6.3891]

    参数:
        x (Tensor): 输入张量。

    返回:
        Tensor: 输入张量的逐元素对称指数。
    """
    return torch.sign(x) * (torch.exp(x.abs()) - 1)
