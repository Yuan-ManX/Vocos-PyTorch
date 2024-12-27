import numpy as np
import torch
import torchaudio
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass


# 设置 PyTorch 的线程数为1，以避免在数据加载时使用多线程，防止潜在的冲突
torch.set_num_threads(1)


@dataclass
class DataConfig:
    """
    DataConfig 数据类用于配置数据加载相关的参数。

    参数说明:
        filelist_path (str): 数据文件列表的路径。
        sampling_rate (int): 音频的采样率。
        num_samples (int): 每个样本的样本数。
        batch_size (int): 每个批次的样本数。
        num_workers (int): 数据加载时使用的子进程数。
    """
    filelist_path: str
    sampling_rate: int
    num_samples: int
    batch_size: int
    num_workers: int


class VocosDataModule(LightningDataModule):
    """
    VocosDataModule 类继承自 PyTorch Lightning 的 LightningDataModule，用于管理训练和验证数据的加载。
    该类负责配置数据加载参数，创建数据集和数据加载器，并提供训练和验证数据加载的方法。

    参数说明:
        train_params (DataConfig): 训练数据的配置参数。
        val_params (DataConfig): 验证数据的配置参数。
    """
    def __init__(self, train_params: DataConfig, val_params: DataConfig):
        super().__init__()
        self.train_config = train_params
        self.val_config = val_params

    def _get_dataloder(self, cfg: DataConfig, train: bool):
        """
        私有方法，用于创建数据加载器。

        参数:
            cfg (DataConfig): 数据配置参数。
            train (bool): 是否为训练数据加载器。

        返回:
            DataLoader: 创建好的数据加载器。
        """
        # 创建数据集实例
        dataset = VocosDataset(cfg, train=train)
        dataloader = DataLoader(
            dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=train, pin_memory=True,
        )
        # 返回数据加载器
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """
        训练数据加载器方法，返回训练数据加载器。

        返回:
            DataLoader: 训练数据加载器。
        """
        return self._get_dataloder(self.train_config, train=True)

    def val_dataloader(self) -> DataLoader:
        """
        验证数据加载器方法，返回验证数据加载器。

        返回:
            DataLoader: 验证数据加载器。
        """
        return self._get_dataloder(self.val_config, train=False)


class VocosDataset(Dataset):
    """
    VocosDataset 类继承自 torch.utils.data.Dataset，用于加载音频数据。
    该类负责读取音频文件列表，加载音频文件，应用必要的音频处理（如混音、归一化、重采样等），
    并根据训练或验证阶段的不同，处理音频数据以确保样本长度一致。

    参数说明:
        cfg (DataConfig): 数据配置参数。
        train (bool): 是否为训练模式。如果是训练模式，则在加载数据时应用数据增强。
    """
    def __init__(self, cfg: DataConfig, train: bool):
        """
        初始化 VocosDataset 类实例。

        参数:
            cfg (DataConfig): 数据配置参数。
            train (bool): 是否为训练模式。
        """
        with open(cfg.filelist_path) as f:
            # 读取音频文件列表
            self.filelist = f.read().splitlines()
        self.sampling_rate = cfg.sampling_rate
        # 每个样本的样本数
        self.num_samples = cfg.num_samples
        # 是否为训练模式
        self.train = train

    def __len__(self) -> int:
        """
        获取数据集的长度。

        返回:
            int: 数据集的长度，即音频文件数量。
        """
        return len(self.filelist)

    def __getitem__(self, index: int) -> torch.Tensor:
        """
        获取指定索引的音频样本。

        参数:
            index (int): 样本的索引。

        返回:
            torch.Tensor: 预处理后的音频样本，形状为 (1, T)。
        """
        audio_path = self.filelist[index]
        y, sr = torchaudio.load(audio_path)
        if y.size(0) > 1:
            # 如果音频是立体声，则混合到单声道
            y = y.mean(dim=0, keepdim=True)
        # 在训练阶段，随机应用增益以实现数据增强
        # 在验证阶段，应用固定的增益
        gain = np.random.uniform(-1, -6) if self.train else -3
        # 应用 SoX 效果进行归一化
        y, _ = torchaudio.sox_effects.apply_effects_tensor(y, sr, [["norm", f"{gain:.2f}"]])
        if sr != self.sampling_rate:
            # 如果音频采样率与目标采样率不同，则进行重采样
            y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=self.sampling_rate)
        if y.size(-1) < self.num_samples:
            # 如果音频长度小于目标样本数，则进行填充
            pad_length = self.num_samples - y.size(-1)
            # 重复音频以进行填充
            padding_tensor = y.repeat(1, 1 + pad_length // y.size(-1))
            # 拼接填充后的音频
            y = torch.cat((y, padding_tensor[:, :pad_length]), dim=1)
        elif self.train:
            # 在训练阶段，如果音频长度大于目标样本数，则随机截取
            start = np.random.randint(low=0, high=y.size(-1) - self.num_samples + 1)
            y = y[:, start : start + self.num_samples]
        else:
            # 在验证阶段，始终截取前一段样本以保证一致性
            y = y[:, : self.num_samples]

        # 返回单声道的音频样本
        return y[0]
