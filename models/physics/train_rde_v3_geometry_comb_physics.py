"""带“四点几何约束”和“梳状频谱约束”的旋转多普勒效应（RDE）转速网络。

这份程序要解决的问题是：只观察光电探测器信号的功率谱（PSD），估计旋转物体的
转速，并同时估计会影响频谱的安装几何参数。

可以把完整模型理解为下面五步：

1. ``spectral_features`` 先把时域电压做 FFT，得到 20~1600 Hz 的功率谱；取对数并
   标准化后，每个样本成为一条一维“频谱曲线”。
2. 一维卷积神经网络在频谱上寻找局部峰、峰间距和宽度等模式，把整条曲线压缩为
   1024 个特征数。
3. ``physical_head`` 从这些特征预测转频 ``f_rot`` 以及几何量 ``gamma、d、phi``。
   网络不直接任意预测四个观测频率，而是把上述物理量代入四点解析公式计算它们。
4. ``scene_head`` 表示粗糙表面、接收器响应等未测量因素；它与几何量一起决定各阶
   谐波的强弱，再由 ``comb_forward`` 重建一条可微分的梳状功率谱。
5. 训练时同时比较转速、四点频率、几何量和重建频谱。这样网络不只要“猜对答案”，
   还要让答案能够通过物理模型解释输入频谱。

文中常见的张量形状记号：B 表示一个批次中的样本数，N 表示输入频点数，F 表示
下采样后用于重建的频点数。本文件只增加解释，不改变原有计算流程。
"""

# 让类型注解延迟求值，避免某些 Python 版本在解释类型名时产生兼容性问题。
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

# NumPy/Pandas 负责训练前后的数组和表格处理；PyTorch 负责可求导的网络计算。
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

# 复用基线程序中的数据集划分名、随机种子设置和 FFT 特征提取函数。
from train_rde_v3_spectral_cnn import SPLITS, set_seed, spectral_features


# 梳状谱最多重建 1~64 阶谐波。第 m 阶谱峰的中心位于 m*f_rot。
MAX_ORDER = 64
# 原 PSD 的频率分辨率为 1 Hz；每隔 4 点取一个点，使重建网格间隔为 4 Hz，降低计算量。
PSD_STRIDE = 4
# 涡旋光拓扑荷的绝对值 |l|=10；明环半径为 5 mm。
L_ABS = 10.0
RING_RADIUS_MM = 5.0
# 标签表中四个方位角 theta=0、90、180、270 度所对应的局部调制频率列。
FREQUENCY_COLUMNS = ("f_theta0_hz", "f_theta90_hz", "f_theta180_hz", "f_theta270_hz")


class GeometryCombCNN(nn.Module):
    """从一维 PSD 形状中提取特征，并分头预测物理量与不可观测场景因素。

    输入 ``shape`` 的形状为 [B, N]。卷积编码后得到 [B, 64, 16]，展平后为
    [B, 1024]。随后有两个输出头：

    * 物理头输出 5 个原始数，经 ``physical_four_points`` 变成
      f_rot、gamma、d、sin(phi)、cos(phi)；
    * 场景头输出 12 个没有人工标签的隐变量，用于概括表面纹理和接收器响应。

    “头”只是共享特征后承担不同任务的小型全连接网络。多任务共享前面的卷积特征，
    能让同一组谱峰模式同时接受转速、几何和频谱重建三方面的约束。
    """

    def __init__(self) -> None:
        super().__init__()
        # 一维卷积可看成让一个小窗口沿频率轴滑动。对输入 x，卷积输出近似为
        # y[i] = sum_j w[j] * x[i+j] + b，因此可学习“某段频率附近是否有谱峰”这类局部模式。
        # stride=2 令频率轴长度每层约减半；padding 保持边界信息。通道数 1→24→48→64，
        # 表示网络逐层学出越来越多种特征，例如尖峰、宽峰、谐波间隔和背景趋势。
        self.convolution = nn.Sequential(
            # BatchNorm 对每个通道做 (x-均值)/标准差，再学习缩放和平移，可稳定训练。
            # GELU 是平滑非线性激活，近似 x*Phi(x)；没有非线性，多层线性运算仍等价于一层。
            nn.Conv1d(1, 24, kernel_size=15, stride=2, padding=7), nn.BatchNorm1d(24), nn.GELU(),
            nn.Conv1d(24, 48, kernel_size=9, stride=2, padding=4), nn.BatchNorm1d(48), nn.GELU(),
            nn.Conv1d(48, 64, kernel_size=7, stride=2, padding=3), nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=5, stride=2, padding=2), nn.BatchNorm1d(64), nn.GELU(),
            # 无论原频谱有多少点，都把每个通道分区平均成 16 点，输出固定为 [B,64,16]。
            nn.AdaptiveAvgPool1d(16),
        )
        # 64*16=1024。Linear 实现 y=Wx+b，把全局频谱特征映射到 5 个物理原始量。
        # Dropout(0.20) 在训练时随机把 20% 中间特征置零，减少网络死记训练样本的风险；
        # model.eval() 后 Dropout 自动关闭，因此推理结果是确定的。
        self.physical_head = nn.Sequential(nn.Linear(1024, 128), nn.GELU(), nn.Dropout(0.20), nn.Linear(128, 5))
        # 12 维场景码是网络内部的“干扰因素摘要”，不是需要对外报告或逐维解释的物理量。
        self.scene_head = nn.Sequential(nn.Linear(1024, 64), nn.GELU(), nn.Linear(64, 12))
        # 4 维几何量与 12 维场景码拼成 16 维，再预测 3 个背景参数和 64 个谐波幅值，
        # 所以最终维数为 MAX_ORDER+3=67。
        self.comb_decoder = nn.Sequential(
            nn.Linear(16, 96), nn.GELU(), nn.Linear(96, MAX_ORDER + 3)
        )

    def forward(self, shape: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """完成共享卷积编码，返回物理头原始输出和场景码。

        ``shape[:, None]`` 在 [B,N] 中插入“通道”维，变成 Conv1d 要求的 [B,1,N]；
        ``flatten(1)`` 保留批次维，把 [64,16] 摊平成 1024 维向量。
        """
        encoded = self.convolution(shape[:, None]).flatten(1)
        return self.physical_head(encoded), self.scene_head(encoded)

    def decode_comb(self, geometry_normalized: torch.Tensor, scene_code: torch.Tensor) -> torch.Tensor:
        """根据几何量和场景码，产生重建梳状谱所需的参数。

        显式输入几何量，是为了让谱包络随安装姿态变化；场景码则补充未被标签记录的
        粗糙表面及接收器差异。``dim=1`` 表示沿特征维拼接，不混合不同样本。
        """
        return self.comb_decoder(torch.cat([geometry_normalized, scene_code], dim=1))


def physical_four_points(raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """把网络的 5 个无约束实数变成合法物理量，并计算四点局部频率。

    输入 ``raw`` 为 [B,5]。输出依次为：转速 [B]（RPM）、四点频率 [B,4]（Hz）和
    归一化几何量 [B,4]。这里使用解析物理公式而不是再训练一个黑箱网络，因此只要
    ``f_rot、gamma、d、phi`` 合理，四个频率就必然满足模型中的几何关系。
    """
    # softplus(x)=ln(1+exp(x)) 恒大于 0，确保转频不会为负。系数 30 只是选择合适的
    # 数值尺度，不是 30 Hz 的上限；softplus 没有上界。
    f_rot = 30.0 * F.softplus(raw[:, 0])
    # sigmoid(x)=1/(1+exp(-x)) 的范围为 (0,1)，故 gamma 被限制到 (0,75) 度，
    # 偏心距 d 被限制到 (0,12) mm，避免网络给出明显不合理的几何参数。
    gamma_deg = 75.0 * torch.sigmoid(raw[:, 1])
    d_mm = 12.0 * torch.sigmoid(raw[:, 2])
    # 角度 phi 在 0°/360° 处有跳变，例如 359° 和 1° 数值相差很大但方向很近。
    # 因此预测二维方向向量并除以其长度，使 sin²(phi)+cos²(phi)=1；以后用 atan2 还原角度。
    direction = F.normalize(raw[:, 3:5], dim=1, eps=1e-6)
    sin_phi, cos_phi = direction[:, 0], direction[:, 1]
    # 三角函数要求弧度，所以先把度转弧度。最低截到 1e-3，防止后面 1/cos(gamma)
    # 在 gamma 接近 90° 时除以零并造成数值爆炸。
    cos_gamma = torch.cos(torch.deg2rad(gamma_deg)).clamp(min=1e-3)
    # q=d/r 是无量纲的相对偏心量。
    q = d_mm / RING_RADIUS_MM
    # 对拓扑荷 +l 与 -l 的叠加光束，Omega=2*pi*f_rot，因此理想调制频率
    # f_mod=l*Omega/pi=2*l*f_rot。这里 l 取绝对值 10。
    f_mod = 2.0 * L_ABS * f_rot
    # 将通用四点几何方程分别代入 theta=0°、90°、180°、270°。
    # 倾斜 gamma 通过 cos(gamma) 或其倒数改变频率；偏心 d 通过 q 与
    # sin(phi)/cos(phi) 改变相对方向上的频率。对置的点符号相反，因而包含可辨识的几何信息。
    f0 = f_mod * cos_gamma * (1.0 - q * sin_phi)
    f90 = f_mod * (1.0 / cos_gamma + q * cos_phi)
    f180 = f_mod * cos_gamma * (1.0 + q * sin_phi)
    f270 = f_mod * (1.0 / cos_gamma - q * cos_phi)
    # stack 把四个 [B] 向量排成 [B,4]，顺序与 FREQUENCY_COLUMNS 完全一致。
    local_frequency = torch.stack([f0, f90, f180, f270], dim=1)
    # 缩放到大致相同的数值范围，避免角度（几十）或毫米值在损失中天然占更大权重。
    geometry_normalized = torch.stack([gamma_deg / 75.0, d_mm / 12.0, sin_phi, cos_phi], dim=1)
    # 1 Hz = 1 转/秒 = 60 转/分，所以将 f_rot 乘 60 得到 RPM。
    return 60.0 * f_rot, local_frequency, geometry_normalized


def comb_forward(
    f_rot: torch.Tensor, comb_raw: torch.Tensor, frequency_hz: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """用可微分的高斯谐波梳，重建归一化对数功率谱。

    若物体以 ``f_rot`` Hz 旋转，表面第 m 阶角向结构会在 ``m*f_rot`` Hz 附近产生
    谱峰。因此用 64 个以整数倍转频为中心的高斯峰相加，再加平滑背景。所有运算都是
    PyTorch 张量运算，损失对输出的导数可以沿这套公式反向传回 CNN。

    输入形状：``f_rot``=[B]，``comb_raw``=[B,67]，``frequency_hz``=[F]；
    输出形状：重建谱=[B,F]，64 阶非负幅值=[B,64]。
    """
    # softplus 保证线宽、背景功率和谐波功率均为正。线宽加 1 Hz 是设置物理上合理
    # 的下限；背景加 1e-5 则避免功率恰为零后无法取对数。
    linewidth = 1.0 + 8.0 * F.softplus(comb_raw[:, 0])
    background_level = F.softplus(comb_raw[:, 1]) + 1e-5
    # 背景随频率变化的斜率可正可负，但限制到 [-4,4]，防止指数函数溢出。
    background_slope = comb_raw[:, 2].clamp(-4.0, 4.0)
    amplitude = F.softplus(comb_raw[:, 3:])
    # 广播计算所需的形状：orders=[1,64,1]，frequency=[1,1,F]。
    # PyTorch 会自动扩展批次、阶数和频率维，无需写三重循环。
    orders = torch.arange(1, MAX_ORDER + 1, dtype=f_rot.dtype, device=f_rot.device)[None, :, None]
    frequency = frequency_hz[None, None, :]
    # 第 m 阶峰中心 c_m=m*f_rot，最终 centres 的形状为 [B,64,1]。
    centres = f_rot[:, None, None] * orders
    # 高阶峰容易受转速波动等因素影响而稍宽，这里让宽度随阶数线性增加 1.2%/阶。
    widths = linewidth[:, None, None] * (1.0 + 0.012 * orders)
    # 单个峰使用高斯函数：A_m*exp[-0.5*((f-c_m)/sigma_m)^2]。
    # 它在中心处等于 A_m，离中心越远越接近 0；对 64 阶求和得到总谐波功率。
    harmonic_power = torch.sum(amplitude[:, :, None] * torch.exp(-0.5 * ((frequency - centres) / widths).square()), dim=1)
    # 把频率线性映射到约 [-0.5,0.5]。指数背景 B(f)=B0*exp(s*f_norm) 始终为正，
    # slope>0 表示背景随频率上升，slope<0 表示下降。
    normalized_frequency = (frequency_hz - frequency_hz.mean()) / (frequency_hz.max() - frequency_hz.min())
    background = background_level[:, None] * torch.exp(background_slope[:, None] * normalized_frequency[None])
    # 功率谱动态范围往往很大，取自然对数能压缩强峰，使弱峰也参与训练。
    log_psd = torch.log(harmonic_power + background + 1e-8)
    # 对每个样本做 z-score：(x-均值)/标准差。这里比较的是“频谱形状”而非绝对功率，
    # 与 spectral_features 生成输入时的标准化方式一致；1e-6 防止平坦谱的标准差为零。
    shape = (log_psd - log_psd.mean(dim=1, keepdim=True)) / (log_psd.std(dim=1, keepdim=True) + 1e-6)
    return shape, amplitude


def metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    """计算四种常见转速误差指标。

    MAE 是平均绝对误差；MAPE 是相对真值的平均百分比误差；RMSE 对大误差惩罚更重；
    bias 是有符号平均误差，正值表示整体高估，负值表示整体低估。
    """
    error = prediction - actual
    return {
        "mae_rpm": float(np.abs(error).mean()),
        "mape_pct": float((np.abs(error) / actual).mean() * 100.0),
        "rmse_rpm": float(np.sqrt(np.mean(error**2))),
        "bias_rpm": float(error.mean()),
    }


def predict(model: GeometryCombCNN, shape: np.ndarray, frequency_hz: torch.Tensor, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """分批推理并返回转速、四点频率、几何量和重建 PSD。

    分成每批最多 256 条可控制显存占用；这里不训练，所以关闭 Dropout/固定 BatchNorm，
    并用 ``no_grad`` 停止保存反向传播所需的中间量，以节省内存和时间。
    """
    model.eval()
    rpms, frequencies, geometries, psds = [], [], [], []
    with torch.no_grad():
        for start in range(0, len(shape), 256):
            # 完整推理路径与训练路径相同：CNN编码 → 合法物理量 → 场景解码 → 频谱重建。
            raw, scene = model(torch.from_numpy(shape[start:start + 256]).to(device))
            rpm, local_frequency, geometry = physical_four_points(raw)
            comb_raw = model.decode_comb(geometry, scene)
            psd, _ = comb_forward(rpm / 60.0, comb_raw, frequency_hz)
            # GPU 张量先移回 CPU 才能转成 NumPy；最后沿样本维拼回完整数据集。
            rpms.append(rpm.cpu().numpy())
            frequencies.append(local_frequency.cpu().numpy())
            geometries.append(geometry.cpu().numpy())
            psds.append(psd.cpu().numpy())
    return np.concatenate(rpms), np.concatenate(frequencies), np.concatenate(geometries), np.concatenate(psds)


def main() -> None:
    """组织数据、训练、早停、测试和结果保存。"""
    parent = Path(__file__).resolve().parent
    # 命令行参数允许不改源码就替换数据目录、输出目录、训练轮数和训练转速区间。
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=parent / "data" / "rde_v3_stratified_18000")
    parser.add_argument("--output-dir", type=Path, default=parent / "results" / "rde_v3_geometry_comb_physics")
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--train-rpm-min", type=float, default=700.0)
    parser.add_argument("--train-rpm-max", type=float, default=2400.0)
    args = parser.parse_args()
    # 固定各随机数生成器，使重复实验尽量可复现；若有 CUDA 就用 GPU，否则使用 CPU。
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # spectral_features 对每个数据划分读取时域波形，执行加窗 FFT、求功率、截取
    # 20~1600 Hz、取对数并逐样本标准化。返回的 shape 就是网络看到的唯一输入。
    raw = {split: spectral_features(args.data_root, split, 20.0, 1_600.0) for split in SPLITS}

    def partition(source: str, name: str, lower: float | None, upper: float | None) -> tuple[str, tuple[np.ndarray, pd.DataFrame]]:
        """按转速上下界筛选一个数据子集，并同时筛选频谱和标签表。"""
        # spectral_features 还会返回若干人工汇总特征，此处用下划线丢弃它们，确保模型
        # 真正只使用 PSD 曲线，不会通过额外的标量特征间接读到答案。
        shape, _, frame = raw[source]
        # 初始时全部选择；再用逻辑与 &= 逐项叠加下界和上界条件。
        mask = np.ones(len(frame), dtype=bool)
        if lower is not None:
            mask &= frame["rpm"].to_numpy() >= lower
        if upper is not None:
            mask &= frame["rpm"].to_numpy() <= upper
        if not mask.any():
            raise ValueError(f"No samples in partition {name}")
        return name, (shape[mask], frame.loc[mask].reset_index(drop=True))

    # core 是训练转速范围内的数据；只有 train_core 会更新网络参数，val_core 只负责选模型。
    datasets = dict(
        [
            partition("train", "train_core", args.train_rpm_min, args.train_rpm_max),
            partition("val", "val_core", args.train_rpm_min, args.train_rpm_max),
        ]
    )
    for source in ("test_id", "test_geometry", "test_hard"):
        # 三类测试集各拆成：范围内 core、低于训练范围 ood_low、高于训练范围 ood_high。
        # OOD（out of distribution）用于检查网络向未见转速外推时是否仍遵守物理规律。
        datasets.update(
            [
                partition(source, f"{source}_core", args.train_rpm_min, args.train_rpm_max),
                partition(source, f"{source}_ood_low", None, args.train_rpm_min - 1e-6),
                partition(source, f"{source}_ood_high", args.train_rpm_max + 1e-6, None),
            ]
        )
    train_shape, train_frame = datasets["train_core"]
    # 从标签表构造四组监督目标。astype(float32) 与网络默认精度一致，可减少显存和转换。
    train_rpm = train_frame["rpm"].to_numpy(dtype=np.float32)
    train_frequency = train_frame.loc[:, FREQUENCY_COLUMNS].to_numpy(dtype=np.float32)
    # NumPy 三角函数使用弧度。phi 仍采用 (sin phi, cos phi)，避免角度周期边界不连续。
    train_phi = np.deg2rad(train_frame["phi_deg"].to_numpy(dtype=np.float32))
    train_geometry = np.column_stack(
        [train_frame["gamma_deg"] / 75.0, train_frame["d_mm"] / 12.0, np.sin(train_phi), np.cos(train_phi)]
    ).astype(np.float32)
    # 四个方位的频率尺度不同，分别做 z-score。只用训练集统计量，避免测试信息泄漏。
    frequency_mean = train_frequency.mean(axis=0)
    frequency_std = np.maximum(train_frequency.std(axis=0), 1e-5)
    # 对应 20,24,...,1600 Hz，共 396 个重建点，与 train_shape[:, ::4] 一一对应。
    frequency_hz = torch.arange(20.0, 1600.0 + 1e-4, float(PSD_STRIDE), device=device)
    model = GeometryCombCNN().to(device)
    # TensorDataset 把同一样本的输入和四种目标绑在一起；DataLoader 每次随机抽 72 条。
    # shuffle=True 每轮重排样本，避免参数更新长期受数据原始排列影响。
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_shape), torch.from_numpy(train_rpm),
            # 频谱重建只需每 4 Hz 比较一次，显著减少 64 个高斯峰在频率轴上的计算量。
            torch.from_numpy(train_shape[:, ::PSD_STRIDE]),
            torch.from_numpy((train_frequency - frequency_mean) / frequency_std),
            torch.from_numpy(train_geometry),
        ),
        batch_size=72,
        shuffle=True,
    )
    # AdamW 为每个参数自适应调整步长；lr 是初始学习率，weight_decay 轻微抑制过大权重，
    # 相当于正则化。余弦退火把学习率按半个余弦曲线逐渐降到接近 0，后期更新更细致。
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=8e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    # 损失计算也在 GPU 上进行，因此预先把标准化常数变成同设备的张量。
    frequency_mean_tensor = torch.from_numpy(frequency_mean).to(device)
    frequency_std_tensor = torch.from_numpy(frequency_std).to(device)
    best_state: dict[str, torch.Tensor] | None = None
    # 保存验证 MAE 最小时的参数。patience 统计连续多少轮没有进步，用于提前停止。
    best_mae, patience = float("inf"), 0
    for epoch in range(args.epochs):
        # train() 打开 Dropout，并让 BatchNorm 使用当前批次统计量。
        model.train()
        for shape, rpm_target, psd_target, frequency_target, geometry_target in loader:
            # 给已标准化的 PSD 加很小的高斯噪声 N(0,0.012²)，相当于数据增强：要求网络
            # 对轻微测量扰动保持稳定。randn_like 生成同形状、均值 0、标准差 1 的噪声。
            shape = shape.to(device) + 0.012 * torch.randn_like(shape.to(device))
            # 正向传播的四步：抽取特征；转为物理量；解码谱参数；按转频整数倍重建 PSD。
            raw_output, scene_code = model(shape)
            rpm_prediction, local_frequency, geometry = physical_four_points(raw_output)
            comb_raw = model.decode_comb(geometry, scene_code)
            reconstructed_psd, amplitude = comb_forward(rpm_prediction / 60.0, comb_raw, frequency_hz)
            # Smooth L1（Huber）损失：小误差时约为 0.5*e²，平滑且鼓励精确拟合；
            # 大误差时约为 |e|-0.5，不像纯平方误差那样被少数异常值完全主导。
            # RPM 除以 1000 是数值缩放，否则数百 RPM 的误差会压过其他损失。
            rpm_loss = F.smooth_l1_loss(rpm_prediction / 1000.0, rpm_target.to(device) / 1000.0)
            # 重建损失要求预测参数能够“解释”观测频谱，而不只是碰巧输出正确转速。
            psd_loss = F.smooth_l1_loss(reconstructed_psd, psd_target.to(device))
            # 四点频率先按训练集均值/标准差归一化，使四个方向以相近尺度参与损失。
            frequency_loss = F.smooth_l1_loss(
                (local_frequency - frequency_mean_tensor) / frequency_std_tensor, frequency_target.to(device)
            )
            # 几何目标已是 [gamma/75,d/12,sin(phi),cos(phi)]，因此可直接比较。
            geometry_loss = F.smooth_l1_loss(geometry, geometry_target.to(device))
            # 总损失是多目标加权和。转速是主目标，频谱/四点频率/几何是较弱辅助约束。
            # amplitude.mean() 是 L1 风格的稀疏惩罚：倾向于只启用必要的谐波，避免用许多
            # 虚假小峰拼凑输入。所有权重是超参数，决定各目标对梯度的相对贡献。
            loss = rpm_loss + 0.25 * psd_loss + 0.08 * frequency_loss + 0.05 * geometry_loss + 0.0005 * amplitude.mean()
            # PyTorch 默认累加梯度，所以先清零；backward 用链式法则计算 d(loss)/d(parameter)。
            # 梯度范数裁剪到 3，防止偶发的大梯度让参数一步跳得过远；step 执行 AdamW 更新。
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
        # 每个 epoch（完整看过一次训练集）结束后降低学习率。
        scheduler.step()
        # 验证集不参与梯度更新，只用于衡量当前模型对未用于更新的数据的泛化能力。
        val_shape, val_frame = datasets["val_core"]
        val_prediction, _, _, _ = predict(model, val_shape, frequency_hz, device)
        val_mae = metrics(val_frame["rpm"].to_numpy(dtype=np.float32), val_prediction)["mae_rpm"]
        if val_mae < best_mae:
            # state_dict 存放所有可学习参数和 BatchNorm 状态；deepcopy 保留当时的独立快照。
            best_mae, patience = val_mae, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience += 1
            if patience >= 14:
                # 连续 14 轮验证 MAE 未改善就早停，避免继续过拟合并节省时间。
                break
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}: validation MAE {val_mae:.2f} RPM")
    # 恢复的不是“最后一轮”，而是整个过程中验证 MAE 最低的那一轮。
    assert best_state is not None
    model.load_state_dict(best_state)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # 保存参数而非整个 Python 对象，后续加载时需重新构造 GeometryCombCNN 再 load_state_dict。
    torch.save({"model_state": model.state_dict()}, args.output_dir / "geometry_comb_physics_best.pt")
    # report 记录实验协议和各子集指标。特别说明：几何/四点标签来自仿真监督，真实实验中
    # 若没有这些标签，需要重新考虑监督方式和参数可辨识性。
    report: dict[str, object] = {
        "protocol": "PSD-only input. Public physical head outputs f_rot, gamma, d, sin(phi), cos(phi). Equation (10) generates four local frequencies; a geometry-conditioned private scene code generates the FFT harmonic envelope. Geometry/four-point labels are synthetic training supervision only.",
        "speed_protocol": {"train_core_rpm": [args.train_rpm_min, args.train_rpm_max], "low_speed_ood_rpm": [400.0, args.train_rpm_min], "high_speed_ood_rpm": [args.train_rpm_max, 3000.0], "checkpoint_selection": "val_core only"},
        "results": {},
    }
    saved = []
    for split, (shape, frame) in datasets.items():
        # 统一评估全部 core 与 OOD 子集，既测范围内精度，也测跨转速/几何/噪声泛化。
        prediction, frequency_prediction, geometry_prediction, psd_prediction = predict(model, shape, frequency_hz, device)
        rpm = frame["rpm"].to_numpy(dtype=np.float32)
        frequency = frame.loc[:, FREQUENCY_COLUMNS].to_numpy(dtype=np.float32)
        phi = np.deg2rad(frame["phi_deg"].to_numpy(dtype=np.float32))
        target_geometry = np.column_stack([frame["gamma_deg"] / 75.0, frame["d_mm"] / 12.0, np.sin(phi), np.cos(phi)])
        report["results"][split] = {
            # **metrics(...) 把 MAE、MAPE、RMSE 和 bias 四个键值展开进当前字典。
            "samples": int(len(rpm)), **metrics(rpm, prediction),
            # 除转速外，再报告物理中间量与频谱重建误差，便于发现“转速对但解释不对”的模型。
            "psd_shape_mae": float(np.abs(psd_prediction - shape[:, ::PSD_STRIDE]).mean()),
            "frequency_mae_hz": float(np.abs(frequency_prediction - frequency).mean()),
            "gamma_mae_deg": float(np.abs(geometry_prediction[:, 0] * 75.0 - target_geometry[:, 0] * 75.0).mean()),
            "d_mae_mm": float(np.abs(geometry_prediction[:, 1] * 12.0 - target_geometry[:, 1] * 12.0).mean()),
        }
        # 逐样本结果便于画散点图、误差分布图或定位极端失败样本。
        rows = frame[["rpm", "severity", "gamma_deg", "phi_deg", "d_mm", "record_index"]].copy()
        rows["split"] = split
        rows["prediction_rpm"] = prediction
        rows["absolute_error_rpm"] = np.abs(prediction - rpm)
        rows["predicted_f_rot_hz"] = prediction / 60.0
        rows["predicted_gamma_deg"] = geometry_prediction[:, 0] * 75.0
        rows["predicted_d_mm"] = geometry_prediction[:, 1] * 12.0
        # atan2(sin(phi),cos(phi)) 会根据象限正确还原角度，再由 rad2deg 转回度。
        rows["predicted_phi_deg"] = np.rad2deg(np.arctan2(geometry_prediction[:, 2], geometry_prediction[:, 3]))
        saved.append(rows)
    # CSV 保存每个样本的预测；JSON 保存汇总指标。UTF-8 可正确写入未来可能加入的中文字段。
    pd.concat(saved, ignore_index=True).to_csv(args.output_dir / "per_sample_predictions.csv", index=False)
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    # 只有直接运行本文件时才开始训练；被其他 Python 文件 import 时不会自动执行 main。
    main()
