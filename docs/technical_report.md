# 涡旋光空间感知与 FPGA 边缘推理项目技术文档

## 0. 文档定位

本项目面向涡旋光实验中的光电信号参数估计问题，目标是将仿真产生的光电时域信号转换为频域特征，再利用轻量化 1D-CNN 估计旋转状态，并将推理核心部署到 Zynq-7020 FPGA 上。

当前已经完成的成果属于：

> 面向光电空间感知的轻量化神经网络 FPGA 边缘推理原型。

当前系统已经完成 RPM 回归任务的板级验证，但尚未完成实时 ADC 接入、完整几何参数多任务输出和完整三维重建。因此，汇报时应将当前成果表述为“空间感知前端和边缘推理模块”，不要直接称为完整三维重建系统。

---

## 1. 项目摘要

### 1.1 要解决的问题

涡旋光或旋转目标在不同转速、倾角、偏心距离和方位角条件下，会产生不同的光电探测信号。传统方法通常需要从频谱中手工寻找峰值、谐波或频率组合。当存在噪声、转速变化、几何条件变化或频率成分混叠时，手工规则的泛化能力会下降。

本项目采用数据驱动方法学习：

```text
光电探测信号 -> 频谱特征 -> 1D-CNN -> 转速/几何参数 -> 物理模型解析
```

核心问题可写成：

```text
x = 光电信号或其频域特征
y = [RPM, gamma, d, phi, ...]

利用神经网络学习 f_theta(x) ≈ y
```

其中 `gamma` 表示倾角，`d` 表示偏心距离，`phi` 表示方位角。当前 FPGA 版本首先实现了 `RPM` 单输出回归，用于验证模型量化、HLS 生成和板级部署链路。

### 1.2 当前项目定位

对应老师要求的方向二：

> FPGA 加速空间智能系统中的光电空间感知前端。

具体对应关系如下：

| 老师的要求 | 当前项目对应内容 |
|---|---|
| 前期调研 | 涡旋光光电测量、FFT 频域分析、1D-CNN、模型量化和 FPGA HLS |
| 跑通算法 | PyTorch 训练、Phase 1 CNN、Phase 2 FPGA-friendly CNN |
| 模块分析 | FFT、卷积、池化、全局平均池化、全连接和 PS-PL 数据搬运分析 |
| 初步优化 | 去除 BN/Dropout/GELU、缩减通道、INT8 权重量化、定点激活、循环流水化 |
| 边缘部署 | Vitis HLS IP、Vivado Block Design、Vitis ARM 软件、Zynq-7020 上板运行 |
| 空间智能联系 | 信号受倾角、偏心距离和方位角影响；后续恢复几何参数多任务输出和物理反演 |

---

## 2. 实验数据与物理背景

### 2.1 数据来源

当前数据为仿真数据，不是实时采集数据。项目数据计划包含约 18,000 条样本，覆盖不同转速和几何组合，并按照不同泛化能力划分训练集和测试集。

主要参数范围为：

| 参数 | 当前设置 |
|---|---|
| 总转速范围 | 400 至 3000 RPM |
| 训练转速范围 | 约 700 至 2400 RPM |
| 采样率 | 20 kHz |
| 单条信号时长 | 1 s |
| 单条时域采样点 | 20,000 |
| FFT 输入特征长度 | 512 |
| 倾角 gamma | 0、20、35、45、60 度 |
| 偏心距离 d | 0、1、2.5、5、10 mm |
| 方位角 phi | 0、30、45、90、135 度 |

数据划分包含：

- `train`：用于网络参数学习；
- `val`：用于早停和模型选择；
- `test_id`：与训练分布相近的测试；
- `test_geometry`：几何组合变化测试；
- `test_hard`：强扰动或较难信号测试。

这种划分比单纯随机划分更适合空间感知问题，因为它可以分别检查网络的范围内泛化能力、几何泛化能力和抗干扰能力。

### 2.2 频域特征提取

当前预处理代码位于：

[prepare_rde_dataset.py](../scripts/prepare_rde_dataset.py)

处理流程为：

1. 读取每条 20 kHz 采样的时域波形；
2. 使用实数快速傅里叶变换 `rFFT`；
3. 计算幅度谱；
4. 截取 20 至 1600 Hz 频带；
5. 使用 `log1p` 压缩动态范围；
6. 插值或重采样为 512 维输入向量。

数学形式为：

```text
X[k] = rFFT{x[n]}
S[k] = log(1 + |X[k]|)
input = Resample(S[f_low:f_high], 512)
```

FFT 是项目中使用的第一个经典算法。它将难以直接处理的长时域波形转换为更容易识别的频谱结构，使 CNN 可以学习频率峰值、谐波分布、谱宽、能量变化等特征。

当前系统的一个重要限制是：输入主要使用幅度谱，时间相位信息没有直接输入网络。因此，RPM 估计可以先取得结果，但几何参数估计可能需要补充相位、多通道频谱、双光斑信息或原始时域窗口。

---

## 3. 空间感知的核心含义

### 3.1 空间信息如何进入信号

本项目不是直接输入 RGB 图像或点云，而是通过光电信号间接感知空间状态。

实验中的空间变量会改变探测器看到的局部频率分布：

```text
倾角 gamma       -> 改变旋转轴与光束方向的关系
偏心距离 d       -> 改变旋转中心与光斑环的相对位置
方位角 phi       -> 改变偏心方向
旋转速度 RPM     -> 改变角速度和频率尺度
```

因此，频谱中的峰值和能量分布并不只反映 RPM，也携带了几何状态信息。

### 3.2 物理前向模型

项目中已经存在可微分的双光斑 RDE 前向模型：

[dualspot_rde_forward.py](../models/physics/dualspot_rde_forward.py)

模型使用局部频率关系：

```text
f = Delta_l / (2*pi) * ((rho x (Omega x r)) dot b) / |rho|^2
```

其中：

- `Omega` 是旋转角速度，和 RPM 相关；
- `b` 是由倾角 `gamma` 和光束方位确定的方向向量；
- `rho` 是环上局部位置向量；
- `r` 是包含偏心距离和方位角的目标位置；
- `Delta_l` 是拓扑荷差；
- 局部频率随环上位置变化，最终形成探测器可观察的频谱分布。

这个物理模型说明了项目中的“空间感知”不是简单把一个 CNN 用在任意信号上，而是存在明确的物理链路：

```text
空间几何参数
    -> 旋转和光场作用
    -> 局部频率分布
    -> 探测器频谱
    -> CNN 反演空间状态
```

### 3.3 当前已经实现和尚未实现的空间能力

已经实现：

- 数据生成时覆盖了倾角、偏心距离和方位角组合；
- 测试集包含几何泛化测试；
- 已有物理前向模型和物理参数脚本；
- CNN 已经能够从频域特征中估计 RPM；
- CNN 已经部署到 Zynq-7020 的 PL 侧进行推理。

尚未完成：

- 当前 FPGA IP 只输出 RPM；
- 几何参数尚未全部部署到 FPGA；
- FFT 仍主要在电脑或软件预处理阶段完成；
- 尚未接入实时 ADC 或真实光电探测器；
- 尚未形成完整的距离、姿态或三维重建结果。

因此，当前最准确的说法是：

> 已完成面向空间感知任务的光电特征提取和 RPM 边缘推理模块，下一步将扩展为几何参数多任务估计和物理模型重建。

---

## 4. Phase 1：软件基线 CNN

### 4.1 目标

Phase 1 的目标是先验证：

- 512 维 FFT 特征是否包含足够的 RPM 信息；
- 1D-CNN 是否比简单基线更适合频谱局部模式提取；
- 几何变化和强扰动下是否仍有一定泛化能力。

相关文件：

[phase1/README.md](../phase1/README.md)

[train_phase1_cnn.py](../phase1/train_phase1_cnn.py)

### 4.2 网络结构

Phase 1 网络结构为：

```text
输入：512
  -> Conv1D(1, 16, kernel=7)
  -> ReLU
  -> MaxPool1D(2)
  -> Conv1D(16, 32, kernel=5)
  -> ReLU
  -> MaxPool1D(2)
  -> Conv1D(32, 64, kernel=3)
  -> ReLU
  -> AdaptiveAvgPool1D(1)
  -> Linear(64, 32)
  -> ReLU
  -> Linear(32, 1)
  -> RPM
```

使用全局平均池化的优点是避免将完整特征图直接展平，降低全连接层参数量，并增强对局部频谱位置小幅变化的鲁棒性。

### 4.3 Phase 1 结果

当前保存结果位于：

[phase1 results](../phase1/results/run_summary.json)

主要指标如下：

| 数据集 | MAE RPM | RMSE RPM | MAPE | R2 |
|---|---:|---:|---:|---:|
| train | 166.93 | 248.37 | 14.08% | 0.891 |
| val | 166.99 | 231.72 | 14.40% | 0.904 |
| test_id | 166.68 | 235.08 | 14.51% | 0.904 |
| test_geometry | 179.55 | 246.79 | 15.61% | 0.894 |
| test_hard | 226.39 | 288.00 | 20.65% | 0.855 |

Phase 1 的结论不是“模型已经足够好”，而是：

1. FFT 特征具有可学习的 RPM 信息；
2. CNN 可以在几何变化和强扰动测试上保持一定泛化；
3. 网络仍需要面向 FPGA 的结构压缩和算子简化。

---

## 5. 原始 CNN 与 FPGA 部署问题

队友已有的 CNN 版本使用了更宽、更复杂的网络，典型结构为：

```text
通道：1 -> 24 -> 48 -> 64 -> 64
卷积核：15、9、7、5
步长：2
激活：GELU
归一化：BatchNorm
分类/回归头：展平后连接较大的全连接层
```

该结构在软件训练上具有较强表达能力，但直接部署到 xc7z020 存在以下问题：

- GELU 需要较复杂的非线性运算；
- BatchNorm 会增加参数和运行时处理；
- 较宽卷积层产生大量乘加；
- 展平后的全连接层会产生大量权重访问和 DSP 使用；
- 大特征图会增加 BRAM 和片上数据搬运压力；
- 浮点版本会造成较高的 LUT、DSP 和时序压力。

因此没有直接推倒重来，而是以原模型为精度参考，设计了更适合 FPGA 的 Phase 2 版本。

---

## 6. Phase 2：FPGA-friendly 1D-CNN

### 6.1 设计目标

Phase 2 的设计目标为：

- 保持相同的 512 维输入和 RPM 回归任务；
- 取消硬件不友好的 BN、Dropout 和 GELU；
- 缩小通道数和全连接层宽度；
- 使用 ReLU 和固定池化；
- 减少控制复杂度，便于 HLS 调度；
- 保持模型输出和软件流程可对齐。

### 6.2 网络结构

Phase 2 网络结构为：

```text
输入：512
  -> Conv1D(1, 8, kernel=7, padding=3)
  -> ReLU
  -> MaxPool1D(2)
  -> Conv1D(8, 16, kernel=5, padding=2)
  -> ReLU
  -> MaxPool1D(2)
  -> Conv1D(16, 32, kernel=3, padding=1)
  -> ReLU
  -> Global Mean Pooling
  -> Linear(32, 16)
  -> ReLU
  -> Linear(16, 1)
  -> 反归一化
  -> RPM
```

网络参数规格保存在：

[fpga_friendly_cnn_spec.json](../phase2/export/fpga_friendly_cnn_spec.json)

### 6.3 Phase 1 与 Phase 2 的精度变化

Phase 2 结果位于：

[phase2 results](../phase2/results/run_summary.json)

| 数据集 | Phase 1 MAE | Phase 2 MAE | 变化 |
|---|---:|---:|---:|
| val | 166.99 | 206.14 | 增加约 39.15 RPM |
| test_id | 166.68 | 203.67 | 增加约 36.99 RPM |
| test_geometry | 179.55 | 213.14 | 增加约 33.59 RPM |
| test_hard | 226.39 | 253.36 | 增加约 26.97 RPM |

这说明 Phase 2 用更低的硬件复杂度换取了一定精度损失。该结果本身很重要，因为它给出了模型压缩的第一条基线：后续剪枝、量化感知训练和蒸馏都需要以 Phase 1 精度作为参考，以 Phase 2 资源为部署参考。

---

## 7. 量化、定点化和 HLS 实现

### 7.1 量化策略

当前导出流程位于：

[export_phase2_fpga_bundle.py](../phase2/export_phase2_fpga_bundle.py)

[prepare_hls_assets.py](../phase2/prepare_hls_assets.py)

当前使用的思路是：

```text
PyTorch 浮点权重
    -> 按层估计尺度
    -> INT8 权重和偏置
    -> HLS 中使用定点激活和累加器
    -> 输出反量化并恢复 RPM
```

权重使用 `int8_t` 存储，激活、尺度和累加器使用不同位宽的 `ap_fixed` 类型。

当前 HLS 顶层文件为：

[phase2_rpm_cnn_top.cpp](../phase2/hls/phase2_rpm_cnn_top.cpp)

### 7.2 定点数据类型

当前设计中使用了类似以下的数据类型：

```cpp
typedef ap_fixed<18, 8, AP_RND, AP_SAT> data_t;
typedef ap_fixed<24, 4, AP_RND, AP_SAT> scale_t;
typedef ap_fixed<40, 24, AP_RND, AP_SAT> acc_t;
typedef ap_fixed<32, 16, AP_RND, AP_SAT> out_t;
```

其中：

- `data_t` 保存归一化输入和中间激活；
- `scale_t` 保存权重量化和偏置反量化系数；
- `acc_t` 防止卷积和全连接累加溢出；
- `out_t` 用于最终 RPM 反归一化。

这里最容易出现的错误是累加器位宽过小。即使输出仍然是一个“看起来合理”的浮点数，内部溢出也可能导致模型结果偏离。当前版本通过扩大累加器和使用饱和定点，解决了早期输出错误问题。

### 7.3 HLS 顶层结构

HLS 顶层执行顺序为：

```text
AXI DDR 输入
    -> normalize_input
    -> conv1d_same_q
    -> ReLU
    -> maxpool1d_2
    -> conv1d_same_q
    -> ReLU
    -> maxpool1d_2
    -> conv1d_same_q
    -> ReLU
    -> global_mean_pool
    -> dense_q
    -> ReLU
    -> dense_q
    -> RPM 反归一化
    -> AXI DDR 输出
```

接口为：

- `m_axi input`：从 DDR 读取 512 维输入；
- `m_axi output_rpm`：向 DDR 写回一个浮点 RPM；
- `s_axilite`：由 ARM PS 配置输入地址、输出地址和启动寄存器；
- `ap_clk/ap_rst_n`：来自 Zynq PS 的时钟和复位。

HLS C++ 不是直接在 ARM 上运行。Vitis HLS 会对 C/C++ 代码进行高层次综合，完成循环分析、操作调度、存储推断、乘法器/DSP 映射、流水线和 RTL 生成，最后封装成 Vivado 可使用的 IP。之后 Vivado 再将 IP 和 PS、AXI、DDR、GPIO 等连接起来，生成 bitstream。

---

## 8. HLS 优化过程

### 8.1 初始浮点版本

第一版 HLS 直接保留了大量浮点计算，综合后资源压力较大，难以适配 xc7z020。问题主要来自：

- 浮点乘法和除法复杂；
- 中间特征图较大；
- 卷积内层循环计算量高；
- 全连接层存在连续乘加；
- 浮点资源和控制逻辑同时占用 LUT/DSP。

### 8.2 算子简化

随后进行了以下修改：

1. GELU 改为 ReLU；
2. BatchNorm 和 Dropout 从硬件路径中移除；
3. 自适应池化改为固定 MaxPool 和 Global Mean Pool；
4. 通道数由较宽结构缩小；
5. 全连接层由更宽结构改为 `32 -> 16 -> 1`；
6. 权重改用 INT8 存储；
7. 激活和累加使用 `ap_fixed`；
8. 对卷积和全连接循环进行流水线调度。

### 8.3 DSP 优化尝试

早期为了提高并行度，尝试增加循环展开和更小的 II。这样可以增加 DSP 并行计算单元，但也会带来：

- DSP 占用增加；
- LUT 和寄存器增加；
- BRAM 端口压力增加；
- 布线和时序压力增加；
- 可能超过 xc7z020 的资源上限。

后续将卷积内层循环和全连接循环设置为较保守的 II，并根据综合报告平衡延迟和资源。当前设计的目标不是单纯追求 DSP 越多越好，而是寻找：

```text
资源可实现 + 时序可收敛 + 延迟可接受 + 输出精度稳定
```

### 8.4 当前硬件验证结果

针对同一个测试向量，当前记录的结果为：

| 项目 | 输出 RPM |
|---|---:|
| 真实标签 | 2308.932 |
| Phase 2 PyTorch 浮点模型 | 2203.136 |
| HLS INT8 浮点仿真 | 2214.897 |
| HLS C Simulation | 2216.635 |
| RTL Co-simulation | 通过 |
| FPGA 板级输出 | 2216.635 |

HLS C Simulation 与 INT8 浮点参考的差值约为 `1.738 RPM`，相对差异约 `0.08%`。板级输出与 HLS C Simulation 一致，说明以下链路已经打通：

```text
PyTorch 权重
 -> INT8 导出
 -> HLS C Simulation
 -> RTL Co-simulation
 -> IP 封装
 -> Vivado bitstream
 -> Vitis ARM 调用
 -> FPGA 输出
```

需要注意，真实标签和预测值之间约 92 RPM 的差异主要属于当前模型误差，不应误判为 FPGA 硬件误差。

### 8.5 综合资源和时延记录

此前 HLS 综合优化过程中记录到的代表性结果如下。不同代码版本、综合策略和目标时钟可能产生不同结果，正式汇报时应以最终版本的 `csynth` 报告为准。

| 指标 | 代表性结果 | 说明 |
|---|---:|---|
| 目标时钟周期 | 10 ns | 对应约 100 MHz 目标时钟 |
| 估计推理延迟 | 约 1.226 ms | 单次 512 维输入推理 |
| DSP | 154 / 220，约 70% | 主要来自卷积和全连接乘加 |
| LUT | 46,865 / 53,200，约 88% | 控制、定点运算和逻辑资源压力较大 |
| BRAM | 40 / 280，约 14% | 主要用于中间特征图和权重/缓存 |
| 关键问题 | LUT 偏高 | 后续优先考虑结构化剪枝和通道缩减 |

这组结果说明当前设计已经可以上板，但距离资源裕量充足还有差距。后续优化的优先级应为：

```text
先降低 LUT 和 DSP 压力
    -> 再优化端到端延迟
    -> 最后考虑增加多任务输出
```

---

## 9. Vivado、Vitis 和板级系统

### 9.1 Vivado Block Design

当前板级系统使用 Zynq-7020 的 PS-PL 架构：

```text
Zynq PS
 ├─ DDR
 ├─ UART
 ├─ I2C0 -> OLED
 ├─ AXI GPIO -> LED/BTN
 ├─ AXI GPIO -> SW
 └─ AXI GP0
       ├─ CNN AXI-Lite 控制
       └─ SmartConnect/AXI Interconnect

CNN HLS IP
 ├─ s_axi_control
 ├─ m_axi_gmem -> DDR
 ├─ ap_clk
 ├─ ap_rst_n
 └─ interrupt
```

PS 负责：

- 准备或接收输入数据；
- 将输入写入 DDR；
- 设置 CNN 输入和输出地址；
- 启动 CNN IP；
- 等待完成；
- 读取结果并进行显示。

PL 负责：

- 归一化；
- 卷积；
- ReLU；
- 池化；
- 全连接；
- RPM 反归一化。

### 9.2 Vitis 软件程序

当前应用程序完成了：

- CNN IP 初始化；
- DDR 输入输出缓冲区配置；
- Cache flush/invalidate；
- CNN 启动和完成轮询；
- 输出原始比特和浮点 RPM；
- GPIO 读取；
- OLED 初始化和 I2C 通信；
- 按键触发 CNN；
- LED 显示运行、完成和错误状态；
- 输出推理时间。

当前 OLED 交互状态为：

```text
待机：显示 READY 和 GPIO 状态
BTN0：触发 CNN
运行：显示 RUNNING
完成：显示 RPM 和推理时间
错误：显示 ERROR
BTN1：返回 READY
```

OLED 属于系统人机交互和调试显示，不应单独包装成“先进显示加速”。它的作用是让 FPGA 推理结果脱离电脑串口，形成可现场演示的边缘设备原型。

---

## 10. 涉及的经典算法与工程方法

### 10.1 FFT

用途：将时域光电信号转为频域特征。

优点：计算复杂度低、物理含义清晰、适合观察旋转频率和谐波。

后续方向：

- 在 PS 侧使用 ARM NEON 或 CMSIS-DSP 优化；
- 使用 FPGA IP 实现实时 FFT；
- 采用滑动窗口实现连续推理；
- 比较原始波形、FFT 幅度、相位和多通道输入。

### 10.2 1D-CNN

用途：提取相邻频率点之间的局部模式和多尺度谱结构。

它比手工寻找单一峰值更适合处理：

- 谐波变化；
- 频谱峰值偏移；
- 多峰和噪声；
- 几何条件引起的频率分布变化。

### 10.3 量化

量化将浮点权重和激活转换为低位宽表示，减少存储和乘法资源。当前已完成 INT8 权重和 `ap_fixed` 激活的部署验证。

下一步应从当前的后训练量化继续到量化感知训练，即在训练期间模拟量化误差，使模型主动适应硬件数值范围。

### 10.4 结构化剪枝

结构化剪枝通过删除整个通道、卷积核或神经元减少规则计算量。它比随机稀疏权重更适合 HLS，因为硬件循环边界可以直接缩小。

推荐实验：

```text
Phase 1/2 baseline
 -> 剪枝 25% 通道
 -> 微调
 -> 剪枝 50% 通道
 -> 微调
 -> 比较精度和 FPGA 资源
```

### 10.5 MobileNet 风格深度可分离卷积

MobileNet 的经典思想是将普通卷积分解为：

```text
Depthwise Conv1D
    -> Pointwise Conv1D(1x1)
```

它适合作为 Phase 2 之后的结构改进方案。优点是乘加量和参数量较少，缺点是 HLS 中需要专门设计 depthwise 和 pointwise 的访存、并行和流水结构。

### 10.6 知识蒸馏

可以将精度较高的 Phase 1 或原始 CNN 作为 Teacher，将剪枝后的轻量网络作为 Student：

```text
L = alpha * L_label(Student, ground_truth)
  + beta  * L_distill(Student, Teacher)
```

对于回归任务，可以直接蒸馏教师模型输出，或者蒸馏中间特征。它适合解决“模型变轻后 MAE 变差”的问题。

### 10.7 物理约束和物理模型

项目已有可微分双光斑 RDE 前向模型。后续可以使用：

```text
预测参数 -> 物理前向模型 -> 预测频率分布
```

并加入物理一致性损失：

```text
L_total = L_parameter + lambda * L_physics
```

这会使项目从普通黑盒回归进一步发展为“物理模型与神经网络结合的空间参数估计”。

---

## 11. 已经尝试和完成的优化过程

### 11.1 从复杂 CNN 到 Phase 1

完成内容：

- 使用 FFT 特征作为统一输入；
- 使用轻量 1D-CNN 替代较宽的原始网络；
- 使用 ReLU、MaxPool 和全局平均池化；
- 建立 train、val、test_id、test_geometry、test_hard 评估流程；
- 保存模型、预测结果和指标。

得到的经验：

- 512 维频谱输入可以支持 RPM 回归；
- 几何泛化测试有必要单独保留；
- 单看平均误差不足以证明空间泛化能力。

### 11.2 从 Phase 1 到 Phase 2

完成内容：

- 进一步减少通道数；
- 减少全连接层规模；
- 固定池化方式；
- 移除 BN、Dropout 和 GELU；
- 保留统一输入和输出接口；
- 使网络便于导出和 HLS 重写。

得到的经验：

- 模型越轻，软件精度存在下降风险；
- 不能只看参数量，还要观察频谱特征是否被过度压缩；
- 需要以资源、延迟和精度三者的 Pareto 平衡为目标。

### 11.3 从浮点 HLS 到定点 HLS

完成内容：

- 权重 INT8 化；
- `ap_fixed` 激活、尺度和输出；
- 扩大累加器避免溢出；
- 卷积和全连接循环流水化；
- 根据 DSP 使用情况调整 II；
- 完成 C Simulation 和 RTL Co-simulation。

得到的经验：

- HLS C Simulation 通过不代表 RTL 和板级一定正确；
- 需要使用同一测试向量逐级比对；
- 必须区分模型误差、量化误差、HLS 数值误差和 PS-PL 传输错误。

### 11.4 从 IP 到板级系统

完成内容：

- HLS IP 封装；
- Vivado AXI 互联和 DDR 配置；
- Vitis platform/domain 更新；
- bitstream/XSA 导出；
- ARM 端调用 CNN；
- UART 输出结果；
- OLED、LED、BTN、SW 外设联调；
- OLED I2C0 和 1.8 V/3.3 V 电平链路验证。

得到的经验：

- 软件程序、XSA、bitstream 和 platform 必须来自同一版本；
- 替换 IP 后需要重新生成 bitstream 并更新 Vitis platform；
- 硬件地址、BSP 宏、启动文件和 bitstream 不一致时，可能出现程序能运行但输出错误的情况。

---

## 12. 当前成果清单

### 12.1 算法和数据

- 完成仿真数据整理；
- 完成 FFT 频域特征生成；
- 建立 512 维输入格式；
- 建立多种测试划分；
- 完成 Phase 1 CNN；
- 完成 Phase 2 FPGA-friendly CNN；
- 保存模型、权重、预测结果和指标。

### 12.2 FPGA 和 HLS

- 完成 HLS 顶层函数；
- 完成 INT8 权重导出；
- 完成定点激活和累加；
- 完成 C Simulation；
- 完成 C/RTL Co-simulation；
- 完成 HLS IP 封装；
- 完成 Vivado Block Design；
- 完成 Zynq-7020 bitstream/XSA；
- 完成 Vitis ARM 端调用。

### 12.3 板级演示

- CNN IP 可由 ARM 端启动；
- 输出 RPM 与 HLS 仿真一致；
- OLED 可正常初始化和显示；
- UART 可打印原始输出和结果；
- LED、按键、开关已经接入演示流程；
- 可显示 CNN 运行状态和推理延迟。

---

## 13. 当前不足与需要补全的实验

### 13.1 数据闭环不足

目前主要使用仿真数据和固定测试向量。后续应增加：

- 真实光电探测器数据；
- ADC 或采集卡接口；
- 实时 FFT；
- 连续帧输入；
- 真实噪声和器件误差。

### 13.2 空间输出不足

当前硬件只输出 RPM。建议先在软件上完成 5 维输出：

```text
[RPM, gamma, d, sin(phi), cos(phi)]
```

其中使用 `sin(phi)` 和 `cos(phi)` 可以避免角度在 0/360 度处的不连续问题。

### 13.3 缺少完整性能基线

需要补充同一测试向量和同一批测试集上的：

- ARM 纯软件推理时间；
- PyTorch CPU 推理时间；
- HLS C Simulation 时间；
- FPGA PL 推理时间；
- PS-PL 数据搬运时间；
- 端到端延迟；
- 功耗和资源利用率。

### 13.4 缺少量化误差分解

建议分别记录：

```text
Float32 PyTorch
 -> Float32 HLS
 -> INT8 浮点仿真
 -> ap_fixed C Simulation
 -> RTL Co-simulation
 -> FPGA 板级输出
```

只有这样才能判断误差来自模型设计、量化策略、定点位宽还是软件接口。

---

## 14. 下一阶段优化路线

### 阶段 A：结构化剪枝

目标：在不改变硬件接口的情况下减少资源。

步骤：

1. 以 Phase 1 和 Phase 2 为基线；
2. 对卷积通道计算 L1 范数或 BN 缩放系数重要性；
3. 删除 25% 通道并微调；
4. 删除 50% 通道并微调；
5. 导出新的权重和网络规格；
6. 比较 MAE、RMSE、DSP、LUT、延迟。

推荐优先尝试的通道结构：

```text
1 -> 8 -> 16 -> 32 -> 16
1 -> 6 -> 12 -> 24 -> 12
1 -> 8 -> 12 -> 24 -> 12
```

### 阶段 B：量化感知训练

目标：降低剪枝和低位宽带来的精度损失。

步骤：

1. 在 PyTorch 中模拟 INT8 权重和激活；
2. 训练阶段加入 fake quantization；
3. 对比后训练量化和 QAT；
4. 选择误差最小的量化尺度；
5. 重新导出 HLS 权重。

### 阶段 C：MobileNet-1D 对照模型

目标：验证深度可分离卷积是否比普通卷积更适合该频谱任务。

建议结构：

```text
Conv1D(1 -> 8)
    -> DepthwiseConv1D(8, k=5)
    -> PointwiseConv1D(8 -> 16)
    -> DepthwiseConv1D(16, k=3)
    -> PointwiseConv1D(16 -> 32)
    -> GlobalMeanPool
    -> FC
```

需要重点观察：虽然理论 MAC 数减少，但 HLS 中的访存和控制逻辑是否抵消了收益。最终以实际 DSP、LUT 和端到端延迟为准。

### 阶段 D：知识蒸馏

如果轻量模型精度下降明显，可使用原始较大 CNN 或 Phase 1 CNN 作为教师模型，将知识迁移给剪枝后的学生模型。

### 阶段 E：多任务空间参数估计

将单输出改为：

```text
RPM、gamma、d、sin(phi)、cos(phi)
```

损失函数可以写为：

```text
L = L_rpm + w1*L_gamma + w2*L_d
    + w3*L_sinphi + w4*L_cosphi
    + lambda*L_physics
```

第一版可以先在 PyTorch 中完成，硬件上先部署 RPM 和 `d` 两个输出，再逐步增加输出维度，避免一次性增加太多 IP 和软件验证工作。

### 阶段 F：实时系统闭环

最终目标为：

```text
光电探测器/ADC
    -> PS 或 PL FFT
    -> DDR/AXI Stream
    -> CNN IP
    -> 物理模型解析
    -> OLED/串口/上位机
```

---

## 15. 推荐的研究问题和成果表述

### 15.1 推荐研究问题

> 面向资源受限的 Zynq-7020 边缘设备，如何通过结构化剪枝、定点量化和 HLS 并行化，将涡旋光光电频谱的 1D-CNN 空间状态估计模型部署到 FPGA，并在精度、资源和延迟之间取得平衡？

### 15.2 当前成果表述

可以向老师这样介绍：

> 我将涡旋光实验中的光电频域信号作为空间感知输入，首先在 PyTorch 中验证 1D-CNN 对转速参数的回归能力，然后针对 Zynq-7020 的资源约束去除 BN、Dropout 和 GELU，缩减通道和全连接层，并进行 INT8 权重量化和定点 HLS 实现。当前已经完成 Vitis HLS C/RTL 仿真、Vivado IP 集成、Vitis ARM 调用以及板级验证，CNN 输出与 HLS 参考结果基本一致。下一阶段计划引入结构化剪枝和量化感知训练，进一步降低 FPGA 资源，并将 RPM 单输出扩展为倾角、偏心距离和方位角等空间参数多任务估计。

### 15.3 不建议使用的表述

当前不建议直接说：

- 已经完成实时三维重建；
- 已经完成完整空间智能系统；
- OLED 本身属于先进显示 FPGA 加速；
- FPGA 已经加速了整个光电采集链路；
- 已经完成真实传感器闭环。

更准确的说法是：

> 已完成光电空间感知任务中的轻量化 CNN 边缘推理核心和板级演示原型。

---

## 16. 参考资料和开源工具

1. FFT：用于时域信号到频域特征的转换，当前项目使用 NumPy `rFFT`。
2. MobileNet：深度可分离卷积和移动端轻量化网络，<https://arxiv.org/abs/1704.04861>。
3. Deep Compression：剪枝、训练量化和权重压缩，<https://arxiv.org/abs/1510.00149>。
4. Knowledge Distillation：教师模型向学生模型迁移知识，<https://arxiv.org/abs/1503.02531>。
5. hls4ml：将机器学习模型转换为 HLS 实现，可作为当前自研 HLS 路线的对比工具，<https://github.com/fastmachinelearning/hls4ml>。
6. FINN：面向量化神经网络的 FPGA 数据流编译框架，<https://github.com/Xilinx/finn>。
7. PointNet：面向点云分类和分割的经典空间智能模型，<https://github.com/charlesq34/pointnet>。

---

## 17. 项目当前结论

当前项目已经完成了从仿真数据、FFT 特征、1D-CNN、量化、HLS、IP 封装、Vivado 集成到 Zynq-7020 板级调用的完整最小闭环。

项目最有价值的地方不是“在 FPGA 上运行了一个 CNN”，而是形成了以下可继续研究的技术链：

```text
物理空间参数
    -> 光电频谱
    -> CNN 空间状态估计
    -> 模型轻量化
    -> HLS 硬件映射
    -> PS-PL 协同
    -> 边缘设备原型
```

下一阶段最合理的主线是：

```text
结构化剪枝 + INT8/QAT
    -> 降低 FPGA 资源
    -> 保持或恢复预测精度
    -> 扩展几何参数多任务输出
    -> 加入物理一致性损失
    -> 完成光电空间状态重建演示
```

这条路线能够自然地把已有的涡旋光项目、1D-CNN、FPGA 加速和老师提出的空间智能方向连接起来。
