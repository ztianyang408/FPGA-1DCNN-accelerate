# Vortex Optical FPGA Edge Inference

基于 Zynq-7020 的涡旋光光电信号空间感知与轻量化 1D-CNN FPGA 边缘推理原型。

## 整体项目框架
<img width="1081" height="605" alt="image" src="https://github.com/user-attachments/assets/2cf71150-a7a7-4d1e-8c20-cb2160e4db03" />

## 1DCNN STURCTURE
<img width="1069" height="264" alt="e9238cd50065ed4c82ac09ec032e193a" src="https://github.com/user-attachments/assets/35efdff5-4609-49ba-9352-beaf3798bdd5" />


## 已完成

- 仿真光电信号的 FFT 频域特征提取
- Phase 1 软件 1D-CNN RPM 回归基线
- Phase 2 FPGA-friendly 1D-CNN
- INT8 权重量化与 `ap_fixed` 定点 HLS 实现
- Vitis HLS C Simulation 和 C/RTL Co-simulation
- Vivado IP 集成、AXI/DDR/PS-PL 协同
- Zynq-7020 板级运行
- UART、OLED、LED、按键和开关演示

当前 FPGA 版本主要完成 RPM 单输出推理。倾角、偏心距离和方位角等几何参数已经进入数据和物理建模范围，下一阶段将扩展为多任务空间参数估计。

## 目录

```text
data/                 处理后的 FFT 数据和板级测试向量
scripts/              数据预处理
phase1/               软件 CNN 和基线模型
phase2/               FPGA-friendly CNN、导出和 HLS 文件
models/physics/       涡旋光物理前向模型及几何实验脚本
fpga/hls_ip/          已封装的 HLS IP
fpga/vivado/          Block Design、工程入口和板级约束
fpga/vitis_app/       ARM 端板级演示程序
fpga/release/         最终 XSA 和 bitstream
docs/                 技术文档
```

## 环境

- Python 3.10+
- PyTorch
- NumPy、Pandas、scikit-learn、joblib、matplotlib
- Vivado/Vitis HLS 2024.2
- Zynq-7020 开发板

安装 Python 依赖：

```bash
pip install -r requirements.txt
```

## 运行软件模型

仓库内已经包含处理后的数据（基于FFT 输出512维度。训练 Phase 1 CNN：

```bash
python phase1/train_phase1_cnn.py
```

训练 Phase 2 FPGA-friendly CNN：

```bash
python phase2/train_phase2_fpga_cnn.py
```

导出 HLS 资产：

```bash
python phase2/export_phase2_fpga_bundle.py
python phase2/prepare_hls_assets.py
```

注意：`scripts/prepare_rde_dataset.py` 用于从原始时域仿真数据生成 FFT 数据。原始数据没有放入本仓库，需要时通过外部存储获取后再运行该脚本。

## HLS 验证

HLS 文件位于 `phase2/hls/`：

- `phase2_rpm_cnn_top.cpp`：HLS 顶层实现
- `phase2_rpm_cnn_tb.cpp`：测试平台
- `phase2_weights_int8.h`：INT8 权重和归一化参数

在 Vitis HLS 中将 `phase2_rpm_cnn` 设置为 top function，依次运行 C Simulation、C Synthesis 和 C/RTL Co-simulation。

## 板级运行

直接使用 `fpga/release/` 中的 XSA 和 bitstream，或在 Vivado 中打开 `fpga/vivado/cnn1d.bd` 重新生成硬件平台。Vitis 应用源码位于：

```text
fpga/vitis_app/src/helloworld.c
fpga/vitis_app/src/phase3_test_vector.h
```

板级演示流程：

```text
BTN0 -> 触发 CNN
LED  -> 显示运行/完成/错误状态
OLED -> 显示 RPM 和推理时间
BTN1 -> 返回待机状态
UART -> 输出原始结果和调试信息
```

## 代表性结果

同一测试向量下：

| 参考 | RPM |
|---|---:|
| 真实标签 | 2308.932 |
| Phase 2 PyTorch | 2203.136 |
| HLS INT8 浮点仿真 | 2214.897 |
| HLS C Simulation | 2216.635 |
| FPGA 板级输出 | 2216.635 |

HLS 与板级输出一致，说明从量化模型到 FPGA 推理的软硬件验证链路已经打通。真实标签与预测值的差异主要是模型误差，不是 FPGA 传输误差。

## 后续路线

```text
结构化剪枝 -> 量化感知训练 -> MobileNet-1D 对照
    -> 降低 LUT/DSP/延迟
    -> RPM + gamma + d + sin(phi) + cos(phi) 多任务输出
    -> 物理一致性损失和空间状态重建
    -> 实时 ADC/FFT/边缘闭环
```

详细说明见 [docs/technical_report.md](docs/technical_report.md)。

## 说明

本仓库保留的是研究记录和最终可用产物。Vivado/Vitis 的缓存、综合中间文件、自动生成 BSP、历史 bitstream 和原始大数据集不纳入版本控制。
