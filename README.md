# FP32-to-INT8/INT4 Quantizer

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Python Version](https://img.shields.io/badge/Python-3.7+-green.svg)](https://www.python.org/downloads/)
[![PyPI Version](https://img.shields.io/badge/PyPI-1.0.0-yellow.svg)](https://pypi.org/project/fp32-to-int-quantizer/)

[English](#english) | [中文](#中文)

---

## English

### Overview

A high-performance FP32-to-INT8/INT4 quantization toolkit with full workflow support. This library provides end-to-end quantization from FP32 floating-point data to INT8/INT4 integers, supporting dual-precision quantization, multiple quantization modes, seamless framework integration, batch processing, compact serialization, and comprehensive precision evaluation.

### Key Features

- **Dual-Precision Support (INT8/INT4)**
  - INT8: Classic low-precision quantization, range -128~127, balanced accuracy and performance
  - INT4: Extreme compression quantization, default symmetric range -8~7, 1 byte stores 2 INT4 values, 75% storage reduction

- **Flexible Quantization Algorithms**
  - Quantization Modes: Symmetric, Asymmetric
  - Quantization Granularity: Per-Tensor, Per-Channel
  - Calibration Strategies: Extremes, Percentile, Entropy (optimized for INT4)

- **Full Engineering Capabilities**
  - Batch processing with automatic parameter reuse
  - PyTorch/TensorFlow model weight quantization
  - Compact binary storage (1 byte = 2 INT4 values)
  - Robust input validation and error handling

- **Comprehensive Precision Evaluation**
  - Basic metrics: MSE, MAE, RMSE, PSNR, SNR
  - Statistical significance metrics: Std, Variance, Skewness, Kurtosis, Entropy
  - Multi-precision comparison visualization

### Project Structure

```text
fp32_to_int_quantizer/
├── core/               # Core quantization logic and calibration strategies
├── frameworks/         # Optional PyTorch/TensorFlow integration
├── serialization/      # Binary data handling (INT4 packed storage)
├── visualization/      # Plotting and metric calculation
└── utils/              # Helper functions
```


### Installation

```bash
python setup.py install
pip install fp32-to-int-quantizer
```

### Quick Start

```python
from fp32_to_int_quantizer import FP32ToLowBitQuantizer, generate_batch_test_data

# Configuration (shared by INT4/INT8)
config = {
    "quant_mode": "symmetric",
    "quant_level": "per_channel",
    "calib_mode": "entropy",
    "entropy_bins": 2048,
    "seed": 42
}

# Generate test data
batch_data = generate_batch_test_data(
    batch_size=4,
    data_shape=(3, 32, 32),
    data_range=(0.0, 1.0),
    is_image_like=True
)

# INT8 Quantization
int8_quantizer = FP32ToLowBitQuantizer(quant_bit=8, **config)
int8_results = int8_quantizer.quantize(batch_data)

# INT4 Quantization
int4_quantizer = FP32ToLowBitQuantizer(quant_bit=4, **config)
int4_results = int4_quantizer.quantize(batch_data)

# Visualize comparison
int4_quantizer.visualize_analysis(
    fp32_original=batch_data[0],
    fp32_recovered=int4_results[0][1],
    save_path="comparison.png",
    ref_metrics={"quant_bit": 8, "scales": int8_quantizer.scales, "zero_points": int8_quantizer.zero_points}
)
```

### Model Quantization (PyTorch Example)

```python
import torch
from fp32_to_int_quantizer import FP32ToLowBitQuantizer

# 1. Prepare Model and Dummy Input
model = ... # Your PyTorch Model
dummy_input = torch.randn(1, 3, 224, 224)

# 2. Initialize Quantizer (INT4)
quantizer = FP32ToLowBitQuantizer(
    quant_bit=4,
    quant_mode="symmetric",
    quant_level="per_channel",
    calib_mode="entropy"
)

# 3. Quantize Weights
quantized_model, layer_params = quantizer.quantize_torch_model(model, dummy_input)
print(f"Quantized {len(layer_params)} layers.")
```

### Documentation

See [docs/API.md](docs/API.md) for detailed API documentation.

### License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

## 中文

### 简介

一款面向模型极致压缩与边缘部署的高性能FP32→INT8/INT4量化工具库。在原生INT8量化基础上扩展完整INT4支持，实现FP32浮点数据到INT8/INT4整型的端到端量化。支持多精度、多模式量化算法，无缝对接PyTorch/TensorFlow框架，提供批量处理、紧凑序列化、全维度精度评估与多精度对比可视化能力。

### 核心特性

- **双精度量化支持（INT8/INT4）**
  - INT8：经典低精度量化，范围-128~127，平衡精度与性能
  - INT4：极致压缩量化，默认对称范围-8~7，1字节存储2个INT4值，存储量减少75%

- **灵活的量化算法体系**
  - 量化模式：对称量化、非对称量化
  - 量化粒度：逐张量量化、逐通道量化
  - 智能校准策略：极值校准、分位数校准、熵校准（INT4专属优化）

- **全链路工程化能力**
  - 批量处理，自动复用量化参数
  - PyTorch/TensorFlow模型权重量化
  - 紧凑二进制存储（1字节=2个INT4值）
  - 鲁棒性校验与错误提示

- **完善的精度评估与可视化**
  - 基础指标：MSE、MAE、RMSE、PSNR、SNR
  - 统计显著性指标：标准差、方差、偏度、峰度、信息熵
  - 多精度对比可视化

### 项目结构

```text
fp32_to_int_quantizer/
├── core/               # 核心量化逻辑与校准策略
├── frameworks/         # 可选的 PyTorch/TensorFlow 集成
├── serialization/      # 二进制数据处理 (INT4 紧凑存储)
├── visualization/      # 绘图与指标计算
└── utils/              # 工具函数
```

### 安装

```bash
python setup.py install
pip install fp32-to-int-quantizer
```

### 快速开始

```python
from fp32_to_int_quantizer import FP32ToLowBitQuantizer, generate_batch_test_data

# 配置（INT4/INT8共用）
config = {
    "quant_mode": "symmetric",
    "quant_level": "per_channel",
    "calib_mode": "entropy",
    "entropy_bins": 2048,
    "seed": 42
}

# 生成测试数据
batch_data = generate_batch_test_data(
    batch_size=4,
    data_shape=(3, 32, 32),
    data_range=(0.0, 1.0),
    is_image_like=True
)

# INT8量化
int8_quantizer = FP32ToLowBitQuantizer(quant_bit=8, **config)
int8_results = int8_quantizer.quantize(batch_data)

# INT4量化
int4_quantizer = FP32ToLowBitQuantizer(quant_bit=4, **config)
int4_results = int4_quantizer.quantize(batch_data)

# 可视化对比
int4_quantizer.visualize_analysis(
    fp32_original=batch_data[0],
    fp32_recovered=int4_results[0][1],
    save_path="comparison.png",
    ref_metrics={"quant_bit": 8, "scales": int8_quantizer.scales, "zero_points": int8_quantizer.zero_points}
)
```

### 模型量化 (PyTorch 示例)

```python
import torch
from fp32_to_int_quantizer import FP32ToLowBitQuantizer

# 1. 准备模型与虚拟输入
model = ... # 您的 PyTorch 模型
dummy_input = torch.randn(1, 3, 224, 224)

# 2. 初始化量化器 (INT4)
quantizer = FP32ToLowBitQuantizer(
    quant_bit=4,
    quant_mode="symmetric",
    quant_level="per_channel",
    calib_mode="entropy"
)

# 3. 量化权重
quantized_model, layer_params = quantizer.quantize_torch_model(model, dummy_input)
print(f"Quantized {len(layer_params)} layers.")
```

### 文档

详细API文档请参阅 [docs/API.md](docs/API.md)。

### 许可证

本项目采用 Apache License 2.0 许可证 - 详见 [LICENSE](LICENSE) 文件。
