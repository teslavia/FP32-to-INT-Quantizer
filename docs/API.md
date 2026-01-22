# API Documentation / API文档

## FP32ToLowBitQuantizer Class

### Constructor Parameters / 构造函数参数

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| quant_bit | int | Quantization precision: 4 or 8 | 8 |
| quant_mode | str | "symmetric" or "asymmetric" | "symmetric" |
| quant_level | str | "per_tensor" or "per_channel" | "per_tensor" |
| calib_mode | str | Calibration mode: None/"percentile"/"entropy" | None |
| calib_percentile | float | Percentile for percentile calibration | 99.9 |
| entropy_bins | int | Bins for entropy calibration | 2048 |
| int_min | int | Custom quantization range min | None (auto) |
| int_max | int | Custom quantization range max | None (auto) |
| seed | int | Random seed for reproducibility | 42 |

### Methods / 方法

#### quantize()

```python
quantize(fp32_data) -> Union[Tuple[np.ndarray, np.ndarray], List[Tuple[np.ndarray, np.ndarray]]]
```

Core quantization interface. Supports single np.ndarray or batch List/Tuple[np.ndarray].

#### evaluate_precision()

```python
evaluate_precision(fp32_original, fp32_recovered, sig_analysis=True) -> dict
```

Calculate precision metrics. Returns basic metrics + statistical significance metrics when sig_analysis=True.

#### visualize_analysis()

```python
visualize_analysis(fp32_original, fp32_recovered, save_path=None, ref_metrics=None, sig_analysis=True)
```

Generate visualization report with all English labels.

#### save_quant_data() / load_quant_data()

```python
save_quant_data(quant_data, save_path: str) -> None
load_quant_data(load_path: str, target_shape: Tuple[int, ...]) -> np.ndarray
```

Save/load quantized data. INT4 uses compact storage (1 byte = 2 values).

#### save_quant_params() / load_quant_params()

```python
save_quant_params(save_path: str) -> None
load_quant_params(load_path: str) -> None
```

Save/load quantization parameters in .npy format.

#### quantize_torch_tensor() / quantize_torch_model()

```python
quantize_torch_tensor(torch_tensor) -> Tuple[torch.Tensor, torch.Tensor]
quantize_torch_model(model, dummy_input, device=None) -> Tuple[torch.nn.Module, dict]
```

PyTorch tensor and model quantization.

## Utility Functions / 工具函数

### generate_test_data()

```python
generate_test_data(shape, data_range=(0.0, 1.0), is_image_like=False) -> np.ndarray
```

Generate single FP32 test data.

### generate_batch_test_data()

```python
generate_batch_test_data(batch_size, data_shape, data_range=(0.0, 1.0), is_image_like=False) -> List[np.ndarray]
```

Generate batch FP32 test data.
