# Changelog / 更新日志

## [1.0.0] - 2024-01-22 / 2024年1月22日

### Added / 新增
- Initial release / 首次发布
- INT8 and INT4 quantization support / INT8和INT4量化支持
- Symmetric and asymmetric quantization modes / 对称和非对称量化模式
- Per-tensor and per-channel quantization granularity / 逐张量和逐通道量化粒度
- Multiple calibration strategies (extremes, percentile, entropy) / 多种校准策略（极值、分位数、熵）
- PyTorch and TensorFlow framework integration / PyTorch和TensorFlow框架对接
- Compact binary storage for INT4 data / INT4数据紧凑二进制存储
- Comprehensive precision evaluation metrics / 完善的精度评估指标
- Multi-precision comparison visualization / 多精度对比可视化
- Batch processing support / 批量处理支持
- Complete unit tests / 完整单元测试

### Fixed / 修复
- Fixed font issues in visualization (all labels in English) / 修复可视化中的字体问题（所有标签改为英文）
- Improved error handling and validation / 改进错误处理和验证
- Fixed type annotations for better IDE support / 修复类型注解以获得更好的IDE支持

### Changed / 更改
- Modular project structure for better maintainability / 模块化项目结构以提高可维护性
- Separated calibration logic from quantization core / 将校准逻辑与量化核心分离
- Refactored visualization into standalone module / 将可视化重构为独立模块

### Removed / 移除
- None / 无
