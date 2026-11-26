# 验证记录

本机 Windows CPU 的 pytest 7.1.2 使用 Python 3.8.10、PyTorch 1.10.2+cpu、NumPy 1.21.4 和 scikit-learn 1.1.2。最终测试为 **9 passed**，交付复跑耗时 16.25 秒。三个 MLP 告警来自合成配置仅运行 5 次迭代。另一次 unittest 独立执行同一套 9 个测试通过。

- ID 对齐、输入哈希核对，以及基因型、环境、地点、LOEO 划分和内层 OOF 成员断言通过。
- 折内均值、填补、按方差选列、PCA、缺失标签过滤、常数与样本不足指标验证通过。
- 外层标签及其它外层环境扰动没有改变训练权重或训练预处理器。固定外测查询的预测在合理容差内保持一致。度量可以由保存的 CSV 重算。
- 历史模型组件固定权重等价性和单-key 退化已独立测量。修正版注意力的形状、掩码、归一化、Q/K 敏感性和梯度，以及超图数值测试通过。
- 合成跑通主实验、四个可用基线、五个消融对照、两个基因组 CNN 重建、STL/MTL/HyperGEP、K 参数、LOEO、归因和 S6 遮蔽。检查点保存、加载、复用、模拟失败后恢复和配对转移已验证。
- 正式隔离之外的真实 G2F 摘录完成 CPU 随机记录级单折，以及一个完整环境留出折。真实 LOEO 检查点回放核对 16 个留出记录和 3 个性状。
- Git Bash 在本机执行了 Code Ocean demo 入口和汇总。缺少正式 bundle 时，默认 reproduce 模式写出 BLOCKED 状态并退出 2。
- 另一个 Anaconda 运行时上的重复 pytest 因长时间卡住终止。上面的通过结果来自可运行的本机 CPU 环境。CI 所列 Python 3.11/PyTorch 2.5.1、Docker、GPU 和实际 Capsule 无本次运行证据。可选依赖安装因 PyPI TLS 连接中断，XGBoost、LightGBM 和 TabNet 未跑通。

测试原始日志见 artifacts/pytest_delivery.log、artifacts/unittest_final.log。正式科学验收结果见 RELEASE_MANIFEST.json。
