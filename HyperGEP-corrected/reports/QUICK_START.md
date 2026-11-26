# 快速开始

在修正版根目录执行以下命令。已有合成和真实数据短跑的目录见 FINAL_AUDIT.md。

```powershell
python -m experiments.run_main --dataset demo --fold 0 --output-dir artifacts/demo/new_run
python -m experiments.run_main --dataset maizegep --data-dir data --dry-run
python -m experiments.run_negative_transfer --dataset g2f --fold 0 --variant MTL --data-dir data --device cpu --resume
python -m scripts.verify_release
```

补跑时使用相同配置、数据、代码、设备、fold 和 seed，并带上 `--resume`。基于模型训练的检查点按配置定期写入，恢复到最近一次成功保存的 epoch。已完成任务通过 SHA256 核对后直接复用。正式数据准备、单折资源估算与全部运行命令见 README.md 和 docs/EXPERIMENTS.md。
