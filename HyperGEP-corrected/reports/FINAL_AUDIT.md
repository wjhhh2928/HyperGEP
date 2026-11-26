# HyperGEP 重构与复现审计

**修正版是可执行的研究候选。正式论文发布状态为 BLOCKED。**

保留源目录中的原始代码、数据、结果和权重。私有快照位于 artifacts/legacy，SOURCE_INVENTORY.tsv 记录 326 项来源 SHA256，补充清单 SOURCE_INVENTORY_SUPPLEMENT.tsv 另记录 106 项。部分大规模相邻数据和权重原地保留并记录散列。公开 HEAD 已查到，本地两次 clone 被连接重置。固定输入与共享权重下，本地原始脚本和拆分版的九个组件均实现数值等价。该验证覆盖模型组件和工具，训练协议有实质差异。

修正版在 hypergep/ 提供 ID 对齐、折内填补、标准化、PCA/筛选、标签掩码、归纳环境关联图、基因型 token 交叉注意力、分支和 OOF Meta-Fusion。在 experiments/ 提供统一入口。checkpoint、数据和代码配置散列、OOF 谱系、真实缺失掩码以及测试预测保存在独立的运行目录。早期耦合、STL、DeepGS 和 DNNGP 是重建候选。新的网络与评估协议须由论文作者审核后冻结。

正式测试矩阵见 EXPERIMENT_STATUS.tsv。合成数据、基线、消融、迁移、K 敏感性、LOEO 和归因入口接受单折测试。G2F 真实短跑从本地候选原始表按明确 ID 提取 160 条记录、10 个环境、143 个基因型和 64 个 SNP。使用 CPU 和短跑预算，完整训练了一个外层测试折。还完成一次真实 G2F 留一环境单折和 16 条记录、三个性状的权重推理回放。结果位于 artifacts/real_smoke。正式研究队列、资源和训练指标未锁定。

论文 Fig. 2a 的写法与补充数据 1 的 MaizeGEP 产量汇总存在一处明显数值差异。历史负迁移 fold CSV 可以重算汇总，但模型、标签和划分谱系仍未验证。补充数据 2 无法用标准 XLSX 阅读器打开。稿件更新建议见 MANUSCRIPT_UPDATE_REQUIRED.md。发布前必须处理 RELEASE_BLOCKERS.md 内列出的问题。

修正版代码、配置、测试和文档是待审核的 GitHub 候选。私有产物、源表清单和数据权重通过忽略规则隔离。Code Ocean 资产区仅应放入许可已核实且带散列的正式数据输入、预测结果或权重。本机 Git Bash 已验证 demo 及正式资产缺失时的阻塞路径。实际 Capsule 和 Docker 尚未验证。执行、补跑与恢复见 QUICK_START.md、README.md 和 docs/EXPERIMENTS.md。
