## 2026-04-18 12:07:02 | 代码 | PE-Anchor3DLane++ 初始脚手架
- 摘要：新增profile-aware anchor subclass, endpoint-aware loss, V1 execution plan, and experiment journal tooling.
- 标签：v1, code
- 模块改动 1：主要改动：新增profile-aware anchor subclass, endpoint-aware loss, V1 execution plan, and experiment journal tooling.
- 模块改动 2：涉及模块：车道检测器模块、损失函数模块、OpenLane v2 配置、V1 执行规划、实验记录工具
- 涉及文件：mmseg/models/lane_detector/anchor_3dlane_pe.py, mmseg/models/losses/lane_loss_pe.py, ../configs_v2/openlane/pe_anchor3dlanepp_r18.py, V1_EXECUTION_PLAN.md, tools/research_journal.py
- Git 快照：`main` @ `a98f7246c05a`
## 2026-04-18 14:19:27 | 代码 | Baseline 复现与环境盘点
- 摘要：盘点local baseline result files, identify missing reproductions, and pin the recommended conda env and working directory.
- 标签：audit, baseline
- 模块改动 1：主要改动：盘点local baseline result files, identify missing reproductions, and pin the recommended conda env and working directory.
- 模块改动 2：涉及模块：baseline 复现台账
- 涉及文件：BASELINE_REPRO_STATUS.md
- Git 快照：`main` @ `a98f7246c05a`
## 2026-04-18 16:36:50 | 代码 | Anchor3DLane++ baseline 清理与复现说明
- 摘要：清理the stray repo-root mmseg, delete non-Anchor3DLane++ local eval outputs, and rewrite the baseline ledger around OpenLane and ApolloSim Anchor3DLane++ reproduction.
- 标签：baseline, cleanup, anchor3dlanepp
- 模块改动 1：主要改动：清理the stray repo-root mmseg, delete non-Anchor3DLane++ local eval outputs, and rewrite the baseline ledger around OpenLane and ApolloSim Anchor3DLane++ reproduction.
- 模块改动 2：涉及模块：baseline 复现台账
- 涉及文件：BASELINE_REPRO_STATUS.md
- Git 快照：`main` @ `a98f7246c05a`
## 2026-04-18 17:23:46 | 实验 | apollosim-anchor3dlane-eval
- 摘要：评测官方 ApolloSim Anchor3DLane checkpoint.
- 标签：baseline, apollosim, eval
- 关键进展 1：评测官方 ApolloSim Anchor3DLane checkpoint.
- 关键进展 2：输出目录：output/eval_apollosim_anchor3dlane
- 涉及文件：configs/apollosim/anchor3dlane.py, pretrained/apollo_anchor3dlane.pth
- 执行命令：`/ssd-data3/ztc2025/.conda/envs/anchor3dlane-cu121/bin/python tools/test.py configs/apollosim/anchor3dlane.py pretrained/apollo_anchor3dlane.pth --show-dir output/eval_apollosim_anchor3dlane`
- 工作目录：`output/eval_apollosim_anchor3dlane`
- 日志文件：`/ssd-data3/ztc2025/Anchor3DLanePM/Anchor3dLane/experiments/research_journal/logs/20260418_172152_apollosim-anchor3dlane-eval.log`
- 退出码：0
- Git 快照：`main` @ `a98f7246c05a`
## 2026-04-18 17:25:36 | 代码 | 官方 baseline 批量运行脚本
- 摘要：新增a baseline runner that checks dataset assets, skips blocked variants, and records official evaluations through the research journal.
- 标签：baseline, tooling
- 模块改动 1：主要改动：新增a baseline runner that checks dataset assets, skips blocked variants, and records official evaluations through the research journal.
- 模块改动 2：涉及模块：baseline 批量运行工具
- 涉及文件：tools/run_official_baselines.py
- Git 快照：`main` @ `a98f7246c05a`
## 2026-04-18 17:26:19 | 实验 | apollosim-anchor3dlane-iter-eval
- 摘要：评测官方 baseline apollosim-anchor3dlane-iter-eval.
- 标签：baseline, apollosim, eval, official
- 关键进展 1：评测官方 baseline apollosim-anchor3dlane-iter-eval.
- 关键进展 2：输出目录：output/eval_apollosim_anchor3dlane_iter
- 涉及文件：configs/apollosim/anchor3dlane_iter.py, pretrained/apollo_anchor3dlane_iter.pth
- 执行命令：`/ssd-data3/ztc2025/.conda/envs/anchor3dlane-cu121/bin/python tools/test.py configs/apollosim/anchor3dlane_iter.py pretrained/apollo_anchor3dlane_iter.pth --show-dir output/eval_apollosim_anchor3dlane_iter`
- 工作目录：`output/eval_apollosim_anchor3dlane_iter`
- 日志文件：`/ssd-data3/ztc2025/Anchor3DLanePM/Anchor3dLane/experiments/research_journal/logs/20260418_172500_apollosim-anchor3dlane-iter-eval.log`
- 退出码：0
- Git 快照：`main` @ `a98f7246c05a`
## 2026-04-18 17:39:12 | 实验 | openlane-v12-anchor3dlane-eval
- 摘要：评测官方 baseline openlane-v12-anchor3dlane-eval.
- 标签：baseline, openlane, v1.2, eval, official
- 关键进展 1：评测官方 baseline openlane-v12-anchor3dlane-eval.
- 关键进展 2：输出目录：output/eval_openlanev2_anchor3dlane
- 涉及文件：configs/openlane/anchor3dlane.py, pretrained/openlanev2_anchor3dlane.pth
- 执行命令：`/ssd-data3/ztc2025/.conda/envs/anchor3dlane-cu121/bin/python tools/test.py configs/openlane/anchor3dlane.py pretrained/openlanev2_anchor3dlane.pth --show-dir output/eval_openlanev2_anchor3dlane`
- 工作目录：`output/eval_openlanev2_anchor3dlane`
- 日志文件：`/ssd-data3/ztc2025/Anchor3DLanePM/Anchor3dLane/experiments/research_journal/logs/20260418_172813_openlane-v12-anchor3dlane-eval.log`
- 退出码：0
- Git 快照：`main` @ `a98f7246c05a`
## 2026-04-18 17:39:15 | 实验 | openlane-v11-anchor3dlane-eval
- 摘要：评测官方 baseline openlane-v11-anchor3dlane-eval.
- 标签：baseline, openlane, v1.1, eval, official
- 关键进展 1：评测官方 baseline openlane-v11-anchor3dlane-eval.
- 关键进展 2：输出目录：output/eval_openlane_v11_anchor3dlane
- 涉及文件：configs/openlane/anchor3dlane.py, pretrained/openlane_anchor3dlane.pth
- 执行命令：`/ssd-data3/ztc2025/.conda/envs/anchor3dlane-cu121/bin/python tools/test.py configs/openlane/anchor3dlane.py pretrained/openlane_anchor3dlane.pth --show-dir output/eval_openlane_v11_anchor3dlane`
- 工作目录：`output/eval_openlane_v11_anchor3dlane`
- 日志文件：`/ssd-data3/ztc2025/Anchor3DLanePM/Anchor3dLane/experiments/research_journal/logs/20260418_172652_openlane-v11-anchor3dlane-eval.log`
- 退出码：0
- Git 快照：`main` @ `a98f7246c05a`
## 2026-04-18 17:39:20 | 实验 | openlane-v11-anchor3dlane-effb3-eval
- 摘要：评测官方 baseline openlane-v11-anchor3dlane-effb3-eval.
- 标签：baseline, openlane, v1.1, eval, official
- 关键进展 1：评测官方 baseline openlane-v11-anchor3dlane-effb3-eval.
- 关键进展 2：输出目录：output/eval_openlane_v11_anchor3dlane_effb3
- 涉及文件：configs/openlane/anchor3dlane_effb3.py, pretrained/openlane_anchor3dlane_effb3.pth
- 执行命令：`/ssd-data3/ztc2025/.conda/envs/anchor3dlane-cu121/bin/python tools/test.py configs/openlane/anchor3dlane_effb3.py pretrained/openlane_anchor3dlane_effb3.pth --show-dir output/eval_openlane_v11_anchor3dlane_effb3`
- 工作目录：`output/eval_openlane_v11_anchor3dlane_effb3`
- 日志文件：`/ssd-data3/ztc2025/Anchor3DLanePM/Anchor3dLane/experiments/research_journal/logs/20260418_172813_openlane-v11-anchor3dlane-effb3-eval.log`
- 退出码：0
- Git 快照：`main` @ `a98f7246c05a`
## 2026-04-18 17:49:25 | 实验 | openlane-v12-anchor3dlane-iter-eval
- 摘要：评测官方 baseline openlane-v12-anchor3dlane-iter-eval.
- 标签：baseline, openlane, v1.2, eval, official
- 关键进展 1：评测官方 baseline openlane-v12-anchor3dlane-iter-eval.
- 关键进展 2：输出目录：output/eval_openlanev2_anchor3dlane_iter
- 涉及文件：configs/openlane/anchor3dlane_iter.py, pretrained/openlanev2_anchor3dlane_iter.pth
- 执行命令：`/ssd-data3/ztc2025/.conda/envs/anchor3dlane-cu121/bin/python tools/test.py configs/openlane/anchor3dlane_iter.py pretrained/openlanev2_anchor3dlane_iter.pth --show-dir output/eval_openlanev2_anchor3dlane_iter`
- 工作目录：`output/eval_openlanev2_anchor3dlane_iter`
- 日志文件：`/ssd-data3/ztc2025/Anchor3DLanePM/Anchor3dLane/experiments/research_journal/logs/20260418_173912_openlane-v12-anchor3dlane-iter-eval.log`
- 退出码：0
- Git 快照：`main` @ `a98f7246c05a`
## 2026-04-18 17:49:29 | 实验 | openlane-v12-anchor3dlane-iter-r50x2-eval
- 摘要：评测官方 baseline openlane-v12-anchor3dlane-iter-r50x2-eval.
- 标签：baseline, openlane, v1.2, eval, official
- 关键进展 1：评测官方 baseline openlane-v12-anchor3dlane-iter-r50x2-eval.
- 关键进展 2：本次运行退出码为 1，请结合日志继续排查。
- 涉及文件：configs/openlane/anchor3dlane_iter_r50.py, pretrained/openlanev2_anchor3dlane_iter_r50x2.pth
- 执行命令：`/ssd-data3/ztc2025/.conda/envs/anchor3dlane-cu121/bin/python tools/test.py configs/openlane/anchor3dlane_iter_r50.py pretrained/openlanev2_anchor3dlane_iter_r50x2.pth --show-dir output/eval_openlanev2_anchor3dlane_iter_r50x2`
- 工作目录：`output/eval_openlanev2_anchor3dlane_iter_r50x2`
- 日志文件：`/ssd-data3/ztc2025/Anchor3DLanePM/Anchor3dLane/experiments/research_journal/logs/20260418_174925_openlane-v12-anchor3dlane-iter-r50x2-eval.log`
- 退出码：1
- Git 快照：`main` @ `a98f7246c05a`
## 2026-04-18 17:49:44 | 实验 | openlane-v11-anchor3dlane-iter-eval
- 摘要：评测官方 baseline openlane-v11-anchor3dlane-iter-eval.
- 标签：baseline, openlane, v1.1, eval, official
- 关键进展 1：评测官方 baseline openlane-v11-anchor3dlane-iter-eval.
- 关键进展 2：输出目录：output/eval_openlane_v11_anchor3dlane_iter
- 涉及文件：configs/openlane/anchor3dlane_iter.py, pretrained/openlane_anchor3dlane_iter.pth
- 执行命令：`/ssd-data3/ztc2025/.conda/envs/anchor3dlane-cu121/bin/python tools/test.py configs/openlane/anchor3dlane_iter.py pretrained/openlane_anchor3dlane_iter.pth --show-dir output/eval_openlane_v11_anchor3dlane_iter`
- 工作目录：`output/eval_openlane_v11_anchor3dlane_iter`
- 日志文件：`/ssd-data3/ztc2025/Anchor3DLanePM/Anchor3dLane/experiments/research_journal/logs/20260418_173920_openlane-v11-anchor3dlane-iter-eval.log`
- 退出码：0
- Git 快照：`main` @ `a98f7246c05a`
## 2026-04-18 17:50:12 | 实验 | openlane-v11-anchor3dlane-temporal-iter-eval
- 摘要：评测官方 baseline openlane-v11-anchor3dlane-temporal-iter-eval.
- 标签：baseline, openlane, v1.1, eval, official
- 关键进展 1：评测官方 baseline openlane-v11-anchor3dlane-temporal-iter-eval.
- 关键进展 2：本次运行退出码为 1，请结合日志继续排查。
- 涉及文件：configs/openlane/anchor3dlane_mf_iter.py, pretrained/openlane_anchor3dlane_temporal_iter.pth
- 执行命令：`/ssd-data3/ztc2025/.conda/envs/anchor3dlane-cu121/bin/python tools/test.py configs/openlane/anchor3dlane_mf_iter.py pretrained/openlane_anchor3dlane_temporal_iter.pth --show-dir output/eval_openlane_v11_anchor3dlane_temporal_iter`
- 工作目录：`output/eval_openlane_v11_anchor3dlane_temporal_iter`
- 日志文件：`/ssd-data3/ztc2025/Anchor3DLanePM/Anchor3dLane/experiments/research_journal/logs/20260418_174944_openlane-v11-anchor3dlane-temporal-iter-eval.log`
- 退出码：1
- Git 快照：`main` @ `a98f7246c05a`
## 2026-04-18 17:51:19 | 实验 | openlane-v11-anchor3dlane-effb3-eval
- 摘要：评测官方 baseline openlane-v11-anchor3dlane-effb3-eval.
- 标签：baseline, openlane, v1.1, eval, official
- 关键进展 1：评测官方 baseline openlane-v11-anchor3dlane-effb3-eval.
- 关键进展 2：输出目录：output/eval_openlane_v11_anchor3dlane_effb3
- 涉及文件：configs/openlane/anchor3dlane_effb3.py, pretrained/openlane_anchor3dlane_effb3.pth
- 执行命令：`/ssd-data3/ztc2025/.conda/envs/anchor3dlane-cu121/bin/python tools/test.py configs/openlane/anchor3dlane_effb3.py pretrained/openlane_anchor3dlane_effb3.pth --show-dir output/eval_openlane_v11_anchor3dlane_effb3`
- 工作目录：`output/eval_openlane_v11_anchor3dlane_effb3`
- 日志文件：`/ssd-data3/ztc2025/Anchor3DLanePM/Anchor3dLane/experiments/research_journal/logs/20260418_173915_openlane-v11-anchor3dlane-effb3-eval.log`
- 退出码：0
- Git 快照：`main` @ `a98f7246c05a`
## 2026-04-25 10:56:27 | 代码 | 训练脚本 yapf 兼容补丁
- 摘要：新增a safe config dump fallback in train.py and train_dist.py so official training can start even when the environment's yapf lacks FormatCode(verify=...).
- 标签：infra, training, repro
- 模块改动 1：主要改动：新增a safe config dump fallback in train.py and train_dist.py so official training can start even when the environment's yapf lacks FormatCode(verify=...).
- 模块改动 2：涉及模块：训练入口、分布式训练入口
- 涉及文件：tools/train.py, tools/train_dist.py
- Git 快照：`main` @ `a98f7246c05a`
## 2026-04-25 10:58:49 | 实验 | openlane-anchor3dlanepp-r18-official-train-launch
- 摘要：启动官方 OpenLane v1.2 Anchor3DLane++ R18 reproduction training on GPUs 0 and 1.
- 标签：baseline, openlane, train, official, anchor3dlanepp
- 关键进展 1：启动官方 OpenLane v1.2 Anchor3DLane++ R18 reproduction training on GPUs 0 and 1.
- 关键进展 2：输出目录：output/repro/openlane_anchor3dlanepp_r18_official_20260425
- 执行命令：`/bin/bash -lc 'source /home/ztc2025/anaconda3/etc/profile.d/conda.sh && conda activate anchor3dlane-cu121 && export PYTHONPATH=$PYTHONPATH:/ssd-data3/ztc2025/Anchor3DLanePM/Anchor3dLane/gen-efficientnet-pytorch && mkdir -p output/repro/openlane_anchor3dlanepp_r18_official_20260425 && CUDA_VISIBLE_DEVICES=0,1 PORT=29601 nohup bash tools/dist_train.sh ../configs_v2/openlane/anchor3dlane++_r18.py 2 --work-dir output/repro/openlane_anchor3dlanepp_r18_official_20260425 > output/repro/openlane_anchor3dlanepp_r18_official_20260425/launcher.log 2>&1 & echo TRAIN_PID:$!'`
- 工作目录：`output/repro/openlane_anchor3dlanepp_r18_official_20260425`
- 日志文件：`/ssd-data3/ztc2025/Anchor3DLanePM/Anchor3dLane/experiments/research_journal/logs/20260425_105840_openlane-anchor3dlanepp-r18-official-train-launch.log`
- 退出码：0
- Git 快照：`main` @ `a98f7246c05a`
## 2026-04-25 11:00:02 | 代码 | 分布式启动参数兼容修复
- 摘要：更新train/test entrypoints to accept both --local_rank and --local-rank so PyTorch 2.x distributed launch works with the official scripts.
- 标签：infra, training, distributed, repro
- 模块改动 1：主要改动：更新train/test entrypoints to accept both --local_rank and --local-rank so PyTorch 2.x distributed launch works with the official scripts.
- 模块改动 2：涉及模块：训练入口、分布式训练入口、评测入口、部署评测入口
- 涉及文件：tools/train.py, tools/train_dist.py, tools/test.py, tools/deploy_test.py
- Git 快照：`main` @ `a98f7246c05a`
## 2026-04-25 11:20:18 | 实验 | openlane-anchor3dlanepp-r18-official-eval-launch
- 摘要：启动官方 OpenLane v1.2 Anchor3DLane++ R18 checkpoint evaluation on GPU 2.
- 标签：baseline, openlane, eval, official, anchor3dlanepp
- 关键进展 1：启动官方 OpenLane v1.2 Anchor3DLane++ R18 checkpoint evaluation on GPU 2.
- 关键进展 2：输出目录：output/eval_openlane_anchor3dlanepp_r18
- 执行命令：`/bin/bash -lc 'source /home/ztc2025/anaconda3/etc/profile.d/conda.sh && conda activate anchor3dlane-cu121 && export PYTHONPATH=$PYTHONPATH:/ssd-data3/ztc2025/Anchor3DLanePM/Anchor3dLane/gen-efficientnet-pytorch && mkdir -p output/eval_openlane_anchor3dlanepp_r18 && CUDA_VISIBLE_DEVICES=2 nohup python -u tools/test.py ../configs_v2/openlane/anchor3dlane++_r18.py pretrained/openlane_anchor3dlane++_r18.pth --show-dir output/eval_openlane_anchor3dlanepp_r18 > output/eval_openlane_anchor3dlanepp_r18/launcher.log 2>&1 & echo EVAL_PID:$!'`
- 工作目录：`output/eval_openlane_anchor3dlanepp_r18`
- 日志文件：`/ssd-data3/ztc2025/Anchor3DLanePM/Anchor3dLane/experiments/research_journal/logs/20260425_105840_openlane-anchor3dlanepp-r18-official-eval-launch.log`
- 退出码：0
- Git 快照：`main` @ `a98f7246c05a`
## 2026-04-25 12:03:40 | 代码 | 实验日志中文化与模块化记录
- 摘要：将 journal.md 改为中文渲染，补充按模块记录的自然语言说明，并加入 Markdown 重建命令与本地 skill 约束。
- 模块改动 1：实验记录工具：重写 Markdown 渲染逻辑，journal.md 现在默认输出中文标签、中文摘要和更精简的 Git 快照。
- 模块改动 2：代码改动记录：新增 --module-note 字段，要求按模块解释改了什么，而不是只看文件或行数统计。
- 模块改动 3：实验进度记录：新增 --progress-note 与中文标题/摘要字段，便于记录关键实验节点和阶段性结论。
- 模块改动 4：工作流固化：新增本地 skill，约束后续在 Anchor3dLane 项目里统一使用中文 journal 和模块化自然语言总结。
- 涉及文件：tools/research_journal.py
- Git 快照：`main` @ `a98f7246c05a`
## 2026-04-26 17:39:16 | 实验 | openlane-anchor3dlanepp-r18-iter5000-eval
- 摘要：Evaluate current official OpenLane v1.2 Anchor3DLane++ R18 training checkpoint at iter 5000.
- 标签：baseline, openlane, eval, checkpoint, anchor3dlanepp
- 关键进展 1：Evaluate current official OpenLane v1.2 Anchor3DLane++ R18 training checkpoint at iter 5000.
- 关键进展 2：输出目录：output/eval_openlane_anchor3dlanepp_r18_iter5000
- 执行命令：`python tools/test.py ../configs_v2/openlane/anchor3dlane++_r18.py output/repro/openlane_anchor3dlanepp_r18_official_20260425/iter_5000.pth --show-dir output/eval_openlane_anchor3dlanepp_r18_iter5000`
- 工作目录：`output/eval_openlane_anchor3dlanepp_r18_iter5000`
- 日志文件：`/ssd-data3/ztc2025/Anchor3DLanePM/Anchor3dLane/experiments/research_journal/logs/20260426_172857_openlane-anchor3dlanepp-r18-iter5000-eval.log`
- 退出码：0
- Git 快照：`main` @ `a98f7246c05a`
## 2026-04-26 18:38:40 | 代码 | BundleLane方法主线重写
- 摘要：围绕BundleLane的局部车道束内禀表示，重写V1方法设计文档与代码实现草案。
- 模块改动 1：方法设计：放弃profile+endpoint的补丁式主线，改为scene-conditioned local lane-bundle frame与frame-relative decoding的统一表述。
- 模块改动 2：工程落地：新增基于Anchor3DLane++代码骨架的实现草案，明确模型子类、intrinsic matcher、loss拆分、配置与开发顺序。
- 涉及文件：V1_METHOD_DESIGN.md, V1_EXECUTION_PLAN.md
- Git 快照：`main` @ `a98f7246c05a`
## 2026-04-26 18:43:55 | 实验 | 官方 OpenLane Anchor3DLane++ R18 断点续训
- 摘要：按官方论文配置，从 iter_5000 断点继续训练 OpenLane v1.2 Anchor3DLane++ R18。
- 标签：baseline, openlane, train, official, resume, anchor3dlanepp
- 关键进展 1：已从 output/repro/openlane_anchor3dlanepp_r18_official_20260425/iter_5000.pth 恢复，继续使用官方 configs_v2/openlane/anchor3dlane++_r18.py 和原 work_dir。
- 关键进展 2：当前已进入 5000+ 迭代，最新日志显示到 Iter [5210/60000]，loss 约 1.6809，双卡 RTX 4090 正在占用。
- 涉及文件：../configs_v2/openlane/anchor3dlane++_r18.py, output/repro/openlane_anchor3dlanepp_r18_official_20260425/iter_5000.pth, output/repro/openlane_anchor3dlanepp_r18_official_20260425/20260426_184119.log, output/repro/openlane_anchor3dlanepp_r18_official_20260425/launcher_resume.log
- 执行命令：`/bin/true`
- 工作目录：`output/repro/openlane_anchor3dlanepp_r18_official_20260425`
- 日志文件：`/ssd-data3/ztc2025/Anchor3DLanePM/Anchor3dLane/experiments/research_journal/logs/20260426_184355_openlane-anchor3dlanepp-r18-official-train-resume.log`
- 退出码：0
- Git 快照：`main` @ `a98f7246c05a`
## 2026-04-26 18:53:23 | 代码 | BundleLane术语说明补充
- 摘要：在BundleLane方法文档中新增术语说明小节，并对关键专业术语补充行内解释。
- 模块改动 1：文档可读性：新增术语表，解释absolute curve regression、Frenet、gauge freedom、nuisance factor、local chart、intrinsic space等概念，降低跨方向阅读门槛。
- 模块改动 2：表述修订：在方法描述中将support interval、intrinsic space等关键术语与正文显式对应，避免术语只出现不解释。
- 涉及文件：V1_METHOD_DESIGN.md
- Git 快照：`main` @ `a98f7246c05a`
## 2026-04-26 19:37:25 | 代码 | BundleLane精简版训练与算量说明
- 摘要：将BundleLane第一版训练策略收敛到先复用原始loss和matcher，并补充方法的收敛性与计算量取舍说明。
- 模块改动 1：方法设计：将V1与V1.1明确分层，V1仅保留base absolute loss、frame loss、span loss和residual_small约束，暂不启用intrinsic matcher、intrinsic vis和结构loss。
- 模块改动 2：工程实现：执行稿改为先复用LaneLossV2与HungarianMatcher，只在detector loss中增量接入少量bundle专属监督，推迟专用loss类与matcher文件到V1.1。
- 模块改动 3：复杂度分析：补充相对Anchor3DLane++的前向开销与训练开销来源说明，强调新增成本主要来自target building而非模型主体。
- 涉及文件：V1_METHOD_DESIGN.md, V1_EXECUTION_PLAN.md
- Git 快照：`main` @ `a98f7246c05a`
## 2026-04-26 23:57:33 | 代码 | BundleLane首轮frame骨架落地
- 摘要：基于Anchor3DLane++实现BundleLane首轮代码骨架，包含bundle frame预测、frame-conditioned anchor注入与frame loss。
- 模块改动 1：模型骨架：新增BundleLaneDetector、BundleFrameHead与BundleAnchorGenerator，以子类形式复用Anchor3DLane++主链，并在stage0支持bundle frame注入。
- 模块改动 2：监督接入：在不改原始LaneLossV2与HungarianMatcher的前提下，为detector新增bundle frame target构造与frame loss计算，保持absolute proposal训练链路稳定。
- 模块改动 3：配置与注册：新增bundlelane_r18配置并注册BundleLaneDetector，完成首轮可构建实验入口。
- 模块改动 4：基础验证：使用anchor3dlane-cu121环境验证新配置可成功build模型，并用伪造GT样本跑通bundle frame target与frame loss。
- 涉及文件：mmseg/models/lane_detector/anchor_3dlane_bundle.py, mmseg/models/lane_detector/__init__.py, ../configs_v2/openlane/bundlelane_r18.py
- Git 快照：`main` @ `caf839884b9a`
