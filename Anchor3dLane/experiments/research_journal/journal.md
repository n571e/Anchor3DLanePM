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
## 2026-04-27 15:43:36 | 实验 | OpenLane Anchor3DLane++ R18 iter_20000 评测启动
- 摘要：在 GPU2 上启动官方 OpenLane v1.2 Anchor3DLane++ R18 的 iter_20000 checkpoint 评测。
- 标签：baseline, openlane, eval, checkpoint, anchor3dlanepp
- 关键进展 1：使用 setsid + nohup 将评测进程彻底脱离当前终端会话，减少因会话退出导致的中断风险。
- 关键进展 2：输出目录为 output/eval_openlane_anchor3dlanepp_r18_iter20000，待 evaluation_result.json 生成后再判断是否继续训练。
- 涉及文件：../configs_v2/openlane/anchor3dlane++_r18.py, output/repro/openlane_anchor3dlanepp_r18_official_20260425/iter_20000.pth, output/eval_openlane_anchor3dlanepp_r18_iter20000/launcher.log
- 执行命令：`/bin/true`
- 工作目录：`output/eval_openlane_anchor3dlanepp_r18_iter20000`
- 日志文件：`/ssd-data3/ztc2025/Anchor3DLanePM/Anchor3dLane/experiments/research_journal/logs/20260427_154336_openlane-anchor3dlanepp-r18-iter20000-eval-launch.log`
- 退出码：0
- Git 快照：`main` @ `ff401181c5a1`
## 2026-04-27 15:55:05 | 实验 | OpenLane Anchor3DLane++ R18 iter_20000 评测结果
- 摘要：记录 OpenLane v1.2 Anchor3DLane++ R18 的 iter_20000 评测指标，并与 iter_5000 和官方发布 checkpoint 对比。
- 标签：baseline, openlane, eval, checkpoint, anchor3dlanepp
- 关键进展 1：iter_20000 的 F1 为 54.25，已明显高于 iter_5000 的 47.58，也已经超过旧版 Anchor3DLane R18 baseline。
- 关键进展 2：当前离官方发布 checkpoint 的 57.89 还差约 3.64 个 F1，且 recall 已较接近，主要差距更多体现在 precision、cate_acc 和几何误差上，因此继续训练是有意义的。
- 涉及文件：output/eval_openlane_anchor3dlanepp_r18_iter20000/evaluation_result.json, output/eval_openlane_anchor3dlanepp_r18_iter20000/lane3d_prediction.json, output/repro/openlane_anchor3dlanepp_r18_official_20260425/iter_20000.pth
- 执行命令：`/bin/true`
- 工作目录：`output/eval_openlane_anchor3dlanepp_r18_iter20000`
- 日志文件：`/ssd-data3/ztc2025/Anchor3DLanePM/Anchor3dLane/experiments/research_journal/logs/20260427_155505_openlane-anchor3dlanepp-r18-iter20000-eval-result.log`
- 退出码：0
- 指标：F1=0.542489，Recall=0.525340，Precision=0.560795，类别准确率=0.885020，近距离 x 误差=0.304690，远距离 x 误差=0.307634，近距离 z 误差=0.082641，远距离 z 误差=0.113405
- Git 快照：`main` @ `ff401181c5a1`
## 2026-04-27 15:55:24 | 实验 | OpenLane Anchor3DLane++ R18 单卡断点续训
- 摘要：由于当前仅 GPU2 空闲，从 iter_20000 以单卡方式继续训练 OpenLane v1.2 Anchor3DLane++ R18，并使用脱离终端的启动方式降低中断风险。
- 标签：baseline, openlane, train, resume, single-gpu, anchor3dlanepp
- 关键进展 1：采用 setsid + nohup 启动，进程与当前终端会话分离；日志写入 launcher_resume_gpu2_single.log。
- 关键进展 2：这是在资源受限条件下的单卡续训，便于保持训练连续性，但从这一点开始不再是严格的双卡官方 batch size 口径。
- 涉及文件：../configs_v2/openlane/anchor3dlane++_r18.py, output/repro/openlane_anchor3dlanepp_r18_official_20260425/iter_20000.pth, output/repro/openlane_anchor3dlanepp_r18_official_20260425/launcher_resume_gpu2_single.log
- 执行命令：`/bin/true`
- 工作目录：`output/repro/openlane_anchor3dlanepp_r18_official_20260425`
- 日志文件：`/ssd-data3/ztc2025/Anchor3DLanePM/Anchor3dLane/experiments/research_journal/logs/20260427_155524_openlane-anchor3dlanepp-r18-single-gpu-train-resume.log`
- 退出码：0
- Git 快照：`main` @ `ff401181c5a1`
## 2026-04-27 16:21:28 | 代码 | BundleLane框架图与基线改进说明补充
- 摘要：在方法设计与执行计划文档中补充相对Anchor3DLane++的框架图，并明确当前V1与V1.1的实现边界。
- 模块改动 1：方法设计：新增Anchor3DLane++与BundleLane的对照框架图，突出共享bundle frame、frame-conditioned anchor、intrinsic表示与训练迁移主线。
- 模块改动 2：工程计划：新增当前落地框架图，并按已落地、V1待补、V1.1再做三层划分实现边界，避免计划与现状混淆。
- 涉及文件：Anchor3dLane/V1_METHOD_DESIGN.md, Anchor3dLane/V1_EXECUTION_PLAN.md
- Git 快照：`main` @ `ff401181c5a1`
## 2026-04-27 21:27:03 | 实验 | OpenLane Anchor3DLane++ R18 单卡续训完成
- 摘要：记录单卡续训已达到官方 60000 iter 终点，并成功保存 iter_60000.pth。
- 标签：baseline, openlane, train, complete, single-gpu, anchor3dlanepp
- 关键进展 1：launcher_resume_gpu2_single.log 显示已在 2026-04-27 20:36 保存 iter_60000.pth，latest.pth 也已更新到最终 checkpoint。
- 关键进展 2：当前训练进程已经退出，GPU2 已释放；下一步应评测 iter_60000 的最终验证集效果。
- 涉及文件：output/repro/openlane_anchor3dlanepp_r18_official_20260425/iter_60000.pth, output/repro/openlane_anchor3dlanepp_r18_official_20260425/latest.pth, output/repro/openlane_anchor3dlanepp_r18_official_20260425/launcher_resume_gpu2_single.log, output/repro/openlane_anchor3dlanepp_r18_official_20260425/20260427_155519.log
- 执行命令：`/bin/true`
- 工作目录：`output/repro/openlane_anchor3dlanepp_r18_official_20260425`
- 日志文件：`/ssd-data3/ztc2025/Anchor3DLanePM/Anchor3dLane/experiments/research_journal/logs/20260427_212703_openlane-anchor3dlanepp-r18-single-gpu-train-complete.log`
- 退出码：0
- Git 快照：`main` @ `ff401181c5a1`
## 2026-04-27 21:28:53 | 实验 | OpenLane Anchor3DLane++ R18 iter_60000 最终评测启动
- 摘要：在 GPU2 上启动 OpenLane v1.2 Anchor3DLane++ R18 最终 iter_60000 checkpoint 的评测。
- 标签：baseline, openlane, eval, final-checkpoint, anchor3dlanepp
- 关键进展 1：采用 setsid + 重定向日志的方式启动，尽量避免终端会话结束导致的中断。
- 关键进展 2：输出目录为 output/eval_openlane_anchor3dlanepp_r18_iter60000，待 evaluation_result.json 生成后汇总最终指标。
- 涉及文件：../configs_v2/openlane/anchor3dlane++_r18.py, output/repro/openlane_anchor3dlanepp_r18_official_20260425/iter_60000.pth, output/eval_openlane_anchor3dlanepp_r18_iter60000/launcher.log
- 执行命令：`/bin/true`
- 工作目录：`output/eval_openlane_anchor3dlanepp_r18_iter60000`
- 日志文件：`/ssd-data3/ztc2025/Anchor3DLanePM/Anchor3dLane/experiments/research_journal/logs/20260427_212853_openlane-anchor3dlanepp-r18-iter60000-eval-launch.log`
- 退出码：0
- Git 快照：`main` @ `ff401181c5a1`
## 2026-04-27 21:31:55 | 代码 | BundleLane首轮消融配置切分
- 摘要：新增BundleLane首轮消融配置，并将执行稿改成模块落地后立即训练的ablation-first节奏。
- 模块改动 1：实验配置：新增baseline、frame-only、frame+anchor三条20k单卡消融配置，统一batch size与checkpoint节奏，便于后续做同口径对照。
- 模块改动 2：执行规划：在V1执行稿中明确每接入一个模块就立刻补对应ablation并启动训练，避免先堆功能再统一验收。
- 涉及文件：../configs_v2/openlane/anchor3dlanepp_ablation_baseline_r18_20k.py, ../configs_v2/openlane/bundlelane_ablation_frame_only_r18_20k.py, ../configs_v2/openlane/bundlelane_ablation_frame_anchor_r18_20k.py, V1_EXECUTION_PLAN.md
- Git 快照：`main` @ `ff401181c5a1`
## 2026-04-27 21:33:12 | 实验 | BundleLane frame-only 首轮消融训练启动
- 摘要：记录 BundleLane 第一条首轮消融训练已在 GPU2 上启动，只保留 frame supervision，关闭 frame-conditioned anchor 注入。
- 关键进展 1：实际训练主进程 PID 为 4105008，当前以单卡 batch size 8、20000 iter 口径运行。
- 关键进展 2：launcher.log 已出现前 50 iter 日志，bundle_frame_x/h/bank/smooth 四项 loss 均正常出值，当前显存约 11.5GB。
- 关键进展 3：本次先记录训练已启动这一状态，后续在 5000/10000/20000 iter checkpoint 处补充评测结果。
- 执行命令：`/bin/true`
- 工作目录：`output/ablation/openlane/bundlelane_frame_only_r18_20k_bs8`
- 日志文件：`/ssd-data3/ztc2025/Anchor3DLanePM/Anchor3dLane/experiments/research_journal/logs/20260427_213312_bundlelane-frame-only-r18-20k-launch-record.log`
- 退出码：0
- Git 快照：`main` @ `ff401181c5a1`
## 2026-04-27 21:55:52 | 实验 | OpenLane Anchor3DLane++ R18 iter_60000 最终评测结果
- 摘要：记录 OpenLane v1.2 Anchor3DLane++ R18 最终 iter_60000 的评测指标，并与 iter_20000 和官方发布 checkpoint 对比。
- 标签：baseline, openlane, eval, final-checkpoint, anchor3dlanepp
- 关键进展 1：iter_60000 的 F1 为 56.35，相比 iter_20000 的 54.25 继续提升，主要收益来自 precision、cate_acc 和几何误差收敛。
- 关键进展 2：当前结果距离官方发布 checkpoint 的 57.89 仍差约 1.54 个 F1，但 precision 与 cate_acc 已非常接近，说明这次单卡续训整体是有效的。
- 涉及文件：output/eval_openlane_anchor3dlanepp_r18_iter60000/evaluation_result.json, output/eval_openlane_anchor3dlanepp_r18_iter60000/launcher.log, output/repro/openlane_anchor3dlanepp_r18_official_20260425/iter_60000.pth
- 执行命令：`/bin/true`
- 工作目录：`output/eval_openlane_anchor3dlanepp_r18_iter60000`
- 日志文件：`/ssd-data3/ztc2025/Anchor3DLanePM/Anchor3dLane/experiments/research_journal/logs/20260427_215552_openlane-anchor3dlanepp-r18-iter60000-eval-result.log`
- 退出码：0
- 指标：F1=0.563496，Recall=0.519926，Precision=0.615036，类别准确率=0.910308，近距离 x 误差=0.256216，远距离 x 误差=0.258284，近距离 z 误差=0.076981，远距离 z 误差=0.105563
- Git 快照：`main` @ `ff401181c5a1`
## 2026-04-28 00:36:06 | 实验 | BundleLane frame-only 首轮消融训练完成
- 摘要：记录 BundleLane frame-only 首轮 20000 iter 单卡消融训练已完成，并整理当前训练日志中的主要收敛现象。
- 关键进展 1：iter_5000、iter_10000、iter_15000、iter_20000 checkpoint 均已生成，latest.pth 已指向 iter_20000.pth。
- 关键进展 2：总 loss 从 iter_10 的 36.34 下降到 iter_20000 的 2.92，bundle_frame_x_loss 从 0.80 量级降到约 0.47，说明 frame 分支已被主训练链路有效驱动。
- 关键进展 3：中后期仍可见 grad_norm 尖峰与 frame_h/frame_x 轻微回摆，因此是否真正带来指标收益仍需结合验证集评测判断。
- 执行命令：`/bin/true`
- 工作目录：`output/ablation/openlane/bundlelane_frame_only_r18_20k_bs8`
- 日志文件：`/ssd-data3/ztc2025/Anchor3DLanePM/Anchor3dLane/experiments/research_journal/logs/20260428_003606_bundlelane-frame-only-r18-20k-train-complete.log`
- 退出码：0
- Git 快照：`main` @ `ff401181c5a1`
## 2026-04-28 00:36:29 | 实验 | BundleLane frame-only iter_20000 评测启动
- 摘要：在 GPU2 上启动 BundleLane frame-only 首轮消融训练的 iter_20000 checkpoint 验证集评测。
- 关键进展 1：评测输出目录为 output/eval_ablation/openlane/bundlelane_frame_only_r18_20k_bs8_iter20000，待 evaluation_result.json 生成后补录最终指标。
- 关键进展 2：本次评测对应的训练策略为 frame-only，不开启 frame-conditioned anchor 注入。
- 执行命令：`/bin/true`
- 工作目录：`output/eval_ablation/openlane/bundlelane_frame_only_r18_20k_bs8_iter20000`
- 日志文件：`/ssd-data3/ztc2025/Anchor3DLanePM/Anchor3dLane/experiments/research_journal/logs/20260428_003629_bundlelane-frame-only-r18-20k-eval-launch.log`
- 退出码：0
- Git 快照：`main` @ `ff401181c5a1`
## 2026-04-28 00:43:49 | 实验 | BundleLane frame-only iter_20000 评测结果
- 摘要：记录 BundleLane frame-only 首轮消融训练 iter_20000 的验证集指标，并与现有 Anchor3DLane++ iter_20000/iter_60000 结果做初步对照。
- 关键进展 1：frame-only 当前 F1 为 45.36，明显低于现有 Anchor3DLane++ iter_20000 的 54.25，也低于 iter_60000 的 56.35。
- 关键进展 2：当前掉点主要体现在 recall、cate_acc 以及 x/z 几何误差，说明仅加 frame supervision 且关闭 frame-conditioned anchor 注入的版本还没有形成有效增益。
- 关键进展 3：但这次训练不是从已有 Anchor3DLane++ checkpoint warm-start，而是仅继承 ResNet18 backbone 预训练，因此该结果更适合用于否定当前训练策略，不适合直接否定 BundleLane 表示本身。
- 执行命令：`/bin/true`
- 工作目录：`output/eval_ablation/openlane/bundlelane_frame_only_r18_20k_bs8_iter20000`
- 日志文件：`/ssd-data3/ztc2025/Anchor3DLanePM/Anchor3dLane/experiments/research_journal/logs/20260428_004349_bundlelane-frame-only-r18-20k-eval-result.log`
- 退出码：0
- 指标：F1=0.4535694308110872，Recall=0.3866883510588974，Precision=0.5484241010982398，类别准确率=0.8288382611609012，近距离 x 误差=0.4756174505759024，远距离 x 误差=0.4302614478620708，近距离 z 误差=0.097689771661311，远距离 z 误差=0.13568987234678992
- Git 快照：`main` @ `ff401181c5a1`
## 2026-04-28 01:43:17 | 代码 | BundleLane warm-start 消融配置补充
- 摘要：新增基于官方 OpenLane R18 最优权重的 warm-start 消融配置，分别用于 BundleLane frame+anchor 主实验和 Anchor3DLane++ fine-tune 对照实验。
- 模块改动 1：实验配置：新增 BundleLane frame+anchor warm-start 配置，改用 load_from 官方最佳 Anchor3DLane++ 权重，并将 fine-tune 口径收敛到单卡 bs8、10k iter、较保守 lr=5e-5。
- 模块改动 2：对照准备：同步新增 Anchor3DLane++ warm-start control 配置，后续可在相同单卡与迭代预算下判断增益到底来自新模块还是单纯继续微调。
- 涉及文件：../configs_v2/openlane/anchor3dlanepp_ablation_baseline_warm_r18_10k_bs8.py, ../configs_v2/openlane/bundlelane_ablation_frame_anchor_warm_r18_10k_bs8.py
- Git 快照：`main` @ `ff401181c5a1`
## 2026-04-28 01:44:02 | 实验 | BundleLane frame+anchor warm-start 主实验启动
- 摘要：记录 BundleLane frame+anchor warm-start 主实验已在 GPU2 上启动，并确认官方 Anchor3DLane++ 权重已成功作为 load_from 加载。
- 关键进展 1：实际训练主进程 PID 为 76185，当前采用单卡 batch size 8、10000 iter、lr=5e-5 的保守 fine-tune 口径。
- 关键进展 2：launcher.log 已确认 load checkpoint from local path: pretrained/openlane_anchor3dlane++_r18.pth，缺失项仅为新增 bundle_frame_head 参数。
- 关键进展 3：前 20 iter 的 absolute detection loss 已明显低于从头训练版本，说明 warm-start 策略已生效；后续重点观察 frame loss 是否在不破坏原始检测性能的前提下带来正增益。
- 执行命令：`/bin/true`
- 工作目录：`output/ablation/openlane/bundlelane_frame_anchor_warm_r18_10k_bs8`
- 日志文件：`/ssd-data3/ztc2025/Anchor3DLanePM/Anchor3dLane/experiments/research_journal/logs/20260428_014402_bundlelane-frame-anchor-warm-r18-10k-launch-record.log`
- 退出码：0
- Git 快照：`main` @ `ff401181c5a1`
## 2026-04-28 10:11:12 | 实验 | BundleLane frame+anchor warm-start iter_10000 评测结果
- 摘要：记录 BundleLane frame+anchor warm-start 主实验 iter_10000 的验证集指标，并与 frame-only、本地 Anchor3DLane++ iter_60000 以及官方最好 checkpoint 对比。
- 关键进展 1：当前 F1 为 57.21，显著高于 frame-only 的 45.36，也高于本地 Anchor3DLane++ iter_60000 的 56.35，说明 warm-start + frame-conditioned anchor 的组合已经出现明确正增益。
- 关键进展 2：相对官方最好 checkpoint 的 57.89，当前仍差约 0.67 个 F1；主要差距体现在 precision、cate_acc 以及近距离 x/z 误差，说明方法已接近强基座上限，但仍需更严谨的对照与后续微调。
- 关键进展 3：由于这次还没有同步跑完 Anchor3DLane++ warm-start control，因此现阶段最稳妥的结论是：问题不在 BundleLane 表示本身，而在此前 frame-only 与从头训练策略。
- 执行命令：`/bin/true`
- 工作目录：`output/eval_ablation/openlane/bundlelane_frame_anchor_warm_r18_10k_bs8_iter10000`
- 日志文件：`/ssd-data3/ztc2025/Anchor3DLanePM/Anchor3dLane/experiments/research_journal/logs/20260428_101112_bundlelane-frame-anchor-warm-r18-10k-eval-result.log`
- 退出码：0
- 指标：F1=0.5721226608660757，Recall=0.5412467098927907，Precision=0.6067344228428081，类别准确率=0.897299290614962，近距离 x 误差=0.2599229786093317，远距离 x 误差=0.28302513057058376，近距离 z 误差=0.07980308823856118，远距离 z 误差=0.106726682716731
- Git 快照：`main` @ `ff401181c5a1`
## 2026-04-28 10:19:02 | 代码 | BundleLane 30k 续训配置补充
- 摘要：新增 BundleLane frame+anchor 的 30k 续训配置，从 warm-start 10k checkpoint 继续训练到 30000 iter。
- 模块改动 1：实验配置：新增 30k 续训配置，保持单卡 bs8 与 lr=5e-5，不再重新 load_from，而是显式 resume_from 已完成的 10k warm-start checkpoint。
- 模块改动 2：实验策略：为缓解单卡资源受限，计划将 Anchor3DLane++ warm-start control 与 BundleLane 30k 续训串成接力任务，先拿关键对照，再继续验证 BundleLane 是否还会继续上涨。
- 涉及文件：../configs_v2/openlane/bundlelane_ablation_frame_anchor_warm_r18_30k_resume_bs8.py
- Git 快照：`main` @ `ff401181c5a1`
## 2026-04-28 10:58:58 | 实验 | Anchor3DLane++ warm-start control 与 BundleLane 30k 接力启动
- 摘要：在 GPU2 上启动串行接力任务：先运行 Anchor3DLane++ warm-start control 10k，对照完成后自动续接 BundleLane frame+anchor 的 30k 续训。
- 关键进展 1：由于当前只有 GPU2 空闲，这里不做不安全的并行占卡，而是采用单卡串行接力。
- 关键进展 2：第一阶段使用官方最佳 Anchor3DLane++ 权重做 10k warm-start control；第二阶段若第一阶段成功退出，则自动从 BundleLane warm-start 10k checkpoint 续训到 30k。
- 执行命令：`/bin/bash -lc 'source /home/ztc2025/anaconda3/etc/profile.d/conda.sh && conda activate anchor3dlane-cu121 && export PYTHONPATH=$PYTHONPATH:/ssd-data3/ztc2025/Anchor3DLanePM/Anchor3dLane/gen-efficientnet-pytorch && mkdir -p output/ablation/openlane/anchor3dlanepp_baseline_warm_r18_10k_bs8 output/ablation/openlane/bundlelane_frame_anchor_warm_r18_30k_resume_bs8 && CUDA_VISIBLE_DEVICES=2 setsid nohup bash -lc "python -u tools/train.py ../configs_v2/openlane/anchor3dlanepp_ablation_baseline_warm_r18_10k_bs8.py --work-dir output/ablation/openlane/anchor3dlanepp_baseline_warm_r18_10k_bs8 --no-validate --gpu-id 0 --seed 3407 > output/ablation/openlane/anchor3dlanepp_baseline_warm_r18_10k_bs8/launcher.log 2>&1; status=$?; if [ $status -eq 0 ]; then python -u tools/train.py ../configs_v2/openlane/bundlelane_ablation_frame_anchor_warm_r18_30k_resume_bs8.py --work-dir output/ablation/openlane/bundlelane_frame_anchor_warm_r18_30k_resume_bs8 --no-validate --gpu-id 0 --seed 3407 > output/ablation/openlane/bundlelane_frame_anchor_warm_r18_30k_resume_bs8/launcher.log 2>&1; fi" > output/ablation/openlane/chain_launcher.log 2>&1 < /dev/null & echo CHAIN_PID:$!'`
- 工作目录：`output/ablation/openlane`
- 日志文件：`/ssd-data3/ztc2025/Anchor3DLanePM/Anchor3dLane/experiments/research_journal/logs/20260428_101914_anchor3dlanepp-warm-control-then-bundle30k-chain.log`
- 退出码：0
- Git 快照：`main` @ `ff401181c5a1`
## 2026-04-28 11:26:11 | 实验 | BundleLane 当前状态核对
- 摘要：汇总当前 BundleLane warm-start 的最好结果、control 训练完成状态，以及 30k 自动接力卡住的问题。
- 标签：bundlelane, ablation, status
- 关键进展 1：当前最好结果仍是 bundlelane_frame_anchor_warm_r18_10k_bs8 的 iter_10000，F1=0.5721，低于官方最好权重 0.5789。
- 关键进展 2：anchor3dlanepp_baseline_warm_r18_10k_bs8 已完成训练，但尚未完成评测，因此还不能判断 warm-start 微调本身会不会自然回落。
- 关键进展 3：bundlelane 30k 自动接力未真正启动，watcher 使用 pgrep 匹配训练命令时会匹配到自身，导致一直停留在等待状态。
- 涉及文件：output/eval_ablation/openlane/bundlelane_frame_anchor_warm_r18_10k_bs8_iter10000/evaluation_result.json, output/ablation/openlane/anchor3dlanepp_baseline_warm_r18_10k_bs8/launcher.log, output/ablation/openlane/bundlelane_30k_watcher.log
- 执行命令：`bash -lc true`
- 工作目录：`output/ablation/openlane/bundlelane_frame_anchor_warm_r18_30k_resume_bs8`
- 日志文件：`/ssd-data3/ztc2025/Anchor3DLanePM/Anchor3dLane/experiments/research_journal/logs/20260428_112611_bundlelane-status-20260428.log`
- 退出码：0
- 指标：F1=0.5721226608660757
- Git 快照：`main` @ `ff401181c5a1`
## 2026-04-30 16:59:56 | 实验 | Anchor3DLane++ warm-start control iter_10000 评测
- 摘要：评测 Anchor3DLane++ warm-start control 的 iter_10000 checkpoint，用于判断官方权重继续微调本身是否会回落。
- 标签：baseline, openlane, eval, warm-start
- 关键进展 1：使用与 BundleLane warm-start 主实验相同的单卡 bs8、10k 微调口径。
- 关键进展 2：评测结果将作为 BundleFrame 是否带来增益的关键 control。
- 涉及文件：../configs_v2/openlane/anchor3dlanepp_ablation_baseline_warm_r18_10k_bs8.py, output/ablation/openlane/anchor3dlanepp_baseline_warm_r18_10k_bs8/iter_10000.pth
- 执行命令：`bash -lc 'set -o pipefail; source /home/ztc2025/anaconda3/etc/profile.d/conda.sh && conda activate anchor3dlane-cu121 && export PYTHONPATH=$PYTHONPATH:/ssd-data3/ztc2025/Anchor3DLanePM/Anchor3dLane/gen-efficientnet-pytorch && CUDA_VISIBLE_DEVICES=1 python -u tools/test.py ../configs_v2/openlane/anchor3dlanepp_ablation_baseline_warm_r18_10k_bs8.py output/ablation/openlane/anchor3dlanepp_baseline_warm_r18_10k_bs8/iter_10000.pth --show-dir output/eval_ablation/openlane/anchor3dlanepp_baseline_warm_r18_10k_bs8_iter10000 2>&1 | tee output/eval_ablation/openlane/anchor3dlanepp_baseline_warm_r18_10k_bs8_iter10000/launcher.log'`
- 工作目录：`output/eval_ablation/openlane/anchor3dlanepp_baseline_warm_r18_10k_bs8_iter10000`
- 日志文件：`/ssd-data3/ztc2025/Anchor3DLanePM/Anchor3dLane/experiments/research_journal/logs/20260430_164818_anchor3dlanepp-baseline-warm-r18-10k-eval-result.log`
- 退出码：0
- Git 快照：`main` @ `ff401181c5a1`
## 2026-04-30 17:10:09 | 实验 | BundleLane frame+anchor warm-start 30k iter_30000 评测
- 摘要：评测 BundleLane frame+anchor warm-start 30k 续训的 iter_30000 checkpoint，判断延长训练是否相对 warm-start control 带来真实收益。
- 标签：bundlelane, openlane, eval, warm-start, resume-30k
- 关键进展 1：该 checkpoint 已于 2026-04-30 11:33 完成 30000/30000 iter 续训并保存。
- 关键进展 2：本次结果将与 BundleLane 10k、Anchor3DLane++ warm-start control 10k 和官方最好权重做同表对照。
- 涉及文件：../configs_v2/openlane/bundlelane_ablation_frame_anchor_warm_r18_30k_resume_bs8.py, output/ablation/openlane/bundlelane_frame_anchor_warm_r18_30k_resume_bs8/iter_30000.pth
- 执行命令：`bash -lc 'set -o pipefail; source /home/ztc2025/anaconda3/etc/profile.d/conda.sh && conda activate anchor3dlane-cu121 && export PYTHONPATH=$PYTHONPATH:/ssd-data3/ztc2025/Anchor3DLanePM/Anchor3dLane/gen-efficientnet-pytorch && CUDA_VISIBLE_DEVICES=1 python -u tools/test.py ../configs_v2/openlane/bundlelane_ablation_frame_anchor_warm_r18_30k_resume_bs8.py output/ablation/openlane/bundlelane_frame_anchor_warm_r18_30k_resume_bs8/iter_30000.pth --show-dir output/eval_ablation/openlane/bundlelane_frame_anchor_warm_r18_30k_resume_bs8_iter30000 2>&1 | tee output/eval_ablation/openlane/bundlelane_frame_anchor_warm_r18_30k_resume_bs8_iter30000/launcher.log'`
- 工作目录：`output/eval_ablation/openlane/bundlelane_frame_anchor_warm_r18_30k_resume_bs8_iter30000`
- 日志文件：`/ssd-data3/ztc2025/Anchor3DLanePM/Anchor3dLane/experiments/research_journal/logs/20260430_170011_bundlelane-frame-anchor-warm-r18-30k-eval-result.log`
- 退出码：0
- Git 快照：`main` @ `ff401181c5a1`
## 2026-04-30 17:11:14 | 实验 | BundleFrame 可行性阶段判断
- 摘要：汇总 Anchor3DLane++ warm-start control 与 BundleLane 10k/30k 评测结果，判断当前 BundleFrame 证据强度与下一步消融路线。
- 标签：bundlelane, ablation, feasibility, openlane
- 关键进展 1：Anchor3DLane++ warm-start control iter_10000 的 F1=57.41，高于 BundleLane frame+anchor warm-start iter_10000 的 F1=57.21，因此 10k 单点结果不能证明 BundleFrame 已带来正增益。
- 关键进展 2：BundleLane 30k 续训 iter_30000 的 F1=56.75，低于自身 10k 和 control 10k；虽然 cate_acc 与几何误差有所改善，但 recall/precision 回落，说明当前注入与损失权重组合会伤害检测面。
- 关键进展 3：评测加载时出现 bundle_basis_x/h/b unexpected key；当前代码将这些基底注册为 persistent=False 且由配置确定，初判不是 learned 参数问题，但未来正式实验需要统一 checkpoint 与当前代码口径。
- 关键进展 4：下一步不建议直接进入 intrinsic 完全版，应先补 control 30k、no-injection、低权重 frame loss 与注入时机消融，再决定是否扩展到完整 Bundle。
- 涉及文件：output/eval_ablation/openlane/anchor3dlanepp_baseline_warm_r18_10k_bs8_iter10000/evaluation_result.json, output/eval_ablation/openlane/bundlelane_frame_anchor_warm_r18_10k_bs8_iter10000/evaluation_result.json, output/eval_ablation/openlane/bundlelane_frame_anchor_warm_r18_30k_resume_bs8_iter30000/evaluation_result.json, mmseg/models/lane_detector/anchor_3dlane_bundle.py
- 执行命令：`bash -lc true`
- 工作目录：`output/eval_ablation/openlane`
- 日志文件：`/ssd-data3/ztc2025/Anchor3DLanePM/Anchor3dLane/experiments/research_journal/logs/20260430_171114_bundleframe-feasibility-review-20260430.log`
- 退出码：0
- 指标：control10k_F_score=0.5740823756409947，bundle10k_F_score=0.5721226608660757，bundle30k_F_score=0.5675103882615904，bundle30k_recall=0.5339995577525268，bundle30k_precision=0.6055087354729469
- Git 快照：`main` @ `ff401181c5a1`
## 2026-05-01 22:58:45 | 代码 | BundleFrame 作用诊断与小步消融配置
- 摘要：新增 BundleFrame 作用路径诊断脚本、串行消融启动脚本，并补齐 control30k、无注入、低 frame loss、延后注入四个小步消融配置。
- 标签：bundlelane, ablation, diagnostics
- 模块改动 1：诊断工具：在同一 checkpoint 和同一 batch 上强制开/关注入，统计 anchor/proposal 差异、frame target 误差、frame loss 以及 bundle head 梯度，避免用不起作用的模块做可行性判断。
- 模块改动 2：消融配置：补齐 warm-start control 30k、BundleFrame 无注入、低 frame loss 和 inject_iters=[1] 四个配置，用来分别判断训练时长、辅助监督、监督强度和注入时机。
- 模块改动 3：启动脚本：使用单 GPU 串行 train->eval 链路替代 watcher，避免 pgrep 自匹配导致等待卡住。
- 涉及文件：tools/diagnose_bundleframe_effect.py, tools/launch_bundleframe_ablation_chain.sh, ../configs_v2/openlane/anchor3dlanepp_ablation_baseline_warm_r18_30k_resume_bs8.py, ../configs_v2/openlane/bundlelane_ablation_frame_anchor_warm_r18_10k_noinject_bs8.py, ../configs_v2/openlane/bundlelane_ablation_frame_anchor_warm_r18_10k_lowloss_bs8.py, ../configs_v2/openlane/bundlelane_ablation_frame_anchor_warm_r18_10k_inject1_bs8.py
- Git 快照：`main` @ `ff401181c5a1`
## 2026-05-01 22:58:56 | 实验 | BundleFrame 作用路径诊断
- 摘要：对 10k 和 30k BundleFrame warm-start checkpoint 做真实 batch 诊断，确认模块不是死路径，并量化当前主要问题。
- 标签：bundlelane, diagnostics, ablation
- 关键进展 1：10k checkpoint：effect_classification=active_but_needs_ablation，final proposal delta mean=0.4818/max=20.0812，frame loss mean=0.2320，bundle supervision grad norm=1.1709。
- 关键进展 2：30k checkpoint：effect_classification=active_but_needs_ablation，final proposal delta mean=0.5261/max=14.1046，frame loss mean=0.2391，bundle supervision grad norm=1.4437。
- 关键进展 3：诊断判断：BundleFrame 注入和监督均有作用，但 x_ref target MAE 约 1.18/1.22m，属于强干预且 frame 预测仍偏粗，下一步优先做无注入、低损失、注入时机和 control 30k 消融，不直接进入 intrinsic 完全版。
- 涉及文件：output/diagnostics/bundleframe_effect_warm10k_train2.json, output/diagnostics/bundleframe_effect_warm30k_train2.json, tools/diagnose_bundleframe_effect.py
- 执行命令：`bash -lc true`
- 工作目录：`output/diagnostics`
- 日志文件：`/ssd-data3/ztc2025/Anchor3DLanePM/Anchor3dLane/experiments/research_journal/logs/20260501_225856_bundleframe-effect-diagnosis-20260501.log`
- 退出码：0
- 指标：warm10k_final_delta_mean=0.48183897137641907，warm10k_final_delta_max=20.08115005493164，warm10k_frame_loss_mean=0.23198749662151386，warm30k_final_delta_mean=0.5261051207780838，warm30k_final_delta_max=14.104634284973145，warm30k_frame_loss_mean=0.23910209516975556
- Git 快照：`main` @ `ff401181c5a1`
## 2026-05-01 22:59:06 | 实验 | BundleFrame 小步消融串行实验启动
- 摘要：在 GPU0 启动串行小步消融链：control 30k、无注入、低 frame loss、延后注入，每个训练完成后立即评测。
- 标签：bundlelane, ablation, launch
- 关键进展 1：已启动 tools/launch_bundleframe_ablation_chain.sh，shell PID=2583372，当前首个任务为 anchor3dlanepp_baseline_warm_r18_30k_resume_bs8，train PID=2583392。
- 关键进展 2：链路顺序：baseline warm 10k->30k control；BundleFrame warm 10k noinject；BundleFrame warm 10k lowloss；BundleFrame warm 10k inject_iters=[1]。
- 关键进展 3：主日志：output/ablation/openlane/bundleframe_ablation_chain_20260501.log；各训练日志和评测日志分别落在对应 work_dir 与 eval_dir 的 launcher.log。
- 涉及文件：tools/launch_bundleframe_ablation_chain.sh, ../configs_v2/openlane/anchor3dlanepp_ablation_baseline_warm_r18_30k_resume_bs8.py, ../configs_v2/openlane/bundlelane_ablation_frame_anchor_warm_r18_10k_noinject_bs8.py, ../configs_v2/openlane/bundlelane_ablation_frame_anchor_warm_r18_10k_lowloss_bs8.py, ../configs_v2/openlane/bundlelane_ablation_frame_anchor_warm_r18_10k_inject1_bs8.py, output/ablation/openlane/bundleframe_ablation_chain_20260501.log
- 执行命令：`bash -lc true`
- 工作目录：`output/ablation/openlane`
- 日志文件：`/ssd-data3/ztc2025/Anchor3DLanePM/Anchor3dLane/experiments/research_journal/logs/20260501_225906_bundleframe-small-ablation-chain-20260501.log`
- 退出码：0
- 指标：chain_pid=2583372，current_train_pid=2583392
- Git 快照：`main` @ `ff401181c5a1`
## 2026-05-02 11:27:53 | 代码 | BundleFrame soft 注入消融
- 摘要：新增 BundleFrame 可配置 soft 注入强度，用线性混合替代全量替换，并补齐 0.25 与 0.5 两个小步消融配置和串行启动脚本。
- 标签：bundlelane, ablation, soft-injection
- 模块改动 1：BundleLane 检测器：新增 bundle_cfg.inject_strength，生成普通 anchor 与 BundleFrame anchor 后按强度线性混合；默认值 1.0 保持旧实验口径不变。
- 模块改动 2：诊断工具：输出 inject_strength，便于把 proposal delta 与注入强度对应起来。
- 模块改动 3：消融配置：新增 soft025 与 soft05 两个 10k warm-start 配置，专门验证原 hard injection 是否过强。
- 涉及文件：mmseg/models/lane_detector/anchor_3dlane_bundle.py, tools/diagnose_bundleframe_effect.py, tools/launch_bundleframe_softinject_chain.sh, ../configs_v2/openlane/bundlelane_ablation_frame_anchor_warm_r18_10k_soft025_bs8.py, ../configs_v2/openlane/bundlelane_ablation_frame_anchor_warm_r18_10k_soft05_bs8.py
- Git 快照：`main` @ `ff401181c5a1`
## 2026-05-02 11:28:19 | 实验 | BundleFrame soft 注入 sanity check
- 摘要：用旧的 BundleFrame 10k checkpoint 验证 soft 注入强度确实会压低 proposal 扰动，为后续训练确认代码路径有效。
- 标签：bundlelane, diagnostics, soft-injection
- 关键进展 1：soft025：inject_strength=0.25，final proposal delta mean=0.2196，max=2.5164，明显低于原 hard injection。
- 关键进展 2：soft05：inject_strength=0.5，final proposal delta mean=0.3862，max=7.4319，介于 soft025 与 hard injection 之间。
- 关键进展 3：诊断判断：soft 注入路径可控，适合继续作为小步消融验证 hard injection 是否过强。
- 涉及文件：output/diagnostics/bundleframe_effect_warm10k_soft025_sanity_train1.json, output/diagnostics/bundleframe_effect_warm10k_soft05_sanity_train1.json, tools/diagnose_bundleframe_effect.py
- 执行命令：`bash -lc true`
- 工作目录：`output/diagnostics`
- 日志文件：`/ssd-data3/ztc2025/Anchor3DLanePM/Anchor3dLane/experiments/research_journal/logs/20260502_112819_bundleframe-softinject-sanity-20260502.log`
- 退出码：0
- 指标：soft025_final_delta_mean=0.2196178138256073，soft025_final_delta_max=2.516447067260742，soft05_final_delta_mean=0.3861815333366394，soft05_final_delta_max=7.431910514831543
- Git 快照：`main` @ `ff401181c5a1`
## 2026-05-02 11:28:29 | 实验 | BundleFrame soft 注入消融训练启动
- 摘要：在 GPU0 启动 soft025 与 soft05 两个 BundleFrame warm-start 10k 消融训练，每个训练完成后自动评测。
- 标签：bundlelane, ablation, soft-injection, launch
- 关键进展 1：已启动 tools/launch_bundleframe_softinject_chain.sh，shell PID=3022745，当前首个任务为 bundlelane_frame_anchor_warm_r18_10k_soft025_bs8，train PID=3022763。
- 关键进展 2：链路顺序：soft025 10k train/eval，然后 soft05 10k train/eval。
- 关键进展 3：主日志：output/ablation/openlane/bundleframe_softinject_chain_20260502.log；各阶段日志写入对应 work_dir 和 eval_dir 的 launcher.log。
- 涉及文件：tools/launch_bundleframe_softinject_chain.sh, ../configs_v2/openlane/bundlelane_ablation_frame_anchor_warm_r18_10k_soft025_bs8.py, ../configs_v2/openlane/bundlelane_ablation_frame_anchor_warm_r18_10k_soft05_bs8.py, output/ablation/openlane/bundleframe_softinject_chain_20260502.log
- 执行命令：`bash -lc true`
- 工作目录：`output/ablation/openlane`
- 日志文件：`/ssd-data3/ztc2025/Anchor3DLanePM/Anchor3dLane/experiments/research_journal/logs/20260502_112829_bundleframe-softinject-chain-20260502.log`
- 退出码：0
- 指标：soft_chain_pid=3022745，current_train_pid=3022763
- Git 快照：`main` @ `ff401181c5a1`
