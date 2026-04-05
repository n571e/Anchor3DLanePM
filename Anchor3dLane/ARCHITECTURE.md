# Anchor3DLane 架构概览

## 一、项目简介

Anchor3DLane 是一个基于锚框回归的单目 3D 车道检测算法，包含两个版本：
- **Anchor3DLane** (CVPR 2023): 基础单帧检测模型
- **Anchor3DLane++** (TPAMI 2024): 引入稀疏锚框回归和多模态融合

**核心思想**：在 3D 空间定义车道锚框，通过投影到透视视图 (FV) 特征提取特征，直接回归 3D 车道坐标，无需 BEV 转换。

---

## 二、目录结构与逐文件职责

说明：本节按当前仓库实际代码整理（Anchor3DLanePM），重点覆盖“训练/推理主路径会用到的文件”。

### 2.1 顶层文件

| 文件 | 作用 |
|------|------|
| `ARCHITECTURE.md` | 架构说明文档（本文件）。 |
| `README.md` | 安装、训练、测试与复现实验入口说明。 |
| `requirements.txt` | Python 依赖清单。 |
| `setup.py` | 项目可编辑安装入口（`python setup.py develop`）。 |
| `setup.cfg` | 代码风格/工具链配置。 |

### 2.2 `configs/`（训练实验定义）

#### `configs/openlane/`

| 文件 | 作用 |
|------|------|
| `anchor3dlane.py` | OpenLane 单帧基线（ResNet18 + 单阶段回归）。 |
| `anchor3dlane_iter.py` | OpenLane 迭代回归版本（`iter_reg=1`，带 `loss_aux`）。 |
| `anchor3dlane_effb3.py` | OpenLane EfficientNet-B3 骨干版本。 |
| `anchor3dlane_iter_r50.py` | OpenLane 高分辨率 + ResNet50 + Deform 版本（`Anchor3DLaneDeform`）。 |
| `anchor3dlane_mf.py` | OpenLane 多帧版本（`OpenlaneMFDataset` + `Anchor3DLaneMF`）。 |
| `anchor3dlane_mf_iter.py` | OpenLane 多帧 + 迭代回归版本。 |

#### `configs/apollosim/`

| 文件 | 作用 |
|------|------|
| `anchor3dlane.py` | ApolloSim 单帧基线，类别数为 2。 |
| `anchor3dlane_iter.py` | ApolloSim 两阶段/迭代回归版本。 |

#### `configs/once/`

| 文件 | 作用 |
|------|------|
| `anchor3dlane.py` | ONCE 单帧基线，`anchor_y_steps` 为近距分布（2~50m）。 |
| `anchor3dlane_iter.py` | ONCE 迭代回归版本。 |
| `anchor3dlane_effb3.py` | ONCE EfficientNet-B3 骨干版本。 |

### 2.3 `mmseg/apis/`（训练/测试 API）

| 文件 | 作用 |
|------|------|
| `__init__.py` | 统一导出训练、推理和三个数据集测试函数。 |
| `train.py` | 通用训练 API（构建 dataloader、runner、hook）。 |
| `test.py` | MMSeg 通用测试流程（single/multi-gpu）。 |
| `test_openlane.py` | OpenLane 专用后处理 + 指标评测 + 可视化。 |
| `test_apollosim.py` | ApolloSim 专用后处理与评测。 |
| `test_once.py` | ONCE 专用后处理与评测。 |
| `inference.py` | 离线推理接口（`init_segmentor` / `inference_segmentor`）。 |

### 2.4 `mmseg/models/lane_detector/`（核心）

| 文件 | 作用 |
|------|------|
| `__init__.py` | 注册并导出三种 detector。 |
| `anchor_3dlane.py` | 单帧主模型：锚框采样、回归头、NMS、loss 入口都在这里。 |
| `anchor_3dlane_deform.py` | Anchor3DLane++ 版本，用 `MSDALayer` 替换标准 Transformer 编码。 |
| `anchor_3dlane_multiframe.py` | 多帧版本，使用 TransformerDecoderLayer 融合历史帧锚特征。 |
| `transformer.py` | 标准 Transformer 编码/解码层实现（基于 DETR 风格改造）。 |
| `position_encoding.py` | 2D/3D 正弦位置编码实现。 |
| `msda.py` | 多尺度可变形注意力编码层封装（调用 `MSDeformAttn`）。 |
| `tools.py` | 几何与可视化辅助函数（含 `homography_crop_resize`）。 |

#### `mmseg/models/lane_detector/assigner/`

| 文件 | 作用 |
|------|------|
| `__init__.py` | 导出多种 anchor-gt 匹配器。 |
| `distance_metric.py` | 距离度量：Euclidean/Manhattan/Partial/FV。 |
| `thresh_assigner.py` | 基于阈值划分正负样本。 |
| `topk_assigner.py` | 基于 Top-K 的正样本分配（主用）。 |
| `topk_fv_assigner.py` | 融合 2D-FV 与 3D 距离的 Top-K 分配。 |

#### `mmseg/models/lane_detector/utils/`

| 文件 | 作用 |
|------|------|
| `__init__.py` | 导出锚框生成与 NMS。 |
| `anchor.py` | 3D 锚框生成器（pitch/yaw/x 起点离散化）。 |
| `nms.py` | 3D 几何 NMS。 |

### 2.5 `mmseg/models/losses/`（损失）

| 文件 | 作用 |
|------|------|
| `lane_loss.py` | 车道检测主损失：分类 + x/z/vis 回归。 |
| `kornia_focal.py` | FocalLoss 实现。 |
| `focal_loss.py` / `cross_entropy_loss.py` / `dice_loss.py` / `lovasz_loss.py` / `accuracy.py` / `utils.py` | MMSeg 通用损失与工具。 |

### 2.6 `mmseg/datasets/lane_datasets/`（数据集）

| 文件 | 作用 |
|------|------|
| `openlane.py` | OpenLane 单帧数据集：读取 `.pkl` 标注、格式转换与评测。 |
| `openlane_temporal.py` | OpenLane 多帧数据集：采样前后帧与位姿。 |
| `apollosim.py` | ApolloSim 数据集封装与评测。 |
| `once.py` | ONCE-3DLane 数据集封装与评测。 |
| `__init__.py` | 为空（不做额外导出逻辑）。 |

### 2.7 `mmseg/datasets/pipelines/`（预处理流水线）

| 文件 | 作用 |
|------|------|
| `compose.py` | pipeline 组合器。 |
| `loading.py` | 读图/读标注（支持多帧图片堆叠）。 |
| `transforms.py` | Resize/Normalize/PhotoMetricDistortion 等增强。 |
| `lane_format.py` | 车道任务专用格式打包（DataContainer）。 |
| `formatting.py` | MMSeg 通用格式化组件（Collect/ToTensor 等）。 |
| `formating.py` | 兼容旧命名的弃用入口。 |
| `test_time_aug.py` | 测试时多尺度/翻转增强。 |
| `__init__.py` | 汇总导出 pipeline 组件。 |

### 2.8 `tools/`（命令行入口与辅助脚本）

| 文件 | 作用 |
|------|------|
| `train.py` | 推荐训练入口，按配置构建 model/dataset/runner。 |
| `test.py` | 推荐测试入口，自动按数据集选择 `test_openlane/once/apollosim`。 |
| `train_dist.py` | 另一份分布式训练入口（与 `train.py` 功能重叠）。 |
| `dist_train.sh` / `dist_test.sh` / `slurm_train.sh` / `slurm_test.sh` | 多机多卡脚本封装。 |
| `benchmark.py` / `analyze_logs.py` / `get_flops.py` / `print_config.py` | 训练日志与性能分析。 |
| `pytorch2onnx.py` / `onnx2tensorrt.py` / `pytorch2torchscript.py` | 导出与部署转换。 |
| `publish_model.py` / `deploy_test.py` / `confusion_matrix.py` | 发布与评测辅助工具。 |

#### `tools/convert_datasets/`

| 文件 | 作用 |
|------|------|
| `openlane.py` | OpenLane 原始标注 -> 训练 `.pkl` 缓存（含可选平滑）。 |
| `apollosim.py` | ApolloSim 原始 JSON -> 训练 `.pkl`。 |
| `once.py` | ONCE 原始标注合并与 `.pkl` 生成。 |

---
---

## 三、核心模块详解

### 3.1 模型架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        输入图像 (B, 3, H, W)                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Backbone (ResNet/EfficientNet)                │
│                         输出多尺度特征                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    1x1 Conv 特征投影 (512→64)                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              Transformer Encoder (自注意力 + 位置编码)            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              Anchor Projection (Conv2d, 64→64 通道)              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│   3D 锚框投影 → Grid Sample → 锚框特征 (B, N, C, L)              │
│   (N=锚框数，L=Y 轴采样点数)                                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    迭代回归头 (可选多阶段)                        │
│   ┌─────────────┬─────────────┬─────────────┬─────────────┐    │
│   │ 分类头      │ X 回归头    │ Z 回归头    │ 可见性头    │    │
│   │ (21 类)     │ (L 点)      │ (L 点)      │ (L 点)      │    │
│   └─────────────┴─────────────┴─────────────┴─────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    3D NMS → 最终车道输出                         │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 模型变体

| 模型 | 文件 | 特点 |
|------|------|------|
| **Anchor3DLane** | `anchor_3dlane.py` | 基础单帧模型，使用 Transformer Encoder |
| **Anchor3DLane++** | `anchor_3dlane_deform.py` | 使用 MSDA(多尺度可变形注意力)，支持迭代回归 |
| **Anchor3DLane-MF** | `anchor_3dlane_multiframe.py` | 多帧版本，融合时序信息 |

### 3.3 锚框设计

**实现层面说明（事实校正）**

代码里的训练/推理张量并不是 `[pitch, yaw, ...]` 这种显式参数化形式，而是：

```
[meta(5), x_1...x_L, z_1...z_L, vis_1...vis_L, cls_logits(C)]
```

- `meta(5)`：数据集相关元字段（不同数据集的具体语义略有差异，常用到类别、有效范围等）。
- `x/z/vis`：沿 `y_steps` 的离散采样值。
- `cls_logits(C)`：分类输出（`C=num_category`），在 `get_proposals()` 末尾拼接。

几何先验（pitch/yaw/x 起点）主要体现在 `AnchorGenerator.generate_anchor()` 的锚线生成过程中，而不是直接作为训练目标向量前几维。

**生成策略** (`utils/anchor.py`):
- **X 范围**: -30m ~ 30m (45 个起始位置)
- **Pitches**: [5°, 2°, 1°, 0°, -1°, -2°, -5°] (7 种)
- **Yaws**: [30°, 20°, ..., -30°] (17 种)
- **总锚框数**: 45 × 7 × 17 ≈ 5,355 个

**Y 轴采样点**:
- 特征采样: `[5, 10, 15, 20, 30, 40, 50, 60, 80, 100]` (10 点)
- 输出采样: `[5, 10, ..., 100]` (20 点，更密集)

---

## 四、数据流

### 4.1 训练流程

```
1. 数据加载 (OpenlaneDataset)
   └─> 加载图像 + 3D 车道标注 + 相机参数

2. 数据增强 (Compose Pipeline)
   ├─> LoadImageFromFile
   ├─> Resize (to 360x480)
   ├─> Normalize (ImageNet mean/std)
   ├─> MaskGenerate (生成有效区域掩码)
   ├─> LaneFormat (格式化车道标注)
   └─> Collect (收集数据)

3. 前向传播 (Anchor3DLane.forward_train)
   └─> encoder_decoder → loss

4. 损失计算 (LaneLoss)
   ├─> 锚框 - 真值匹配 (TopkAssigner)
   ├─> 分类损失 (Focal Loss)
   ├─> X/Z 回归损失 (Smooth L1)
   └─> 可见性损失 (Smooth L1)

5. 反向传播 & 优化
```

### 4.2 推理流程

```
1. 图像预处理 (同训练)

2. 前向传播 (Anchor3DLane.forward_test)
   └─> encoder_decoder → reg_proposals

3. NMS 后处理
   ├─> 置信度过滤
   ├─> 3D NMS (基于 X/Z 距离)
   └─> 可见性优化

4. 输出去标准化
   └─> (x_norm=30, y_norm=100, z_norm=10)
```

---

## 五、损失函数

### LaneLoss (`mmseg/models/losses/lane_loss.py`)

```python
Loss = w_cls * L_cls + w_x * L_x + w_z * L_z + w_vis * L_vis
```

| 损失项 | 类型 | 公式 | 说明 |
|--------|------|------|------|
| **L_cls** | Focal Loss | `-α(1-p)^γ log(p)` | 分类 (前景/背景) |
| **L_x** | Smooth L1 | 可见性加权 | X 坐标回归 |
| **L_z** | Smooth L1 | 可见性加权 | Z 坐标回归 |
| **L_vis** | Smooth L1 | - | 可见性回归 |

**默认权重**: 均为 1.0

---

## 六、配置系统

### 6.1 配置文件结构

```python
# 数据集配置
dataset_type = 'OpenlaneDataset'
data_root = './data/OpenLane'
train_pipeline = [...]
test_pipeline = [...]

# 模型配置
model = dict(
    type='Anchor3DLane',
    backbone=dict(type='ResNetV1c', depth=18, ...),
    y_steps=[...],              # Y 轴采样点
    feat_y_steps=[...],         # 特征采样点
    anchor_cfg=dict(            # 锚框生成配置
        pitches=[...],
        yaws=[...],
        num_x=45,
        distances=[3,]
    ),
    db_cfg=dict(                # 数据集特定配置
        org_h=1280, org_w=1920,
        ipm_h=208, ipm_w=128,
        cam_height=1.55, pitch=3,
        K=[[...], [...], [...]]
    ),
    # Transformer 配置
    attn_dim=64,
    num_heads=2,
    dim_feedforward=128,
    feat_size=(45, 60),
    num_category=21,
    # 损失配置
    loss_lane=dict(type='LaneLoss', ...)
)

# 训练配置
optimizer = dict(type='Adam', lr=2e-4)
lr_config = dict(policy='step', step=[50000,])
runner = dict(type='IterBasedRunner', max_iters=60000)
```

### 6.2 模型注册

模型通过 `@LANENET2S.register_module()` 装饰器注册:

```python
# mmseg/models/builder.py
LANENET2S = Registry('models')

# 使用
model = dict(type='Anchor3DLane', ...)  # 自动查找注册类
```

---

## 七、评估指标

### 7.1 ApolloSim / OpenLane

| 指标 | 说明 |
|------|------|
| **F1 Score** | 精确率与召回率的调和平均 |
| **AP (Accuracy)** | 分类准确率 |
| **X Error (Close/Far)** | X 方向误差 (近/远距离) |
| **Z Error (Close/Far)** | Z 方向误差 (近/远距离) |

### 7.2 ONCE-3DLane

| 指标 | 说明 |
|------|------|
| **F1 Score** | 调和平均 |
| **Precision** | 精确率 |
| **Recall** | 召回率 |
| **CD Error** | Chamfer Distance |

---

## 八、关键模块代码级分析

### 8.1 单帧模型主干：`Anchor3DLane`

核心文件：`mmseg/models/lane_detector/anchor_3dlane.py`

关键调用链：

1. `forward_train()` / `forward_test()`
2. `encoder_decoder()`
3. `feature_extractor()` + `anchor_projection`
4. `get_proposals()`（可迭代）
5. 训练走 `loss()`，测试走 `nms()`

`feature_extractor()` 的真实行为：

- 用 backbone（可选 neck）取 feature map。
- `input_proj(1x1 conv)` 统一通道到 `attn_dim`。
- 构建 mask 与 `PositionEmbeddingSine`。
- 将 `[B,C,H,W]` 拉平成 `[HW,B,C]` 送入 `TransformerEncoderLayer`。

这一步决定了后续锚特征采样的空间语义质量。

### 8.2 锚特征采样与回归：`cut_anchor_features()` + `get_proposals()`

关键点：

1. `projection_transform()` 把 `(x,y,z)` 通过 `3x4` 投影矩阵映射到特征图坐标 `(u,v)`。
2. 坐标归一化到 `[-1,1]` 后，用 `F.grid_sample` 直接采样锚点序列特征。
3. 采样结果经 `DecodeLayer` 分别输出分类、`x/z` 偏移和可见性。
4. 若 `iter_reg>0`，第 `t+1` 阶段以上一阶段 proposal 作为新锚继续细化。

这套设计的实质是“几何先验引导的序列采样 + 逐步残差回归”。

### 8.3 匹配策略：`TopkAssigner`

核心文件：`mmseg/models/lane_detector/assigner/topk_assigner.py`

算法要点：

- 先构建 proposal-target 的全连接距离矩阵 `D ∈ R^{Np×Nt}`。
- 距离由 `distance_metric.py` 提供，默认是可见性加权的 3D Euclidean。
- 每个 GT 取 Top-K 最近 proposal 作为候选正样本。
- 再做一次“同一 proposal 只归属最近 GT”的冲突消解。
- 负样本从其余 proposal 中随机采样，数量受 `neg_k` 限制。

相比固定阈值匹配，Top-K 在车道密集场景更稳，不容易出现正样本不足。

### 8.4 损失实现：`LaneLoss`

核心文件：`mmseg/models/losses/lane_loss.py`

代码行为拆解：

1. 先把 GT 从原始密集长度（如 200）按 `anchor_steps` 下采样到锚长度 `L`。
2. 用 assigner 得到 `positives/negatives/gt_idx`。
3. 分类：`FocalLoss(cls_pred, cls_target)`。
4. 回归：`SmoothL1(x,z,vis)`，其中 `x/z` 会乘以 `vis_target` 做可见性加权。
5. batch 内求均值后，再乘配置中的 `loss_weights`。

### 8.5 后处理：`nms_3d`

核心文件：`mmseg/models/lane_detector/utils/nms.py`

NMS 不是 2D 框 IoU，而是“可见点重合区域上的 3D 点距”抑制：

- 先按得分降序取当前 lane。
- 对剩余 lane 计算重叠可见点处的平均欧氏距离。
- 距离小于阈值视为重复，剔除。

该策略与车道几何更一致，能避免 2D NMS 的视角偏置。

### 8.6 多帧版本：`Anchor3DLaneMF`

核心文件：`mmseg/models/lane_detector/anchor_3dlane_multiframe.py`

关键机制：

- 当前帧与历史帧都做锚点采样。
- 当前帧锚特征作为 query，历史帧锚特征作为 memory。
- 用 `nn.TransformerDecoderLayer` 做时序融合后再回归。

因此多帧增益主要来自“锚级别时序上下文”，而不是简单图像级拼接。

### 8.7 可变形注意力版本：`Anchor3DLaneDeform`

核心文件：

- `mmseg/models/lane_detector/anchor_3dlane_deform.py`
- `mmseg/models/lane_detector/msda.py`
- `mmseg/models/utils/ops/modules/ms_deform_attn.py`

与基础版差异：

- 编码器由标准 self-attention 改为 `MSDeformAttn`。
- 通过多尺度 reference points 做稀疏采样，降低高分辨率下的注意力成本。
- 配置中常配合更强 backbone（如 ResNet50 + DCN）使用。

### 8.8 数据流代码落点

训练样本是如何进入模型的：

1. 数据集 `__getitem__()`：读取图像路径、`.pkl` 标注、相机参数并组装 `results`。
2. `Compose` 顺序执行 pipeline。
3. `LaneFormat` 将 `img/gt_3dlanes/gt_project_matrix/mask` 打包成 DataContainer。
4. `Collect` 只保留模型前向所需 key。

多帧场景下，`OpenlaneMFDataset` 还会补充：

- `prev_images`
- `prev_poses`

并由 `LoadImageFromFile(extra_keys='prev_images')` 一并加载并堆叠。

### 8.9 核心索引（便于跳转）

| 模块 | 入口文件 | 关键函数/类 |
|------|----------|-------------|
| 单帧检测器 | `anchor_3dlane.py` | `feature_extractor`, `cut_anchor_features`, `get_proposals`, `encoder_decoder` |
| 可变形检测器 | `anchor_3dlane_deform.py` | `Anchor3DLaneDeform.feature_extractor` |
| 多帧检测器 | `anchor_3dlane_multiframe.py` | `Anchor3DLaneMF.get_proposals`, `Anchor3DLaneMF.encoder_decoder` |
| 匹配器 | `assigner/topk_assigner.py` | `match_proposals_with_targets` |
| 损失 | `losses/lane_loss.py` | `LaneLoss.forward` |
| NMS | `utils/nms.py` | `nms_3d` |
| 数据集 | `lane_datasets/*.py` | `__getitem__`, `format_results`, `eval` |
| 数据流水线 | `pipelines/*.py` | `LoadImageFromFile`, `Normalize`, `LaneFormat`, `Collect` |

---

## 九、快速开始

### 9.1 环境安装

```bash
# 创建环境
conda create -n lane3d python=3.7
conda activate lane3d

# 安装 PyTorch
conda install pytorch==1.9.1 torchvision==0.10.1 cudatoolkit=11.1 -c pytorch

# 安装依赖
pip install -U openmim
mim install mmcv-full
pip install -r requirements.txt

# 安装 Anchor3DLane
python setup.py develop

# 编译可变形注意力 (可选)
cd mmseg/models/utils/ops
sh make.sh
```

### 9.2 训练

```bash
# 单 GPU 训练
python tools/train.py configs/openlane/anchor3dlane.py

# 多 GPU 训练
bash tools/dist_train.sh configs/openlane/anchor3dlane.py 8
```

### 9.3 测试

```bash
# 单 GPU 测试
python tools/test.py configs/openlane/anchor3dlane.py checkpoint.pth --show-dir output/

# 多 GPU 测试
bash tools/dist_test.sh configs/openlane/anchor3dlane.py checkpoint.pth 8
```

### 9.4 可视化

添加 `--show` 参数生成可视化结果:

```bash
python tools/test.py configs/openlane/anchor3dlane.py checkpoint.pth --show-dir output/ --show
```

可视化结果保存在 `output/vis/` 目录。

---

## 十、模型性能

### OpenLane 数据集 (ResNet-18 Backbone)

| 模型 | F1 | X Error (Close) | Z Error (Close) |
|------|----|-----------------|-----------------|
| Anchor3DLane | 53.1 | 0.300m | 0.103m |
| Anchor3DLane-Iter | 53.7 | 0.276m | 0.107m |
| Anchor3DLane++ | 57.9 | 0.232m | 0.076m |

### ApolloSim 数据集

| 模型 | F1 | AP | X Error (Far) | Z Error (Far) |
|------|----|----|---------------|---------------|
| Anchor3DLane | 95.6 | 97.2 | 0.306m | 0.223m |
| Anchor3DLane-Iter | 97.1 | 95.4 | 0.300m | 0.223m |

---

## 十一、引用

```bibtex
@inproceedings{huang2023anchor3dlane,
  title={Anchor3DLane: Learning to Regress 3D Anchors for Monocular 3D Lane Detection},
  author={Huang, Shaofei and Shen, Zhenwei and Huang, Zehao and Ding, Zi-han and Dai, Jiao and Han, Jizhong and Wang, Naiyan and Liu, Si},
  booktitle={CVPR},
  year={2023}
}

@article{huang2024anchor3dlane++,
  title={Anchor3DLane++: 3D Lane Detection via Sample-Adaptive Sparse 3D Anchor Regression},
  author={Huang, Shaofei and Shen, Zhenwei and Huang, Zehao and Liao, Yue and Han, Jizhong and Wang, Naiyan and Liu, Si},
  journal={TPAMI},
  year={2024}
}
```

---

## 十二、联系方式

- **论文作者**: Shaofei Huang
- **邮箱**: nowherespyfly@gmail.com
- **GitHub**: https://github.com/tusen-ai/Anchor3DLane

---

*文档生成时间：2026-04-04*
