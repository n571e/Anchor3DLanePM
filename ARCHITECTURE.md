# Anchor3DLane 架构概览

## 一、项目简介

Anchor3DLane 是一个基于锚框回归的单目 3D 车道检测算法，包含两个版本：
- **Anchor3DLane** (CVPR 2023): 基础单帧检测模型
- **Anchor3DLane++** (TPAMI 2024): 引入稀疏锚框回归和多模态融合

**核心思想**：在 3D 空间定义车道锚框，通过投影到透视视图 (FV) 特征提取特征，直接回归 3D 车道坐标，无需 BEV 转换。

---

## 二、目录结构

```
Anchor3DLane-main/
├── configs/                     # 配置文件
│   ├── apollosim/              # ApolloSim 数据集配置
│   │   ├── anchor3dlane.py
│   │   └── anchor3dlane_iter.py
│   ├── openlane/               # OpenLane 数据集配置
│   │   ├── anchor3dlane.py
│   │   ├── anchor3dlane_effb3.py
│   │   ├── anchor3dlane_iter.py
│   │   └── anchor3dlane_mf.py  # 多帧版本
│   └── once/                   # ONCE-3DLane 数据集配置
│
├── data/                        # 数据目录
│   ├── Apollosim/
│   ├── OpenLane/
│   └── ONCE/
│
├── mmseg/                       # 核心代码库 (基于 MMSegmentation v0.26)
│   ├── apis/                   # API 接口
│   │   ├── train.py            # 训练 API
│   │   ├── test.py             # 测试 API
│   │   ├── test_openlane.py    # OpenLane 评测
│   │   ├── test_apollosim.py   # ApolloSim 评测
│   │   └── test_once.py        # ONCE 评测
│   │
│   ├── core/                   # 核心功能
│   │   ├── evaluation/         # 评估指标 (F1, AP, 等)
│   │   ├── hook/               # 训练钩子
│   │   ├── optimizers/         # 优化器构建
│   │   └── seg/                # 分割相关
│   │
│   ├── datasets/               # 数据集
│   │   ├── lane_datasets/      # 3D 车道数据集
│   │   │   ├── openlane.py
│   │   │   ├── apollosim.py
│   │   │   ├── once.py
│   │   │   └── openlane_temporal.py  # 多帧数据
│   │   └── pipelines/          # 数据增强管道
│   │       ├── loading.py
│   │       ├── transforms.py
│   │       ├── lane_format.py
│   │       └── formatting.py
│   │
│   ├── models/                 # 模型定义
│   │   ├── lane_detector/      # 【核心】Anchor3DLane 检测器
│   │   │   ├── anchor_3dlane.py        # 基础版本
│   │   │   ├── anchor_3dlane_deform.py # Anchor3DLane++ (可变形注意力)
│   │   │   ├── anchor_3dlane_multiframe.py  # 多帧版本
│   │   │   ├── transformer.py          # Transformer 编码器
│   │   │   ├── position_encoding.py    # 位置编码
│   │   │   ├── msda.py                 # 多尺度可变形注意力
│   │   │   ├── tools.py                # 工具函数 (单应性变换)
│   │   │   ├── utils/                  # 工具模块
│   │   │   │   ├── anchor.py           # 锚框生成器
│   │   │   │   └── nms.py              # 3D NMS
│   │   │   └── assigner/               # 锚框 - 真值匹配
│   │   │       ├── topk_assigner.py
│   │   │       └── distance_metric.py
│   │   │
│   │   ├── losses/             # 损失函数
│   │   │   ├── lane_loss.py    # 车道检测损失
│   │   │   ├── kornia_focal.py # Focal Loss
│   │   │   └── ...
│   │   │
│   │   ├── backbones/          # 主干网络
│   │   │   ├── resnet.py
│   │   │   ├── efficientnet.py
│   │   │   └── ...
│   │   ├── necks/              # 颈部网络
│   │   └── builder.py          # 模型构建器
│   │
│   └── utils/                  # 工具函数
│
├── tools/                       # 工具脚本
│   ├── train.py                # 训练入口
│   ├── test.py                 # 测试入口
│   ├── dist_train.sh           # 分布式训练脚本
│   ├── dist_test.sh            # 分布式测试脚本
│   └── convert_datasets/       # 数据集转换工具
│       ├── openlane.py
│       ├── apollosim.py
│       └── once.py
│
├── gen-efficientnet-pytorch/    # EfficientNet 实现 (第三方依赖)
├── pretrained/                  # 预训练权重目录
├── requirements.txt             # Python 依赖
├── setup.py                     # 安装脚本
└── README.md                    # 项目说明
```

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

**锚框结构** (每个锚框 5 + 3×L 维，L=20):
```
[pitch, yaw, start_y, end_y, confidence, x_1...x_L, z_1...z_L, vis_1...vis_L]
```

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

## 八、关键类/函数索引

### 8.1 核心类

| 类名 | 文件 | 说明 |
|------|------|------|
| `Anchor3DLane` | `anchor_3dlane.py` | 基础检测器 |
| `Anchor3DLaneDeform` | `anchor_3dlane_deform.py` | 可变形注意力版本 |
| `Anchor3DLaneMF` | `anchor_3dlane_multiframe.py` | 多帧检测器 |
| `AnchorGenerator` | `utils/anchor.py` | 锚框生成器 |
| `LaneLoss` | `losses/lane_loss.py` | 损失函数 |
| `TopkAssigner` | `assigner/topk_assigner.py` | Top-K 匹配 |

### 8.2 关键函数

| 函数 | 文件 | 说明 |
|------|------|------|
| `feature_extractor()` | `anchor_3dlane.py:226` | 特征提取 (Backbone+Transformer) |
| `cut_anchor_features()` | `anchor_3dlane.py:201` | 锚框特征采样 |
| `get_proposals()` | `anchor_3dlane.py:249` | 生成回归提议 |
| `encoder_decoder()` | `anchor_3dlane.py:289` | 编码器 - 解码器前向 |
| `nms_3d()` | `utils/nms.py:12` | 3D 非极大值抑制 |
| `homography_crop_resize()` | `tools.py` | 单应性变换 |

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

*文档生成时间：2026-04-03*
