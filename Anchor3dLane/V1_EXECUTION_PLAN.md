# BundleLane V1 代码实现草案

## 1. 文档定位

本文档不再讨论方法动机，而专门回答一个问题：

**如何基于当前 `Anchor3DLane++` 代码，以最小侵入方式实现 `BundleLane`。**

目标不是一次性把最终论文系统全写完，而是先做一版：

- 张量定义清晰
- 代码改动路径可控
- 可以快速验证核心判断

## 2. 当前代码中最值得复用的部分

当前 `Anchor3DLane++` 已经提供了很强的工程骨架：

- backbone + FPN 特征提取
- `PAAG / ExpertDecode` 初始 query 生成
- 基于 3D anchors 的投影采样
- iterative regression 主路径
- proposal tensor 与 NMS/evaluator 对接

因此第一版不应推翻以下模块：

- `Anchor3DLanePP.feature_extractor()`
- `obtain_projection_matrix()`
- `cut_anchor_features()`
- `nms()`
- `pred2lanes()` 与 OpenLane evaluator

应当重点改的只有三块：

1. anchor 的几何定义
2. regression 的表示方式
3. 额外监督如何尽量少而稳地接入

## 3. 推荐目录与文件改动

### 3.1 新增文件

- `Anchor3dLane/mmseg/models/lane_detector/anchor_3dlane_bundle.py`
- `Anchor3dLane/tools/vis_bundle_frame.py`
- `configs_v2/openlane/bundlelane_r18.py`

以下文件属于 `V1.1` 再引入，不属于第一轮最小补丁：

- `Anchor3dLane/mmseg/models/losses/lane_loss_bundle.py`
- `Anchor3dLane/mmseg/models/losses/matcher_bundle.py`

### 3.2 需要修改的注册入口

- `Anchor3dLane/mmseg/models/lane_detector/__init__.py`

`V1` 第一轮通常不需要改：

- `Anchor3dLane/mmseg/models/losses/__init__.py`

### 3.3 尽量不改的文件

- `mmseg/datasets/lane_datasets/openlane.py`
- `tools/test.py`
- `tools/deploy_test.py`
- 现有 evaluator 相关脚本

理由很简单：

- 当前评测接口已经稳定
- 第一版不需要改 GT 缓存格式
- 只要最后输出绝对 `x/z/vis` proposal，就能继续走原链路

## 4. 推荐的类设计

### 4.1 模型主类

新模型类建议命名：

`BundleLaneDetector`

注册方式：

```python
@LANENET2S.register_module()
class BundleLaneDetector(Anchor3DLanePP):
    ...
```

不建议直接改 `Anchor3DLanePP` 原类。

原因：

- 方便与 baseline 并行维护
- 方便做逐步 ablation
- 当前 `PEAnchor3DLanePP` 已经验证“以子类形式扩展 detector”是可行路径

### 4.2 参考的代码结构

实现风格建议优先参考：

- `anchor_3dlane_pp.py`
- `anchor_3dlane_pe.py`

其中：

- `anchor_3dlane_pp.py` 提供原始 forward / iterative regression 主链
- `anchor_3dlane_pe.py` 提供“新增共享几何分支并注入 anchor”的范式

## 5. 核心张量设计

### 5.1 保持 proposal 输出格式不变

为了最大限度复用现有后处理和评测，第一版 proposal tensor 仍保持：

```text
[meta(5), x_1..x_L, z_1..z_L, vis_1..vis_L, cls_logits(C)]
```

这意味着：

- 模型内部可以在 intrinsic space 工作
- 但输出给 loss/NMS/evaluator 时，必须能恢复绝对 `x/z/vis`

这条约束非常重要，因为它意味着：

- `V1` 可以继续直接复用原始 `LaneLossV2`
- 第一版不需要一开始就重写整套 loss / matcher
- 新表示先通过“恢复后的 absolute proposal”接受稳定监督

### 5.2 新增的中间输出

建议在 `encoder_decoder()` 的输出字典中新增：

```python
output = {
    'reg_proposals': ...,
    'anchors': ...,
    'bundle_frame': {
        'x_ref': Tensor[B, L],
        'h': Tensor[B, L],
        'bank': Tensor[B, L],
        'alpha_x': Tensor[B, Kx],
        'alpha_h': Tensor[B, Kh],
        'alpha_b': Tensor[B, Kb],
    },
    'intrinsic_preds': [
        [
            {
                'd': Tensor[B, N, L],
                'r': Tensor[B, N, L],
                'span': Tensor[B, N, 2],
                'v': Tensor[B, N, L],
            }
            for feat_idx in ...
        ]
        for iter_idx in ...
    ],
}
```

这里：

- `bundle_frame` 是全图共享量
- `intrinsic_preds` 保存中间内禀变量，供新 loss 和 matcher 使用

## 6. 子模块设计

### 6.1 `BundleFrameHead`

建议新增：

```python
class BundleFrameHead(nn.Module):
    def __init__(self, in_channels, hidden_dim, basis_dims):
        ...
    def forward(self, feat):
        return alpha_x, alpha_h, alpha_b
```

功能：

- 从深层特征预测 frame basis 系数

辅助函数：

- `_build_basis(name, dim, y_steps)`
- `_decode_frame(alpha_x, alpha_h, alpha_b)`

解码结果：

- `x_ref`
- `h`
- `bank`

### 6.2 `BundleAnchorGenerator`

建议在新模型文件中实现，不必第一版就抽到公共 utils。

接口可设计为：

```python
class BundleAnchorGenerator(AnchorGenerator_torch):
    def generate_anchors_batch(
        self,
        d0s,
        yaws,
        pitches,
        x_ref,
        h_ref,
        bank,
    ):
        ...
```

输出仍返回绝对 proposal-compatible anchors。

### 6.3 `SpanHead`

新增一个非常轻的 head：

```python
self.reg_span_layer = nn.ModuleDict()
```

每层每次迭代输出：

```python
span_logits -> [B, N, 2]
```

第一版建议：

- 直接回归 `s,e` 的连续索引
- 再通过 clamp 限制到 `[0, L-1]`

不要第一版就把 span 做成离散分类，工程负担更高。

## 7. `BundleLaneDetector` 的实现草图

### 7.1 `__init__`

需要新增的配置项：

```python
bundle_cfg = dict(
    enabled=True,
    basis_dim_x=6,
    basis_dim_h=4,
    basis_dim_b=3,
    hidden_dim=128,
    center_tau=8.0,
    inject_iters=[0],
    span_tau=1.0,
)
```

初始化中需要：

- build `BundleFrameHead`
- build basis buffers
- build `BundleAnchorGenerator`
- build `reg_span_layer`

### 7.2 `_build_bundle_targets()`

建议写成模型内部或 loss 内部共用函数。

输入：

- 单张图 `gt_3dlanes`

输出：

```python
{
    'x_ref': Tensor[L],
    'h': Tensor[L],
    'bank': Tensor[L],
    'frame_mask': Tensor[L],
    'd_gt': Tensor[M, L],
    'r_gt': Tensor[M, L],
    'span_gt': Tensor[M, 2],
    'vis_inside_span': Tensor[M, L],
}
```

关键细节：

- `x_ref`：加权中位数
- `h/bank`：按 `z = h + bank * (x - x_ref)` 拟合
- `d_gt = x - x_ref`
- `r_gt = z - h - bank * d_gt`
- `span_gt = [first_visible_idx, last_visible_idx]`

### 7.3 `_predict_bundle_frame()`

输入：

- `anchor_feats[-1]`

输出：

- `x_ref`
- `h`
- `bank`
- 对应系数

伪代码：

```python
alpha_x, alpha_h, alpha_b = self.bundle_frame_head(anchor_feats[-1])
x_ref = alpha_x @ self.bundle_basis_x.T
h = alpha_h @ self.bundle_basis_h.T
bank = alpha_b @ self.bundle_basis_b.T
```

### 7.4 `get_proposals()`

这是最关键的改造点。

当前 `get_proposals()` 直接预测：

- `cls_logits`
- `reg_x`
- `reg_z`
- `reg_vis`

新逻辑建议改为：

1. 预测：
   - `cls_logits`
   - `delta_d`
   - `delta_r`
   - `v`
   - `span`
2. 若 `reg_prior=True`：
   - 用 `BundleAnchorGenerator` 结合 `x_ref/h/bank` 生成绝对 anchors
3. 在 intrinsic space 解码：
   - `d_pred = d_anchor + delta_d`
   - `r_pred = delta_r`
4. 恢复绝对 `x/z/vis`
5. 将绝对 `x/z/vis` 写入 proposal tensor
6. 将 `d/r/span/v` 存入额外字典返回

这里需要特别强调：

- `V1` 的关键不是让 loss 直接吃 `d/r/span`
- 而是先确保“通过 intrinsic 解码恢复出来的 absolute proposal”本身是稳定的
- 因此第一版优先保证 proposal tensor 兼容和前向可解释，再考虑更激进的 intrinsic supervision

返回值建议从：

```python
return reg_proposals, cur_anchors
```

改成：

```python
return reg_proposals, cur_anchors, intrinsic_dict
```

然后在 `encoder_decoder()` 里把 `intrinsic_dict` 收集到 `output['intrinsic_preds']`。

### 7.5 `encoder_decoder()`

新增流程：

```python
bundle_frame = self._predict_bundle_frame(anchor_feats[-1])
```

然后在第一个 stage 的 first feature level 中，将 bundle frame 注入：

- anchor generation
- intrinsic decoding

注意：

- 第一版只在 `iter_idx == 0 and feat_idx == 0` 用 frame-conditioned anchor
- 后续 stage 先保持 residual refinement，不做更复杂耦合

## 8. Loss 实现草案

### 8.1 新损失类

`V1` 第一轮不建议先写新的总损失类。

更稳的做法是：

- 继续复用原始 `LaneLossV2`，把它作为 `L_base_abs`
- 在模型类自己的 `loss()` 里，额外加：
  - `frame loss`
  - `span loss`
  - `residual_small loss`

也就是说，第一版先走和 `PEAnchor3DLanePP` 类似的路线：

- 主 proposal loss 仍走原链路
- 新表示专属 loss 在 detector `loss()` 里增量加入

只有当表示本身已经证明有效时，再把 bundle 专属逻辑沉淀成新损失类。

因此，下面这个新损失类属于 `V1.1` 规划，而不是 `V1` 的第一轮必做项。

建议命名：

`LaneLossBundle`

不要在原 `LaneLossV2` 里硬塞逻辑，否则很快会失控。

### 8.2 输入接口

建议 `forward()` 输入：

```python
def forward(self, proposals_list, targets, bundle_frame=None, intrinsic_preds=None):
    ...
```

其中：

- `proposals_list` 保持兼容原逻辑
- `bundle_frame` 提供 `x_ref/h/bank`
- `intrinsic_preds` 提供 `d/r/span/v`

### 8.3 损失拆分

对于 `V1`，建议只实现：

- `compute_frame_loss()`
- `compute_span_loss()`
- `compute_residual_small_loss()`

它们直接在 detector `loss()` 中调用即可。

完整 `LaneLossBundle` 里再考虑实现：

- `compute_frame_loss()`
- `compute_offset_loss()`
- `compute_residual_loss()`
- `compute_span_loss()`
- `compute_vis_loss()`
- `compute_order_loss()`
- `compute_nocross_loss()`

第一版优先级：

1. `frame`
2. `span`
3. `residual_small`
4. 原始 absolute `cls/x/z/vis`

以下内容推迟到 `V1.1`：

5. `offset`
6. `residual`
7. `vis`
8. `order`
9. `nocross`

### 8.4 建议初始权重

```python
loss_weights = dict(
    cls_loss=1.0,
    frame_x_loss=0.3,
    frame_h_loss=0.3,
    frame_bank_loss=0.2,
    frame_smooth_loss=0.1,
    span_loss=0.2,
    residual_small_loss=0.05,
)
```

若进入 `V1.1`，再扩展为完整权重表。

## 9. Matcher 实现草案

### 9.1 新 matcher 类

`V1` 第一轮不建议先改 matcher。

更稳的做法是：

- 继续复用原始 `HungarianMatcher`
- 先验证 bundle frame 与 intrinsic decoding 是否对最终 absolute proposal 有帮助

因此 `IntrinsicHungarianMatcher` 属于 `V1.1` 扩展项。

建议命名：

`IntrinsicHungarianMatcher`

可以新建文件 `matcher_bundle.py`，也可以放在现有 `matcher.py` 里。

为了避免影响旧逻辑，`V1.1` 建议新文件。

### 9.2 接口

```python
def forward(
    self,
    cls_logits,
    d_pred,
    r_pred,
    span_pred,
    targets,
    use_sigmoid=False,
):
    ...
```

### 9.3 目标构造

对每个 GT lane 构造：

- `d_gt`
- `r_gt`
- `span_gt`
- `category`

### 9.4 cost 设计

```python
C_cls  = focal or softmax class cost
C_d    = masked L1(d_pred, d_gt)
C_r    = masked L1(r_pred, r_gt)
C_span = 1 - interval_iou(span_pred, span_gt)
C_ord  = |mean(d_pred) - mean(d_gt)|
```

建议总 cost：

```python
C = 3.0 * C_cls + 1.0 * C_d + 0.5 * C_r + 0.5 * C_span + 0.2 * C_ord
```

`V1.1` 中也先不把 `nocross` 放进 matcher，只放进 loss。

## 10. 配置文件草案

新配置文件：

`configs_v2/openlane/bundlelane_r18.py`

建议基于现有：

- `configs_v2/openlane/anchor3dlane++_r18.py`

主要变化：

```python
model = dict(
    type='BundleLaneDetector',
    ...
    bundle_cfg=dict(
        enabled=True,
        basis_dim_x=6,
        basis_dim_h=4,
        basis_dim_b=3,
        hidden_dim=128,
        center_tau=8.0,
        inject_iters=[0],
        span_tau=1.0,
    ),
    loss_lane=[
        [lane_loss_prior, lane_loss_prior],
        [lane_loss, lane_loss],
        [lane_loss, lane_loss],
    ],
)
```

建议额外准备三个 ablation config：

- `bundlelane_frame_only_r18.py`
- `bundlelane_decode_only_r18.py`
- `bundlelane_match_r18.py`  `# V1.1`

## 11. 建议的开发顺序

### P0. 设计验证

先不写全量功能，只做：

- `BundleFrameHead`
- bundle target 构造
- `frame loss`

目标：

- 验证 `x_ref/h/bank` 可学
- 写可视化脚本

### P1. Frame-conditioned anchor

新增：

- `BundleAnchorGenerator`
- stage 0 注入

目标：

- 不改 matcher
- 不改 proposal 格式
- 先看远距 `z` 与整体稳定性

### P2. Intrinsic decoding

新增：

- `span head`
- `d/r` 解码
- proposal 绝对坐标恢复

目标：

- forward 跑通
- evaluator 结果可正常输出
- 在继续使用原始 `LaneLossV2` 的情况下，验证 absolute proposal 是否已受益

### P2.5 精简版损失接入

新增：

- `frame loss`
- `span loss`
- `residual_small loss`

目标：

- 不重写 matcher
- 不重写总损失类
- 先验证“新表示 + 少量附加监督”是否可稳定收敛

### P3. Intrinsic matcher（V1.1）

新增：

- `IntrinsicHungarianMatcher`
- `LaneLossBundle`

目标：

- 看短车道和复杂结构是否改善

### P4. Structure regularization（V1.1）

新增：

- `order_loss`
- `nocross_loss`

目标：

- 提升 merge/split 和相邻 lane 稳定性

## 12. 可视化与调试建议

必须补的可视化：

1. `x_ref/h/bank` 曲线
2. `d_gt` 与 `d_pred`
3. `r_gt` 与 `r_pred`
4. `span_gt` 与 `span_pred`
5. 解码前后绝对 `x/z` lane 对比

如果这些图看不懂，loss 再漂亮也不要急着跑大训练。

## 13. 成功标准

第一版不以最终 SOTA 为唯一目标，而以“问题分解是否成立”为主。

优先判断：

1. `frame` 是否稳定可视化
2. `r` 是否明显小于直接回归 `z`
3. 在保持原 matcher 的前提下，是否已经改善：
   - `z_error_far`
   - 短车道稳定性
   - merge/split 附近的形状质量

如果这三点里至少两点成立，这条线就值得继续。

## 14. 第一轮最小补丁集合

如果只做最小可运行版本，建议先改这些文件：

- `mmseg/models/lane_detector/anchor_3dlane_bundle.py`
- `mmseg/models/lane_detector/__init__.py`
- `configs_v2/openlane/bundlelane_r18.py`
- `tools/vis_bundle_frame.py`

第一轮先不要动 dataset cache。

第一轮也先不要新增：

- `lane_loss_bundle.py`
- `matcher_bundle.py`
- `mmseg/models/losses/__init__.py`

## 15. 一句话总结

`BundleLane` 的代码落地关键不在于“加几个头”，而在于：

- 内部工作空间切到 intrinsic space
- 外部接口保持 absolute proposal 兼容

而 `V1` 最关键的工程策略则是：

- 先复用原始 `LaneLossV2 + HungarianMatcher`
- 只少量加入 bundle 专属监督
- 等表示本身验证有效后，再逐步把 loss 和 matcher 真正迁到 intrinsic space

只要这条边界守住，工程复杂度就能控制住，同时还能保留当前 `Anchor3DLane++` 代码库的大部分资产。
