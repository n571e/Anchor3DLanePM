import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import force_fp32

from ..builder import LANENET2S
from .anchor_3dlane_pp import Anchor3DLanePP
from .utils import AnchorGenerator_torch


class BundleAnchorGenerator(AnchorGenerator_torch):
    def generate_anchors_batch(self,
                               xs=None,
                               yaws=None,
                               pitches=None,
                               x_ref=None,
                               h_ref=None,
                               bank=None):
        if x_ref is None or h_ref is None or bank is None:
            return super().generate_anchors_batch(xs=xs, yaws=yaws, pitches=pitches)

        bs, num_anchors = xs.shape
        anchors = torch.zeros(
            bs, num_anchors, 5 + self.anchor_len * 3, device=xs.device)
        y_steps = self.y_steps[None, None, :].repeat(bs, num_anchors, 1).to(xs.device)
        xs = torch.clamp(xs, -1, 1)
        yaws = torch.clamp(yaws, -1, 1)
        pitches = torch.clamp(pitches, -1, 1)
        x_ref = x_ref[:, None, :].repeat(1, num_anchors, 1)
        h_ref = h_ref[:, None, :].repeat(1, num_anchors, 1)
        bank = bank[:, None, :].repeat(1, num_anchors, 1)

        anchors[..., 2] = yaws
        anchors[..., 3] = pitches
        anchors[..., 4] = xs

        start_x = (xs[..., None] + 1) / 2 * (self.x_max - self.x_min) + self.x_min
        start_offset = start_x - x_ref[..., :1]
        yaw_delta = (y_steps - self.y_steps[0]) * torch.tan(yaws[..., None] * math.pi)
        x_coords = x_ref + start_offset + yaw_delta
        anchors[..., 5:5 + self.anchor_len] = x_coords

        pitch_delta = (y_steps - self.y_steps[0]) * torch.tan(pitches[..., None] * math.pi)
        z_coords = h_ref + bank * (x_coords - x_ref) + pitch_delta
        anchors[..., 5 + self.anchor_len:5 + self.anchor_len * 2] = z_coords
        return anchors


class BundleFrameHead(nn.Module):
    def __init__(self, in_channels, hidden_dim, basis_dims):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.shared = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.head_x = nn.Linear(hidden_dim, basis_dims['x'])
        self.head_h = nn.Linear(hidden_dim, basis_dims['h'])
        self.head_b = nn.Linear(hidden_dim, basis_dims['b'])

    def forward(self, feature):
        pooled = self.pool(feature).flatten(1)
        hidden = self.shared(pooled)
        return self.head_x(hidden), self.head_h(hidden), self.head_b(hidden)


@LANENET2S.register_module()
class BundleLaneDetector(Anchor3DLanePP):
    def __init__(self, bundle_cfg=None, **kwargs):
        anchor_cfg = copy.deepcopy(kwargs.get('anchor_cfg'))
        super().__init__(**kwargs)
        self.bundle_cfg = copy.deepcopy(bundle_cfg) if bundle_cfg is not None else {}
        self.use_bundle_frame = bool(self.bundle_cfg.get('enabled', False))
        self.bundle_center_tau = float(self.bundle_cfg.get('center_tau', 8.0))
        self.bundle_target_smooth_kernel = int(self.bundle_cfg.get('target_smooth_kernel', 3))
        self.bundle_inject_iters = set(self.bundle_cfg.get('inject_iters', [0]))
        self.frame_x_loss_weight = float(self.bundle_cfg.get('frame_x_loss_weight', 0.3))
        self.frame_h_loss_weight = float(self.bundle_cfg.get('frame_h_loss_weight', 0.3))
        self.frame_bank_loss_weight = float(self.bundle_cfg.get('frame_bank_loss_weight', 0.2))
        self.frame_smooth_loss_weight = float(self.bundle_cfg.get('frame_smooth_loss_weight', 0.1))

        if self.use_bundle_frame:
            if anchor_cfg is None:
                raise ValueError('bundle_cfg.enabled=True requires anchor_cfg in the model config.')
            self.anchor_generator = BundleAnchorGenerator(
                anchor_cfg,
                x_min=self.x_min,
                x_max=self.x_max,
                y_max=int(self.y_steps[-1]),
                y_steps=self.y_steps,
                norm=(self.x_norm, self.y_norm, self.z_norm),
            )
            basis_dims = dict(
                x=int(self.bundle_cfg.get('basis_dim_x', 6)),
                h=int(self.bundle_cfg.get('basis_dim_h', 4)),
                b=int(self.bundle_cfg.get('basis_dim_b', 3)),
            )
            in_channels = int(self.bundle_cfg.get('in_channels', self.anchor_feat_channels))
            hidden_dim = int(self.bundle_cfg.get('hidden_dim', max(self.anchor_feat_channels, 64)))
            self.bundle_frame_head = BundleFrameHead(in_channels, hidden_dim, basis_dims)
            self.register_buffer(
                'bundle_basis_x',
                self._build_bundle_basis(basis_dims['x'], normalize_center=True),
                persistent=False,
            )
            self.register_buffer(
                'bundle_basis_h',
                self._build_bundle_basis(basis_dims['h'], normalize_center=True),
                persistent=False,
            )
            self.register_buffer(
                'bundle_basis_b',
                self._build_bundle_basis(basis_dims['b'], normalize_center=False),
                persistent=False,
            )
        else:
            self.bundle_frame_head = None
            self.register_buffer('bundle_basis_x', torch.empty(0), persistent=False)
            self.register_buffer('bundle_basis_h', torch.empty(0), persistent=False)
            self.register_buffer('bundle_basis_b', torch.empty(0), persistent=False)

    def _build_bundle_basis(self, basis_dim, normalize_center=True):
        y_steps = torch.as_tensor(self.y_steps, dtype=torch.float32)
        if normalize_center:
            y_norm = y_steps - y_steps.mean()
        else:
            y_norm = y_steps - y_steps[0]
        y_norm = y_norm / y_norm.abs().max().clamp_min(1.0)
        basis = [torch.ones_like(y_norm)]
        for degree in range(1, basis_dim):
            basis.append(y_norm ** degree)
        return torch.stack(basis, dim=1)

    def _predict_bundle_frame(self, feature):
        alpha_x, alpha_h, alpha_b = self.bundle_frame_head(feature)
        x_ref = alpha_x @ self.bundle_basis_x.t()
        h_ref = alpha_h @ self.bundle_basis_h.t()
        bank = alpha_b @ self.bundle_basis_b.t()
        return {
            'x_ref': x_ref,
            'h': h_ref,
            'bank': bank,
            'alpha_x': alpha_x,
            'alpha_h': alpha_h,
            'alpha_b': alpha_b,
        }

    def _use_bundle_injection(self, iter_idx, bundle_frame):
        return self.use_bundle_frame and bundle_frame is not None and iter_idx in self.bundle_inject_iters

    def _weighted_median(self, values, weights):
        sort_idx = torch.argsort(values)
        values = values[sort_idx]
        weights = weights[sort_idx]
        cdf = torch.cumsum(weights, dim=0) / weights.sum().clamp_min(1e-6)
        median_idx = torch.searchsorted(cdf, cdf.new_tensor([0.5])).item()
        return values[min(median_idx, values.numel() - 1)]

    def _fit_height_bank(self, x_vals, z_vals, x_ref):
        if x_vals.numel() == 1:
            return z_vals[0], z_vals.new_tensor(0.0)

        centered_x = x_vals - x_ref
        weights = torch.exp(-torch.abs(centered_x) / self.bundle_center_tau).clamp_min(1e-4)
        design = torch.stack([torch.ones_like(centered_x), centered_x], dim=1)
        weighted_design = design * weights[:, None]
        lhs = design.t() @ weighted_design
        rhs = design.t() @ (weights * z_vals)
        lhs = lhs + torch.eye(2, device=lhs.device, dtype=lhs.dtype) * 1e-4
        params = torch.linalg.solve(lhs, rhs)
        return params[0], params[1]

    def _interpolate_target(self, values, mask):
        valid_idx = torch.nonzero(mask > 0.5, as_tuple=False).flatten()
        if valid_idx.numel() < 2:
            return values, mask
        interpolated = values.clone()
        new_mask = mask.clone()
        for start, end in zip(valid_idx[:-1], valid_idx[1:]):
            gap = int(end.item() - start.item())
            if gap <= 1:
                continue
            start_val = values[start]
            end_val = values[end]
            for offset in range(1, gap):
                ratio = offset / gap
                interpolated[start + offset] = start_val + ratio * (end_val - start_val)
                new_mask[start + offset] = 1.0
        return interpolated, new_mask

    def _smooth_target(self, values, mask):
        if self.bundle_target_smooth_kernel <= 1:
            return values
        radius = self.bundle_target_smooth_kernel // 2
        smoothed = values.clone()
        for idx in range(values.shape[0]):
            if mask[idx] < 0.5:
                continue
            left = max(0, idx - radius)
            right = min(values.shape[0], idx + radius + 1)
            valid = mask[left:right] > 0.5
            if valid.any():
                smoothed[idx] = values[left:right][valid].mean()
        return smoothed

    @torch.no_grad()
    def _build_bundle_frame_target(self, target, device):
        target = target.to(device)
        target = target[target[:, 1] > 0]
        x_ref = torch.zeros(self.anchor_len, device=device)
        h_ref = torch.zeros(self.anchor_len, device=device)
        bank = torch.zeros(self.anchor_len, device=device)
        mask = torch.zeros(self.anchor_len, device=device)
        if target.numel() == 0:
            return x_ref, h_ref, bank, mask

        gt_anchor_len = max((target.shape[1] - 5) // 3, 0)
        if gt_anchor_len == 0:
            return x_ref, h_ref, bank, mask

        sample_idx = torch.as_tensor(
            np.round(self.y_steps).astype(np.int64) - 1,
            device=device,
            dtype=torch.long,
        ).clamp_(0, gt_anchor_len - 1)
        x_all = target[:, 5:5 + gt_anchor_len].index_select(1, sample_idx)
        z_all = target[:, 5 + gt_anchor_len:5 + 2 * gt_anchor_len].index_select(1, sample_idx)
        vis_all = target[:, 5 + 2 * gt_anchor_len:5 + 3 * gt_anchor_len].index_select(1, sample_idx) > 0.5

        for step in range(self.anchor_len):
            valid = vis_all[:, step]
            if not valid.any():
                continue
            x_vals = x_all[valid, step]
            z_vals = z_all[valid, step]
            weights = torch.exp(-torch.abs(x_vals) / self.bundle_center_tau).clamp_min(1e-4)
            x_center = self._weighted_median(x_vals, weights)
            h_val, bank_val = self._fit_height_bank(x_vals, z_vals, x_center)
            x_ref[step] = x_center
            h_ref[step] = h_val
            bank[step] = bank_val
            mask[step] = 1.0

        x_ref, mask = self._interpolate_target(x_ref, mask)
        h_ref, _ = self._interpolate_target(h_ref, mask)
        bank, _ = self._interpolate_target(bank, mask)
        x_ref = self._smooth_target(x_ref, mask)
        h_ref = self._smooth_target(h_ref, mask)
        bank = self._smooth_target(bank, mask)
        return x_ref, h_ref, bank, mask

    def _compute_bundle_frame_losses(self, bundle_frame, gt_3dlanes):
        if not self.use_bundle_frame or bundle_frame is None:
            return {}

        targets_x = []
        targets_h = []
        targets_bank = []
        masks = []
        for target in gt_3dlanes:
            x_ref, h_ref, bank, mask = self._build_bundle_frame_target(target, bundle_frame['x_ref'].device)
            targets_x.append(x_ref)
            targets_h.append(h_ref)
            targets_bank.append(bank)
            masks.append(mask)
        targets_x = torch.stack(targets_x, dim=0)
        targets_h = torch.stack(targets_h, dim=0)
        targets_bank = torch.stack(targets_bank, dim=0)
        masks = torch.stack(masks, dim=0)
        mask_sum = masks.sum()
        zero = bundle_frame['x_ref'].sum() * 0
        if mask_sum <= 0:
            return {
                'bundle_frame_x_loss': zero,
                'bundle_frame_h_loss': zero,
                'bundle_frame_bank_loss': zero,
                'bundle_frame_smooth_loss': zero,
            }

        x_loss = F.smooth_l1_loss(bundle_frame['x_ref'], targets_x, reduction='none')
        h_loss = F.smooth_l1_loss(bundle_frame['h'], targets_h, reduction='none')
        bank_loss = F.smooth_l1_loss(bundle_frame['bank'], targets_bank, reduction='none')
        x_loss = (x_loss * masks).sum() / mask_sum.clamp_min(1.0)
        h_loss = (h_loss * masks).sum() / mask_sum.clamp_min(1.0)
        bank_loss = (bank_loss * masks).sum() / mask_sum.clamp_min(1.0)

        if self.anchor_len > 2:
            smooth_mask = masks[:, 2:] * masks[:, 1:-1] * masks[:, :-2]
            second_diff_x = bundle_frame['x_ref'][:, 2:] - 2 * bundle_frame['x_ref'][:, 1:-1] + bundle_frame['x_ref'][:, :-2]
            second_diff_h = bundle_frame['h'][:, 2:] - 2 * bundle_frame['h'][:, 1:-1] + bundle_frame['h'][:, :-2]
            second_diff_b = bundle_frame['bank'][:, 2:] - 2 * bundle_frame['bank'][:, 1:-1] + bundle_frame['bank'][:, :-2]
            if smooth_mask.sum() > 0:
                smooth_loss = (
                    second_diff_x.abs() + second_diff_h.abs() + second_diff_b.abs()
                )
                smooth_loss = (smooth_loss * smooth_mask).sum() / smooth_mask.sum().clamp_min(1.0)
            else:
                smooth_loss = zero
        else:
            smooth_loss = zero

        return {
            'bundle_frame_x_loss': x_loss * self.frame_x_loss_weight,
            'bundle_frame_h_loss': h_loss * self.frame_h_loss_weight,
            'bundle_frame_bank_loss': bank_loss * self.frame_bank_loss_weight,
            'bundle_frame_smooth_loss': smooth_loss * self.frame_smooth_loss_weight,
        }

    @force_fp32()
    def get_proposals(self,
                      project_matrixes,
                      anchor_feat,
                      feat_idx,
                      proposals_prev,
                      feat_size,
                      iter_idx,
                      reg_prior=False,
                      bundle_frame=None):
        batch_size = project_matrixes.shape[0]
        xs, ys, zs = self.compute_anchor_cut_indices(proposals_prev, self.feat_y_steps)
        batch_anchor_features, _ = self.cut_anchor_features(
            anchor_feat, project_matrixes, xs, ys, zs, self.anchor_feat_len, feat_size)

        if self.with_pos != 'none':
            xs = xs / self.x_norm
            ys = ys / self.y_norm
            zs = zs / self.z_norm
            xyz = torch.stack([xs, ys, zs], -1)
            batch_pos_features = self.position_encoder(xyz)
            batch_pos_features = batch_pos_features.transpose(1, 2).reshape(
                batch_size, self.anchor_feat_channels, self.anchor_num, self.anchor_feat_len)
            if self.with_pos == 'add':
                batch_anchor_features = batch_anchor_features + batch_pos_features
            elif self.with_pos == 'pcat':
                batch_anchor_features = torch.cat([batch_anchor_features, batch_pos_features], 1)
                batch_anchor_features = batch_anchor_features.permute(0, 2, 3, 1)
                batch_anchor_features = self.fuse_pos[f'layer_{feat_idx}'][iter_idx](batch_anchor_features)
                batch_anchor_features = batch_anchor_features.permute(0, 3, 1, 2)
            else:
                batch_anchor_features = torch.cat([batch_anchor_features, batch_pos_features], 1)

        batch_anchor_features = batch_anchor_features.transpose(1, 2)
        batch_anchor_features = batch_anchor_features.flatten(2, 3)
        batch_anchor_features = self.dynamic_head[f'layer_{feat_idx}'][iter_idx](batch_anchor_features)
        batch_anchor_features = batch_anchor_features.flatten(0, 1)

        cls_logits = self.cls_layer[f'layer_{feat_idx}'][iter_idx](batch_anchor_features)
        cls_logits = cls_logits.reshape(batch_size, -1, cls_logits.shape[1])
        reg_x = self.reg_x_layer[f'layer_{feat_idx}'][iter_idx](batch_anchor_features)
        reg_x = reg_x.reshape(batch_size, -1, reg_x.shape[1])
        reg_z = self.reg_z_layer[f'layer_{feat_idx}'][iter_idx](batch_anchor_features)
        reg_z = reg_z.reshape(batch_size, -1, reg_z.shape[1])
        reg_vis = self.reg_vis_layer[f'layer_{feat_idx}'][iter_idx](batch_anchor_features)
        reg_vis = torch.sigmoid(reg_vis).reshape(batch_size, -1, reg_vis.shape[1])

        if reg_prior:
            reg_lane_priors = self.reg_prior_layer[iter_idx](batch_anchor_features)
            reg_lane_priors = torch.tanh(reg_lane_priors.reshape(batch_size, -1, 3))
            lane_priors = reg_lane_priors + proposals_prev[..., 2:5]
            if self._use_bundle_injection(iter_idx, bundle_frame):
                cur_anchors = self.anchor_generator.generate_anchors_batch(
                    lane_priors[:, :, 2],
                    lane_priors[:, :, 0],
                    lane_priors[:, :, 1],
                    x_ref=bundle_frame['x_ref'],
                    h_ref=bundle_frame['h'],
                    bank=bundle_frame['bank'],
                )
            else:
                cur_anchors = self.anchor_generator.generate_anchors_batch(
                    lane_priors[:, :, 2], lane_priors[:, :, 0], lane_priors[:, :, 1])
            reg_proposals = torch.zeros(
                batch_size,
                self.anchor_num,
                5 + self.anchor_len * 3 + self.num_category,
                device=project_matrixes.device,
            )
            reg_proposals[:, :, :5 + self.anchor_len * 3] += cur_anchors[:, :, :5 + self.anchor_len * 3]
        else:
            reg_proposals = torch.zeros(
                batch_size,
                self.anchor_num,
                5 + self.anchor_len * 3 + self.num_category,
                device=project_matrixes.device,
            )
            reg_proposals[:, :, :5 + self.anchor_len * 3] += proposals_prev[:, :, :5 + self.anchor_len * 3]

        reg_proposals[:, :, 5:5 + self.anchor_len] += reg_x
        reg_proposals[:, :, 5 + self.anchor_len:5 + self.anchor_len * 2] += reg_z
        reg_proposals[:, :, 5 + self.anchor_len * 2:5 + self.anchor_len * 3] = reg_vis
        reg_proposals[:, :, 5 + self.anchor_len * 3:5 + self.anchor_len * 3 + self.num_category] = cls_logits

        if reg_prior:
            return reg_proposals, cur_anchors
        return reg_proposals, None

    def encoder_decoder(self, img, mask, gt_project_matrix, **kwargs):
        batch_size = img.shape[0]
        anchor_feats = self.feature_extractor(img, mask)
        bundle_frame = None
        if self.use_bundle_frame:
            bundle_frame = self._predict_bundle_frame(anchor_feats[-1])

        reg_proposals_all = []
        anchors_all = []

        for iter_idx in range(self.iter_reg):
            reg_proposals_layer = []
            anchors_layer = []
            for feat_idx, feat_size in enumerate(self.feat_sizes[::-1]):
                project_matrixes = self.obtain_projection_matrix(gt_project_matrix, feat_size)
                project_matrixes = torch.stack(project_matrixes, dim=0)
                select_idx = self.feat_num - 1 - feat_idx
                if iter_idx == 0:
                    if feat_idx == 0:
                        yaw_weights, pitch_weights, x_weights = self.expert_layer(anchor_feats[-1])
                        init_yaw = self.init_proposals_yaws.weight.unsqueeze(0).repeat(batch_size, 1, 1)
                        init_pitch = self.init_proposals_pitches.weight.unsqueeze(0).repeat(batch_size, 1, 1)
                        init_xs = self.init_proposals_xs.weight.unsqueeze(0).repeat(batch_size, 1, 1)
                        yaws = (yaw_weights @ init_yaw).squeeze(-1)
                        pitches = (pitch_weights @ init_pitch).squeeze(-1)
                        xs = (x_weights @ init_xs).squeeze(-1)
                        if self._use_bundle_injection(iter_idx, bundle_frame):
                            anchors = self.anchor_generator.generate_anchors_batch(
                                xs,
                                yaws,
                                pitches,
                                x_ref=bundle_frame['x_ref'],
                                h_ref=bundle_frame['h'],
                                bank=bundle_frame['bank'],
                            )
                        else:
                            anchors = self.anchor_generator.generate_anchors_batch(xs, yaws, pitches)
                        reg_proposals, update_anchors = self.get_proposals(
                            project_matrixes,
                            anchor_feats[select_idx],
                            feat_idx,
                            anchors,
                            feat_size,
                            iter_idx,
                            reg_prior=True,
                            bundle_frame=bundle_frame,
                        )
                        reg_proposals_layer.append(reg_proposals)
                        anchors_layer.append(update_anchors)
                    else:
                        proposals_prev = reg_proposals_layer[feat_idx - 1]
                        reg_proposals, _ = self.get_proposals(
                            project_matrixes,
                            anchor_feats[select_idx],
                            feat_idx,
                            proposals_prev,
                            feat_size,
                            iter_idx,
                            reg_prior=False,
                            bundle_frame=bundle_frame,
                        )
                        reg_proposals_layer.append(reg_proposals)
                        anchors_layer.append(proposals_prev[:, :, :5 + self.anchor_len * 3])
                else:
                    if feat_idx == 0:
                        proposals_prev = reg_proposals_all[iter_idx - 1][0]
                        reg_proposals, update_anchors = self.get_proposals(
                            project_matrixes,
                            anchor_feats[select_idx],
                            feat_idx,
                            proposals_prev,
                            feat_size,
                            iter_idx,
                            reg_prior=True,
                            bundle_frame=bundle_frame,
                        )
                        reg_proposals_layer.append(reg_proposals)
                        anchors_layer.append(update_anchors)
                    else:
                        proposals_prev = reg_proposals_layer[feat_idx - 1]
                        reg_proposals, _ = self.get_proposals(
                            project_matrixes,
                            anchor_feats[select_idx],
                            feat_idx,
                            proposals_prev,
                            feat_size,
                            iter_idx,
                            reg_prior=False,
                            bundle_frame=bundle_frame,
                        )
                        reg_proposals_layer.append(reg_proposals)
                        anchors_layer.append(proposals_prev[:, :, :5 + self.anchor_len * 3])
            reg_proposals_all.append(reg_proposals_layer)
            anchors_all.append(anchors_layer)

        output = {'reg_proposals': reg_proposals_all, 'anchors': anchors_all}
        if self.use_bundle_frame:
            output['bundle_frame'] = bundle_frame
        return output

    def loss(self, output, gt_3dlanes):
        losses, other_vars = super().loss(output, gt_3dlanes)
        losses.update(self._compute_bundle_frame_losses(output.get('bundle_frame'), gt_3dlanes))
        return losses, other_vars
