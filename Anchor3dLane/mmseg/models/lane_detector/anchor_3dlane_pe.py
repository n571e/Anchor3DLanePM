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


class ProfileAnchorGenerator(AnchorGenerator_torch):
    def generate_anchors_batch(self, xs=None, yaws=None, pitches=None, base_z=None):
        bs, num_anchors = xs.shape
        anchors = torch.zeros(
            bs, num_anchors, 5 + self.anchor_len * 3, device=xs.device)
        y_steps = self.y_steps[None, None, :].repeat(bs, num_anchors, 1).to(xs.device)
        xs = torch.clamp(xs, -1, 1)
        yaws = torch.clamp(yaws, -1, 1)
        pitches = torch.clamp(pitches, -1, 1)
        anchors[..., 2] = yaws
        anchors[..., 3] = pitches
        anchors[..., 4] = xs
        anchors[..., 5:5 + self.anchor_len] = (
            (xs[..., None] + 1) / 2 * (self.x_max - self.x_min)
            + self.x_min
            + (y_steps - 1) * torch.tan(yaws[..., None] * math.pi)
        )
        if base_z is None:
            anchors[..., 5 + self.anchor_len:5 + self.anchor_len * 2] = (
                self.start_z + (y_steps - 1) * torch.tan(pitches[..., None] * math.pi)
            )
        else:
            if base_z.dim() == 2:
                base_z = base_z[:, None, :].repeat(1, num_anchors, 1)
            elif base_z.dim() != 3:
                raise ValueError('base_z must have shape [B, L] or [B, N, L].')
            pitch_residual = (y_steps - self.y_steps[0]) * torch.tan(pitches[..., None] * math.pi)
            anchors[..., 5 + self.anchor_len:5 + self.anchor_len * 2] = base_z + pitch_residual
        return anchors


class ProfileHead(nn.Module):
    def __init__(self, in_channels, hidden_dim, basis_dim):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, basis_dim),
        )

    def forward(self, feature):
        pooled = self.pool(feature).flatten(1)
        return self.mlp(pooled)


@LANENET2S.register_module()
class PEAnchor3DLanePP(Anchor3DLanePP):
    def __init__(self, profile_cfg=None, **kwargs):
        anchor_cfg = copy.deepcopy(kwargs.get('anchor_cfg'))
        super().__init__(**kwargs)
        self.profile_cfg = copy.deepcopy(profile_cfg) if profile_cfg is not None else {}
        self.use_profile_anchor = bool(self.profile_cfg.get('enabled', False))
        self.profile_loss_weight = float(self.profile_cfg.get('loss_weight', 0.5))
        self.profile_smooth_weight = float(self.profile_cfg.get('smooth_weight', 0.1))
        self.profile_center_tau = float(self.profile_cfg.get('center_tau', 8.0))
        self.profile_target_smooth_kernel = int(self.profile_cfg.get('target_smooth_kernel', 3))
        self.profile_inject_iters = set(self.profile_cfg.get('inject_iters', [0]))

        if self.use_profile_anchor:
            if anchor_cfg is None:
                raise ValueError('profile_cfg.enabled=True requires anchor_cfg in the model config.')
            self.anchor_generator = ProfileAnchorGenerator(
                anchor_cfg,
                x_min=self.x_min,
                x_max=self.x_max,
                y_max=int(self.y_steps[-1]),
                y_steps=self.y_steps,
                norm=(self.x_norm, self.y_norm, self.z_norm),
            )
            basis_dim = int(self.profile_cfg.get('basis_dim', 4))
            in_channels = int(self.profile_cfg.get('in_channels', self.anchor_feat_channels))
            hidden_dim = int(
                self.profile_cfg.get('hidden_dim', max(self.anchor_feat_channels, basis_dim * 8)))
            self.profile_head = ProfileHead(in_channels, hidden_dim, basis_dim)
            self.register_buffer(
                'profile_basis',
                self._build_profile_basis(basis_dim),
                persistent=False,
            )
        else:
            self.profile_head = None
            self.register_buffer('profile_basis', torch.empty(0), persistent=False)

    def _build_profile_basis(self, basis_dim):
        y_steps = torch.as_tensor(self.y_steps, dtype=torch.float32)
        y_norm = y_steps - y_steps.mean()
        y_norm = y_norm / y_norm.abs().max().clamp_min(1.0)
        basis = [torch.ones_like(y_norm)]
        for degree in range(1, basis_dim):
            basis.append(y_norm ** degree)
        return torch.stack(basis, dim=1)

    def _predict_profile(self, feature):
        alpha = self.profile_head(feature)
        road_profile = alpha @ self.profile_basis.t()
        return road_profile, alpha

    def _use_profile_injection(self, iter_idx, road_profile):
        return self.use_profile_anchor and road_profile is not None and iter_idx in self.profile_inject_iters

    def _interpolate_profile_target(self, profile, mask):
        valid_idx = torch.nonzero(mask > 0.5, as_tuple=False).flatten()
        if valid_idx.numel() < 2:
            return profile, mask
        interpolated = profile.clone()
        new_mask = mask.clone()
        for start, end in zip(valid_idx[:-1], valid_idx[1:]):
            gap = int(end.item() - start.item())
            if gap <= 1:
                continue
            start_val = profile[start]
            end_val = profile[end]
            for offset in range(1, gap):
                ratio = offset / gap
                interpolated[start + offset] = start_val + ratio * (end_val - start_val)
                new_mask[start + offset] = 1.0
        return interpolated, new_mask

    def _smooth_profile_target(self, profile, mask):
        if self.profile_target_smooth_kernel <= 1:
            return profile
        radius = self.profile_target_smooth_kernel // 2
        smoothed = profile.clone()
        for idx in range(profile.shape[0]):
            if mask[idx] < 0.5:
                continue
            left = max(0, idx - radius)
            right = min(profile.shape[0], idx + radius + 1)
            valid = mask[left:right] > 0.5
            if valid.any():
                smoothed[idx] = profile[left:right][valid].mean()
        return smoothed

    @torch.no_grad()
    def _build_profile_target(self, target, device):
        target = target.to(device)
        target = target[target[:, 1] > 0]
        profile = torch.zeros(self.anchor_len, device=device)
        mask = torch.zeros(self.anchor_len, device=device)
        if target.numel() == 0:
            return profile, mask

        gt_anchor_len = max((target.shape[1] - 5) // 3, 0)
        if gt_anchor_len == 0:
            return profile, mask

        sample_idx = torch.as_tensor(
            np.round(self.y_steps).astype(np.int64) - 1,
            device=device,
            dtype=torch.long)
        sample_idx = sample_idx.clamp_(0, gt_anchor_len - 1)
        x_all = target[:, 5:5 + gt_anchor_len].index_select(1, sample_idx)
        z_all = target[:, 5 + gt_anchor_len:5 + 2 * gt_anchor_len].index_select(1, sample_idx)
        vis_all = target[:, 5 + 2 * gt_anchor_len:5 + 3 * gt_anchor_len].index_select(1, sample_idx) > 0.5

        for step in range(self.anchor_len):
            valid = vis_all[:, step]
            if not valid.any():
                continue
            z_vals = z_all[valid, step]
            if z_vals.numel() == 1:
                profile[step] = z_vals[0]
            else:
                x_vals = x_all[valid, step]
                weights = torch.exp(-torch.abs(x_vals) / self.profile_center_tau).clamp_min(1e-4)
                sort_idx = torch.argsort(z_vals)
                z_sorted = z_vals[sort_idx]
                weight_sorted = weights[sort_idx]
                cdf = torch.cumsum(weight_sorted, dim=0) / weight_sorted.sum().clamp_min(1e-6)
                median_idx = torch.searchsorted(cdf, cdf.new_tensor([0.5])).item()
                profile[step] = z_sorted[min(median_idx, z_sorted.numel() - 1)]
            mask[step] = 1.0

        profile, mask = self._interpolate_profile_target(profile, mask)
        profile = self._smooth_profile_target(profile, mask)
        return profile, mask

    def _compute_profile_losses(self, road_profile, gt_3dlanes):
        if not self.use_profile_anchor or road_profile is None:
            return {}

        targets = []
        masks = []
        for target in gt_3dlanes:
            profile_target, profile_mask = self._build_profile_target(target, road_profile.device)
            targets.append(profile_target)
            masks.append(profile_mask)
        targets = torch.stack(targets, dim=0)
        masks = torch.stack(masks, dim=0)
        mask_sum = masks.sum()
        zero = road_profile.sum() * 0
        if mask_sum <= 0:
            return {
                'profile_loss': zero,
                'profile_smooth_loss': zero,
            }

        profile_loss = F.smooth_l1_loss(road_profile, targets, reduction='none')
        profile_loss = (profile_loss * masks).sum() / mask_sum.clamp_min(1.0)

        if self.anchor_len > 2:
            second_diff = road_profile[:, 2:] - 2 * road_profile[:, 1:-1] + road_profile[:, :-2]
            smooth_mask = masks[:, 2:] * masks[:, 1:-1] * masks[:, :-2]
            if smooth_mask.sum() > 0:
                smooth_loss = (second_diff.abs() * smooth_mask).sum() / smooth_mask.sum().clamp_min(1.0)
            else:
                smooth_loss = zero
        else:
            smooth_loss = zero

        return {
            'profile_loss': profile_loss * self.profile_loss_weight,
            'profile_smooth_loss': smooth_loss * self.profile_smooth_weight,
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
                      road_profile=None):
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
            if self._use_profile_injection(iter_idx, road_profile):
                cur_anchors = self.anchor_generator.generate_anchors_batch(
                    lane_priors[:, :, 2], lane_priors[:, :, 0], lane_priors[:, :, 1],
                    base_z=road_profile)
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
        road_profile = None
        profile_alpha = None
        if self.use_profile_anchor:
            road_profile, profile_alpha = self._predict_profile(anchor_feats[-1])

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
                        if self._use_profile_injection(iter_idx, road_profile):
                            anchors = self.anchor_generator.generate_anchors_batch(
                                xs, yaws, pitches, base_z=road_profile)
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
                            road_profile=road_profile,
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
                            road_profile=road_profile,
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
                            road_profile=road_profile,
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
                            road_profile=road_profile,
                        )
                        reg_proposals_layer.append(reg_proposals)
                        anchors_layer.append(proposals_prev[:, :, :5 + self.anchor_len * 3])
            reg_proposals_all.append(reg_proposals_layer)
            anchors_all.append(anchors_layer)

        output = {'reg_proposals': reg_proposals_all, 'anchors': anchors_all}
        if self.use_profile_anchor:
            output['road_profile'] = road_profile
            output['profile_alpha'] = profile_alpha
        return output

    def loss(self, output, gt_3dlanes):
        losses, other_vars = super().loss(output, gt_3dlanes)
        losses.update(self._compute_profile_losses(output.get('road_profile'), gt_3dlanes))
        return losses, other_vars
