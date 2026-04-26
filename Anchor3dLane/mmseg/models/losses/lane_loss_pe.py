import warnings

import torch
import torch.nn as nn

from ..builder import LOSSES
from .focal_loss import FocalLossSigmoid
from .kornia_focal import FocalLoss
from .lane_loss_v2 import LaneLossV2


@LOSSES.register_module()
class LaneLossPE(LaneLossV2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.endpoint = 'end_losses' in self.loss_weights
        self._warned_gt_anchor_len = False

    def _compute_endpoint_loss(self, x_pred, z_pred, x_target, z_target, vis_target, smooth_l1_loss):
        vis_target = vis_target > 0.5
        valid_lanes = vis_target.sum(dim=1) >= 2
        if not valid_lanes.any():
            return x_pred.sum() * 0

        x_pred = x_pred[valid_lanes]
        z_pred = z_pred[valid_lanes]
        x_target = x_target[valid_lanes]
        z_target = z_target[valid_lanes]
        vis_target = vis_target[valid_lanes]

        lane_idx = torch.arange(x_pred.shape[0], device=x_pred.device)
        point_idx = torch.arange(self.anchor_len, device=x_pred.device)[None, :].repeat(x_pred.shape[0], 1)
        start_idx = torch.where(
            vis_target, point_idx, torch.full_like(point_idx, self.anchor_len)).min(dim=1)[0]
        end_idx = torch.where(
            vis_target, point_idx, torch.full_like(point_idx, -1)).max(dim=1)[0]

        start_pred = torch.stack([x_pred[lane_idx, start_idx], z_pred[lane_idx, start_idx]], dim=-1)
        end_pred = torch.stack([x_pred[lane_idx, end_idx], z_pred[lane_idx, end_idx]], dim=-1)
        start_target = torch.stack([x_target[lane_idx, start_idx], z_target[lane_idx, start_idx]], dim=-1)
        end_target = torch.stack([x_target[lane_idx, end_idx], z_target[lane_idx, end_idx]], dim=-1)

        start_loss = smooth_l1_loss(start_pred, start_target).mean()
        end_loss = smooth_l1_loss(end_pred, end_target).mean()
        return 0.5 * (start_loss + end_loss)

    def forward(self, proposals_list, targets):
        if self.use_sigmoid:
            focal_loss = FocalLossSigmoid(alpha=self.focal_alpha, gamma=self.focal_gamma, reduction='none')
        else:
            focal_loss = FocalLoss(alpha=self.focal_alpha, gamma=self.focal_gamma)
        smooth_l1_loss = nn.SmoothL1Loss(reduction='none')
        cls_losses = 0
        reg_losses_x = 0
        reg_losses_z = 0
        reg_losses_vis = 0
        if self.lane_prior:
            reg_losses_prior = 0
        if self.consist:
            consist_losses = 0
        if self.endpoint:
            end_losses = 0
        valid_imgs = len(targets)
        total_positives = 0
        total_negatives = 0
        for idx in range(len(proposals_list)):
            proposals = proposals_list[idx][0]
            num_clses = proposals.shape[1] - 5 - self.anchor_len * 3
            anchors = proposals_list[idx][1]
            target = targets[idx]
            target = target[target[:, 1] > 0]
            if len(target) == 0:
                cls_target = proposals.new_zeros(len(proposals)).long()
                cls_pred = proposals[:, 5 + self.anchor_len * 3:]
                cls_losses += focal_loss(cls_pred, cls_target).sum()
                reg_losses_x += smooth_l1_loss(cls_pred, cls_pred).sum() * 0
                reg_losses_z += smooth_l1_loss(cls_pred, cls_pred).sum() * 0
                reg_losses_vis += smooth_l1_loss(cls_pred, cls_pred).sum() * 0
                if self.lane_prior:
                    reg_losses_prior += smooth_l1_loss(cls_pred, cls_pred).sum() * 0
                if self.consist:
                    consist_losses += smooth_l1_loss(cls_pred, cls_pred).sum() * 0
                if self.endpoint:
                    end_losses += smooth_l1_loss(cls_pred, cls_pred).sum() * 0
                continue

            x_indices = torch.tensor(self.anchor_steps).to(torch.long).to(target.device) + 5
            gt_anchor_len = max((target.shape[1] - 5) // 3, 0)
            if gt_anchor_len != self.gt_anchor_len and not self._warned_gt_anchor_len:
                warnings.warn(
                    f'LaneLossPE detected gt_anchor_len={gt_anchor_len}, '
                    f'while config gt_anchor_len={self.gt_anchor_len}. Using the target-derived length.',
                    UserWarning,
                )
                self._warned_gt_anchor_len = True
            z_indices = x_indices + gt_anchor_len
            vis_indices = x_indices + gt_anchor_len * 2
            x_target = target.index_select(1, x_indices)
            z_target = target.index_select(1, z_indices)
            vis_target = target.index_select(1, vis_indices)
            target = torch.cat((target[:, :5], x_target, z_target, vis_target), dim=1)
            with torch.no_grad():
                if self.anchor_assign:
                    anchor_assign = torch.cat([anchors, proposals[:, 65:]], 1)
                    indices_src, indices_tgt = self.assigner(anchor_assign, target, use_sigmoid=self.use_sigmoid)
                else:
                    indices_src, indices_tgt = self.assigner(proposals, target, use_sigmoid=self.use_sigmoid)

            positives = proposals[indices_src]
            num_positives = len(positives)
            total_positives += num_positives
            negatives_mask = torch.ones(proposals.shape[0], dtype=torch.bool, device=proposals.device)
            negatives_mask[indices_src] = False
            negatives = proposals[negatives_mask]
            num_negatives = len(negatives)
            total_negatives += num_negatives

            if num_positives == 0:
                cls_target = proposals.new_zeros(len(proposals)).long()
                cls_pred = proposals[:, :2]
                cls_losses += focal_loss(cls_pred, cls_target).sum()
                reg_losses_x += smooth_l1_loss(cls_pred, cls_pred).sum() * 0
                reg_losses_z += smooth_l1_loss(cls_pred, cls_pred).sum() * 0
                reg_losses_vis += smooth_l1_loss(cls_pred, cls_pred).sum() * 0
                if self.lane_prior:
                    reg_losses_prior += smooth_l1_loss(cls_pred, cls_pred).sum() * 0
                if self.consist:
                    consist_losses += smooth_l1_loss(cls_pred, cls_pred).sum() * 0
                if self.endpoint:
                    end_losses += smooth_l1_loss(cls_pred, cls_pred).sum() * 0
                continue

            all_proposals = torch.cat([positives, negatives], 0)
            cls_target = proposals.new_zeros(num_positives + num_negatives).long()
            cls_target[:num_positives] = target[indices_tgt][:, 1]
            cls_pred = all_proposals[:, 5 + self.anchor_len * 3:]

            x_pred = positives[:, 5:5 + self.anchor_len]
            z_pred = positives[:, 5 + self.anchor_len:5 + self.anchor_len * 2]
            vis_pred = positives[:, 5 + self.anchor_len * 2:5 + self.anchor_len * 3]
            prior_pred = positives[:, 2:5]

            with torch.no_grad():
                target = target[indices_tgt]
                x_target = target[:, 5:5 + self.anchor_len]
                z_target = target[:, 5 + self.anchor_len:5 + self.anchor_len * 2]
                vis_target = target[:, 5 + self.anchor_len * 2:5 + self.anchor_len * 3]
                prior_target = target[:, 2:5]
                valid_points = vis_target.sum().clamp_min(1.0)

            reg_loss_x = smooth_l1_loss(x_pred, x_target)
            reg_loss_x = reg_loss_x * vis_target
            reg_losses_x += reg_loss_x.sum() / valid_points

            reg_loss_z = smooth_l1_loss(z_pred, z_target)
            reg_loss_z = reg_loss_z * vis_target
            reg_losses_z += reg_loss_z.sum() / valid_points

            reg_loss_vis = smooth_l1_loss(vis_pred, vis_target)
            reg_losses_vis += reg_loss_vis.mean()
            cls_loss = focal_loss(cls_pred, cls_target)

            if self.lane_prior:
                prior_loss = smooth_l1_loss(prior_pred, prior_target).mean()
                reg_losses_prior += prior_loss
            if self.consist:
                xr = x_pred.unsqueeze(1)
                xl = x_pred.unsqueeze(0)
                cos = x_pred.new_zeros(x_pred.shape[0], self.anchor_len - 1)
                cos = 5 / ((x_pred[..., 1:] - x_pred[..., :-1]) ** 2 + 25) ** 0.5
                cos = torch.cat([cos[..., 0:1].clone(), cos], -1)
                distance = (xr - xl) * cos.detach()
                distance_mean = distance.mean(-1, keepdims=True)
                distance_delta = (distance - distance_mean).abs()[..., self.ds:]
                distance_mask = distance_delta < self.delta
                consist_loss = (distance_delta * distance_mask).sum(-1) / (distance_mask.sum(-1) + 1e-6)
                consist_loss = consist_loss.triu(diagonal=1)
                consist_losses += consist_loss.sum() / (num_positives * (num_positives - 1) / 2 + 1e-6)
            if self.endpoint:
                end_losses += self._compute_endpoint_loss(
                    x_pred, z_pred, x_target, z_target, vis_target, smooth_l1_loss)

            if self.use_sigmoid:
                cls_losses += cls_loss.sum() / num_positives / num_clses
            else:
                cls_losses += cls_loss.sum() / num_positives

        cls_losses = cls_losses / valid_imgs
        reg_losses_x = reg_losses_x / valid_imgs
        reg_losses_z = reg_losses_z / valid_imgs
        reg_losses_vis = reg_losses_vis / valid_imgs

        losses = {
            'cls_loss': cls_losses,
            'reg_losses_x': reg_losses_x,
            'reg_losses_z': reg_losses_z,
            'reg_losses_vis': reg_losses_vis,
        }

        if self.lane_prior:
            reg_losses_prior = reg_losses_prior / valid_imgs
            losses['reg_losses_prior'] = reg_losses_prior

        if self.consist:
            consist_losses = consist_losses / valid_imgs
            losses['consist_losses'] = consist_losses

        if self.endpoint:
            end_losses = end_losses / valid_imgs
            losses['end_losses'] = end_losses

        for key in losses.keys():
            losses[key] = losses[key] * self.loss_weights[key]

        batch_size = len(proposals_list)
        return {
            'losses': losses,
            'batch_positives': total_positives / batch_size,
            'batch_negatives': total_negatives / batch_size,
        }
