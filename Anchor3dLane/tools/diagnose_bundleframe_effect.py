import argparse
import json
import os
from collections import defaultdict

import mmcv
import torch
from mmcv.parallel import scatter
from mmcv.runner import load_checkpoint

from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_lanedetector


def parse_int_list(value):
    if value is None:
        return None
    value = value.strip()
    if value == '':
        return []
    return [int(item) for item in value.split(',')]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Diagnose whether BundleFrame changes anchors/proposals and receives gradients.')
    parser.add_argument('config', help='Config file.')
    parser.add_argument('checkpoint', help='Checkpoint file.')
    parser.add_argument('--split', default='train', choices=['train', 'test'])
    parser.add_argument('--gpu-id', default=0, type=int)
    parser.add_argument('--num-batches', default=2, type=int)
    parser.add_argument('--samples-per-gpu', default=1, type=int)
    parser.add_argument('--workers-per-gpu', default=0, type=int)
    parser.add_argument(
        '--inject-iters',
        default=None,
        help='Comma-separated injected refinement iterations for the "on" pass. '
             'Defaults to the config/model value.')
    parser.add_argument('--output-json', default=None)
    return parser.parse_args()


def to_float(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().cpu().item())
    return float(value)


def tensor_stats(tensor, prefix):
    tensor = tensor.detach().float()
    return {
        f'{prefix}_mean': float(tensor.mean().cpu()),
        f'{prefix}_abs_mean': float(tensor.abs().mean().cpu()),
        f'{prefix}_std': float(tensor.std().cpu()),
        f'{prefix}_min': float(tensor.min().cpu()),
        f'{prefix}_max': float(tensor.max().cpu()),
    }


def diff_stats(a, b, prefix):
    diff = (a.detach().float() - b.detach().float()).abs()
    return {
        f'{prefix}_delta_mean': float(diff.mean().cpu()),
        f'{prefix}_delta_max': float(diff.max().cpu()),
    }


def grad_norm(module):
    total = 0.0
    tensors = 0
    for param in module.parameters():
        if param.grad is None:
            continue
        total += float(param.grad.detach().float().norm().cpu())
        tensors += 1
    return total, tensors


@torch.no_grad()
def frame_target_stats(model, bundle_frame, gt_3dlanes):
    targets_x = []
    targets_h = []
    targets_bank = []
    masks = []
    device = bundle_frame['x_ref'].device
    for target in gt_3dlanes:
        x_ref, h_ref, bank, mask = model._build_bundle_frame_target(target, device)
        targets_x.append(x_ref)
        targets_h.append(h_ref)
        targets_bank.append(bank)
        masks.append(mask)

    targets_x = torch.stack(targets_x, dim=0)
    targets_h = torch.stack(targets_h, dim=0)
    targets_bank = torch.stack(targets_bank, dim=0)
    masks = torch.stack(masks, dim=0)
    mask_sum = masks.sum().clamp_min(1.0)

    metrics = {
        'target_valid_ratio': float(masks.float().mean().cpu()),
        'target_valid_count': float(masks.sum().cpu()),
    }
    for name, pred, target in (
        ('x_ref', bundle_frame['x_ref'], targets_x),
        ('h', bundle_frame['h'], targets_h),
        ('bank', bundle_frame['bank'], targets_bank),
    ):
        abs_err = (pred.detach() - target).abs() * masks
        metrics[f'{name}_target_mae'] = float((abs_err.sum() / mask_sum).cpu())
        metrics.update(tensor_stats(target[masks > 0.5], f'{name}_target'))
    return metrics


def run_encoder(model, batch, inject_iters):
    model.bundle_inject_iters = set(inject_iters)
    project_matrix = batch['gt_project_matrix'].squeeze(1)
    return model.encoder_decoder(batch['img'], batch['mask'], project_matrix)


def summarize_injection_effect(model, batch, on_iters):
    with torch.no_grad():
        out_on = run_encoder(model, batch, on_iters)
        out_off = run_encoder(model, batch, [])

    metrics = {}
    bundle_frame = out_on.get('bundle_frame')
    if bundle_frame is not None:
        for name in ('x_ref', 'h', 'bank', 'alpha_x', 'alpha_h', 'alpha_b'):
            metrics.update(tensor_stats(bundle_frame[name], name))
        metrics.update(frame_target_stats(model, bundle_frame, batch['gt_3dlanes']))
        frame_losses = model._compute_bundle_frame_losses(bundle_frame, batch['gt_3dlanes'])
        for name, value in frame_losses.items():
            metrics[name] = to_float(value)
        metrics['bundle_frame_loss_sum'] = sum(
            value for key, value in metrics.items() if key.startswith('bundle_frame_') and key.endswith('_loss'))

    anchor_delta_max = 0.0
    anchor_delta_mean_sum = 0.0
    anchor_delta_count = 0
    for iter_idx, (anchors_on, anchors_off) in enumerate(zip(out_on['anchors'], out_off['anchors'])):
        for feat_idx, (anchor_on, anchor_off) in enumerate(zip(anchors_on, anchors_off)):
            prefix = f'anchor_i{iter_idx}_f{feat_idx}'
            cur = diff_stats(anchor_on, anchor_off, prefix)
            metrics.update(cur)
            anchor_delta_max = max(anchor_delta_max, cur[f'{prefix}_delta_max'])
            anchor_delta_mean_sum += cur[f'{prefix}_delta_mean']
            anchor_delta_count += 1

    metrics['anchor_delta_mean_across_stages'] = (
        anchor_delta_mean_sum / max(anchor_delta_count, 1))
    metrics['anchor_delta_max_across_stages'] = anchor_delta_max

    final_on = out_on['reg_proposals'][-1][-1]
    final_off = out_off['reg_proposals'][-1][-1]
    metrics.update(diff_stats(final_on, final_off, 'final_proposal'))
    geo_slice = slice(5, 5 + model.anchor_len * 3)
    metrics.update(diff_stats(final_on[..., geo_slice], final_off[..., geo_slice], 'final_geometry'))
    return metrics


def summarize_gradients(model, batch, on_iters):
    if not model.use_bundle_frame or model.bundle_frame_head is None:
        return {}

    project_matrix = batch['gt_project_matrix'].squeeze(1)
    metrics = {}

    model.zero_grad(set_to_none=True)
    model.bundle_inject_iters = set(on_iters)
    out = model.encoder_decoder(batch['img'], batch['mask'], project_matrix)
    frame_losses = model._compute_bundle_frame_losses(out.get('bundle_frame'), batch['gt_3dlanes'])
    frame_loss = sum(value.mean() for value in frame_losses.values())
    frame_loss.backward()
    norm, tensors = grad_norm(model.bundle_frame_head)
    metrics['bundle_supervision_loss'] = float(frame_loss.detach().cpu())
    metrics['bundle_supervision_grad_norm_sum'] = norm
    metrics['bundle_supervision_grad_tensors'] = tensors

    model.zero_grad(set_to_none=True)
    out = model.encoder_decoder(batch['img'], batch['mask'], project_matrix)
    losses, _ = model.loss(out, batch['gt_3dlanes'])
    total_loss = sum(value.mean() for key, value in losses.items() if 'loss' in key)
    total_loss.backward()
    norm, tensors = grad_norm(model.bundle_frame_head)
    metrics['total_loss'] = float(total_loss.detach().cpu())
    metrics['bundle_total_grad_norm_sum'] = norm
    metrics['bundle_total_grad_tensors'] = tensors

    model.zero_grad(set_to_none=True)
    return metrics


def aggregate_batches(batch_metrics):
    grouped = defaultdict(list)
    for metrics in batch_metrics:
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                grouped[key].append(float(value))

    summary = {}
    for key, values in grouped.items():
        summary[f'{key}_mean'] = sum(values) / len(values)
        summary[f'{key}_max'] = max(values)
    return summary


def classify_effect(model, summary):
    if not model.use_bundle_frame:
        return 'bundle_frame_disabled'
    if summary.get('anchor_delta_max_across_stages_max', 0.0) <= 1e-6:
        return 'injection_path_inactive'
    if summary.get('bundle_frame_loss_sum_mean', 0.0) <= 1e-8:
        return 'frame_supervision_inactive'
    if summary.get('bundle_supervision_grad_norm_sum_mean', 0.0) <= 1e-8:
        return 'frame_supervision_has_no_gradient'
    return 'active_but_needs_ablation'


def main():
    args = parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    if hasattr(cfg.model, 'pretrained'):
        cfg.model.pretrained = None

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        device_ids = [args.gpu_id]
    else:
        device_ids = [-1]

    data_cfg = cfg.data[args.split]
    if args.split == 'test':
        data_cfg.test_mode = True
    dataset = build_dataset(data_cfg)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=args.samples_per_gpu,
        workers_per_gpu=args.workers_per_gpu,
        num_gpus=1,
        dist=False,
        shuffle=False,
        persistent_workers=False)

    model = build_lanedetector(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    if torch.cuda.is_available():
        model = model.cuda(args.gpu_id)
    model.eval()

    configured_iters = sorted(getattr(model, 'bundle_inject_iters', []))
    on_iters = parse_int_list(args.inject_iters)
    if on_iters is None:
        on_iters = configured_iters

    original_iters = set(getattr(model, 'bundle_inject_iters', []))
    batch_metrics = []
    for batch_idx, raw_batch in enumerate(data_loader):
        if batch_idx >= args.num_batches:
            break
        if torch.cuda.is_available():
            batch = scatter(raw_batch, device_ids)[0]
        else:
            batch = raw_batch
        metrics = summarize_injection_effect(model, batch, on_iters)
        if batch_idx == 0:
            metrics.update(summarize_gradients(model, batch, on_iters))
        metrics['batch_idx'] = batch_idx
        batch_metrics.append(metrics)

    model.bundle_inject_iters = original_iters
    summary = aggregate_batches(batch_metrics)
    report = {
        'config': args.config,
        'checkpoint': args.checkpoint,
        'split': args.split,
        'num_batches': len(batch_metrics),
        'samples_per_gpu': args.samples_per_gpu,
        'configured_inject_iters': configured_iters,
        'measured_inject_iters': on_iters,
        'inject_strength': getattr(model, 'bundle_inject_strength', None),
        'effect_classification': classify_effect(model, summary),
        'summary': summary,
        'batches': batch_metrics,
    }

    if args.output_json is not None:
        mmcv.mkdir_or_exist(os.path.dirname(args.output_json))
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
