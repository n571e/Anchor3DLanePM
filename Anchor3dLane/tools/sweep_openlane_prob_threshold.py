#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import mmcv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mmseg.datasets import build_dataset
from mmseg.datasets.tools import eval_openlane


def parse_args():
    parser = argparse.ArgumentParser(
        description='Sweep OpenLane probability thresholds for a saved prediction file.')
    parser.add_argument('config', help='Config file used to build the OpenLane dataset.')
    parser.add_argument('prediction', help='Saved lane3d_prediction.json.')
    parser.add_argument(
        '--thresholds',
        nargs='+',
        type=float,
        help='Explicit probability thresholds to evaluate.')
    parser.add_argument('--start', type=float, default=0.4)
    parser.add_argument('--stop', type=float, default=0.65)
    parser.add_argument('--step', type=float, default=0.025)
    parser.add_argument('--output-json', default='')
    return parser.parse_args()


def build_thresholds(args):
    if args.thresholds:
        return sorted(set(round(th, 6) for th in args.thresholds))
    thresholds = []
    value = args.start
    while value <= args.stop + args.step * 0.5:
        thresholds.append(round(value, 6))
        value += args.step
    return thresholds


def load_eval_inputs(dataset, prediction):
    with open(prediction, 'r') as f:
        json_pred = [json.loads(line, strict=False) for line in f]
    with open(dataset.eval_file, 'r') as f:
        json_gt = [json.loads(line) for line in f]
    if dataset.test_list is not None:
        with open(dataset.test_list, 'r') as f:
            test_list = {line.strip().split('.')[0] for line in f}
        json_pred = [item for item in json_pred if item['file_path'][:-4] in test_list]
        json_gt = [item for item in json_gt if item['file_path'][:-4] in test_list]
    return json_pred, {item['file_path']: item for item in json_gt}


def stats_to_result(threshold, stats):
    keys = [
        'F_score',
        'recall',
        'precision',
        'cate_acc',
        'x_error_close',
        'x_error_far',
        'z_error_close',
        'z_error_far',
    ]
    result = {'prob_th': threshold}
    result.update({key: float(value) for key, value in zip(keys, stats)})
    return result


def main():
    args = parse_args()
    prediction = Path(args.prediction)
    if not prediction.exists():
        raise FileNotFoundError(prediction)

    cfg = mmcv.Config.fromfile(args.config)
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    json_pred, gts = load_eval_inputs(dataset, prediction)
    evaluator = eval_openlane.OpenLaneEval(dataset)

    results = []
    for threshold in build_thresholds(args):
        stats = evaluator.bench_one_submit(json_pred, gts, prob_th=threshold)
        result = stats_to_result(threshold, stats)
        results.append(result)
        print(
            f"prob_th={threshold:.3f} "
            f"F1={result['F_score']:.9f} "
            f"R={result['recall']:.9f} "
            f"P={result['precision']:.9f}",
            flush=True)

    best = max(results, key=lambda item: item['F_score'])
    print(
        f"best prob_th={best['prob_th']:.3f} "
        f"F1={best['F_score']:.9f} "
        f"R={best['recall']:.9f} "
        f"P={best['precision']:.9f}",
        flush=True)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open('w', encoding='utf-8') as f:
            json.dump({
                'config': args.config,
                'prediction': str(prediction),
                'results': results,
                'best': best,
            }, f, indent=2)


if __name__ == '__main__':
    main()
