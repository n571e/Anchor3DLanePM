#!/usr/bin/env python3
"""Run official baseline evaluations with journal logging.

This script is intentionally conservative:
- it only evaluates checkpoints that already exist locally
- it skips baselines whose required dataset assets are missing
- it records each evaluation through tools/research_journal.py
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/ssd-data3/ztc2025/.conda/envs/anchor3dlane-cu121/bin/python")


@dataclass(frozen=True)
class Baseline:
    name: str
    config: str
    checkpoint: str
    output_dir: str
    tags: tuple[str, ...]
    dataset: str


BASELINES: tuple[Baseline, ...] = (
    Baseline(
        name="apollosim-anchor3dlane-eval",
        config="configs/apollosim/anchor3dlane.py",
        checkpoint="pretrained/apollo_anchor3dlane.pth",
        output_dir="output/eval_apollosim_anchor3dlane",
        tags=("baseline", "apollosim", "eval", "official"),
        dataset="apollosim",
    ),
    Baseline(
        name="apollosim-anchor3dlane-iter-eval",
        config="configs/apollosim/anchor3dlane_iter.py",
        checkpoint="pretrained/apollo_anchor3dlane_iter.pth",
        output_dir="output/eval_apollosim_anchor3dlane_iter",
        tags=("baseline", "apollosim", "eval", "official"),
        dataset="apollosim",
    ),
    Baseline(
        name="openlane-v11-anchor3dlane-eval",
        config="configs/openlane/anchor3dlane.py",
        checkpoint="pretrained/openlane_anchor3dlane.pth",
        output_dir="output/eval_openlane_v11_anchor3dlane",
        tags=("baseline", "openlane", "v1.1", "eval", "official"),
        dataset="openlane",
    ),
    Baseline(
        name="openlane-v11-anchor3dlane-effb3-eval",
        config="configs/openlane/anchor3dlane_effb3.py",
        checkpoint="pretrained/openlane_anchor3dlane_effb3.pth",
        output_dir="output/eval_openlane_v11_anchor3dlane_effb3",
        tags=("baseline", "openlane", "v1.1", "eval", "official"),
        dataset="openlane",
    ),
    Baseline(
        name="openlane-v11-anchor3dlane-iter-eval",
        config="configs/openlane/anchor3dlane_iter.py",
        checkpoint="pretrained/openlane_anchor3dlane_iter.pth",
        output_dir="output/eval_openlane_v11_anchor3dlane_iter",
        tags=("baseline", "openlane", "v1.1", "eval", "official"),
        dataset="openlane",
    ),
    Baseline(
        name="openlane-v11-anchor3dlane-temporal-iter-eval",
        config="configs/openlane/anchor3dlane_mf_iter.py",
        checkpoint="pretrained/openlane_anchor3dlane_temporal_iter.pth",
        output_dir="output/eval_openlane_v11_anchor3dlane_temporal_iter",
        tags=("baseline", "openlane", "v1.1", "eval", "official"),
        dataset="openlane",
    ),
    Baseline(
        name="openlane-v12-anchor3dlane-eval",
        config="configs/openlane/anchor3dlane.py",
        checkpoint="pretrained/openlanev2_anchor3dlane.pth",
        output_dir="output/eval_openlanev2_anchor3dlane",
        tags=("baseline", "openlane", "v1.2", "eval", "official"),
        dataset="openlane",
    ),
    Baseline(
        name="openlane-v12-anchor3dlane-iter-eval",
        config="configs/openlane/anchor3dlane_iter.py",
        checkpoint="pretrained/openlanev2_anchor3dlane_iter.pth",
        output_dir="output/eval_openlanev2_anchor3dlane_iter",
        tags=("baseline", "openlane", "v1.2", "eval", "official"),
        dataset="openlane",
    ),
    Baseline(
        name="openlane-v12-anchor3dlane-iter-r50x2-eval",
        config="configs/openlane/anchor3dlane_iter_r50.py",
        checkpoint="pretrained/openlanev2_anchor3dlane_iter_r50x2.pth",
        output_dir="output/eval_openlanev2_anchor3dlane_iter_r50x2",
        tags=("baseline", "openlane", "v1.2", "eval", "official"),
        dataset="openlane",
    ),
    Baseline(
        name="openlane-v12-anchor3dlanepp-r18-eval",
        config="../configs_v2/openlane/anchor3dlane++_r18.py",
        checkpoint="pretrained/openlane_anchor3dlane++_r18.pth",
        output_dir="output/eval_openlane_anchor3dlanepp_r18",
        tags=("baseline", "openlane", "v1.2", "anchor3dlanepp", "eval", "official"),
        dataset="openlane",
    ),
    Baseline(
        name="openlane-v12-anchor3dlanepp-r50-eval",
        config="../configs_v2/openlane/anchor3dlane++_r50.py",
        checkpoint="pretrained/openlane_anchor3dlane++_r50.pth",
        output_dir="output/eval_openlane_anchor3dlanepp_r50",
        tags=("baseline", "openlane", "v1.2", "anchor3dlanepp", "eval", "official"),
        dataset="openlane",
    ),
    Baseline(
        name="openlane-v12-anchor3dlanepp-r50x2-eval",
        config="../configs_v2/openlane/anchor3dlane++_r50x2.py",
        checkpoint="pretrained/openlane_anchor3dlane++_r50x2.pth",
        output_dir="output/eval_openlane_anchor3dlanepp_r50x2",
        tags=("baseline", "openlane", "v1.2", "anchor3dlanepp", "eval", "official"),
        dataset="openlane",
    ),
    Baseline(
        name="openlane-v12-anchor3dlanepp-r18-se-eval",
        config="../configs_v2/openlane/anchor3dlane++_r18_SE.py",
        checkpoint="pretrained/openlane_anchor3dlane++_r18_se.pth",
        output_dir="output/eval_openlane_anchor3dlanepp_r18_se",
        tags=("baseline", "openlane", "v1.2", "anchor3dlanepp", "fusion", "eval", "official"),
        dataset="openlane",
    ),
    Baseline(
        name="openlane-v12-anchor3dlanepp-r50-se-eval",
        config="../configs_v2/openlane/anchor3dlane++_r50_SE.py",
        checkpoint="pretrained/openlane_anchor3dlane++_r50_se.pth",
        output_dir="output/eval_openlane_anchor3dlanepp_r50_se",
        tags=("baseline", "openlane", "v1.2", "anchor3dlanepp", "fusion", "eval", "official"),
        dataset="openlane",
    ),
    Baseline(
        name="openlane-v12-anchor3dlanepp-r50x2-se-eval",
        config="../configs_v2/openlane/anchor3dlane++_r50x2_SE.py",
        checkpoint="pretrained/openlane_anchor3dlane++_r50x2_se.pth",
        output_dir="output/eval_openlane_anchor3dlanepp_r50x2_se",
        tags=("baseline", "openlane", "v1.2", "anchor3dlanepp", "fusion", "eval", "official"),
        dataset="openlane",
    ),
    Baseline(
        name="openlane-v12-anchor3dlanepp-r18-pp-eval",
        config="../configs_v2/openlane/anchor3dlane++_r18_PP.py",
        checkpoint="pretrained/openlane_anchor3dlane++_r18_pp.pth",
        output_dir="output/eval_openlane_anchor3dlanepp_r18_pp",
        tags=("baseline", "openlane", "v1.2", "anchor3dlanepp", "fusion", "eval", "official"),
        dataset="openlane",
    ),
    Baseline(
        name="openlane-v12-anchor3dlanepp-r50-pp-eval",
        config="../configs_v2/openlane/anchor3dlane++_r50_PP.py",
        checkpoint="pretrained/openlane_anchor3dlane++_r50_pp.pth",
        output_dir="output/eval_openlane_anchor3dlanepp_r50_pp",
        tags=("baseline", "openlane", "v1.2", "anchor3dlanepp", "fusion", "eval", "official"),
        dataset="openlane",
    ),
    Baseline(
        name="openlane-v12-anchor3dlanepp-r50x2-pp-eval",
        config="../configs_v2/openlane/anchor3dlane++_r50x2_PP.py",
        checkpoint="pretrained/openlane_anchor3dlane++_r50x2_pp.pth",
        output_dir="output/eval_openlane_anchor3dlanepp_r50x2_pp",
        tags=("baseline", "openlane", "v1.2", "anchor3dlanepp", "fusion", "eval", "official"),
        dataset="openlane",
    ),
    Baseline(
        name="once-anchor3dlane-eval",
        config="configs/once/anchor3dlane.py",
        checkpoint="pretrained/once_anchor3dlane.pth",
        output_dir="output/eval_once_anchor3dlane",
        tags=("baseline", "once", "eval", "official"),
        dataset="once",
    ),
    Baseline(
        name="once-anchor3dlane-effb3-eval",
        config="configs/once/anchor3dlane_effb3.py",
        checkpoint="pretrained/once_anchor3dlane_effb3.pth",
        output_dir="output/eval_once_anchor3dlane_effb3",
        tags=("baseline", "once", "eval", "official"),
        dataset="once",
    ),
    Baseline(
        name="once-anchor3dlane-iter-eval",
        config="configs/once/anchor3dlane_iter.py",
        checkpoint="pretrained/once_anchor3dlane_iter.pth",
        output_dir="output/eval_once_anchor3dlane_iter",
        tags=("baseline", "once", "eval", "official"),
        dataset="once",
    ),
    Baseline(
        name="once-anchor3dlanepp-r18-eval",
        config="../configs_v2/once/anchor3dlane++_r18.py",
        checkpoint="pretrained/once_anchor3dlane++_r18.pth",
        output_dir="output/eval_once_anchor3dlanepp_r18",
        tags=("baseline", "once", "anchor3dlanepp", "eval", "official"),
        dataset="once",
    ),
    Baseline(
        name="once-anchor3dlanepp-r50x2-eval",
        config="../configs_v2/once/anchor3dlane++_r50x2.py",
        checkpoint="pretrained/once_anchor3dlane++_r50x2.pth",
        output_dir="output/eval_once_anchor3dlanepp_r50x2",
        tags=("baseline", "once", "anchor3dlanepp", "eval", "official"),
        dataset="once",
    ),
)


def dataset_ready(dataset: str) -> tuple[bool, str]:
    if dataset == "apollosim":
        need = [
            REPO_ROOT / "data/ApolloSim/images",
            REPO_ROOT / "data/ApolloSim/cache_dense",
        ]
    elif dataset == "openlane":
        need = [
            REPO_ROOT / "data/OpenLane/images",
            REPO_ROOT / "data/OpenLane/cache_dense",
        ]
    elif dataset == "once":
        need = [
            REPO_ROOT / "data/ONCE/raw_data",
            REPO_ROOT / "data/ONCE/cache_dense",
        ]
    else:
        return False, f"unknown dataset: {dataset}"

    missing = [str(p) for p in need if not p.exists()]
    if missing:
        return False, "missing dataset assets: " + ", ".join(missing)
    return True, "ok"


def run_baseline(baseline: Baseline, dry_run: bool) -> int:
    checkpoint = REPO_ROOT / baseline.checkpoint
    if not checkpoint.exists():
        print(f"[skip] {baseline.name}: missing checkpoint {checkpoint}")
        return 2

    ready, reason = dataset_ready(baseline.dataset)
    if not ready:
        print(f"[skip] {baseline.name}: {reason}")
        return 3

    env = os.environ.copy()
    extra = str(REPO_ROOT / "gen-efficientnet-pytorch")
    env["PYTHONPATH"] = extra if not env.get("PYTHONPATH") else env["PYTHONPATH"] + ":" + extra

    cmd = [
        str(PYTHON),
        "tools/research_journal.py",
        "run-exp",
        "--name",
        baseline.name,
        "--summary",
        f"Evaluate official baseline {baseline.name}.",
    ]
    for tag in baseline.tags:
        cmd.extend(["--tag", tag])
    cmd.extend(
        [
            "--files",
            baseline.config,
            baseline.checkpoint,
            "--work-dir",
            baseline.output_dir,
            "--",
            str(PYTHON),
            "tools/test.py",
            baseline.config,
            baseline.checkpoint,
            "--show-dir",
            baseline.output_dir,
        ]
    )

    print("[run]", " ".join(cmd))
    if dry_run:
        return 0
    return subprocess.run(cmd, cwd=REPO_ROOT, env=env).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Run official Anchor3DLane baselines.")
    parser.add_argument("--dataset", choices=["all", "apollosim", "openlane", "once"], default="all")
    parser.add_argument("--name-contains", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    selected = []
    for baseline in BASELINES:
        if args.dataset != "all" and baseline.dataset != args.dataset:
            continue
        if args.name_contains and args.name_contains not in baseline.name:
            continue
        selected.append(baseline)

    if not selected:
        print("No baselines selected.")
        return 1

    failures = []
    for baseline in selected:
        rc = run_baseline(baseline, dry_run=args.dry_run)
        if rc not in (0, 2, 3):
            failures.append((baseline.name, rc))

    if failures:
        print("Failures:")
        for name, rc in failures:
            print(f"- {name}: rc={rc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
