# Anchor3DLane++ Baseline Ledger

Updated: 2026-04-18

## Scope

This ledger now keeps only the baseline line that matters for the current V1:

- `OpenLane-v1.2 Anchor3DLane++ ResNet-18`
- `ApolloSim Anchor3DLane++ ResNet-18`

The following local artifacts were intentionally cleaned on 2026-04-18:

- repo-root stray directory `../mmseg/`
- [output/eval_apollo_anchor3dlane](/ssd-data3/ztc2025/Anchor3DLanePM/Anchor3dLane/output/eval_apollo_anchor3dlane)
- [output/eval_apollo_anchor3dlane_iter](/ssd-data3/ztc2025/Anchor3DLanePM/Anchor3dLane/output/eval_apollo_anchor3dlane_iter)
- [output/eval_openlanev2_anchor3dlane](/ssd-data3/ztc2025/Anchor3DLanePM/Anchor3dLane/output/eval_openlanev2_anchor3dlane)

Raw datasets and pretrained weights were kept.

## Current status

| Dataset / baseline | Config | Local checkpoint | Local eval status | Note |
|---|---|---|---|---|
| OpenLane-v1.2 `Anchor3DLane++ R18` | [configs_v2/openlane/anchor3dlane++_r18.py](/ssd-data3/ztc2025/Anchor3DLanePM/configs_v2/openlane/anchor3dlane++_r18.py) | [pretrained/openlane_anchor3dlane++_r18.pth](/ssd-data3/ztc2025/Anchor3DLanePM/Anchor3dLane/pretrained/openlane_anchor3dlane++_r18.pth) | Not yet reproduced locally | Only kept result directory is [output/eval_openlane_anchor3dlanepp_r18](/ssd-data3/ztc2025/Anchor3DLanePM/Anchor3dLane/output/eval_openlane_anchor3dlanepp_r18), and it is still empty. |
| ApolloSim `Anchor3DLane++ R18` | [configs_v2/apollosim/anchor3dlane++_r18.py](/ssd-data3/ztc2025/Anchor3DLanePM/configs_v2/apollosim/anchor3dlane++_r18.py) | Not found locally | Not yet reproduced locally | The repo provides the config, but there is no local `Anchor3DLane++` ApolloSim checkpoint to evaluate directly. |

## Environment decision

Use:

- conda env: `anchor3dlane-cu121`
- code root: [Anchor3dLane](/ssd-data3/ztc2025/Anchor3DLanePM/Anchor3dLane)
- mmseg package: [Anchor3dLane/mmseg](/ssd-data3/ztc2025/Anchor3DLanePM/Anchor3dLane/mmseg)

Why this is now stable:

- the duplicate repo-root `mmseg/` has been removed
- `import mmseg` now resolves to [Anchor3dLane/mmseg/__init__.py](/ssd-data3/ztc2025/Anchor3DLanePM/Anchor3dLane/mmseg/__init__.py) both from the repo root and from `Anchor3dLane/`
- `mmsegmentation` is installed in editable mode against this project

Known caveat:

- `mmcv.ops` still misses `mmcv._ext` in this env, so this is a workable research env, not a perfectly pristine official env

## Reproduction order

Use this order and do not start PE experiments before step 3 is done.

1. Reproduce `OpenLane-v1.2 Anchor3DLane++ R18` by direct evaluation with the existing official checkpoint.
2. If needed, add `--eval-splits` on OpenLane to inspect failure modes before PE ablations.
3. Reproduce `ApolloSim Anchor3DLane++ R18` by training from scratch with the repo config, because there is no local `++` checkpoint.
4. Evaluate the best ApolloSim checkpoint and store the JSON result.
5. Only after both baselines are stable, start `PE-Anchor3DLane++` on OpenLane first, then ApolloSim.

Why this order:

- `OpenLane++` is the paper-table baseline and already has a local checkpoint, so it is the fastest anchor point.
- `ApolloSim++` is still valuable for geometry generalization, but in this repo it is a train-first baseline instead of a ready-made eval-first baseline.

## Commands

### 0. One-time shell setup

```bash
conda activate anchor3dlane-cu121
cd /ssd-data3/ztc2025/Anchor3DLanePM/Anchor3dLane
export PYTHONPATH=$PYTHONPATH:./gen-efficientnet-pytorch
```

### 1. Sanity check imports and data roots

```bash
python -c "import mmseg; print(mmseg.__file__)"
test -d data/OpenLane && echo "OpenLane OK"
test -d data/ApolloSim && echo "ApolloSim OK"
```

Expected `mmseg` target:

- [Anchor3dLane/mmseg/__init__.py](/ssd-data3/ztc2025/Anchor3DLanePM/Anchor3dLane/mmseg/__init__.py)

### 2. OpenLane-v1.2 Anchor3DLane++ R18 evaluation

```bash
python tools/research_journal.py run-exp \
  --name "openlane-anchor3dlanepp-r18-eval" \
  --summary "Evaluate official OpenLane-v1.2 Anchor3DLane++ R18 checkpoint." \
  --tag baseline --tag openlane --tag anchor3dlanepp --tag eval \
  --files ../configs_v2/openlane/anchor3dlane++_r18.py \
          pretrained/openlane_anchor3dlane++_r18.pth \
  --work-dir output/eval_openlane_anchor3dlanepp_r18 \
  -- python tools/test.py \
       ../configs_v2/openlane/anchor3dlane++_r18.py \
       pretrained/openlane_anchor3dlane++_r18.pth \
       --show-dir output/eval_openlane_anchor3dlanepp_r18
```

Optional split analysis:

```bash
python tools/research_journal.py run-exp \
  --name "openlane-anchor3dlanepp-r18-eval-splits" \
  --summary "Evaluate OpenLane-v1.2 Anchor3DLane++ R18 with official split reports." \
  --tag baseline --tag openlane --tag anchor3dlanepp --tag eval \
  --work-dir output/eval_openlane_anchor3dlanepp_r18 \
  -- python tools/test.py \
       ../configs_v2/openlane/anchor3dlane++_r18.py \
       pretrained/openlane_anchor3dlane++_r18.pth \
       --show-dir output/eval_openlane_anchor3dlanepp_r18 \
       --eval-splits
```

Result files to check:

- `output/eval_openlane_anchor3dlanepp_r18/evaluation_result.json`
- `output/eval_openlane_anchor3dlanepp_r18/lane3d_prediction.json`

README target for this paper baseline:

- `F1 57.9`
- `Recall 91.4`
- `x_error_close/far 0.232 / 0.265`
- `z_error_close/far 0.076 / 0.102`

### 3. ApolloSim Anchor3DLane++ R18 training

```bash
python tools/research_journal.py run-exp \
  --name "apollosim-anchor3dlanepp-r18-train" \
  --summary "Train ApolloSim Anchor3DLane++ R18 baseline from repo config." \
  --tag baseline --tag apollosim --tag anchor3dlanepp --tag train \
  --files ../configs_v2/apollosim/anchor3dlane++_r18.py \
  --work-dir output/apollosim/anchor3dlane++_r18 \
  -- python tools/train.py \
       ../configs_v2/apollosim/anchor3dlane++_r18.py \
       --work-dir output/apollosim/anchor3dlane++_r18
```

If you later resume an interrupted run:

```bash
python tools/research_journal.py run-exp \
  --name "apollosim-anchor3dlanepp-r18-resume" \
  --summary "Resume ApolloSim Anchor3DLane++ R18 baseline training." \
  --tag baseline --tag apollosim --tag anchor3dlanepp --tag train \
  --work-dir output/apollosim/anchor3dlane++_r18 \
  -- python tools/train.py \
       ../configs_v2/apollosim/anchor3dlane++_r18.py \
       --work-dir output/apollosim/anchor3dlane++_r18 \
       --auto-resume
```

Checkpoint path to use after training:

- `output/apollosim/anchor3dlane++_r18/latest.pth`

### 4. ApolloSim Anchor3DLane++ R18 evaluation

```bash
python tools/research_journal.py run-exp \
  --name "apollosim-anchor3dlanepp-r18-eval" \
  --summary "Evaluate ApolloSim Anchor3DLane++ R18 baseline checkpoint." \
  --tag baseline --tag apollosim --tag anchor3dlanepp --tag eval \
  --work-dir output/eval_apollosim_anchor3dlanepp_r18 \
  -- python tools/test.py \
       ../configs_v2/apollosim/anchor3dlane++_r18.py \
       output/apollosim/anchor3dlane++_r18/latest.pth \
       --show-dir output/eval_apollosim_anchor3dlanepp_r18
```

Result files to check:

- `output/eval_apollosim_anchor3dlanepp_r18/evaluation_result.json`
- `output/eval_apollosim_anchor3dlanepp_r18/lane3d_prediction.json`

## What is still unresolved

- `OpenLane++ R18` has not been run locally yet, even though the official checkpoint already exists here.
- `ApolloSim++ R18` has no local pretrained checkpoint, so that baseline currently depends on a local train run.
- The public README table in this repo exposes `OpenLane++` paper metrics, but it does not expose a ready-to-use `ApolloSim++` checkpoint entry the way it does for the older ApolloSim baselines.

## Recommendation for the next action

Run exactly one command first:

```bash
python tools/research_journal.py run-exp \
  --name "openlane-anchor3dlanepp-r18-eval" \
  --summary "Evaluate official OpenLane-v1.2 Anchor3DLane++ R18 checkpoint." \
  --tag baseline --tag openlane --tag anchor3dlanepp --tag eval \
  --files ../configs_v2/openlane/anchor3dlane++_r18.py \
          pretrained/openlane_anchor3dlane++_r18.pth \
  --work-dir output/eval_openlane_anchor3dlanepp_r18 \
  -- python tools/test.py \
       ../configs_v2/openlane/anchor3dlane++_r18.py \
       pretrained/openlane_anchor3dlane++_r18.pth \
       --show-dir output/eval_openlane_anchor3dlanepp_r18
```

That will give the fastest clean baseline anchor for the rest of V1.
