# MORPH: Cross-Morphology Motion Retargeting

This repository is a standalone project for teacher-model retargeting across robot morphologies.

## Environment Setup

From repository root:

```bash
conda create -n morph python=3.10 -y
conda activate morph
pip install -e .
```

This installs core dependencies including `torch`, `mujoco`, `pytorch-kinematics`, `wandb`, and `tensorboard`.

If you need CUDA-specific PyTorch wheels, reinstall `torch` after this step using the official PyTorch install command for your CUDA version.

No `gmr` environment is required.

Install package metadata name: `morph`

Runtime module namespace remains `csmt` for now, so commands stay as `python -m csmt....`

## Repository Layout

- `assets/robots/`: robot XML assets (source simulation XMLs)
- `assets/fk/`: stripped FK XMLs used by PyTorch Kinematics
- `configs/robots/*.yaml`: robot metadata + limits + XML paths
- `configs/tasks/<family>/defaults.yaml`: task-level defaults
- `configs/tasks/<family>/pairs/*.yaml`: pair-specific correspondences and loss overrides
- `src/csmt/pipelines/`: training/inference/data utilities

## 1) Bootstrap a Robot Config

```bash
python -m csmt.tools.bootstrap_robot_from_xml \
  --xml ./assets/robots/unitree_go2/go2.xml \
  --robot-id go2 \
  --output-root .
```

This creates:
- `configs/robots/<robot_id>.yaml`
- `assets/fk/<robot_id>_fk.xml`

Then manually set task-specific indices later (feet/EE) in pair YAMLs.

## 2) Bootstrap a Task Pair

```bash
python -m csmt.tools.bootstrap_task_pair \
  --output-root . \
  --task-family locomotion \
  --pair-id g1_to_go2_with_arm \
  --src-robot g1 \
  --dst-robot go2_with_arm
```

Edit the generated pair file under:
- `configs/tasks/locomotion/pairs/g1_to_go2_with_arm.yaml`

## 3) Create Dataset

Dual-domain:

```bash
python -m csmt.pipelines.create_dataset \
  --output-root . \
  --task-family locomotion \
  --pair-id g1_to_go2_with_arm \
  --src-pkl-dir ./data/raw/g1/locomotion \
  --dst-pkl-dir ./data/raw/go2_with_arm/locomotion \
  --processed-dir ./data/processed/loco_g1_go2arm \
  --window-size 64 \
  --stride 20
```

Single-domain (source-only):

```bash
python -m csmt.pipelines.create_dataset \
  --output-root . \
  --task-family locomotion \
  --pair-id g1_to_go2_with_arm \
  --src-pkl-dir ./data/raw/g1/locomotion \
  --processed-dir ./data/processed/g1_only \
  --single-domain src
```

## 4) Train Teacher

```bash
python -m csmt.pipelines.train_teacher \
  --output-root . \
  --processed-dir ./data/processed/loco_g1_go2arm \
  --task-family locomotion \
  --pair-id g1_to_go2_with_arm \
  --save-dir ./runs/teacher_loco_g1_go2arm \
  --device cuda:0 \
  --batch-size 128 \
  --epoch-num 3000
```

## 5) Run Teacher Inference

```bash
python -m csmt.pipelines.infer_teacher \
  --output-root . \
  --processed-dir ./data/processed/loco_g1_go2arm \
  --task-family locomotion \
  --pair-id g1_to_go2_with_arm \
  --teacher-dir ./runs/teacher_loco_g1_go2arm \
  --teacher-epoch 600 \
  --input-pkl ./data/raw/g1/locomotion/walk1_subject1.pkl \
  --output-pkl ./demo_output/retargeted_go2_with_arm.pkl \
  --device cuda:0 \
  --save-src-debug
```

`--save-src-debug` also writes source reconstruction/cycle outputs for debugging.

## 6) Visualize Any Motion

```bash
python -m csmt.pipelines.visualize_motion \
  --output-root . \
  --robot-id go2_with_arm \
  --pkl ./demo_output/retargeted_go2_with_arm.pkl \
  --loop
```

Optional XML override:

```bash
python -m csmt.pipelines.visualize_motion \
  --output-root . \
  --robot-id go2_with_arm \
  --xml ./assets/robots/go2_with_arm/scene.xml \
  --pkl ./demo_output/retargeted_go2_with_arm.pkl
```



## 6.1) Visualize Contact Debug Overlay

If you ran inference with `--save-contact-debug`, visualize per-foot contact confidence and source gating in the MuJoCo viewer:

```bash
python -m csmt.pipelines.visualize_motion \
  --output-root . \
  --robot-id go2 \
  --pkl ./demo_output/retargeted_go2.pkl \
  --contact-debug-npz ./demo_output/retargeted_go2_contact_debug.npz \
  --loop
```

Viewer controls:
- `c`: toggle contact overlay
- `space`: pause/play
- `r`: reset

## 6.2) Visualize EE Target Overlay (Manip Debug)

Use source-motion EE targets against the retargeted destination motion:

```bash
python -m csmt.pipelines.visualize_motion \
  --output-root . \
  --robot-id go2_with_arm \
  --pkl ./demo_output/retargeted_go2_with_arm.pkl \
  --ee-source-pkl ./data/raw/g1/manipulation/salut.pkl \
  --ee-task-family manipulation \
  --ee-pair-id g1_to_go2_with_arm \
  --ee-target-mode displacement \
  --ee-ref-frames 10 \
  --ee-disp-scale-mode loss_ratio \
  --loop
```

Viewer controls:
- `e`: toggle EE overlay
- `space`: pause/play
- `r`: reset

Notes:
- If pair EE indices are empty, EE overlay is skipped.
- `--ee-target-mode displacement` matches the displacement-style EE objective.

## 6.3) Inference Flags for Debug Artifacts

To generate debug artifacts used by overlays:

```bash
python -m csmt.pipelines.infer_teacher \
  --output-root . \
  --processed-dir ./data/processed/loco_g1_go2 \
  --task-family locomotion \
  --pair-id g1_to_go2 \
  --teacher-dir ./runs/teacher_loco_g1_go2 \
  --input-pkl ./data/raw/g1/locomotion/walk1_subject1.pkl \
  --output-pkl ./demo_output/retargeted_go2.pkl \
  --device cuda:0 \
  --save-src-debug \
  --save-contact-debug
```

This writes:
- `*_src_rec.pkl`
- `*_src_cyc.pkl`
- `*_contact_debug.npz`
- `*_contact_debug.png`
- `*_contact_debug_z.png`
- `*_contact_debug_xyz.png`

## 7) Resample Dataset FPS

```bash
python -m csmt.pipelines.resample_dataset \
  --input-dir ./data/raw/go2_with_arm/locomotion \
  --output-dir ./data/raw/go2_with_arm/locomotion_30fps \
  --target-fps 30
```

If input PKLs do not store FPS metadata:

```bash
python -m csmt.pipelines.resample_dataset \
  --input-dir ./data/raw/go2_with_arm/locomotion \
  --output-dir ./data/raw/go2_with_arm/locomotion_30fps \
  --target-fps 30 \
  --src-fps 50
```
