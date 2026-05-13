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

## 8) Student RT Pipeline (Distill -> Train -> Infer)

This section is the recommended baseline before moving to SMPL-input work.

### 8.1 Create Distillation Dataset

```bash
python -m csmt.pipelines.create_distill_dataset \
  --teacher_dir ./runs/teacher_mix_g1_go2_with_d1_v7 \
  --processed-dir ./data/processed/mix_g1_go2_with_d1 \
  --output_dir ./data/processed/distill_rt_g1_go2_d1_v2 \
  --window 24 \
  --prev_frames 4 \
  --batch_size 64 \
  --val_ratio 0.1
```

Notes:
- `--prev_frames` should match `prev_len` in `configs/models/student_rt.yaml`.
- Recreate distill data if you change `prev_len`.

### 8.2 Train Student

```bash
python -m csmt.pipelines.train_student \
  --output-root . \
  --data-dir ./data/processed/distill_rt_g1_go2_d1_v2 \
  --save-dir ./runs/student_rt_g1_go2_d1 \
  --device cuda:0
```

Important student config knobs in `configs/models/student_rt.yaml`:
- Capacity: `conv_channels`, `gru_hidden`
- Robustness: `prev_context_mode`, `y_prev_noise_std`, `y_prev_noise_prob`
- Root supervision: `root_motion_target_mode` (`source|teacher|blend`), `root_motion_blend_alpha`
- Losses: `lambda_imitation`, `lambda_src_motion`, `lambda_smooth`, `lambda_joint_limit`

### 8.3 Run Student Inference

```bash
python -m csmt.pipelines.infer_student_rt \
  --output-root . \
  --processed-dir ./data/processed/mix_g1_go2_with_d1 \
  --task-family manipulation \
  --pair-id g1_to_go2_with_d1 \
  --student-ckpt ./runs/student_rt_g1_go2_d1/best.pt \
  --input-pkl ./data/raw/g1/mixed_small/General_A1_-_Stand_stageii.pkl \
  --output-pkl ./demo_output/student_go2_with_d1.pkl \
  --device cuda:0 \
  --root-motion-mode source \
  --dst-start-height 0.28
```

Notes:
- `--dst-start-height` can matter a lot for robots with different base offsets.
- `--root-motion-mode` is inference-only and does not change training.

### 8.4 Visualize Student Output

```bash
python -m csmt.pipelines.visualize_motion \
  --output-root . \
  --robot-id go2_with_d1 \
  --pkl ./demo_output/student_go2_with_d1.pkl \
  --loop
```

## 9) Baseline Before SMPL Refactor

Before starting SMPL-input changes, keep one tagged baseline:
- merge this stable RT student pipeline to `main`
- create a git tag (example): `student-rt-g1-baseline`
- start SMPL work in a new branch from that tag/commit
