# MORPH: Cross-Morphology Motion Retargeting

MORPH is a standalone codebase for learning and running cross-morphology motion retargeting. The current primary example in this repo is **G1 humanoid -> Go2 quadruped locomotion**.

The usual workflow is:

```text
G1 PKLs + Go2 PKLs -> processed stats/windows -> teacher
G1 PKLs -> teacher -> corrector/skate-comp -> cleaned Go2 PKLs
paired G1 PKLs + cleaned Go2 PKLs -> RT student
paired SMPL motions + cleaned Go2 PKLs -> SMPL student
```

The current student pipelines are direct paired-data pipelines. They train on final target PKLs and do **not** require precomputed distillation shard NPZs.

## Environment Setup

From repository root:

```bash
conda create -n morph python=3.10 -y
conda activate morph
pip install -e .
```

This installs the package metadata name `morph`. The runtime module namespace is still `csmt`, so commands use `python -m csmt....`.

If you need CUDA-specific PyTorch wheels, reinstall `torch` after this step using the official PyTorch command for your CUDA version.

## Repository Layout

- `assets/robots/`: robot XML assets
- `assets/fk/`: stripped FK XMLs used by PyTorch Kinematics
- `configs/robots/*.yaml`: robot metadata, joint limits, nominal base height, XML paths
- `configs/tasks/<family>/defaults.yaml`: task-level defaults
- `configs/tasks/<family>/pairs/*.yaml`: pair-specific correspondences and losses
- `configs/models/*.yaml`: teacher, corrector, and student model configs
- `src/csmt/pipelines/`: dataset, training, inference, visualization utilities

## 1) Robot And Pair Setup

The G1 -> Go2 locomotion pair is expected to exist as:

```text
configs/robots/g1.yaml
configs/robots/go2.yaml
configs/tasks/locomotion/pairs/g1_to_go2.yaml
```

To bootstrap a new robot config from XML:

```bash
python -m csmt.tools.bootstrap_robot_from_xml \
  --xml ./assets/robots/unitree_go2/go2.xml \
  --robot-id go2 \
  --output-root .
```

To bootstrap a new pair:

```bash
python -m csmt.tools.bootstrap_task_pair \
  --output-root . \
  --task-family locomotion \
  --pair-id g1_to_go2 \
  --src-robot g1 \
  --dst-robot go2
```

After bootstrapping, edit the pair YAML manually for body/foot correspondences and loss weights.

## 2) Create Processed Dataset

Use this for teacher training and for robot statistics used by inference/student code.

```bash
python -m csmt.pipelines.create_dataset \
  --output-root . \
  --task-family locomotion \
  --pair-id g1_to_go2 \
  --src-pkl-dir ./data/raw/g1/locomotion \
  --dst-pkl-dir ./data/raw/go2/locomotion \
  --processed-dir ./data/processed/loco_g1_go2 \
  --window-size 64 \
  --stride 20
```

The processed directory contains per-robot normalization stats such as `g1_stats.npz` and `go2_stats.npz`. Keep `--processed-dir` consistent between teacher, corrector, student, and inference.

## 3) Train Teacher

```bash
python -m csmt.pipelines.train_teacher \
  --output-root . \
  --processed-dir ./data/processed/loco_g1_go2 \
  --task-family locomotion \
  --pair-id g1_to_go2 \
  --save-dir ./runs/teacher_loco_g1_go2 \
  --device cuda:0 \
  --batch-size 128 \
  --epoch-num 3000
```

Teacher runs should contain `refactor_teacher_run.json`; legacy teacher dirs without it are not supported by the current inference utilities.

## 4) Teacher Inference

Single clip:

```bash
python -m csmt.pipelines.infer_teacher \
  --output-root . \
  --processed-dir ./data/processed/loco_g1_go2 \
  --task-family locomotion \
  --pair-id g1_to_go2 \
  --teacher-dir ./runs/teacher_loco_g1_go2 \
  --input-pkl ./data/raw/g1/locomotion/walk1_subject1.pkl \
  --output-pkl ./demo_output/teacher_go2.pkl \
  --device cuda:0 \
  --dst-start-height 0.28 \
  --save-src-debug
```

Batch mode:

```bash
python -m csmt.pipelines.infer_teacher \
  --output-root . \
  --processed-dir ./data/processed/loco_g1_go2 \
  --task-family locomotion \
  --pair-id g1_to_go2 \
  --teacher-dir ./runs/teacher_loco_g1_go2 \
  --input-pkl-dir ./data/student/loco_g1_go2/g1 \
  --output-pkl-dir ./data/student/loco_g1_go2/go2 \
  --device cuda:0 \
  --dst-start-height 0.28 \
  --no-save-src-debug
```

Useful final-cleaning flags:

```text
--corrector-ckpt ./runs/corrector_loco_g1_go2/best.pt
--apply-root-skate-comp
```

If both are used, the output order is:

```text
teacher -> corrector -> root skate compensation -> saved Go2 PKL
```

## 5) Corrector: Offline Teacher Cleanup

The corrector is an offline post-processor for teacher outputs. It is meant for cleaning the teacher-generated Go2 data before training students or downstream tracking policies.

The current corrector operates on long source PKL clips in memory:

```text
source G1 PKL -> teacher Go2 output -> corrector residual -> cleaned Go2 PKL
```

It does **not** train from prebuilt NPZ windows. During training it loads source PKLs from `--src-pkl-dir`, runs the teacher once per clip, caches the teacher output, and learns residual corrections over variable-length sequences.

### 5.1 Train Corrector

```bash
python -m csmt.pipelines.train_corrector \
  --output-root . \
  --processed-dir ./data/processed/loco_g1_go2 \
  --task-family locomotion \
  --pair-id g1_to_go2 \
  --teacher-dir ./runs/teacher_loco_g1_go2 \
  --src-pkl-dir ./data/student/loco_g1_go2/g1 \
  --save-dir ./runs/corrector_loco_g1_go2 \
  --device cuda:0 \
  --max-frames 0 \
  --train-seq-len 0 \
  --eval-seq-len 0 \
  --wandb
```

For very long clips, cap or crop training:

```bash
python -m csmt.pipelines.train_corrector \
  --output-root . \
  --processed-dir ./data/processed/loco_g1_go2 \
  --task-family locomotion \
  --pair-id g1_to_go2 \
  --teacher-dir ./runs/teacher_loco_g1_go2 \
  --src-pkl-dir ./data/student/loco_g1_go2/g1 \
  --save-dir ./runs/corrector_loco_g1_go2 \
  --device cuda:0 \
  --max-frames 3000 \
  --train-seq-len 1024 \
  --eval-seq-len 0 \
  --wandb
```

Main config:

```text
configs/models/corrector.yaml
```

Important config/CLI knobs:

- `joint_delta_max`: max residual on target joint angles
- `root_pos_delta_max`: max residual on root trajectory position in trajectory mode
- `yaw_delta_max`: max residual on root yaw
- `correct_root_motion`: master switch for root trajectory correction
- `correct_root_xy`, `correct_root_z`, `correct_root_yaw`: independently enable/disable root channels
- `lambda_preserve_joints`: keeps corrected joints close to teacher joints
- `lambda_preserve_root_vel`: preserves teacher root local velocity/yaw-rate while still allowing direct root-position correction
- `lambda_grounding`: penalizes floating/penetration during source-gated contact windows
- `lambda_skating`: penalizes stance-foot sliding
- `lambda_smooth`: temporal smoothness on residuals
- `ground_margin`: foot height margin for grounding
- `physics_ref_frames`: number of initial frames used for physics/reference estimates

The key design choice is that root correction is trajectory-based: the model edits root position/yaw directly, while the preserve-root loss compares the resulting root velocities/yaw-rate against the original teacher motion. This avoids cumulative velocity drift while still allowing the corrector to remove long-horizon floating or sinking.

### 5.2 Generate Cleaned Go2 PKLs

After training, use the corrector through teacher inference:

```bash
python -m csmt.pipelines.infer_teacher \
  --output-root . \
  --processed-dir ./data/processed/loco_g1_go2 \
  --task-family locomotion \
  --pair-id g1_to_go2 \
  --teacher-dir ./runs/teacher_loco_g1_go2 \
  --corrector-ckpt ./runs/corrector_loco_g1_go2/best.pt \
  --input-pkl-dir ./data/student/loco_g1_go2/g1 \
  --output-pkl-dir ./data/student/loco_g1_go2/go2 \
  --device cuda:0 \
  --dst-start-height 0.28 \
  --apply-root-skate-comp \
  --no-save-src-debug
```

Recommended data rule: train students on this final cleaned Go2 folder, not on raw teacher outputs if the teacher has visible floating, penetration, or skating artifacts.

## 6) Visualize Motions

Visualize any Go2 PKL:

```bash
python -m csmt.pipelines.visualize_motion \
  --output-root . \
  --robot-id go2 \
  --pkl ./demo_output/teacher_go2.pkl \
  --loop
```

Visualize a frame range:

```bash
python -m csmt.pipelines.visualize_motion \
  --output-root . \
  --robot-id go2 \
  --pkl ./demo_output/teacher_go2.pkl \
  --start-frame 300 \
  --end-frame 600 \
  --loop
```

If teacher inference was run with `--save-contact-debug`, you can overlay contact diagnostics:

```bash
python -m csmt.pipelines.visualize_motion \
  --output-root . \
  --robot-id go2 \
  --pkl ./demo_output/teacher_go2.pkl \
  --contact-debug-npz ./demo_output/teacher_go2_contact_debug.npz \
  --loop
```

Viewer controls:

```text
space: pause/play
r: reset
c: toggle contact overlay, if available
```

## 7) RT Student: Paired G1 PKL -> Cleaned Go2 PKL

This is the normal real-time student. It consumes G1-style robot features and predicts Go2 references. It is the path used when the live system still goes through GMR:

```text
video/camera -> FastSAM SMPL -> GMR G1 PKL/frame -> RT student -> Go2
```

Expected paired folders:

```text
data/student/loco_g1_go2/
  g1/
    clip_0001.pkl
    clip_0002.pkl
  go2/
    clip_0001.pkl
    clip_0002.pkl
```

Pairing rule: `g1/<name>.pkl` matches `go2/<name>.pkl` by exact filename.

Train:

```bash
python -m csmt.pipelines.train_student \
  --output-root . \
  --processed-dir ./data/processed/loco_g1_go2 \
  --task-family locomotion \
  --pair-id g1_to_go2 \
  --src-pkl-dir ./data/student/loco_g1_go2/g1 \
  --dst-pkl-dir ./data/student/loco_g1_go2/go2 \
  --save-dir ./runs/student_rt_g1_go2 \
  --device cuda:0 \
  --epochs 25 \
  --batch-size 256
```

Main config:

```text
configs/models/student_rt.yaml
```

Important knobs:

- `balanced_sampling`: samples clips more evenly so long clips do not dominate
- `samples_per_epoch`: `0` means use the full train-window count
- `prev_context_mode`: `student` is usually the realistic autoregressive setting
- `y_prev_noise_std`, `y_prev_noise_prob`: robustness to previous-output errors
- `lambda_imitation`: joint imitation weight
- `lambda_src_motion`: root/local velocity supervision weight
- `lambda_smooth`: output smoothness regularization

Run inference:

```bash
python -m csmt.pipelines.infer_student_rt \
  --output-root . \
  --processed-dir ./data/processed/loco_g1_go2 \
  --task-family locomotion \
  --pair-id g1_to_go2 \
  --student-ckpt ./runs/student_rt_g1_go2/best.pt \
  --input-pkl ./data/student/loco_g1_go2/g1/walk1_subject1.pkl \
  --output-pkl ./demo_output/student_rt_go2.pkl \
  --device cuda:0 \
  --root-motion-mode student \
  --dst-start-height 0.28
```

## 8) SMPL Student: Paired SMPL -> Cleaned Go2 PKL

This is the direct SMPL-input student. It skips GMR at inference time:

```text
video/camera -> FastSAM SMPL -> SMPL student -> Go2
```

Expected paired folders:

```text
data/student/loco_smpl_g1_go2/
  smpl/
    clip_0001.npz
    clip_0002.npz
  go2/
    clip_0001.pkl
    clip_0002.pkl
```

Pairing rule: `smpl/<name>.npz` matches `go2/<name>.pkl` by filename stem.

SMPL files should contain the usual AMASS-style arrays:

```text
pose_body:   [T, 63]
root_orient: [T, 3]
trans:       [T, 3]
```

Train:

```bash
python -m csmt.pipelines.train_student_smpl \
  --output-root . \
  --processed-dir ./data/processed/loco_g1_go2 \
  --task-family locomotion \
  --pair-id g1_to_go2 \
  --smpl-dir ./data/student/loco_smpl_g1_go2/smpl \
  --dst-pkl-dir ./data/student/loco_smpl_g1_go2/go2 \
  --save-dir ./runs/student_smpl_g1_go2 \
  --device cuda:0 \
  --epochs 25 \
  --batch-size 256
```

Main config:

```text
configs/models/student_smpl.yaml
```

Current SMPL student details:

- `smpl_input_dim: 69`
- SMPL is resampled to the paired Go2 PKL fps by default
- SMPL input normalization is computed from the paired training set and saved with the checkpoint
- `smpl_root_map: world_z` keeps vertical velocity in world Z, which is usually better for jumps/squats than treating all root velocity as local horizontal motion
- Targets are the final cleaned Go2 PKLs, not teacher outputs generated on the fly

Run inference:

```bash
python -m csmt.pipelines.infer_student_smpl \
  --output-root . \
  --processed-dir ./data/processed/loco_g1_go2 \
  --task-family locomotion \
  --pair-id g1_to_go2 \
  --student-ckpt ./runs/student_smpl_g1_go2/best.pt \
  --input-smpl ./data/student/loco_smpl_g1_go2/smpl/walking_fast01_stageii.npz \
  --output-pkl ./demo_output/student_smpl_go2.pkl \
  --device cuda:0 \
  --root-motion-mode student \
  --target-fps 30 \
  --dst-start-height 0.28
```
