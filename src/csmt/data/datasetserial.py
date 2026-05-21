"""
Dataset loader for motion retargeting.

Feature vector returned by __getitem__:
    [joint_angles (njoints) | lin_vel_local (3) | yaw_rate (1)]  -> njoints + 4

The global quaternion (base_rot) and translation (base_trans) are loaded as
auxiliary tensors for inference-time trajectory reconstruction but are NOT
part of the normalized model input.
"""

from torch.utils.data import Dataset
import numpy as np
import torch


class MotionDataset(Dataset):
    def __init__(self, config, topology: str, job: str):
        self.config = config
        self.topology = str(topology).lower()

        def _pick(*names, default=None):
            for name in names:
                if hasattr(config, name):
                    value = getattr(config, name)
                    if value is not None:
                        return value
            return default

        if self.topology in ("dst", "dog"):
            self.njoints = int(_pick("dst_njoints", "dog_njoints", default=0))
            self.nbodies = int(_pick("dst_nbodies", "dog_nbodies", default=0))
            stats_path = _pick("dststats_path", "dogstats_path")
            data_path = _pick("dst_train_path", "dog_train_path") if job == "train" else _pick("dst_test_path", "dog_test_path")
        elif self.topology in ("src", "human"):
            self.njoints = int(_pick("src_njoints", "hum_njoints", default=0))
            self.nbodies = int(_pick("src_nbodies", "hum_nbodies", default=0))
            stats_path = _pick("srcstats_path", "humstats_path")
            data_path = _pick("src_train_path", "hum_train_path") if job == "train" else _pick("src_test_path", "hum_test_path")
        else:
            raise ValueError(f"Unknown topology: {topology}")

        _stats = np.load(stats_path, allow_pickle=True)
        _data = np.load(data_path, allow_pickle=True)

        # Skeleton structure (used by SkeletonEncoder and FK)
        self.parents = _stats["parents"]
        self.offsets = _stats["offsets"]

        # Model input features [N, T, njoints+4]
        self.joint_pos = torch.from_numpy(_data["joint_pos"]).float()
        self.lin_vel_local = torch.from_numpy(_data["lin_vel_local"]).float()
        self.yaw_rate = torch.from_numpy(_data["yaw_rate"]).float()

        # Auxiliary data (not model inputs)
        self.base_trans = torch.from_numpy(_data["base_trans"]).float()
        self.base_rot = torch.from_numpy(_data["base_rot"]).float()
        self.yaw = torch.from_numpy(_data["yaw"]).float()
        if "ee_tag" in _data.files:
            self.ee_tag = torch.from_numpy(_data["ee_tag"]).float()
            if self.ee_tag.ndim == 1:
                self.ee_tag = self.ee_tag.unsqueeze(-1)
        else:
            # Backward compatibility for older processed datasets.
            self.ee_tag = torch.ones((self.joint_pos.shape[0], 1), dtype=torch.float32)

        # Normalization statistics (for model input features only)
        self.mean = torch.from_numpy(_stats["mean"]).float()
        self.std = torch.from_numpy(_stats["std"]).float()

        # Precompute normalized motion features once to avoid per-sample cat/norm overhead.
        self.motion_data = torch.cat([self.joint_pos, self.lin_vel_local, self.yaw_rate], dim=-1)
        self.motion_data_norm = (self.motion_data - self.mean) / (self.std + 1e-8)
        self.offsets_flat = torch.from_numpy(self.offsets).float().reshape(-1)

        # Velocity range for retargeting velocity mapping loss. Mapping mode
        # normalizes planar speed, so compute vmax from XY speed as well.
        vel_norms_xy = self.lin_vel_local[..., :2].norm(dim=-1).numpy()
        if self.topology in ("src", "human"):
            vmax_percentile = float(_pick("retar_vel_src_vmax_percentile", default=95.0))
        else:
            vmax_percentile = float(_pick("retar_vel_dst_vmax_percentile", default=95.0))
        vmax_percentile = float(np.clip(vmax_percentile, 1.0, 100.0))
        self.min_vel = float(np.percentile(vel_norms_xy, 5))
        self.max_vel = float(np.percentile(vel_norms_xy, vmax_percentile))
        self.vel_vmax_percentile = vmax_percentile

    def __len__(self):
        return len(self.joint_pos)

    def __getitem__(self, idx):
        motion_data = self.motion_data_norm[idx]
        offsets = self.offsets_flat
        label = self.ee_tag[idx]

        # Keep return structure compatible with training loops (5-tuple).
        return motion_data, label, offsets, offsets, label

    def denorm(self, x: torch.Tensor, transpose: bool = False) -> torch.Tensor:
        if transpose:
            x = x.transpose(1, 2)

        device = x.device
        mean = self.mean.to(device)
        std = self.std.to(device)
        return x * std + mean

    def get_auxiliary(self, idx):
        return {
            "base_trans": self.base_trans[idx],
            "base_rot": self.base_rot[idx],
            "yaw": self.yaw[idx],
        }


class SrcDataset(MotionDataset):
    def __init__(self, config, topology=None, job=None):
        split = "test" if str(job).lower() == "test" else "train"
        super().__init__(config, "src", split)


class SrcTestDataset(MotionDataset):
    def __init__(self, config, topology=None, job=None):
        super().__init__(config, "src", "test")


class DstDataset(MotionDataset):
    def __init__(self, config, topology=None, job=None):
        split = "test" if str(job).lower() == "test" else "train"
        super().__init__(config, "dst", split)


class DstTestDataset(MotionDataset):
    def __init__(self, config, topology=None, job=None):
        super().__init__(config, "dst", "test")


# Backward-compatible aliases.
class HumDataset(MotionDataset):
    def __init__(self, config, topology=None, job=None):
        super().__init__(config, "src", "train")


class HumTestDataset(MotionDataset):
    def __init__(self, config, topology=None, job=None):
        super().__init__(config, "src", "test")


class DogDataset(MotionDataset):
    def __init__(self, config, topology=None, job=None):
        super().__init__(config, "dst", "train")


class DogTestDataset(MotionDataset):
    def __init__(self, config, topology=None, job=None):
        super().__init__(config, "dst", "test")
