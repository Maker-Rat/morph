"""
Fast differentiable Forward Kinematics using pytorch_kinematics.
Supports G1 and Go2 robots loaded from MJCF files.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import os
import numpy as np
import xml.etree.ElementTree as ET

try:
    from pytorch_kinematics import build_chain_from_mjcf
    import pytorch_kinematics.mjcf as pk_mjcf
    PYTORCH_KINEMATICS_AVAILABLE = True
except ImportError:
    PYTORCH_KINEMATICS_AVAILABLE = False
    print("[FK] pytorch_kinematics not available. Install with: pip install pytorch-kinematics")


class ForwardKinematics(nn.Module):
    """
    Fast, differentiable forward kinematics using pytorch_kinematics.
    Supports batched operations and GPU acceleration.
    Loads from MJCF files only.
    """
    
    def __init__(self, model_path: str, robot_name: str = "g1", device: str = 'cuda', 
                 parents: Optional[List[int]] = None):
        """
        Initialize FK from MJCF file.
        
        Args:
            model_path: Path to MJCF (.xml) file
            robot_name: Robot name ('g1' or 'go2')
            device: Device to use ('cuda' or 'cpu')
            parents: Parent indices for local position computation (optional)
        """
        super().__init__()
        
        if not PYTORCH_KINEMATICS_AVAILABLE:
            raise RuntimeError("pytorch_kinematics is not installed. Install with: pip install pytorch-kinematics")
        
        self.robot_name = robot_name.lower()
        self.model_path = model_path
        self.device = device
        self.parents = parents
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        if not model_path.endswith('.xml'):
            raise ValueError(f"Expected MJCF (.xml) file, got: {model_path}")
        
        print(f"[{robot_name.upper()} FK] Loading kinematic chain from {os.path.basename(model_path)} (MJCF)...")
        
        with open(model_path, 'r') as f:
            content = f.read()
        
        try:
            self.chain = build_chain_from_mjcf(content)
        except Exception as e:
            import traceback
            print(f"[{robot_name.upper()} FK] Detailed MJCF error:")
            print(traceback.format_exc())
            raise
        
        self.chain = self.chain.to(dtype=torch.float32, device=torch.device(device))
        
        # Get joint information
        self.joint_names = self.chain.get_joint_parameter_names()
        self.n_joints = len(self.joint_names)
        
        # Get all link names and filter out 'world'
        all_link_names = self.chain.get_link_names()
        self.link_names = [name for name in all_link_names if name != 'world']
        self.n_links = len(self.link_names)
        
        print(f"[{robot_name.upper()} FK] Loaded {self.n_joints} joints, {self.n_links} links (excluding world)")
        print(f"[{robot_name.upper()} FK] Joint order: {self.joint_names[:min(5, len(self.joint_names))]}...")
        print(f"[{robot_name.upper()} FK] Link order: {self.link_names[:min(5, len(self.link_names))]}...")
        print(f"[{robot_name.upper()} FK] Ready ✓")
    
    @staticmethod
    def parse_joint_limits_from_mjcf(xml_path, num_joints):
        """
        Parse joint limits from MJCF XML file.
        Resolves class inheritance for joint ranges.
        Returns: joint_ranges tensor [num_joints, 2] with (min, max) for each joint in degrees/radians
        Falls back to default ranges if parsing fails.
        """
        default_range = 180.0  # degrees
        joint_ranges = torch.ones(num_joints, 2) * torch.tensor([-default_range, default_range])
        
        if not os.path.exists(xml_path):
            print(f"Warning: XML file {xml_path} not found. Using default joint ranges.")
            return joint_ranges
        
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Build a map of class -> joint range (handles class inheritance)
            class_ranges = {}
            for default_elem in root.findall('.//default'):
                class_name = default_elem.get('class', '')
                
                # Check this default element for joint with range
                for joint_elem in default_elem.findall('joint'):
                    range_attr = joint_elem.get('range')
                    if range_attr:
                        try:
                            parts = range_attr.split()
                            min_val = float(parts[0])
                            max_val = float(parts[1])
                            class_ranges[class_name] = (min_val, max_val)
                            break  # Use first joint with range in this class
                        except:
                            pass
                
                # Check nested defaults (e.g., <default class="hip"><default class="front_hip">)
                for nested_default in default_elem.findall('default'):
                    nested_class = nested_default.get('class', '')
                    for joint_elem in nested_default.findall('joint'):
                        range_attr = joint_elem.get('range')
                        if range_attr:
                            try:
                                parts = range_attr.split()
                                min_val = float(parts[0])
                                max_val = float(parts[1])
                                class_ranges[nested_class] = (min_val, max_val)
                                break
                            except:
                                pass
            
            # Find all joints in body hierarchy and resolve their ranges
            joint_count = 0
            for body in root.findall('.//body'):
                for joint in body.findall('joint'):
                    if joint_count >= num_joints:
                        break
                    
                    # Try direct range attribute first
                    range_attr = joint.get('range')
                    if range_attr:
                        try:
                            parts = range_attr.split()
                            min_val = float(parts[0])
                            max_val = float(parts[1])
                            joint_ranges[joint_count, 0] = min_val
                            joint_ranges[joint_count, 1] = max_val
                            joint_count += 1
                            continue
                        except:
                            pass
                    
                    # Try to resolve via class inheritance
                    joint_class = joint.get('class', '')
                    if joint_class and joint_class in class_ranges:
                        min_val, max_val = class_ranges[joint_class]
                        joint_ranges[joint_count, 0] = min_val
                        joint_ranges[joint_count, 1] = max_val
                    
                    joint_count += 1
                if joint_count >= num_joints:
                    break
        except Exception as e:
            print(f"Warning: Failed to parse joint limits from {xml_path}: {e}")
        
        return joint_ranges
    
    def _rotate_by_quaternion(self, points: torch.Tensor, quaternion: torch.Tensor) -> torch.Tensor:
        """
        Rotate points by quaternion (batch operation).
        
        Args:
            points: [batch_size, n_points, 3] or [batch_size, 3]
            quaternion: [batch_size, 4] - XYZW format (xyz first, w last)
            
        Returns:
            Rotated points with same shape as input
        """
        # Handle both [B, N, 3] and [B, 3] inputs
        input_shape = points.shape
        if len(input_shape) == 2:
            points = points.unsqueeze(1)  # [B, 1, 3]
        
        # Extract quaternion components (XYZW format)
        x, y, z, w = quaternion[..., 0], quaternion[..., 1], quaternion[..., 2], quaternion[..., 3]
        
        # Normalize quaternion
        norm = torch.sqrt(x**2 + y**2 + z**2 + w**2)
        x, y, z, w = x/norm, y/norm, z/norm, w/norm
        
        # Build rotation matrix using quaternion
        # R = I + 2*q_v*q_v^T + 2*w*[q_v]_x
        # where [q_v]_x is the skew-symmetric matrix of (x,y,z)
        
        # Rotation matrix elements
        xx, yy, zz = x*x, y*y, z*z
        xy, xz, yz = x*y, x*z, y*z
        wx, wy, wz = w*x, w*y, w*z
        
        # Build rotation matrix [B, 3, 3]
        R = torch.stack([
            torch.stack([1 - 2*(yy + zz), 2*(xy - wz), 2*(xz + wy)], dim=-1),
            torch.stack([2*(xy + wz), 1 - 2*(xx + zz), 2*(yz - wx)], dim=-1),
            torch.stack([2*(xz - wy), 2*(yz + wx), 1 - 2*(xx + yy)], dim=-1)
        ], dim=-2)  # [B, 3, 3]
        
        # Apply rotation: [B, 3, 3] @ [B, N, 3] -> [B, N, 3]
        rotated = torch.einsum('bij,bnj->bni', R, points)
        
        # Restore original shape if needed
        if len(input_shape) == 2:
            rotated = rotated.squeeze(1)
        
        return rotated
    
    def forward(self, motion_data: torch.Tensor,
                dt: float = 1.0/30.0) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute forward kinematics for batched motion data.

        Feature vector: [joint_angles (n_joints) | lin_vel_local (3) | yaw_rate (1)]

        Global positions are computed by:
          1. Running FK in the robot local frame (joint angles only)
          2. Rotating local positions by the integrated yaw
          3. Translating by the integrated world position

        Args:
            motion_data: [B, T, n_joints+4]
            dt:          timestep in seconds (default 1/30)

        Returns:
            global_positions: [B, T, n_bodies, 3]
            base_relative_positions: [B, T, n_bodies, 3]
                FK positions after yaw rotation but before global base translation
                (i.e., relative to base origin in world-aligned frame).
        """
        batch_size, time_steps, _ = motion_data.shape

        joint_angles  = motion_data[..., :self.n_joints]         # [B, T, n_joints]
        lin_vel_local = motion_data[..., self.n_joints:self.n_joints+3]  # [B, T, 3]
        yaw_rate      = motion_data[..., self.n_joints+3]         # [B, T]

        # ── 1. FK in local frame ───────────────────────────────────────────
        joint_angles_flat = joint_angles.reshape(-1, self.n_joints)
        fk_result = self.chain.forward_kinematics(joint_angles_flat)

        local_positions_list = []
        for link_name in self.link_names:
            if link_name in fk_result:
                pos = fk_result[link_name].get_matrix()[:, :3, 3]
                local_positions_list.append(pos)

        if not local_positions_list:
            raise RuntimeError(f"No links found. Available: {list(fk_result.keys())}")

        local_pos_flat = torch.stack(local_positions_list, dim=1)  # [B*T, n_bodies, 3]
        n_bodies = len(local_positions_list)
        local_pos = local_pos_flat.reshape(batch_size, time_steps, n_bodies, 3)

        # ── 2. Integrate yaw_rate → yaw angle ─────────────────────────────
        yaw = torch.cumsum(yaw_rate * dt, dim=1)  # [B, T]

        # Build yaw rotation quaternion: [0, 0, sin(yaw/2), cos(yaw/2)]  (XYZW)
        half = yaw * 0.5
        zeros = torch.zeros_like(half)
        yaw_quat = torch.stack([zeros, zeros, torch.sin(half), torch.cos(half)], dim=-1)  # [B, T, 4]

        # ── 3. Rotate local → world ────────────────────────────────────────
        world_pos_rotated = self._rotate_by_quaternion(
            local_pos.reshape(-1, n_bodies, 3),
            yaw_quat.reshape(-1, 4)
        ).reshape(batch_size, time_steps, n_bodies, 3)

        # Base-link-relative positions (true base-relative coordinates).
        # In both G1 and Go2 chains, link_names[0] is the base link
        # ('pelvis' for G1, 'base' for Go2). Subtract it so EE targets are
        # expressed relative to robot base, not MJCF world offset.
        base_relative_positions = world_pos_rotated - world_pos_rotated[:, :, 0:1, :]

        # ── 4. Integrate local velocity → world position ───────────────────
        # Rotate local velocity to world frame
        cos_yaw = torch.cos(yaw)  # [B, T]
        sin_yaw = torch.sin(yaw)
        world_vx = cos_yaw * lin_vel_local[..., 0] - sin_yaw * lin_vel_local[..., 1]
        world_vy = sin_yaw * lin_vel_local[..., 0] + cos_yaw * lin_vel_local[..., 1]
        world_vz = lin_vel_local[..., 2]
        world_vel = torch.stack([world_vx, world_vy, world_vz], dim=-1)  # [B, T, 3]

        base_position = torch.cumsum(world_vel * dt, dim=1)  # [B, T, 3]

        global_positions = world_pos_rotated + base_position.unsqueeze(2)

        return global_positions, base_relative_positions

    def compute_base_relative_points(self,
                                     motion_data: torch.Tensor,
                                     link_indices: List[int],
                                     local_offsets: Optional[List[List[float]]] = None,
                                     dt: float = 1.0/30.0) -> torch.Tensor:
        """
        Compute base-relative positions for selected links with optional per-link local offsets.

        This is useful when the monitored EE is not at the link origin (e.g., calf -> paw tip).

        Args:
            motion_data:   [B, T, n_joints+4]
            link_indices:  list of link indices in self.link_names
            local_offsets: list of [x, y, z] local offsets per selected link
            dt:            timestep in seconds

        Returns:
            ee_points: [B, T, K, 3] base-relative positions, K=len(link_indices)
        """
        if len(link_indices) == 0:
            bsz, tsz = motion_data.shape[0], motion_data.shape[1]
            return torch.zeros(bsz, tsz, 0, 3, device=motion_data.device, dtype=motion_data.dtype)

        if local_offsets is None:
            local_offsets = [[0.0, 0.0, 0.0] for _ in link_indices]
        if len(local_offsets) < len(link_indices):
            local_offsets = list(local_offsets) + [[0.0, 0.0, 0.0] for _ in range(len(link_indices) - len(local_offsets))]
        local_offsets_t = torch.as_tensor(local_offsets, dtype=motion_data.dtype, device=motion_data.device)

        batch_size, time_steps, _ = motion_data.shape
        joint_angles = motion_data[..., :self.n_joints]      # [B, T, n_joints]
        yaw_rate = motion_data[..., self.n_joints+3]         # [B, T]
        yaw = torch.cumsum(yaw_rate * dt, dim=1)             # [B, T]

        half = yaw * 0.5
        yaw_quat = torch.stack(
            [torch.zeros_like(half), torch.zeros_like(half), torch.sin(half), torch.cos(half)],
            dim=-1
        )  # [B, T, 4] in XYZW

        joint_angles_flat = joint_angles.reshape(-1, self.n_joints)  # [B*T, n_joints]
        fk_result = self.chain.forward_kinematics(
            {name: joint_angles_flat[:, i] for i, name in enumerate(self.joint_names)}
        )

        base_name = self.link_names[0]
        base_tf = fk_result[base_name].get_matrix()[:, :3, :]            # [B*T, 3, 4]
        base_local = base_tf[:, :, 3].reshape(batch_size, time_steps, 1, 3)  # [B, T, 1, 3]
        base_world_rot = self._rotate_by_quaternion(
            base_local.reshape(-1, 1, 3),
            yaw_quat.reshape(-1, 4),
        ).reshape(batch_size, time_steps, 1, 3)  # [B, T, 1, 3]

        ee_local_points = []
        for k, link_idx in enumerate(link_indices):
            idx = int(link_idx)
            if idx < 0 or idx >= len(self.link_names):
                raise IndexError(f"Link index out of range: {idx} for {len(self.link_names)} links")

            link_name = self.link_names[idx]
            tf = fk_result[link_name].get_matrix()[:, :3, :]   # [B*T, 3, 4]
            rot = tf[:, :, :3]                                  # [B*T, 3, 3]
            pos = tf[:, :, 3]                                   # [B*T, 3]

            off = local_offsets_t[k].view(1, 3, 1)
            off = off.expand(pos.shape[0], -1, -1)             # [B*T, 3, 1]
            pos = pos + torch.bmm(rot, off).squeeze(-1)        # [B*T, 3]
            ee_local_points.append(pos)

        ee_local_points = torch.stack(ee_local_points, dim=1)  # [B*T, K, 3]
        ee_world_rot = self._rotate_by_quaternion(ee_local_points, yaw_quat.reshape(-1, 4))
        ee_world_rot = ee_world_rot.reshape(batch_size, time_steps, len(link_indices), 3)

        return ee_world_rot - base_world_rot
