from csmt.models.Intergrated import IntegratedModel
from csmt.models.functions import (
    get_gan_loss,
    get_rec_loss,
    get_root_loss,
    get_kine_loss,
    get_cycle_loss,
    get_cycle_latent_loss,
    get_retar_root_v_loss,
    get_recloss_input,
    get_optimizer,
)
import torch
import os
from csmt.parser.base import try_mkdir
from csmt.models.base_model import BaseModel
from csmt.utils.loss_function import (estimate_contact_from_height,
                                 skating_loss_from_contact,
                                 grounding_loss_from_contact)
import torch.nn as nn
import time

class PAN_model(BaseModel):
    def __init__(self, args, body_parts, joint_parts, datasets, topology_name):
        super(PAN_model, self).__init__(args)
        self.D_parameters = []
        self.G_parameters = []
        self.models = []
        self.args = args
        self.datasets = datasets
        self.n_topology = len(body_parts)
        self.topology_name = topology_name
        
        self.criterion_gan = get_gan_loss(args.dis_loss_type)
        self.criterion_rec = get_rec_loss(args.rec_loss_type)
        # Root loss for non-quaternion targets
        self.criterion_root = get_root_loss(args.root_loss_type)
        # Select quaternion loss type for explicit quaternion supervision
        if hasattr(args, 'quat_loss_type') and args.quat_loss_type == 'geodesic':
            from csmt.utils.loss_function import geodesic_loss_quat
            self.criterion_quat = geodesic_loss_quat
        else:
            self.criterion_quat = self.criterion_root
        self.criterion_kine = get_kine_loss(args.global_kine_loss_type)
        self.criterion_cycle = get_cycle_loss(args.cyc_loss_type)
        self.criterion_cycle_latent = get_cycle_latent_loss(args.cyc_latent_loss_type)
        self.criterion_root_v = get_retar_root_v_loss(args.retar_vel_loss_type)
        self.mse = nn.MSELoss()

        # Canonical physics grounding weight.
        if isinstance(args, dict):
            lambda_ground = args.get('lambda_grounding', 0.0)
            lambda_sk = args.get('lambda_skating', 0.0)
        else:
            try:
                lambda_ground = getattr(args, 'lambda_grounding')
            except (AttributeError, KeyError):
                lambda_ground = 0.0
            try:
                lambda_sk = getattr(args, 'lambda_skating')
            except (AttributeError, KeyError):
                lambda_sk = 0.0
        self.lambda_grounding = float(lambda_ground)

        if isinstance(args, dict):
            src_start_height = float(args.get('src_start_height', 0.0))
            dst_start_height = float(args.get('dst_start_height', 0.0))
            physics_ground_mode = str(args.get('physics_ground_mode', 'first_frames')).lower()
        else:
            src_start_height = float(getattr(args, 'src_start_height', 0.0))
            dst_start_height = float(getattr(args, 'dst_start_height', 0.0))
            physics_ground_mode = str(getattr(args, 'physics_ground_mode', 'first_frames')).lower()
        if physics_ground_mode not in ('first_frames', 'zero_nominal'):
            physics_ground_mode = 'first_frames'
        self._nominal_start_height = {'src': src_start_height, 'dst': dst_start_height}
        self._physics_ground_mode = physics_ground_mode

        # Physics loss flags — only active when lambdas > 0.
        # Uses args.src_end / args.dst_end as foot FK body indices.
        self.use_physics = (float(lambda_sk) > 0 or
                            self.lambda_grounding > 0)
        def _as_list(value):
            if value is None:
                return []
            return list(value)

        # Canonical topology aliases used across legacy/new naming.
        self._topology_alias = {
            "src": "src",
            "human": "src",
            "dst": "dst",
            "dog": "dst",
        }
        # Map topology key -> foot indices
        self._foot_indices = {
            'src': _as_list(getattr(self.args, 'src_end', getattr(self.args, 'hum_end', []))),
            'dst': _as_list(getattr(self.args, 'dst_end', getattr(self.args, 'dog_end', []))),
        }
        # End-effector indices used by EE retargeting loss (e.g., hands ↔ front paws).
        self._ee_indices = {
            'src': _as_list(getattr(self.args, 'src_ee', getattr(self.args, 'hum_ee', []))),
            'dst': _as_list(getattr(self.args, 'dst_ee', getattr(self.args, 'dog_ee', []))),
        }

        for i in range(len(topology_name)):
            model = IntegratedModel(args, body_parts[i], joint_parts[i], datasets[i].njoints, datasets[i].nbodies,
                                    datasets[i].parents, topology_name[i], self.device)
            
            if args.is_train:
                model.train()
            else:
                model.eval()
            self.models.append(model)
            self.D_parameters += model.D_parameters()
            self.G_parameters += model.G_parameters()

        # Cache dataset velocity range tensors on device (avoids per-step tensor allocations).
        self._vel_min_t = []
        self._vel_max_t = []
        for ds in self.datasets:
            self._vel_min_t.append(torch.tensor(float(ds.min_vel), dtype=torch.float32, device=self.device))
            self._vel_max_t.append(torch.tensor(float(ds.max_vel), dtype=torch.float32, device=self.device))

        if args.is_train:
            self.fake_pools = []
            self.optimizerD = get_optimizer(args.optimizer, self.D_parameters, lr=args.lr_d)
            self.optimizerG = get_optimizer(args.optimizer, self.G_parameters, lr=args.lr_g)
            self.optimizers = [self.optimizerD, self.optimizerG]

    def discriminator_requires_grad_(self, requires_grad):
        for model in self.models:
            for para in model.discriminator.parameters():
                para.requires_grad = requires_grad

    def train(self):
        """Set all models to training mode"""
        for model in self.models:
            model.train()
    
    def eval(self):
        """Set all models to evaluation mode"""
        for model in self.models:
            model.eval()

    def set_input(self, input):
        self.motions_input = []
        self.offsets = []
        self.offsets_withend = []
        self.ee_sample_weight_gates = []
        for i, item in enumerate(input):
            if len(item) >= 4:
                motion, offsets, offsets_withend, ee_gate = item[:4]
            else:
                motion, offsets, offsets_withend = item
                ee_gate = torch.ones((motion.shape[0], 1), dtype=motion.dtype, device=motion.device)
            # print("Data Input :", motion.shape)
            self.motions_input.append(motion.float().to(self.device))
            self.offsets.append(offsets.float().to(self.device))
            self.offsets_withend.append(offsets_withend.float().to(self.device))
            self.ee_sample_weight_gates.append(ee_gate.float().to(self.device))
        
    def forward(self):
        self.motion = []
        self.motion_denorm = []
        self.skel_rep = []
        self.latents = []
        self.gt_pos = []
        self.gt_local_pos = []
        self.rec = []
        self.rec_denorm = []
        self.rec_pos = []
        self.rec_local_pos = []

        self.cyc = []
        self.cyc_denorm = []
        self.cyc_pos = []
        self.cyc_local_pos = []
        self.cyc_latents = []

        self.fake_retar = []
        self.fake_retar_denorm = []
        self.fake_pos = []
        self.fake_local_pos = []
        self.fake_latents = []
        self.retar_latents = []


        self.mus      = []   # mu per topology (None if not VAE)
        self.log_vars = []   # log_var per topology (None if not VAE)

        # reconstruct
        for i in range(self.n_topology):
            motion = self.motions_input[i]
            self.skel_rep.append(self.models[i].skel_enc(self.offsets[i]).unsqueeze(-1))
            ae_out = self.models[i].ae(motion, self.skel_rep[i])
            if self.models[i].ae.use_vae:
                latent, mu, log_var, rec = ae_out
            else:
                latent, rec = ae_out
                mu, log_var = None, None
            rec_denorm = self.datasets[i].denorm(rec, transpose=False)

            rec_pos, rec_local_pos = self.models[i].fk.forward(rec_denorm)
            motion_denorm = self.datasets[i].denorm(motion, transpose=True)
            # motion_denorm already has unit quaternions (not denormalized), no need to normalize
            gt_pos, gt_local_pos = self.models[i].fk.forward(motion_denorm)

            self.motion.append(motion)
            self.motion_denorm.append(motion_denorm)
            self.rec.append(rec)
            self.rec_denorm.append(rec_denorm)
            self.rec_pos.append(rec_pos)
            self.latents.append(latent)
            self.mus.append(mu)
            self.log_vars.append(log_var)
            self.gt_pos.append(gt_pos)
            self.gt_local_pos.append(gt_local_pos)
            self.rec_local_pos.append(rec_local_pos)

        # retargeting
        for i in range(self.n_topology):
            a = 0
            for j in range(self.n_topology):
                if j == i:
                    continue
                else:
                    # Use mu for retargeting when VAE (deterministic, no sampling noise)
                    retar_latent = self.mus[i] if self.mus[i] is not None else self.latents[i]
                    fake_retar = self.models[j].ae.dec(retar_latent, self.skel_rep[j])
                    
                    fake_retar_input = self.models[j].ae.outformat2input(fake_retar)
                    fake_latent, _, _ = self.models[j].ae.encode(fake_retar_input)

                    # cycle
                    cyc_latent = fake_latent
                    cyc = self.models[i].ae.dec(cyc_latent, self.skel_rep[i])
                    cyc_denorm = self.datasets[i].denorm(cyc, transpose=False)
                    cyc_pos, cyc_local_pos = self.models[i].fk.forward(cyc_denorm)
                    fake_retar_denorm = self.datasets[j].denorm(fake_retar, transpose=False)
                    fake_pos, fake_local_pos = self.models[j].fk.forward(fake_retar_denorm)

                    self.retar_latents.append(retar_latent)
                    self.fake_retar.append(fake_retar)
                    self.fake_retar_denorm.append(fake_retar_denorm)
                    self.fake_pos.append(fake_pos)
                    self.fake_local_pos.append(fake_local_pos)
                    self.fake_latents.append(fake_latent)

                    self.cyc.append(cyc)
                    self.cyc_latents.append(cyc_latent)
                    self.cyc_denorm.append(cyc_denorm)
                    self.cyc_pos.append(cyc_pos)
                    self.cyc_local_pos.append(cyc_local_pos)

                a += 1

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterion_gan(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterion_gan(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D(self):
        self.loss_D = 0
        """
        A->B, B->A [0, 1]
        """
        p = 0
        for i in range(self.n_topology):
            for j in range(self.n_topology):
                if j == i:
                    continue
                else:
                    # j is the destination topology for the discriminator
                    if self.args.dis_mode == 'denorm_rotation':
                        # Real data: ground truth motion for topology j (includes joint angles, quat, velocities)
                        true_input = self.motion_denorm[j].transpose(1, 2)
                        # Fake data: retargeted motion for topology j
                        fake_input = self.fake_retar_denorm[p].transpose(1, 2)
                    elif self.args.dis_mode == 'denorm_pos':
                        # Real data: ground truth positions for topology j (all bodies)
                        true_input = self.gt_pos[j][:, :, :self.datasets[j].nbodies, :].reshape(
                            self.gt_pos[j].shape[0], self.gt_pos[j].shape[1], -1).transpose(1, 2)
                        # Fake data: retargeted positions from fake_pos[p] (all bodies)
                        fake_input = self.fake_pos[p][:, :, :self.datasets[j].nbodies, :].reshape(
                            self.fake_pos[p].shape[0], self.fake_pos[p].shape[1], -1).transpose(1, 2)
                    elif self.args.dis_mode == 'latent':
                        # Latent space has same dimension, so this should work
                        true_input = self.latents[j]
                        fake_input = self.retar_latents[p]
                    else:
                        raise ValueError(f"Unsupported dis_mode: {self.args.dis_mode}")
                    
                    loss_Ds = self.backward_D_basic(self.models[j].discriminator,
                                                    true_input.detach(), fake_input.detach())
                    self.loss_D += loss_Ds
                    self.loss_recoder.add_scalar('D_loss_{}'.format(i), loss_Ds)
                p += 1

    def compute_joint_limit_loss(self, joint_angles, joint_limits, threshold=0.85):
        """
        Penalize joint angles that exceed 85% of the allowed range.
        
        Args:
            joint_angles: [B, T, num_joints] - joint angles in denormalized space
            joint_limits: [num_joints, 2] - (min, max) limits for each joint
            threshold: penalty applied when position exceeds threshold * range
        
        Returns:
            loss: scalar tensor
        """
        if joint_limits is None:
            return torch.tensor(0.0, device=joint_angles.device, dtype=joint_angles.dtype)
        
        # Compute joint ranges
        joint_range = joint_limits[:, 1] - joint_limits[:, 0]  # [num_joints]
        joint_min = joint_limits[:, 0]  # [num_joints]
        
        # Normalize angles to [0, 1] within their range
        normalized_angles = (joint_angles - joint_min) / (joint_range + 1e-8)  # [B, T, num_joints]
        
        # Compute penalty: values outside [threshold, 1-threshold] are penalized
        # Penalty = max(0, abs(normalized - 0.5) - 0.5*(1-threshold))
        center = 0.5
        limit_dist = 0.5 * (1.0 - threshold)
        
        distance_from_center = torch.abs(normalized_angles - center)  # [B, T, num_joints]
        violation = torch.clamp(distance_from_center - limit_dist, min=0.0)  # [B, T, num_joints]
        
        loss = torch.mean(violation ** 2)
        return loss

    def _canonical_topology_key(self, name):
        return self._topology_alias.get(str(name).lower(), str(name).lower())
    
    def _default_ee_local_offset(self, topology_name, link_name):
        """
        Default local EE offset heuristic.
        For Go2 calf links, shift to a distal point (paw-tip proxy) instead of the
        calf link origin, which is often too static for meaningful EE supervision.
        """
        topo = self._canonical_topology_key(topology_name)
        if topo == 'dst' and isinstance(link_name, str) and link_name.endswith('_calf'):
            return [0.0, 0.0, -0.213]
        return [0.0, 0.0, 0.0]

    @staticmethod
    def _offsets_all_zero(offsets, eps=1e-12):
        for off in offsets:
            if abs(float(off[0])) > eps or abs(float(off[1])) > eps or abs(float(off[2])) > eps:
                return False
        return True

    def _compute_paired_ee_points(self, src_top_idx, dst_top_idx,
                                  src_motion_denorm, dst_motion_denorm,
                                  src_ee_idx, dst_ee_idx,
                                  src_local_cache=None, dst_local_cache=None):
        """
        Build paired source/destination EE points in base-relative space using FK,
        with optional per-link local offsets (auto-applied by topology/link name).
        """
        if src_motion_denorm is None or dst_motion_denorm is None:
            return None, None
        if len(src_ee_idx) == 0 or len(dst_ee_idx) == 0:
            return None, None

        fk_src = self.models[src_top_idx].fk
        fk_dst = self.models[dst_top_idx].fk
        src_name = self._canonical_topology_key(self.topology_name[src_top_idx])
        dst_name = self._canonical_topology_key(self.topology_name[dst_top_idx])

        n = min(len(src_ee_idx), len(dst_ee_idx))
        src_valid_idx = []
        dst_valid_idx = []
        src_offsets = []
        dst_offsets = []
        for i in range(n):
            s_idx = int(src_ee_idx[i])
            d_idx = int(dst_ee_idx[i])
            if s_idx < 0 or s_idx >= len(fk_src.link_names):
                continue
            if d_idx < 0 or d_idx >= len(fk_dst.link_names):
                continue

            s_name = fk_src.link_names[s_idx]
            d_name = fk_dst.link_names[d_idx]
            src_valid_idx.append(s_idx)
            dst_valid_idx.append(d_idx)
            src_offsets.append(self._default_ee_local_offset(src_name, s_name))
            dst_offsets.append(self._default_ee_local_offset(dst_name, d_name))

        if len(src_valid_idx) == 0:
            return None, None

        if src_local_cache is not None and self._offsets_all_zero(src_offsets):
            src_ee = src_local_cache[:, :, src_valid_idx, :]
        else:
            src_ee = fk_src.compute_base_relative_points(
                src_motion_denorm, src_valid_idx, src_offsets, dt=1.0 / 30.0
            )

        if dst_local_cache is not None and self._offsets_all_zero(dst_offsets):
            dst_ee = dst_local_cache[:, :, dst_valid_idx, :]
        else:
            dst_ee = fk_dst.compute_base_relative_points(
                dst_motion_denorm, dst_valid_idx, dst_offsets, dt=1.0 / 30.0
            )
        return src_ee, dst_ee

    def compute_ee_match_loss(self, src_ee, dst_ee,
                              mode='disp',
                              src_scale=None, dst_scale=None,
                              eps=1e-8,
                              sample_weight=None):
        """
        End-effector matching loss on base-relative FK positions.

        Args:
            src_ee: [B, T, K, 3] source EE points (already paired/aligned)
            dst_ee: [B, T, K, 3] destination EE points (already paired/aligned)
            mode: 'disp' | 'pos' | 'direction' (legacy alias: 'position' -> 'disp')
            src_scale, dst_scale: optional manual scales. If None, estimated from
                batch mean signal magnitudes for optional normalization.
        """
        if src_ee is None or dst_ee is None:
            return torch.tensor(0.0, device=self.device)
        if src_ee.shape[2] == 0 or dst_ee.shape[2] == 0:
            return torch.tensor(0.0, device=self.device)

        n = min(src_ee.shape[2], dst_ee.shape[2])
        src_ee = src_ee[:, :, :n, :]
        dst_ee = dst_ee[:, :, :n, :]

        # Backward-compatibility alias.
        if mode == 'position':
            mode = 'disp'

        if mode in ('disp', 'pos'):
            # Build source/destination EE signals to compare.
            # - disp: displacement from first ee_ref_frames (motion intent, offset-invariant)
            # - pos : absolute base-relative positions
            # Optional normalization can be configured via args.ee_norm_mode.
            src_sig = src_ee
            dst_sig = dst_ee
            if mode == 'disp':
                # Displacement-based EE target:
                # match motion intent while ignoring static cross-morphology offsets
                # (e.g. human elbows above pelvis vs dog paws below base at rest).
                ref_frames = int(getattr(self.args, 'ee_ref_frames', 10))
                ref_frames = max(1, min(ref_frames, src_ee.shape[1], dst_ee.shape[1]))
                src_ref = src_ee[:, :ref_frames].mean(dim=1, keepdim=True)  # [B,1,K,3]
                dst_ref = dst_ee[:, :ref_frames].mean(dim=1, keepdim=True)  # [B,1,K,3]
                src_sig = src_ee - src_ref
                dst_sig = dst_ee - dst_ref

            norm_mode = str(getattr(self.args, 'ee_norm_mode', 'per_domain')).lower()
            if src_scale is None:
                src_scale = torch.norm(src_sig, dim=-1).mean().detach().clamp_min(eps)
            else:
                src_scale = torch.tensor(float(src_scale), device=src_ee.device).clamp_min(eps)
            if dst_scale is None:
                dst_scale = torch.norm(dst_sig, dim=-1).mean().detach().clamp_min(eps)
            else:
                dst_scale = torch.tensor(float(dst_scale), device=dst_ee.device).clamp_min(eps)

            if norm_mode == 'none':
                pass
            elif norm_mode == 'per_domain':
                src_sig = src_sig / src_scale
                dst_sig = dst_sig / dst_scale
            elif norm_mode == 'src_only':
                src_sig = src_sig / src_scale
                dst_sig = dst_sig / src_scale
            elif norm_mode == 'shared':
                shared_scale = (src_scale + dst_scale) * 0.5
                src_sig = src_sig / shared_scale
                dst_sig = dst_sig / shared_scale
            else:
                raise ValueError(f"Unsupported ee_norm_mode: {norm_mode}")

            axis_norm = bool(getattr(self.args, 'ee_axis_norm', True))
            if axis_norm:
                # Per-axis normalization prevents one dominant axis (often Z) from
                # overshadowing X/Y during optimization.
                axis_scale = torch.mean(torch.abs(src_sig), dim=(0, 1, 2), keepdim=True).detach().clamp_min(1e-2)
                src_sig = src_sig / axis_scale
                dst_sig = dst_sig / axis_scale
            per_sample = torch.mean((dst_sig - src_sig) ** 2, dim=(1, 2, 3))
            if sample_weight is None:
                return per_sample.mean()
            w = sample_weight.reshape(-1).to(per_sample.device).float()
            w = torch.clamp(w, min=0.0)
            denom = torch.clamp(w.sum(), min=1e-8)
            return (per_sample * w).sum() / denom

        if mode == 'direction':
            src_dir = src_ee / (torch.norm(src_ee, dim=-1, keepdim=True) + eps)
            dst_dir = dst_ee / (torch.norm(dst_ee, dim=-1, keepdim=True) + eps)
            per_sample = torch.mean((dst_dir - src_dir) ** 2, dim=(1, 2, 3))
            if sample_weight is None:
                return per_sample.mean()
            w = sample_weight.reshape(-1).to(per_sample.device).float()
            w = torch.clamp(w, min=0.0)
            denom = torch.clamp(w.sum(), min=1e-8)
            return (per_sample * w).sum() / denom

        raise ValueError(f"Unsupported ee_match_mode: {mode}")

    def backward_G(self):
        # rec_loss and gan loss
        self.rec_losses = []
        self.rec_loss = 0
        self.cycle_loss = 0  # weighted sum kept for legacy logging
        self.cycle_fk_loss = torch.tensor(0.0, device=self.device)
        self.cycle_latent_loss = torch.tensor(0.0, device=self.device)
        self.cycle_motion_loss = torch.tensor(0.0, device=self.device)
        self.loss_G = 0
        self.retar_loss = 0
        self.retar_ang_loss = torch.tensor(0.0, device=self.device)
        self.ee_loss = torch.tensor(0.0, device=self.device)
        self.kl_loss = 0
        self.loss_G_total = 0
        
        # reconstruction loss
        for i in range(self.n_topology):
            input_0, input_1 = get_recloss_input(self, self.args.rec_loss_type, i)
            indices = self.models[i].indices
            indices_withend = self.models[i].indices_withend
            joint_indices = self.models[i].joint_indices
            joint_indices_withend = self.models[i].joint_indices_withend
            # rec_loss1: reconstruct all joint angles (full pose, not filtered by correspondence)
            rec_loss1 = self.criterion_rec(input_0,
                                           input_1,
                                           self.datasets[i].njoints,
                                           indices=joint_indices)

            self.loss_recoder.add_scalar('rec_loss_quater_{}'.format(i), rec_loss1)

            njoints = self.datasets[i].njoints

            # rec_loss_ang: explicit root angular-rate reconstruction loss.
            # Legacy datasets have [yaw_rate]; rpy datasets have [roll_rate, pitch_rate, yaw_rate].
            motion_denorm = self.motion_denorm[i]
            rec_denorm    = self.rec_denorm[i]
            ang_input = motion_denorm[:, :, njoints+3:]
            ang_rec   = rec_denorm[:,   :, njoints+3:]
            rec_loss_yaw = self.mse(ang_input, ang_rec)
            self.loss_recoder.add_scalar('rec_loss_ang_{}'.format(i), rec_loss_yaw)

            # rec_loss2: local velocity reconstruction in normalized feature space
            # self.motion[i]: [B, C, T], self.rec[i]: [B, T, C]
            input_pos = self.motion[i].transpose(1, 2)[:, :, njoints:njoints+3]  # [B, T, 3]
            rec_pos   = self.rec[i][:, :, njoints:njoints+3]                      # [B, T, 3]
            rec_loss2 = self.criterion_root(input_pos, rec_pos)
            self.loss_recoder.add_scalar('rec_loss_global_{}'.format(i), rec_loss2)

            # rec_loss3: reconstruct kinematic positions
            rec_loss3 = self.criterion_kine(self.gt_pos[i], self.rec_pos[i], indices=indices_withend)
            self.loss_recoder.add_scalar('rec_loss_position_{}'.format(i), rec_loss3)

            rec_loss = rec_loss1 + rec_loss2 * 100 + rec_loss3 * 1e-2 + rec_loss_yaw * 25

            self.rec_losses.append(rec_loss)
            self.rec_loss += rec_loss

        # KL divergence loss (VAE only)
        lambda_kl = getattr(self.args, 'lambda_kl', 1e-3)
        for i in range(self.n_topology):
            mu      = self.mus[i]
            log_var = self.log_vars[i]
            if mu is not None and log_var is not None:
                kl_i = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
                self.kl_loss += kl_i
                self.loss_recoder.add_scalar('kl_loss_{}'.format(i), kl_i)

        p = 0
        for src in range(self.n_topology):

            indices = self.models[src].indices
            joint_indices = self.models[src].joint_indices
            for dst in range(self.n_topology):
                src_joints = self.datasets[src].njoints
                dst_joints = self.datasets[dst].njoints
                if dst == src:
                    continue
                else:
                    # Cycle consistency terms are accumulated separately so latent
                    # alignment cannot silently dominate FK or motion-cycle quality.
                    cycle_fk_loss = self.criterion_cycle(self, src, p, indice=indices)
                    cycle_latent_loss = self.criterion_cycle_latent(self, src, p)
                    self.loss_recoder.add_scalar('cycle_fk_loss_{}_{}'.format(src, dst), cycle_fk_loss)
                    self.loss_recoder.add_scalar('cycle_latent_loss_{}_{}'.format(src, dst), cycle_latent_loss)
                    self.cycle_fk_loss = self.cycle_fk_loss + cycle_fk_loss
                    self.cycle_latent_loss = self.cycle_latent_loss + cycle_latent_loss

                    # Cycle consistency loss for denormalized motion features.
                    njoints_src = self.datasets[src].njoints
                    motion_denorm_src = self.motion_denorm[src]
                    cyc_denorm_curr   = self.cyc_denorm[p]

                    # cycle_motion_loss1: joint angles (full pose)
                    cyc_joint_angles = cyc_denorm_curr[:, :, :njoints_src]
                    src_joint_angles = motion_denorm_src[:, :, :njoints_src]
                    cycle_motion_loss1 = self.criterion_rec(src_joint_angles, cyc_joint_angles, njoints_src, indices=joint_indices)
                    self.loss_recoder.add_scalar('cycle_motion_loss1_{}_{}'.format(src, dst), cycle_motion_loss1)

                    # cycle_motion_loss2: local velocity
                    cyc_vel = cyc_denorm_curr[:, :, njoints_src:njoints_src+3]
                    src_vel = motion_denorm_src[:, :, njoints_src:njoints_src+3]
                    cycle_motion_loss2 = self.criterion_root(src_vel, cyc_vel)
                    self.loss_recoder.add_scalar('cycle_motion_loss2_{}_{}'.format(src, dst), cycle_motion_loss2)

                    # cycle_motion_loss3: root angular rates
                    cyc_yaw = cyc_denorm_curr[:, :, njoints_src+3:]
                    src_yaw = motion_denorm_src[:, :, njoints_src+3:]
                    cycle_yaw_loss = self.mse(src_yaw, cyc_yaw)
                    self.loss_recoder.add_scalar('cycle_ang_loss_{}_{}'.format(src, dst), cycle_yaw_loss)

                    # Keep the historical motion-cycle definition as joint angles only.
                    # Velocity/angular cycle terms are logged above for ablations.
                    cycle_motion_loss = cycle_motion_loss1
                    self.cycle_motion_loss = self.cycle_motion_loss + cycle_motion_loss

                    # --- Retargeted root angular-rate transfer losses. ---
                    # These are intentionally outside lambda_retar_vel. Yaw is always
                    # the last angular channel; rpy mode also exposes roll/pitch.
                    motion_denorm_src     = self.motion_denorm[src]
                    fake_retar_denorm_dst = self.fake_retar_denorm[p]
                    src_ang = motion_denorm_src[:, :, src_joints+3:]
                    dst_ang = fake_retar_denorm_dst[:, :, dst_joints+3:]

                    def _loss_weight(name, default=0.0):
                        return float(getattr(self.args, name, default))

                    roll_loss = torch.tensor(0.0, device=self.device)
                    pitch_loss = torch.tensor(0.0, device=self.device)
                    if src_ang.shape[-1] >= 3 and dst_ang.shape[-1] >= 3:
                        roll_loss = self.mse(src_ang[..., 0:1], dst_ang[..., 0:1])
                        pitch_loss = self.mse(src_ang[..., 1:2], dst_ang[..., 1:2])
                        self.retar_ang_loss = self.retar_ang_loss + roll_loss * _loss_weight('lambda_retar_roll_rate', 0.0)
                        self.retar_ang_loss = self.retar_ang_loss + pitch_loss * _loss_weight('lambda_retar_pitch_rate', 0.0)
                    yaw_rate_src = src_ang[..., -1:]
                    yaw_rate_retar = dst_ang[..., -1:]
                    retar_yaw_loss = self.mse(yaw_rate_src, yaw_rate_retar)
                    yaw_weight = _loss_weight('lambda_retar_yaw_rate', -1.0)
                    if yaw_weight < 0.0:
                        yaw_weight = float(getattr(self.args, 'lambda_retar_vel', 0.0)) * 1e-1
                    self.retar_ang_loss = self.retar_ang_loss + retar_yaw_loss * yaw_weight
                    self.loss_recoder.add_scalar('retar_roll_rate_loss_{}_{}'.format(src, dst), roll_loss)
                    self.loss_recoder.add_scalar('retar_pitch_rate_loss_{}_{}'.format(src, dst), pitch_loss)
                    self.loss_recoder.add_scalar('retar_yaw_rate_loss_{}_{}'.format(src, dst), retar_yaw_loss)
                    
                    dst_joint_angles = fake_retar_denorm_dst[:, :, :dst_joints]  # [B, T, num_joints]

                    if hasattr(self.models[dst], 'joint_limits') and self.models[dst].joint_limits is not None:
                        # Use config threshold and loss weight
                        threshold = getattr(self.args, 'joint_limit_threshold', 0.85)
                        retar_joint_limit_loss = self.compute_joint_limit_loss(
                            dst_joint_angles, 
                            self.models[dst].joint_limits,
                            threshold=threshold
                        )
                        self.loss_recoder.add_scalar('retar_joint_limit_loss_{}_{}'.format(src, dst), retar_joint_limit_loss)
                    else:
                        retar_joint_limit_loss = 0.0
                    
                    joint_limit_weight = getattr(self.args, 'joint_limit_loss_weight', 1e-1)
                    self.retar_loss += retar_joint_limit_loss * joint_limit_weight

                    # retargeted root velocity loss
                    src_vector = self.motion_denorm[src][..., src_joints : src_joints + 3]
                    retar_vector = self.fake_retar_denorm[p][..., dst_joints : dst_joints + 3]

                    eps = 1e-8
                    src_max_vel = self._vel_max_t[src]
                    dst_max_vel = self._vel_max_t[dst]

                    src_speed = torch.norm(src_vector, dim=-1, p=2, keepdim=True).clamp_min(eps)
                    dst_speed = torch.norm(retar_vector, dim=-1, p=2, keepdim=True).clamp_min(eps)
                    src_speed_xy = torch.norm(src_vector[..., :2], dim=-1, p=2, keepdim=True).clamp_min(eps)
                    dst_speed_xy = torch.norm(retar_vector[..., :2], dim=-1, p=2, keepdim=True).clamp_min(eps)

                    deadzone = float(getattr(self.args, 'retar_vel_deadzone', 0.05))
                    deadzone_t = torch.tensor(deadzone, device=src_vector.device, dtype=src_vector.dtype)
                    src_vmax_t = src_max_vel.to(device=src_vector.device, dtype=src_vector.dtype).clamp_min(deadzone + eps)
                    dst_vmax_t = dst_max_vel.to(device=retar_vector.device, dtype=retar_vector.dtype).clamp_min(deadzone + eps)

                    # Mapping mode: remove tiny standstill jitter (deadzone), then
                    # normalize by per-topology vmax and clamp to [0, 1].
                    input_vel_scalar = torch.clamp(
                        torch.relu(src_speed_xy - deadzone_t) / (src_vmax_t - deadzone_t).clamp_min(eps),
                        min=0.0,
                        max=1.0,
                    )
                    retar_vel_scalar = torch.clamp(
                        torch.relu(dst_speed_xy - deadzone_t) / (dst_vmax_t - deadzone_t).clamp_min(eps),
                        min=0.0,
                        max=1.0,
                    )

                    input_vel_xy = input_vel_scalar * src_vector[..., :2] / src_speed_xy
                    retar_vel_xy = retar_vel_scalar * retar_vector[..., :2] / dst_speed_xy
                    map_z = bool(getattr(self.args, 'retar_vel_map_z', True))
                    if map_z:
                        # Scale z with the same mapping scalar by preserving z direction sign.
                        input_vel_z = input_vel_scalar * src_vector[..., 2:3] / src_speed
                        retar_vel_z = retar_vel_scalar * retar_vector[..., 2:3] / dst_speed
                    else:
                        # Keep z unscaled (direct) while mapping only x/y.
                        input_vel_z = src_vector[..., 2:3]
                        retar_vel_z = retar_vector[..., 2:3]

                    input_vel = torch.cat([input_vel_xy, input_vel_z], dim=-1)
                    retar_vel = torch.cat([retar_vel_xy, retar_vel_z], dim=-1)

                    if self.args.retar_vel_matching == 'mapping':
                        retar_root_v_loss = self.criterion_root_v(input_vel, retar_vel)
                    elif self.args.retar_vel_matching == 'direct':
                        retar_root_v_loss = self.criterion_root_v(
                            self.motion_denorm[src][..., src_joints:src_joints + 3],
                            self.fake_retar_denorm[p][..., dst_joints:dst_joints + 3],
                        )
                    elif self.args.retar_vel_matching == 'direction':
                        retar_root_v_loss = self.criterion_root_v(src_vector / src_speed,
                                                                  retar_vector / dst_speed)

                    # retar_root_v_loss += self.mse(self.motion_denorm[src][..., src_joints + 7: src_joints + 8],
                    #                          self.fake_retar_denorm[p][..., dst_joints + 7: dst_joints + 8])
                    self.loss_recoder.add_scalar('retar_root_v_loss_{}_{}'.format(src, dst), retar_root_v_loss)
                    self.retar_loss += retar_root_v_loss

                    # End-effector matching loss (base-relative FK space)
                    lambda_ee = getattr(self.args, 'lambda_ee', 0.0)
                    if lambda_ee > 0.0:
                        src_name = self._canonical_topology_key(self.topology_name[src])
                        dst_name = self._canonical_topology_key(self.topology_name[dst])
                        src_ee_idx = self._ee_indices.get(src_name, [])
                        dst_ee_idx = self._ee_indices.get(dst_name, [])
                        ee_mode = getattr(self.args, 'ee_match_mode', 'position')

                        src_ee_pts, dst_ee_pts = self._compute_paired_ee_points(
                            src, dst,
                            self.motion_denorm[src],
                            self.fake_retar_denorm[p],
                            src_ee_idx=src_ee_idx,
                            dst_ee_idx=dst_ee_idx,
                            src_local_cache=self.gt_local_pos[src],
                            dst_local_cache=self.fake_local_pos[p],
                        )
                        gate = self.ee_sample_weight_gates[src]
                        w_manip = float(getattr(self.args, 'ee_weight_manip', 1.0))
                        w_loco = float(getattr(self.args, 'ee_weight_locomotion', 0.0))
                        ee_weight = w_loco + (w_manip - w_loco) * gate
                        ee_i = self.compute_ee_match_loss(
                            src_ee_pts,
                            dst_ee_pts,
                            mode=ee_mode,
                            sample_weight=ee_weight,
                        )
                        self.ee_loss = self.ee_loss + ee_i
                        self.loss_recoder.add_scalar(f'ee_loss_{src_name}2{dst_name}', ee_i)
                        self.loss_recoder.add_scalar(f'ee_weight_mean_{src_name}2{dst_name}', ee_weight.mean())

                    if self.args.dis:
                        # Ensure data dimensions match the destination topology discriminator
                        if self.args.dis_mode == 'denorm_pos':
                            # Use all body positions for discriminator input
                            fake_input = self.fake_pos[p][:, :, :self.datasets[dst].nbodies, :].reshape(
                                self.fake_pos[p].shape[0], self.fake_pos[p].shape[1], -1).transpose(1, 2)
                        elif self.args.dis_mode == 'denorm_rotation':
                            fake_input = self.fake_retar_denorm[p].transpose(1, 2)
                        elif self.args.dis_mode == 'latent':
                            fake_input = self.retar_latents[p]
                        else:
                            raise ValueError(f"Unsupported dis_mode: {self.args.dis_mode}")
                        
                        loss_G = self.criterion_gan(self.models[dst].discriminator(fake_input), True)
                    else:
                        loss_G = torch.tensor(0)
                    self.loss_recoder.add_scalar('G_loss_{}_{}'.format(src, dst), loss_G)
                    self.loss_G += loss_G

                p += 1

        # ── Physics losses (mapping-free, source-phase gated) ───────────────
        # For each retargeting direction (src→dst):
        #   1) Estimate source contact timing from source feet (height-only)
        #   2) Estimate destination contact from destination feet (height-only)
        #   3) Apply source temporal gate × destination per-foot contact
        #      to destination skating and grounding losses (no explicit mapping)
        self.physics_loss = torch.tensor(0.0, device=self.device)

        lambda_skating = getattr(self.args, 'lambda_skating', 0.0)
        lambda_grounding = self.lambda_grounding

        if self.use_physics and len(self.fake_pos) > 0:
            dt = 1.0 / 30.0
            ground_margin = getattr(self.args, 'ground_margin', 0.05)
            if isinstance(self.args, dict):
                contact_ref_frames = max(1, int(self.args.get('physics_ref_frames', 10)))
            else:
                try:
                    contact_ref_frames = max(1, int(getattr(self.args, 'physics_ref_frames')))
                except (AttributeError, KeyError):
                    contact_ref_frames = 10
            physics_ground_mode = self._physics_ground_mode

            p = 0
            for src in range(self.n_topology):
                for dst in range(self.n_topology):
                    if dst == src:
                        continue

                    src_name = self._canonical_topology_key(self.topology_name[src])
                    dst_name = self._canonical_topology_key(self.topology_name[dst])
                    src_pos = self.gt_pos[src]    # [B, T, n_src_bodies, 3]
                    fake_pos_p = self.fake_pos[p]  # [B, T, n_dst_bodies, 3]
                    src_foot_idx = self._foot_indices.get(src_name, [])
                    dst_foot_idx = self._foot_indices.get(dst_name, [])

                    if len(dst_foot_idx) == 0:
                        p += 1
                        continue

                    # Keep contact/ground estimates non-differentiable to prevent
                    # the model from shaping the gates instead of fixing motion.
                    with torch.no_grad():
                        # Align destination FK global z to a nominal world frame where
                        # base z at t=0 matches configured start height.
                        dst_start_height = float(self._nominal_start_height.get(dst_name, 0.0))
                        dst_base_z0 = fake_pos_p.detach()[:, :1, 0:1, 2]
                        dst_z_shift = dst_start_height - dst_base_z0
                        fake_pos_p_phys = fake_pos_p.detach().clone()
                        fake_pos_p_phys[..., 2] = fake_pos_p_phys[..., 2] + dst_z_shift

                        if physics_ground_mode == 'zero_nominal':
                            dst_ground_mode = 'zero'
                            dst_fixed_ground_z = 0.0
                        else:
                            dst_ground_mode = 'first_frames'
                            dst_fixed_ground_z = 0.0

                        dst_contact, dst_ground_z = estimate_contact_from_height(
                            fake_pos_p_phys,
                            dst_foot_idx,
                            ground_margin=ground_margin,
                            ground_mode=dst_ground_mode,
                            fixed_ground_z=dst_fixed_ground_z,
                            ref_frames=contact_ref_frames,
                            smooth_steps=1,
                        )

                        if len(src_foot_idx) > 0:
                            src_pos_phys = src_pos.detach()
                            if physics_ground_mode == 'zero_nominal':
                                src_start_height = float(self._nominal_start_height.get(src_name, 0.0))
                                src_base_z0 = src_pos_phys[:, :1, 0:1, 2]
                                src_z_shift = src_start_height - src_base_z0
                                src_pos_phys = src_pos_phys.clone()
                                src_pos_phys[..., 2] = src_pos_phys[..., 2] + src_z_shift
                                src_ground_mode = 'zero'
                            else:
                                src_ground_mode = 'first_frames'

                            src_contact, _ = estimate_contact_from_height(
                                src_pos_phys,
                                src_foot_idx,
                                ground_margin=ground_margin,
                                ground_mode=src_ground_mode,
                                fixed_ground_z=0.0,
                                ref_frames=contact_ref_frames,
                                smooth_steps=1,
                            )
                            src_time_gate = torch.max(src_contact, dim=-1, keepdim=True).values
                        else:
                            src_time_gate = torch.ones(
                                dst_contact.shape[0], dst_contact.shape[1], 1,
                                device=dst_contact.device, dtype=dst_contact.dtype
                            )

                        gated_contact_skating = dst_contact * src_time_gate

                        # source-gated only for grounding (broadcast over feet)
                        grounding_gate = src_time_gate.expand(-1, -1, dst_contact.shape[-1])

                    self.loss_recoder.add_scalar(f'src_contact_mean_{src_name}2{dst_name}', src_time_gate.mean())
                    self.loss_recoder.add_scalar(f'dst_contact_mean_{src_name}2{dst_name}', dst_contact.mean())
                    self.loss_recoder.add_scalar(f'gated_contact_mean_{src_name}2{dst_name}', gated_contact_skating.mean())
                    self.loss_recoder.add_scalar(f'dst_z_shift_mean_{src_name}2{dst_name}', dst_z_shift.mean())

                    if lambda_skating > 0:
                        sk = skating_loss_from_contact(
                            fake_pos_p_phys,
                            gated_contact_skating,
                            foot_indices=dst_foot_idx,
                            dt=dt,
                            horizontal_only=True,
                        )
                        self.physics_loss = self.physics_loss + sk * lambda_skating
                        self.loss_recoder.add_scalar(f'skating_loss_{src_name}2{dst_name}', sk)

                    if lambda_grounding > 0:
                        gp = grounding_loss_from_contact(
                            fake_pos_p_phys,
                            grounding_gate,
                            foot_indices=dst_foot_idx,
                            ground_z=dst_ground_z,
                            target_clearance=0.0,
                        )
                        self.physics_loss = self.physics_loss + gp * lambda_grounding
                        self.loss_recoder.add_scalar(f'grounding_loss_{src_name}2{dst_name}', gp)

                    p += 1

            self.loss_recoder.add_scalar('physics_loss_total', self.physics_loss)

        def _arg_float(name, default):
            if isinstance(self.args, dict):
                return float(self.args.get(name, default))
            return float(getattr(self.args, name, default))

        legacy_cycle_weight = _arg_float('lambda_cycle', 1e-3)
        lambda_cycle_fk = _arg_float('lambda_cycle_fk', -1.0)
        lambda_cycle_latent = _arg_float('lambda_cycle_latent', -1.0)
        lambda_cycle_motion = _arg_float('lambda_cycle_motion', 0.0)
        if lambda_cycle_fk < 0.0:
            lambda_cycle_fk = legacy_cycle_weight
        if lambda_cycle_latent < 0.0:
            lambda_cycle_latent = legacy_cycle_weight

        self.cycle_loss = (
            self.cycle_fk_loss * lambda_cycle_fk
            + self.cycle_latent_loss * lambda_cycle_latent
            + self.cycle_motion_loss * lambda_cycle_motion
        )

        self.loss_G_total = self.rec_loss   * self.args.lambda_rec       + \
                            self.cycle_loss                              + \
                            self.loss_G     * 1                          + \
                            self.retar_loss * self.args.lambda_retar_vel + \
                            self.retar_ang_loss + \
                            self.ee_loss    * getattr(self.args, 'lambda_ee', 0.0) + \
                            self.kl_loss    * getattr(self.args, 'lambda_kl', 1e-3) + \
                            self.physics_loss

        self.loss_recoder.add_scalar('kl_loss_total', self.kl_loss)
        self.loss_recoder.add_scalar('retargeting_ang_loss_total', self.retar_ang_loss)
        self.loss_recoder.add_scalar('ee_loss_total', self.ee_loss)
        self.loss_recoder.add_scalar('G_loss_total',  self.loss_G_total)
        self.loss_G_total.backward()

    def optimize_parameters(self):
        self.forward()

        # update Gs
        self.discriminator_requires_grad_(False)
        self.optimizerG.zero_grad(set_to_none=True)
        self.backward_G()
        self.optimizerG.step()

        # update Ds
        if self.args.dis:
            self.discriminator_requires_grad_(True)
            self.optimizerD.zero_grad(set_to_none=True)
            self.backward_D()
            self.optimizerD.step()
        else:
            self.loss_D = torch.tensor(0)
        
        # Log metrics to WandB
        self.log_to_wandb()

    def verbose(self):
        kl_val = self.kl_loss.item() if torch.is_tensor(self.kl_loss) else self.kl_loss
        res = {'rec_loss_0': self.rec_losses[0].item(),
               'rec_loss_1': self.rec_losses[1].item(),
               'cycle_loss': self.cycle_loss.item(),
               'cycle_fk_loss': self.cycle_fk_loss.item(),
               'cycle_latent_loss': self.cycle_latent_loss.item(),
               'cycle_motion_loss': self.cycle_motion_loss.item(),
               'kl_loss':    kl_val,
               'ee_loss':    self.ee_loss.item(),
               'D_loss_gan': self.loss_D.item(),
               'G_loss_gan': self.loss_G.item()}
        return sorted(res.items(), key=lambda x: x[0])
    
    def log_to_wandb(self):
        """Log all important metrics to WandB"""
        if not self.use_wandb or not self.wandb_api:
            return
        
        # Aggregate losses
        metrics = {
            'loss/rec_loss_total': self.rec_loss.item(),
            'loss/rec_loss_topology_0': self.rec_losses[0].item(),
            'loss/rec_loss_topology_1': self.rec_losses[1].item(),
            'loss/cycle_loss': self.cycle_loss.item(),
            'loss/cycle_fk_raw': self.cycle_fk_loss.item(),
            'loss/cycle_latent_raw': self.cycle_latent_loss.item(),
            'loss/cycle_motion_raw': self.cycle_motion_loss.item(),
            'loss/discriminator': self.loss_D.item(),
            'loss/generator': self.loss_G.item(),
            'loss/generator_total': self.loss_G_total.item(),
            'loss/retargeting_velocity': self.retar_loss.item(),
            'loss/retargeting_angular': self.retar_ang_loss.item(),
            'loss/end_effector': self.ee_loss.item(),
        }
        
        self.wandb_api.log(metrics)

    def save(self):
        for i, model in enumerate(self.models):
            model.save(os.path.join(self.model_save_dir, self.topology_name[i]), self.epoch_cnt)

        for i, optimizer in enumerate(self.optimizers):
            file_name = os.path.join(self.model_save_dir, 'optimizers/{}/{}.pt'.format(self.epoch_cnt, i))
            try_mkdir(os.path.split(file_name)[0])
            torch.save(optimizer.state_dict(), file_name)

    def load(self, epoch=None):
        # Resolve epoch here so `self.epoch_cnt` always matches the actual
        # checkpoint used (important for IK-prior decay during eval/distill).
        resolved_epoch = epoch
        if resolved_epoch is None and len(self.topology_name) > 0:
            probe_path = os.path.join(self.model_save_dir, self.topology_name[0])
            if os.path.exists(probe_path):
                all_epochs = [int(q) for q in os.listdir(probe_path)
                              if os.path.isdir(os.path.join(probe_path, q)) and str(q).isdigit()]
                if len(all_epochs) > 0:
                    resolved_epoch = sorted(all_epochs)[-1]

        for i, model in enumerate(self.models):
            model.load(os.path.join(self.model_save_dir, self.topology_name[i]), resolved_epoch)

        if self.is_train and not self.args.with_end:

            for i, optimizer in enumerate(self.optimizers):
                file_name = os.path.join(self.model_save_dir, 'optimizers/{}/{}.pt'.format(resolved_epoch, i))
                optimizer.load_state_dict(torch.load(file_name))
        self.epoch_cnt = 0 if resolved_epoch is None else resolved_epoch

    def compute_test_result(self):
        mse = torch.nn.MSELoss()
        rec_err = []
        for i in range(self.n_topology):
            gt_pos = self.gt_pos[i]
            rec_pos = self.rec_pos[i]
            rec_err.append(self.criterion_kine(gt_pos, rec_pos, indices=self.models[i].indices_withend))
        cyc_err = []
        p = 0
        for i in range(self.n_topology):
            gt_pos = self.gt_pos[i]
            mean_err = []
            for j in range(self.n_topology):
                if j == i:
                    continue
                cyc_pos = self.cyc_pos[p]
                mean_err.append(mse(gt_pos[..., self.models[i].indices_withend, :],
                                    cyc_pos[..., self.models[i].indices_withend, :]))

                p += 1
            cyc_err.append(torch.mean(torch.Tensor(mean_err)))

        rec_err = torch.Tensor(rec_err)
        cyc_err = torch.Tensor(cyc_err)

        return rec_err, cyc_err, self.fake_pos[0]
