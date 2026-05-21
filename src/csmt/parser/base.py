# base parser for biped-quadruped retargeting
from argparse import ArgumentParser

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def add_misc_options(parser):
    group = parser.add_argument_group('Miscellaneous options')
    group.add_argument("--save_dir", help="directory name to save models", default='./run')
    group.add_argument('--with_end', type=boolean_string, default=True, help='whether considering robot end-sites')

def add_cuda_options(parser):
    group = parser.add_argument_group('Cuda options')
    group.add_argument('--device', type=str, default='cuda:0')


def adding_cuda(parameters):
    import torch
    if parameters["cuda"] and torch.cuda.is_available():
        parameters["device"] = torch.device("cuda")
    else:
        parameters["device"] = torch.device("cpu")


def add_dataset_options(parser):
    group = parser.add_argument_group('Dataset options')
    # Canonical naming for pairwise retargeting datasets.
    group.add_argument("--srcstats_path", type=str, default=None)
    group.add_argument("--dststats_path", type=str, default=None)
    group.add_argument("--src_train_path", type=str, default=None)
    group.add_argument("--dst_train_path", type=str, default=None)
    group.add_argument("--src_test_path", type=str, default=None)
    group.add_argument("--dst_test_path", type=str, default=None)

    # Backward-compatible aliases retained so older para/config still parse.
    group.add_argument("--humstats_path", type=str, default=None)
    group.add_argument("--dogstats_path", type=str, default=None)
    group.add_argument("--hum_train_path", type=str, default=None)
    group.add_argument("--dog_train_path", type=str, default=None)
    group.add_argument("--hum_test_path", type=str, default=None)
    group.add_argument("--dog_test_path", type=str, default=None)
    group.add_argument("--time_size", type=int, default=64)


def add_losses_options(parser):
    group = parser.add_argument_group('Losses options')
    group.add_argument("--quat_loss_type", type=str, choices=["mse", "geodesic"], default="geodesic", help="Loss for quaternion supervision: mse or geodesic")
    group = parser.add_argument_group('Losses options')

    group.add_argument("--rec_loss_type", type=str,
                       choices=["mse_rec", "quat_rec", "norm_rec"],
                       default='norm_rec')
    group.add_argument("--root_loss_type", type=str, choices=["mse_root"], default='mse_root')
    group.add_argument("--global_kine_loss_type", type=str,
                       choices=["mse_kine", "l1_kine", "part_kine"], default="part_kine")
    group.add_argument("--cyc_loss_type", type=str, default="mse_cycle_motion")
    group.add_argument("--cyc_latent_loss_type", type=str, default="mse_latent")
    group.add_argument("--retar_vel_loss_type", type=str, default='linear')
    group.add_argument("--dis_loss_type", type=str, choices=["bce_gan", "l2_gan"], default='l2_gan')
    group.add_argument("--retar_vel_matching", type=str, default='direct', choices=["mapping", 'direct', 'direction'])
    group.add_argument("--retar_vel_deadzone", type=float, default=0.05,
                       help="Deadzone (m/s) used by retar_vel_matching=mapping before speed normalization.")
    group.add_argument("--retar_vel_map_z", type=boolean_string, default=True,
                       help="If False and retar_vel_matching=mapping, apply mapping-scale only to x/y; z is matched directly.")
    group.add_argument("--retar_vel_src_vmax_percentile", type=float, default=95.0,
                       help="Source XY-speed percentile used as vmax for retar_vel_matching=mapping.")
    group.add_argument("--retar_vel_dst_vmax_percentile", type=float, default=95.0,
                       help="Destination XY-speed percentile used as vmax for retar_vel_matching=mapping.")
    group.add_argument("--root_ang_features", type=str, default="yaw", choices=["yaw", "rpy"],
                       help="Motion root angular feature layout: yaw=[yaw_rate], rpy=[roll_rate,pitch_rate,yaw_rate].")
    group.add_argument("--root_ang_dim", type=int, default=1,
                       help="Number of root angular-rate channels. Usually inferred from processed stats.")
    group.add_argument("--lambda_retar_roll_rate", type=float, default=0.0,
                       help="Separate retarget roll-rate matching weight, outside lambda_retar_vel.")
    group.add_argument("--lambda_retar_pitch_rate", type=float, default=0.0,
                       help="Separate retarget pitch-rate matching weight, outside lambda_retar_vel.")
    group.add_argument("--lambda_retar_yaw_rate", type=float, default=-1.0,
                       help="Separate retarget yaw-rate matching weight. <0 preserves legacy 0.1*lambda_retar_vel behavior.")

    group.add_argument('--use_vae', action='store_true', default=False,
                       help='Enable VAE (adds mu/log_var heads and KL loss)')
    group.add_argument('--lambda_kl', type=float, default=1e-3,
                       help='Weight for KL divergence loss (VAE only)')
    
    group.add_argument('--lambda_rec', type=float, default=1)
    group.add_argument('--lambda_cycle', type=float, default=1e-3)
    group.add_argument('--lambda_cycle_motion', type=float, default=0.0,
                       help='Additional weight for cycle motion reconstruction term '
                            '(joint/vel/yaw feature-space cycle loss).')
    group.add_argument('--lambda_retar_vel', type=float, default=1e3)
    group.add_argument('--lambda_ee', type=float, default=0.0,
                       help='Weight for end-effector matching loss on retargeted motion')
    group.add_argument('--ee_weight_manip', type=float, default=1.0,
                       help='Per-sample multiplier for EE loss when ee_tag=1 (manipulation clip).')
    group.add_argument('--ee_weight_locomotion', type=float, default=0.0,
                       help='Per-sample multiplier for EE loss when ee_tag=0 (locomotion clip).')
    group.add_argument('--joint_limit_threshold', type=float, default=0.85,
                       help='Soft joint-limit margin as a fraction of each joint range '
                            '(e.g., 0.85 starts penalizing near 85% of the limit).')
    group.add_argument('--joint_limit_loss_weight', type=float, default=1e-1,
                       help='Weight for soft joint-limit regularization loss on retargeted joints.')
    group.add_argument('--ee_match_mode', type=str, default='disp',
                       choices=['disp', 'pos', 'position', 'direction'],
                       help='EE loss mode: disp (displacement target), pos (base-relative position target), '
                            'or direction only. Legacy alias: position -> disp.')
    group.add_argument('--ee_ref_frames', type=int, default=5,
                       help='Number of initial frames used as EE displacement reference')
    group.add_argument('--ee_norm_mode', type=str, default='per_domain',
                       choices=['none', 'per_domain', 'src_only', 'shared'],
                       help='Optional EE normalization mode for disp/pos targets.')
    group.add_argument('--ee_axis_norm', type=boolean_string, default=True,
                       help='If True, apply per-axis normalization on source EE signal for disp/pos targets.')

    # Physics losses — contact-gated, applied to retargeted motions only
    group.add_argument('--lambda_skating', type=float, default=0.0,
                       help='Weight for contact-gated anti-skating loss on retargeted feet '
                            '(destination self-contact, mapping-free).')
    group.add_argument('--lambda_grounding', type=float, default=0.0,
                       help='Weight for contact-gated grounding loss on retargeted feet '
                            '(keeps contacting feet near ground, handles both floating and penetration).')
    group.add_argument('--use_ground_plane_contact', type=boolean_string, default=True,
                       help='If True, estimate destination ground from first frames of each sequence; '
                            'if False, assume fixed global ground z=0.')
    group.add_argument('--ground_margin', type=float, default=0.05,
                       help='Height scale (metres) for contact confidence decay from ground.')
    group.add_argument('--physics_ref_frames', type=int, default=5,
                       help='Number of initial frames used for per-sequence ground estimation '
                            'in mapping-free physics contact detection.')
    group.add_argument('--physics_ground_mode', type=str, default='first_frames',
                       choices=['first_frames', 'zero_nominal'],
                       help='Ground reference mode for physics losses: first_frames uses per-sequence ground estimate; '
                            'zero_nominal shifts each topology by nominal start height and uses fixed ground z=0.')
    group.add_argument('--src_start_height', type=float, default=0.0,
                       help='Nominal world-frame base height for source topology used by zero_nominal physics mode.')
    group.add_argument('--dst_start_height', type=float, default=0.0,
                       help='Nominal world-frame base height for destination topology used by zero_nominal physics mode.')

def add_model_options(parser):
    group = parser.add_argument_group('Model options')
    group.add_argument("--architecture_name", type=str, default='pan')
    group.add_argument("--fid_net_name", type=str, default='FIDAutoEncoder')

    group.add_argument("--transformer", type=boolean_string, default=True)
    group.add_argument("--transformer_layers", type=int, default=1)
    group.add_argument("--transformer_latents", type=int, default=32)
    group.add_argument("--transformer_ffsize", type=int, default=256)
    group.add_argument("--transformer_heads", type=int, default=1)
    group.add_argument("--transformer_dropout", type=int, default=0)
    group.add_argument("--transformer_srcdim", type=int, default=1)

    group.add_argument("--conv_input", type=int, default=4)
    group.add_argument("--conv_layers", type=int, default=2)
    group.add_argument("--kernel_size", type=int, default=15)
    group.add_argument("--dim_per_part", type=int, default=32)
    group.add_argument("--padding_mode", type=str, default='reflect')

    group.add_argument('--upsampling', type=str, default='linear', help="'stride2' or 'nearest', 'linear'")
    group.add_argument("--skeleton_info", type=str, default="additive")
    group.add_argument('--root_latent_dim', type=int, default=64, help='latent dimension for dedicated root feature stream')

    group.add_argument("--dis", type=boolean_string, help="use_discriminator", default=True)
    group.add_argument("--diter", type=int, default=3)
    group.add_argument("--dis_mode", type=str,
                        choices=['norm_rotation', 'denorm_rotation', 'denorm_pos', 'latent'], default='denorm_rotation')
    group.add_argument("--dis_hidden", type=int, default=256)
    group.add_argument("--dis_layers", type=int, default=3)
    group.add_argument("--dis_kernel_size", type=int, default=15)


def try_mkdir(path):
    import os
    if not os.path.exists(path):
        os.system('mkdir -p {}'.format(path))


class Dict(dict):
    __setattr__ = dict.__setattr__

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc


def dict_to_object(dictObj):
    if not isinstance(dictObj, dict):
        return dictObj
    inst = Dict()
    for k, v in dictObj.items():
        inst[k] = dict_to_object(v)
    return inst
