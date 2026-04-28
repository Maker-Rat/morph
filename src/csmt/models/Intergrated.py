import torch
from csmt.models.networks import MotionAE, LatentDiscriminator, SkeletonEncoder
import os
from csmt.utils.differentiable_fk import ForwardKinematics

class IntegratedModel:
    def __init__(self, args, body_parts, joint_parts, njoints, nbodies, parents, topology, device, **kwargs):
        self.args = args
        self.body_parts = body_parts
        self.joint_parts = joint_parts
        self.part_num = len(self.body_parts)
        self.njoints = njoints
        self.nbodies = nbodies
        self.topology = topology
        self.device = device
        self.indices = []
        self.joint_indices = []
        top_key = str(topology).lower()
        if top_key in ("src", "human"):
            self.foot_indices = getattr(args, "src_end", getattr(args, "hum_end"))
        elif top_key in ("dst", "dog"):
            self.foot_indices = getattr(args, "dst_end", getattr(args, "dog_end"))
        else:
            self.foot_indices = []

        for part in body_parts:
            self.indices += part
            self.indices_withend = self.indices

        for part in joint_parts:
            self.joint_indices += part
            self.joint_indices_withend = self.joint_indices

        top_key = str(topology).lower()
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
        if top_key in ("src", "human"):
            fk_path = getattr(args, "src_fk_path", None) or os.path.join(repo_root, "assets", "fk", "g1_fk.xml")
            xml_path = getattr(args, "src_xml_path", None) or os.path.join(repo_root, "assets", "robots", "unitree_g1", "g1_mocap_29dof.xml")
        elif top_key in ("dst", "dog"):
            fk_path = getattr(args, "dst_fk_path", None) or os.path.join(repo_root, "assets", "fk", "go2_fk.xml")
            xml_path = getattr(args, "dst_xml_path", None) or os.path.join(repo_root, "assets", "robots", "unitree_go2", "go2.xml")
        else:
            raise ValueError(f"Unsupported topology key: {topology}")

        self.fk = ForwardKinematics(
            model_path=fk_path,
            robot_name=str(topology),
            device=device,
            parents=parents,
        )

        # Prefer pre-resolved YAML limits passed via args; fallback to XML parsing.
        if top_key in ("src", "human"):
            jl_lower = getattr(args, "src_joint_limits_lower", getattr(args, "hum_joint_limits_lower", None))
            jl_upper = getattr(args, "src_joint_limits_upper", getattr(args, "hum_joint_limits_upper", None))
            limit_label = "SRC"
        else:
            jl_lower = getattr(args, "dst_joint_limits_lower", getattr(args, "dog_joint_limits_lower", None))
            jl_upper = getattr(args, "dst_joint_limits_upper", getattr(args, "dog_joint_limits_upper", None))
            limit_label = "DST"

        if (
            isinstance(jl_lower, (list, tuple)) and isinstance(jl_upper, (list, tuple))
            and len(jl_lower) == njoints and len(jl_upper) == njoints
        ):
            self.joint_limits = torch.stack([
                torch.tensor(jl_lower, dtype=torch.float32),
                torch.tensor(jl_upper, dtype=torch.float32),
            ], dim=1).to(device)
            print(f"[{limit_label}] Loaded joint limits for {njoints} joints from robot YAML")
        else:
            self.joint_limits = ForwardKinematics.parse_joint_limits_from_mjcf(xml_path, njoints).to(device)
            print(f"[{limit_label}] Loaded joint limits for {njoints} joints from XML fallback: {xml_path}")

        self.ae = MotionAE(args, self.joint_parts, njoints).to(device)
        self.skel_enc = SkeletonEncoder(args, self.body_parts, nbodies).to(device)

        if self.args.dis:
            if self.args.dis_mode == 'norm_rotation':
                input_dim = self.args.conv_input * self.nbodies + 3
                hidden_dim = self.args.dis_hidden
                self.discriminator = \
                    LatentDiscriminator(args.dis_layers, args.dis_kernel_size,
                                        input_dim, hidden_dim).to(device)
            elif self.args.dis_mode == 'denorm_rotation':
                # denorm_rotation: [njoints + 3 (lin_vel_local) + 1 (yaw_rate)]
                input_dim = njoints + 4
                print("Discriminator input dim (denorm_rotation):", input_dim, topology)
                hidden_dim = self.args.dis_hidden
                self.discriminator = \
                    LatentDiscriminator(args.dis_layers, args.dis_kernel_size,
                                        input_dim, hidden_dim).to(device)
            elif self.args.dis_mode == 'denorm_pos':
                input_dim = 3 * self.nbodies
                print("Discriminator input dim (denorm_pos):", input_dim, topology)
                hidden_dim = self.args.dis_hidden
                self.discriminator = \
                    LatentDiscriminator(args.dis_layers, args.dis_kernel_size,
                                        input_dim, hidden_dim).to(device)


    def parameters(self):
        return self.G_parameters() + self.D_parameters()

    def G_parameters(self):
        parameters = list(self.ae.parameters()) + list(self.skel_enc.parameters())
        return parameters

    def D_parameters(self):
        return list(self.discriminator.parameters())

    def save(self, path, epoch):
        from csmt.parser.base import try_mkdir

        path = os.path.join(path, str(epoch))
        try_mkdir(path)

        torch.save(self.ae.state_dict(), os.path.join(path, 'ae.pth'))
        torch.save(self.skel_enc.state_dict(), os.path.join(path, 'skel_enc.pth'))

        if self.args.dis:
            torch.save(self.discriminator.state_dict(), os.path.join(path, 'discriminator.pth'))

        print('Save at {} succeed!'.format(path))

    def load(self, path, epoch=None):
        print('loading from', path)
        if not os.path.exists(path):
            raise Exception('Unknown loading path')

        if epoch is None:
            all = [int(q) for q in os.listdir(path) if os.path.isdir(os.path.join(path, q))]
            if len(all) == 0:
                raise Exception('Empty loading path')
            epoch = sorted(all)[-1]

        path = os.path.join(path, str(epoch))
        print('loading from epoch {}......'.format(epoch))

        # Use map_location to handle CPU/CUDA compatibility
        map_location = torch.device('cpu')
        
        self.ae.load_state_dict(torch.load(os.path.join(path, 'ae.pth'), 
                                         map_location=map_location))
        self.skel_enc.load_state_dict(torch.load(os.path.join(path, 'skel_enc.pth'),
                                                map_location=map_location))

        if os.path.exists(os.path.join(path, 'discriminator.pth')):
            self.discriminator.load_state_dict(torch.load(os.path.join(path, 'discriminator.pth'),
                                                         map_location=map_location))
        print('load succeed!')

    def train(self):
        self.ae = self.ae.train()
        self.skel_enc = self.skel_enc.train()
        if self.args.dis:
            self.discriminator = self.discriminator.train()

    def eval(self):
        self.ae = self.ae.eval()
        self.skel_enc = self.skel_enc.eval()
        if self.args.dis:
            self.discriminator = self.discriminator.eval()
