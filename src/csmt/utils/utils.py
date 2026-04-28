import torch
import numpy as np
import torch.nn.functional as F

def get_lpos(skel, t_size, njoints, device):
    b_size = skel.shape[0]
    print("skel shape:", skel.unsqueeze(1).repeat(1, t_size, 1).shape)
    lpos = torch.cat([torch.zeros(b_size, t_size, 3).to(device),
                      skel.unsqueeze(1).repeat(1, t_size, 1)],
                     dim=-1)
    print("lpos shape before reshape:", lpos.shape)
    print("b_size:", b_size, "t_size:", t_size, "njoints:", njoints)
    lpos = lpos.reshape(b_size, t_size, njoints, -1)

    return lpos

def get_body_part(correspondence, topology_name):
    part_list = []
    for dic in correspondence:
        part_list.append(dic[topology_name])
    return part_list


def get_part_matrix(part_list, njoints):
    # Body parts + root as virtual body part
    # njoints here is njoints_only (pure joint count, no root features)
    nparts = len(part_list)
    matrix = torch.zeros(nparts, njoints)  # +2 for root features (linvel, quat)
    for i, part in enumerate(part_list):
        matrix[i, part] = 1
    # Root virtual row connects to all joint features
    # matrix[:, -1] = 1
    # matrix[:, -2] = 1
    return matrix


def get_offset_part_matrix(part_list, num_offsets):
    matrix = torch.zeros(len(part_list), num_offsets+1)
    for i, part in enumerate(part_list):
        matrix[i, part] = 1
    return matrix[:, 1:]


def get_transformer_matrix(part_list, njoints):
    """
    Token layout: [body_parts (nparts) | joints (njoints) | linvel (1) | yaw_rate (1)]
    Indices:       0 .. nparts-1        nparts..nparts+njoints-1  -2       -1
    Total tokens = nparts + njoints + 2

    Connectivity:
      - Each body part token <-> its own joint tokens (bidirectional)
      - Joint tokens <-> same-part joint tokens (bidirectional)
      - linvel token  <-> ALL other tokens (fully connected, global context)
      - yaw_rate token <-> ALL other tokens (fully connected, global context)
      - Self-attention for every token
    """
    nparts = len(part_list)
    joint_start  = nparts
    linvel_idx   = nparts + njoints
    yaw_rate_idx = nparts + njoints + 1
    n_total      = nparts + njoints + 2

    matrix = torch.zeros([n_total, n_total])

    # Body part tokens <-> their joints (bidirectional)
    for i in range(nparts):
        for joint_idx in part_list[i]:
            matrix[i, joint_start + joint_idx] = 1
            matrix[joint_start + joint_idx, i] = 1

    # Joint tokens <-> same-part joints (bidirectional, includes self via loop)
    for i in range(nparts):
        for joint_idx in part_list[i]:
            for other_joint_idx in part_list[i]:
                matrix[joint_start + joint_idx, joint_start + other_joint_idx] = 1

    # linvel token <-> ALL tokens (fully connected)
    matrix[linvel_idx, :] = 1
    matrix[:, linvel_idx] = 1

    # yaw_rate token <-> ALL tokens (fully connected)
    matrix[yaw_rate_idx, :] = 1
    matrix[:, yaw_rate_idx] = 1

    # Self-attention for all tokens
    for i in range(n_total):
        matrix[i, i] = 1

    matrix = matrix.float().masked_fill(matrix == 0., float(-1e20)).masked_fill(matrix == 1., float(0.0))
    return matrix


def smooth_yaw_quat(quat_xyzw, window=30):
    x, y, z, w = quat_xyzw.unbind(dim=-1)  # each [B, T]
    
    cos_yaw = 1.0 - 2.0*(y*y + z*z)
    sin_yaw = 2.0*(w*z + x*y)
    
    # avg_pool1d expects [B, C, T] — unsqueeze C dim
    cos_smooth = F.avg_pool1d(
        cos_yaw.unsqueeze(1), kernel_size=window, stride=1, padding=window//2
    ).squeeze(1)
    sin_smooth = F.avg_pool1d(
        sin_yaw.unsqueeze(1), kernel_size=window, stride=1, padding=window//2
    ).squeeze(1)
    
    # Trim to exact T in case padding produces T+1
    T = quat_xyzw.shape[1]
    cos_smooth = cos_smooth[:, :T]
    sin_smooth = sin_smooth[:, :T]
    
    norm = torch.sqrt(cos_smooth**2 + sin_smooth**2).clamp(min=0.3)
    cos_smooth = cos_smooth / norm
    sin_smooth = sin_smooth / norm
    
    half_cos = torch.sqrt(((1.0 + cos_smooth) / 2.0).clamp(min=0))
    half_sin = sin_smooth / (2.0 * half_cos.clamp(min=1e-8))
    
    zeros = torch.zeros_like(half_cos)
    return torch.stack([zeros, zeros, half_sin, half_cos], dim=-1)  # [B, T, 4]


def getbodyparts(edges):
    """Extract body parts from skeleton edge topology"""
    degree = [0] * 100
    
    for edge in edges:
        degree[edge[0]] += 1
        degree[edge[1]] += 1
    
    def find_chains(j, seq, edge_seq_list):
        if degree[j] > 2 and j != 0:
            edge_seq_list.append(seq)
            seq = []
        
        if degree[j] == 1:
            edge_seq_list.append(seq)
            return
        
        for idx, edge in enumerate(edges):
            if edge[0] == j:
                find_chains(edge[1], seq + [idx], edge_seq_list)
    
    edge_seq_list = []
    find_chains(0, [], edge_seq_list)
    
    # Convert edge sequences to joint sequences
    joint_seq = []
    for seq in edge_seq_list:
        joint_chain = []
        for i, edge in enumerate(seq):
            joint_chain.append(edges[edge][0])
            if i == len(seq)-1:
                joint_chain.append(edges[edge][1])
        joint_seq.append(joint_chain)
    
    return joint_seq


def calselfmask(part_list, njoints, edges=None, is_conv=False):
    part_list = part_list.copy()
    nparts = len(part_list)

    matrix = torch.zeros([njoints + nparts, njoints])
    n = 0

    if edges is not None:
        rotation_map = []
        for i, edge in enumerate(edges):
            rotation_map.append(edge[1])
        rotation_map_reverse = []
        for i in range(1, njoints):
            rotation_map_reverse.append(rotation_map.index(i))

    for part in part_list:
        if part[0] == 0:
            part.pop(0)
        for i in range(len(part)):
            if edges is not None:
                part[i] = rotation_map_reverse[part[i]-1]
            else:
                part[i] -= 1

    for part in part_list:
        matrix[n, part] = 1
        for k in part:
            matrix[k + nparts, part] = 1
        n += 1

    matrix = torch.cat((torch.zeros([njoints+nparts, nparts]), matrix), dim=1)
    for p in range(nparts + njoints):
        matrix[p, p] = 1

    matrix[:, -1] = 1
    if not is_conv:
        matrix = matrix.float().masked_fill(matrix == 0., float(-1e20)).masked_fill(matrix == 1., float(0.0))
    else:
        matrix = matrix[:nparts, nparts:]
    return matrix