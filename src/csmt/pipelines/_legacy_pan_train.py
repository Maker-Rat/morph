from __future__ import annotations

import argparse
import os


def build_legacy_default_args() -> dict:
    from csmt.parser.base import (
        add_cuda_options,
        add_dataset_options,
        add_losses_options,
        add_misc_options,
        add_model_options,
    )
    from csmt.parser.training import add_training_options

    p = argparse.ArgumentParser()
    add_misc_options(p)
    add_cuda_options(p)
    add_training_options(p)
    add_dataset_options(p)
    add_model_options(p)
    add_losses_options(p)
    args = p.parse_args([])
    return vars(args)


def run_pan_training(parameters: dict, para_cmd: str | None = None) -> None:
    """
    Run PAN teacher training using a fully materialized arg dict.
    This mirrors train_lafan1dog.py but does not import top-level config.py.
    """
    import torch
    from torch.utils.data.dataloader import DataLoader

    from csmt.data.datasetserial import DstDataset, SrcDataset
    from csmt.models import create_model
    from csmt.parser.base import dict_to_object, try_mkdir
    from csmt.utils.utils import get_body_part

    args = dict_to_object(parameters)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if isinstance(args.device, str) and "cuda" in args.device:
        if ":" in args.device:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device.split(":")[-1]
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if isinstance(args.device, str):
        req = args.device.lower()
        if req.startswith("cuda"):
            args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            args.device = torch.device("cpu")
    elif isinstance(args.device, torch.device):
        if args.device.type == "cuda" and not torch.cuda.is_available():
            args.device = torch.device("cpu")
    else:
        args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    log_path = os.path.join(args.save_dir, "logs/")
    try_mkdir(args.save_dir)
    try_mkdir(log_path)

    with open(os.path.join(args.save_dir, "para.txt"), "w", encoding="utf-8") as para_file:
        if para_cmd is None:
            para_cmd = ""
        para_file.write(para_cmd)

    srcdataset = SrcDataset(args, "src", "train")
    dstdataset = DstDataset(args, "dst", "train")

    num_workers = int(getattr(args, "num_workers", 0))
    pin_memory = (args.device.type == "cuda")
    dl_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "drop_last": True,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        dl_kwargs["persistent_workers"] = True
        dl_kwargs["prefetch_factor"] = 2

    srcloader = DataLoader(srcdataset, **dl_kwargs)
    dstloader = DataLoader(dstdataset, **dl_kwargs)
    dstfeeder = iter(dstloader)
    srcfeeder = iter(srcloader)

    body_src_key = "src_bodies" if "src_bodies" in args.correspondence_bodies[0] else "hum_bodies"
    body_dst_key = "dst_bodies" if "dst_bodies" in args.correspondence_bodies[0] else "dog_bodies"
    joint_src_key = "src_joints" if "src_joints" in args.correspondence_joints[0] else "hum_joints"
    joint_dst_key = "dst_joints" if "dst_joints" in args.correspondence_joints[0] else "dog_joints"

    src_bodies = get_body_part(args.correspondence_bodies, body_src_key)
    dst_bodies = get_body_part(args.correspondence_bodies, body_dst_key)

    src_joints = get_body_part(args.correspondence_joints, joint_src_key)
    dst_joints = get_body_part(args.correspondence_joints, joint_dst_key)

    joint_parts = [src_joints, dst_joints]
    body_parts = [src_bodies, dst_bodies]
    datasets = [srcdataset, dstdataset]

    model = create_model(args, body_parts, joint_parts, datasets, ["src", "dst"])

    if args.epoch_begin:
        model.load(epoch=args.epoch_begin)

    model.setup()

    epoch = args.epoch_begin
    while epoch < args.epoch_num:
        if epoch % args.save_iter == 0 or epoch == args.epoch_num - 1:
            model.save()

        flag = True
        while flag:
            try:
                dst_batch = next(dstfeeder)
                input_d, _, d_offsets, d_offsets_withend = dst_batch[:4]
            except StopIteration:
                dstfeeder = iter(dstloader)
                dst_batch = next(dstfeeder)
                input_d, _, d_offsets, d_offsets_withend = dst_batch[:4]

            try:
                src_batch = next(srcfeeder)
                input_h, _, h_offsets, h_offsets_withend = src_batch[:4]
            except StopIteration:
                epoch += 1
                flag = False
                srcfeeder = iter(srcloader)
                continue

            h_offsets = h_offsets.reshape(h_offsets.shape[0], -1)
            d_offsets = d_offsets.reshape(d_offsets.shape[0], -1)

            vel_dim = 4
            src_njoints = getattr(args, "src_njoints", getattr(args, "hum_njoints"))
            dst_njoints = getattr(args, "dst_njoints", getattr(args, "dog_njoints"))
            input_h_encoder = (input_h[..., : src_njoints + vel_dim]).transpose(1, 2)
            input_d_encoder = (input_d[..., : dst_njoints + vel_dim]).transpose(1, 2)

            input_h_encoder = (input_h_encoder, h_offsets, h_offsets_withend)
            input_d_encoder = (input_d_encoder, d_offsets, d_offsets_withend)

            model.set_input([input_h_encoder, input_d_encoder])
            model.optimize_parameters()

        model.epoch()
