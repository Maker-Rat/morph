try:
    from torch.utils.tensorboard import SummaryWriter as _TorchSummaryWriter
except Exception as exc:
    _TENSORBOARD_IMPORT_ERROR = exc

    class _TorchSummaryWriter:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            logdir = args[0] if len(args) > 0 else kwargs.get("log_dir", "")
            print(
                "⚠ TensorBoard unavailable; falling back to no-op writer. "
                f"(reason: {_TENSORBOARD_IMPORT_ERROR}) log_dir={logdir}"
            )

        def add_scalar(self, *args, **kwargs):
            return None

        def close(self):
            return None

SummaryWriter = _TorchSummaryWriter
import numpy as np
import torch


class SingleLoss:
    def __init__(self, name: str, writer: SummaryWriter, wandb=None):
        self.name = name
        self.loss_step = []
        self.loss_epoch = []
        self.loss_epoch_tmp = []
        self.writer = writer
        self.wandb = wandb

    def add_scalar(self, val, step=None):
        if step is None: step = len(self.loss_step)
        self.loss_step.append(val)
        self.loss_epoch_tmp.append(val)
        self.writer.add_scalar('Train/step_' + self.name, val, step)
        
        # Log to WandB
        if self.wandb:
            self.wandb.log({f'train/step_{self.name}': val, 'step': step})

    def epoch(self, step=None):
        if step is None: step = len(self.loss_epoch)
        loss_avg = sum(self.loss_epoch_tmp) / len(self.loss_epoch_tmp)
        self.loss_epoch_tmp = []
        self.loss_epoch.append(loss_avg)
        self.writer.add_scalar('Train/epoch_' + self.name, loss_avg, step)
        
        # Log to WandB
        if self.wandb:
            self.wandb.log({f'train/epoch_{self.name}': loss_avg, 'epoch': step})

    def save(self, path):
        loss_step = np.array(self.loss_step)
        loss_epoch = np.array(self.loss_epoch)
        np.save(path + self.name + '_step.npy', loss_step)
        np.save(path + self.name + '_epoch.npy', loss_epoch)


class LossRecorder:
    def __init__(self, writer: SummaryWriter, wandb=None):
        self.losses = {}
        self.writer = writer
        self.wandb = wandb

    def add_scalar(self, name, val, step=None):
        if isinstance(val, torch.Tensor): val = val.item()
        if name not in self.losses:
            self.losses[name] = SingleLoss(name, self.writer, self.wandb)
        self.losses[name].add_scalar(val, step)

    def epoch(self, step=None):
        for loss in self.losses.values():
            loss.epoch(step)

    def save(self, path):
        for loss in self.losses.values():
            loss.save(path)
