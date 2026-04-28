import os
import torch
import torch.optim
from abc import ABC, abstractmethod


class BaseModel(ABC):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
    """

    def __init__(self, args):
        self.args = args
        self.is_train = args.is_train
        # Respect explicit device selection from args (e.g., --device cpu).
        # Fall back to CUDA only when requested and available.
        requested_device = None
        try:
            requested_device = getattr(args, "device")
        except (AttributeError, KeyError):
            requested_device = None
        if requested_device is None and isinstance(args, dict):
            requested_device = args.get("device", None)

        if isinstance(requested_device, torch.device):
            if requested_device.type == "cuda" and not torch.cuda.is_available():
                self.device = torch.device("cpu")
            else:
                self.device = requested_device
        elif isinstance(requested_device, str):
            req = requested_device.strip().lower()
            if req.startswith("cuda"):
                self.device = torch.device(requested_device if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Keep args.device aligned to resolved torch.device for downstream modules.
        try:
            args.device = self.device
        except Exception:
            pass
        self.model_save_dir = os.path.join(args.save_dir, 'models')  # save all the checkpoints to save_dir

        if self.is_train:
            from csmt.utils.loss_record import LossRecorder, SummaryWriter
            self.log_path = os.path.join(args.save_dir, 'logs')
            self.writer = SummaryWriter(self.log_path)
            
            # Initialize WandB for experiment tracking
            self.use_wandb = False
            self.wandb_api = None
            try:
                import wandb
                self.use_wandb = True
                self.wandb_api = wandb
                
                # Initialize WandB run
                # Handle both attribute and dict-like access to args
                project_name = 'csmt-ablation_stage3'
                try:
                    if hasattr(args, 'wandb_project'):
                        project_name = args.wandb_project
                    elif isinstance(args, dict) and 'wandb_project' in args:
                        project_name = args['wandb_project']
                except (AttributeError, TypeError, KeyError):
                    pass
                
                run_name = os.path.basename(args.save_dir)
                
                # Convert args to dict for wandb config
                try:
                    args_dict = vars(args)
                except TypeError:
                    args_dict = dict(args) if isinstance(args, dict) else {}
                
                self.wandb_api.init(
                    project=project_name,
                    name=run_name,
                    config={k: v for k, v in args_dict.items() if not k.startswith('_')},
                    dir=args.save_dir,
                    save_code=False
                )
                print(f"✓ WandB initialized: {project_name}/{run_name}")
            except ImportError:
                print("⚠ WandB not installed. Install with: pip install wandb")
                self.use_wandb = False
            except Exception as e:
                print(f"⚠ WandB initialization failed: {e}")
                self.use_wandb = False
            
            # Create LossRecorder with WandB support
            self.loss_recoder = LossRecorder(self.writer, wandb=self.wandb_api if self.use_wandb else None)

        self.epoch_cnt = 0
        self.schedulers = []
        self.optimizers = []

    @abstractmethod
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        pass

    @abstractmethod
    def compute_test_result(self):
        """
        After forward, do something like output bvh, get error value
        """
        pass

    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass


    @abstractmethod
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass

    def get_scheduler(self, optimizer):
        if self.args.scheduler == 'linear':
            def lambda_rule(epoch):
                lr_l = 1.0 - max(0, epoch - self.args.n_epochs_origin) / float(self.args.n_epochs_decay + 1)
                return lr_l
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        if self.args.scheduler == 'Step_LR':
            print('Step_LR scheduler set')
            return torch.optim.lr_scheduler.StepLR(optimizer, 50, 0.5)
        if self.args.scheduler == 'Plateau':
            print('Plateau_LR shceduler set')
            return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5, verbose=True)
        if self.args.scheduler == 'MultiStep':
            return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[])

    def setup(self):
        """Load and print networks; create schedulers
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if self.is_train:
            self.schedulers = [self.get_scheduler(optimizer) for optimizer in self.optimizers]

    def train(self):
        """Set all models to training mode"""
        pass  # Override in subclass

    def eval(self):
        """Set all models to evaluation mode"""
        pass  # Override in subclass

    def epoch(self):
        self.loss_recoder.epoch()
        for scheduler in self.schedulers:
            if scheduler is not None:
                scheduler.step()
        
        # Log epoch to WandB
        if self.use_wandb and self.is_train and self.wandb_api:
            self.wandb_api.log({"epoch": self.epoch_cnt})
        
        self.epoch_cnt += 1

    def test(self):
        """Forward function used in test time.
        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            self.forward()
            self.compute_test_result()
