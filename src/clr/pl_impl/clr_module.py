import torch
import torch.nn as nn
import torch.nn.functional as f

from pytorch_lightning import LightningModule
from pl_bolts.models.self_supervised.resnets import resnet18, resnet50
from pl_bolts.optimizers.lars import LARS
from pl_bolts.optimizers.lr_scheduler import linear_warmup_decay
from typing import Union

from .cw import cw_normality
from .memory_operator import MemoryOperator


class Projection(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=2048, output_dim=128):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim, bias=False),
        )

    def forward(self, x):
        x = self.model(x)
        return x
        # return f.normalize(x, dim=1)


class CLR(LightningModule):
    def __init__(
        self,
        gpus: int,
        num_samples: int,
        batch_size: int,
        dataset: str,
        reg_coeff: float,
        memory_length: int = 0,
        num_nodes: int = 1,
        arch: str = "resnet50",
        hidden_mlp: int = 2048,
        feat_dim: int = 128,
        warmup_epochs: int = 10,
        max_epochs: int = 100,
        first_conv: bool = True,
        maxpool1: bool = True,
        optimizer: str = "lars",
        exclude_bn_bias: bool = False,
        start_lr: float = 0.0,
        learning_rate: float = 1e-3,
        final_lr: float = 0.0,
        weight_decay: float = 1e-6,
        **kwargs
    ):
        """
        Args:
            batch_size: the batch size
            num_samples: num samples in the dataset
            warmup_epochs: epochs to warmup the lr for
            lr: the optimizer learning rate
            opt_weight_decay: the optimizer weight decay
            loss_temperature: the loss temperature
        """
        super().__init__()
        self.save_hyperparameters()

        self.gpus = gpus
        self.num_nodes = num_nodes
        self.arch = arch
        self.dataset = dataset
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.reg_coeff = reg_coeff

        self.hidden_mlp = hidden_mlp
        self.feat_dim = feat_dim
        self.first_conv = first_conv
        self.maxpool1 = maxpool1

        self.optim = optimizer
        self.exclude_bn_bias = exclude_bn_bias
        self.weight_decay = weight_decay

        self.start_lr = start_lr
        self.final_lr = final_lr
        self.learning_rate = learning_rate
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs

        self.mse_loss = nn.MSELoss()
        self.memory_length = memory_length
        self.memory = MemoryOperator(self.memory_length)
        self.encoder = self.init_model()

        self.projection = Projection(input_dim=self.hidden_mlp, hidden_dim=self.hidden_mlp, output_dim=self.feat_dim)

        # compute iters per epoch
        global_batch_size = self.num_nodes * self.gpus * self.batch_size if self.gpus > 0 else self.batch_size
        self.train_iters_per_epoch = self.num_samples // global_batch_size

    def init_model(self):
        if self.arch == "resnet18":
            backbone = resnet18
        elif self.arch == "resnet50":
            backbone = resnet50

        return backbone(first_conv=self.first_conv, maxpool1=self.maxpool1, return_all_feature_maps=False)

    def forward(self, x):
        # bolts resnet returns a list
        return self.encoder(x)[-1]

    def pcgrad(self, g1: torch.Tensor, g2: torch.Tensor) -> tuple[Union[torch.Tensor, float], Union[torch.Tensor, float], torch.Tensor, torch.Tensor, torch.Tensor]:

        lambda1: Union[torch.Tensor, float] = 1.0
        lambda2: Union[torch.Tensor, float] = 1.0

        dot_product = torch.dot(g1, g2)

        g1_norm: torch.Tensor = torch.norm(g1, 2)
        g2_norm: torch.Tensor = torch.norm(g2, 2)

        if dot_product < 0:
            lambda1 = 1.0 - dot_product / g1_norm ** 2
            lambda2 = 1.0 - dot_product / g2_norm ** 2

        return lambda1, lambda2, dot_product, g1_norm, g2_norm

    def shared_step(self, batch, train: bool):
        if self.dataset == "stl10":
            unlabeled_batch = batch[0]
            batch = unlabeled_batch

        # final image in tuple is for online eval
        (img1, img2, _), y = batch

        # get h representations, bolts resnet returns a list
        h1 = self(img1)
        h2 = self(img2)

        # get z representations
        z1 = self.projection(h1)
        z2 = self.projection(h2)

        if train:
            eval_latent = self.memory(z1)
            log_prefix = "train"
        else:
            eval_latent = z1
            log_prefix = "val"

        # g1 = torch.cat([i.flatten() for i in torch.autograd.grad(z1, list(self.parameters()), retain_graph=True,
        #                                                          create_graph=True)]).flatten()
        # g2 = torch.cat([i.flatten() for i in torch.autograd.grad(z2, list(self.parameters()), retain_graph=True,
        #                                                          create_graph=True)]).flatten()

        mse = self.mse_loss(z1, z2)
        cw = cw_normality(eval_latent)
        cw_reg = cw * self.reg_coeff
        loss = mse + cw_reg
        self.log(f"{log_prefix}_loss/mse", mse, on_step=False, on_epoch=True)
        self.log(f"{log_prefix}_loss/cw", cw, on_step=False, on_epoch=True)
        self.log(f"{log_prefix}_loss/cw_reg", cw_reg, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, train=True)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, train=False)
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def exclude_from_wt_decay(self, named_params, weight_decay, skip_list=("bias", "bn")):
        params = []
        excluded_params = []

        for name, param in named_params:
            if not param.requires_grad:
                continue
            elif any(layer_name in name for layer_name in skip_list):
                excluded_params.append(param)
            else:
                params.append(param)

        return [
            {"params": params, "weight_decay": weight_decay},
            {
                "params": excluded_params,
                "weight_decay": 0.0,
            },
        ]

    def configure_optimizers(self):
        if self.exclude_bn_bias:
            params = self.exclude_from_wt_decay(self.named_parameters(), weight_decay=self.weight_decay)
        else:
            params = self.parameters()

        if self.optim == "lars":
            optimizer = LARS(
                params,
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay,
                trust_coefficient=0.001,
            )
        elif self.optim == "adam":
            optimizer = torch.optim.Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay)

        warmup_steps = self.train_iters_per_epoch * self.warmup_epochs
        total_steps = self.train_iters_per_epoch * self.max_epochs

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                linear_warmup_decay(warmup_steps, total_steps, cosine=True),
            ),
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]
