import os
import sys
from argparse import ArgumentParser

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pl_bolts.transforms.dataset_normalizations import (
    cifar10_normalization,
    imagenet_normalization,
    stl10_normalization,
)
from pl_bolts.callbacks.ssl_online import SSLOnlineEvaluator
from pl_bolts.datamodules import CIFAR10DataModule, ImagenetDataModule, STL10DataModule

from pl_impl.transforms import CLREvalDataTransform, CLRTrainDataTransform
from pl_impl.clr_module import CLR


def get_args(args):
    parser = ArgumentParser()

    # model params
    parser.add_argument("--arch", default="resnet50", type=str, help="convnet architecture")
    # specify flags to store false
    parser.add_argument("--first_conv", action="store_false")
    parser.add_argument("--maxpool1", action="store_false")
    parser.add_argument("--hidden_mlp", default=2048, type=int, help="hidden layer dimension in projection head")
    parser.add_argument("--feat_dim", default=128, type=int, help="feature dimension")
    parser.add_argument("--online_ft", action="store_true")
    parser.add_argument("--fp32", action="store_true")

    # transform params
    parser.add_argument("--gaussian_blur", action="store_true", help="add gaussian blur")
    parser.add_argument("--jitter_strength", type=float, default=1.0, help="jitter strength")
    parser.add_argument("--dataset", type=str, default="cifar10", help="stl10, cifar10")
    parser.add_argument("--data_dir", type=str, default=".", help="path to download data")
    parser.add_argument("--model_weights", type=str, default="", help="path to download data")

    # training params
    parser.add_argument("--fast_dev_run", default=1, type=int)
    parser.add_argument("--num_nodes", default=1, type=int, help="number of nodes for training")
    parser.add_argument("--gpus", default=1, type=int, help="number of gpus to train on")
    parser.add_argument("--num_workers", default=4, type=int, help="num of workers per GPU")
    parser.add_argument("--optimizer", default="adam", type=str, help="choose between adam/lars")
    parser.add_argument("--exclude_bn_bias", action="store_true", help="exclude bn/bias from weight decay")
    parser.add_argument("--max_epochs", default=100, type=int, help="number of total epochs to run")
    parser.add_argument("--max_steps", default=-1, type=int, help="max steps")
    parser.add_argument("--warmup_epochs", default=0, type=int, help="number of warmup epochs")
    parser.add_argument("--batch_size", default=128, type=int, help="batch size per gpu")

    parser.add_argument("--reg_coeff", default=1, type=float, help="coeff")
    parser.add_argument("--weight_decay", default=1e-6, type=float, help="weight decay")
    parser.add_argument("--learning_rate", default=1e-3, type=float, help="base learning rate")
    parser.add_argument("--start_lr", default=0, type=float, help="initial warmup learning rate")
    parser.add_argument("--final_lr", type=float, default=1e-6, help="final learning rate")
    parser.add_argument("--exp_name", type=str, default="test", help="exp name")
    parser.add_argument("--mode", type=str, default="both", help="which img should be augmented")

    return parser.parse_args(args)


def cli_main():

    # model args
    args = get_args(sys.argv[1:])

    if args.dataset == "stl10":
        dm = STL10DataModule(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)

        dm.train_dataloader = dm.train_dataloader_mixed
        dm.val_dataloader = dm.val_dataloader_mixed
        args.num_samples = dm.num_unlabeled_samples

        args.maxpool1 = False
        args.first_conv = True
        args.input_height = dm.size()[-1]

        normalization = stl10_normalization()

        args.gaussian_blur = True
        args.jitter_strength = 1.0
    elif args.dataset == "cifar10":
        val_split = 5000
        if args.num_nodes * args.gpus * args.batch_size > val_split:
            val_split = args.num_nodes * args.gpus * args.batch_size

        dm = CIFAR10DataModule(
            data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers, val_split=val_split
        )

        args.num_samples = dm.num_samples

        args.maxpool1 = False
        args.first_conv = False
        args.input_height = dm.size()[-1]

        normalization = cifar10_normalization()

        args.gaussian_blur = False
        args.jitter_strength = 0.5
    elif args.dataset == "imagenet":
        args.maxpool1 = True
        args.first_conv = True
        normalization = imagenet_normalization()

        args.gaussian_blur = True
        args.jitter_strength = 1.0

        # args.batch_size = 64
        # args.num_nodes = 8
        # args.gpus = 8  # per-node
        # args.max_epochs = 800

        # args.optimizer = "lars"
        # args.learning_rate = 4.8
        args.final_lr = 0.0048
        args.start_lr = 0.3
        # args.online_ft = True

        dm = ImagenetDataModule(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)

        args.num_samples = dm.num_samples
        args.input_height = dm.size()[-1]
    else:
        raise NotImplementedError("other datasets have not been implemented till now")

    aug = True if args.mode == "both" else False
    dm.train_transforms = CLRTrainDataTransform(
        input_height=args.input_height,
        gaussian_blur=args.gaussian_blur,
        jitter_strength=args.jitter_strength,
        normalize=normalization,
        augment_both=aug
    )

    dm.val_transforms = CLREvalDataTransform(
        input_height=args.input_height,
        gaussian_blur=args.gaussian_blur,
        jitter_strength=args.jitter_strength,
        normalize=normalization
    )

    model = CLR(**args.__dict__)
    if args.model_weights:
        model = CLR(**args.__dict__).load_from_checkpoint(args.model_weights, strict=False, reg_coeff=args.reg_coeff)

    online_evaluator = None
    if args.online_ft:
        # online eval
        online_evaluator = SSLOnlineEvaluator(
            drop_p=0.0,
            hidden_dim=None,
            z_dim=args.hidden_mlp,
            num_classes=dm.num_classes,
            dataset=args.dataset,
        )

    exp_dir = f"outputs/{args.dataset}/{args.arch}/{args.exp_name}"
    log_dir = f"{exp_dir}/pre_train/logs"
    ckpt_dir = f"{exp_dir}/pre_train/ckpt"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    logger = TensorBoardLogger(log_dir)
    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_checkpoint = ModelCheckpoint(dirpath=ckpt_dir, save_last=True, save_top_k=1, monitor="val_loss")
    callbacks = [model_checkpoint, online_evaluator] if args.online_ft else [model_checkpoint]
    callbacks.append(lr_monitor)

    trainer = Trainer(
        max_epochs=args.max_epochs,
        gpus=args.gpus,
        num_nodes=args.num_nodes,
        # distributed_backend="ddp",
        accelerator="gpu" if args.gpus > 1 else None,
        sync_batchnorm=True if args.gpus > 1 else False,
        precision=32 if args.fp32 else 16,
        callbacks=callbacks,
        logger=logger,
        # plugins=DDPPlugin(find_unused_parameters=True),
        # fast_dev_run=args.fast_dev_run,
    )

    trainer.fit(model, datamodule=dm)

    os.environ["BEST_MODEL_PATH"] = model_checkpoint.best_model_path


if __name__ == "__main__":
    cli_main()
