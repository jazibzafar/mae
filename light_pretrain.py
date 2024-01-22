import argparse
import lightning as L
import models_mae
import torch
from mae_utils import MAETransform, GeoWebDataset
import timm.optim.optim_factory as optim_factory
import logging


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)

    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--in_chans', default=4, type=int,
                        help='Image channels RGB (3) or RGB+NIR (4).')
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--data_list', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='file list path')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    return parser


# Define the Lightning module
class LitModel(L.LightningModule):
    def __init__(self, model, mask_ratio, weight_decay, lr):
        super().__init__()
        self.model = model
        self.mask_ratio = mask_ratio
        self.weight_decay = weight_decay
        self.lr = lr

    def training_step(self, batch):
        # Training step defines the train loop.
        # It is independent of forward calls.
        loss, _, _ = self.model(batch, mask_ratio=self.mask_ratio)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        param_groups = optim_factory.param_groups_weight_decay(self.model, self.weight_decay)
        optimizer = torch.optim.AdamW(param_groups, lr=self.lr, betas=(0.9, 0.95))
        return optimizer


def main(args):
    transform_train = MAETransform(args.input_size)

    dataset_train = GeoWebDataset(root=args.data_path,
                                  n_bands=args.in_chans,
                                  augmentations=transform_train,
                                  num_workers=args.num_workers)

    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        pin_memory=True,
        # drop_last=True,
    )

    model = models_mae.__dict__[args.model](img_size=args.input_size,
                                            in_chans=args.in_chans,
                                            norm_pix_loss=args.norm_pix_loss)

    masked_autoencoder = LitModel(model, args.mask_ratio, args.weight_decay, args.lr)
    # TODO: checkpoint even n training steps.
    # TODO: train for a total of N training steps.
    print("beginning training")
    trainer = L.Trainer(accelerator='gpu', devices=1, max_epochs=-1)
    trainer.fit(model=masked_autoencoder, train_dataloaders=dataloader_train)

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    # if args.output_dir:
    #     Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)




