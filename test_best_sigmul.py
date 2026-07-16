import os
import json
import argparse
import torch
import pytorch_lightning as pl
from argparse import ArgumentParser

from lit_model import LitModel
from dataset import WrapperDataset
from utils.sig_mul_utils import load_data


def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="./checkpoints_sigmul_rmabnormal/AB1101/esm2_650m_AB1101-val_pearson_corr_lr-3e-05_loss-mse.ckpt",
    )
    parser.add_argument("--model_locate", type=str, default="./model/esm2_650m")

    # dataset
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--data_folder", type=str, default="./data/sigmul_data")
    parser.add_argument("--data_name", type=str, default="AB1101")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--rm_abnormal", type=bool, default=True)
    parser.add_argument("--seed", type=int, default=3407)

    # model args（保持和训练时一致）
    parser.add_argument("--hidden_size", type=int, default=1280)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--out_dim", type=int, default=1)
    parser.add_argument("--freeze_backbone", action="store_true")

    parser.add_argument("--use_mamba_branch", type=bool, default=True)
    parser.add_argument("--use_contrastive", type=bool, default=True)
    parser.add_argument("--mamba_layers", type=int, default=2)
    parser.add_argument("--mamba_d_state", type=int, default=16)
    parser.add_argument("--mamba_d_conv", type=int, default=4)
    parser.add_argument("--mamba_expand", type=int, default=2)
    parser.add_argument("--mamba_dropout", type=float, default=0.1)

    parser.add_argument("--proj_dim", type=int, default=256)
    parser.add_argument("--contrastive_temperature", type=float, default=0.1)
    parser.add_argument("--cl1_weight", type=float, default=0.1)
    parser.add_argument("--cl2_weight", type=float, default=0.05)

    # trainer
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--accelerator", type=str, default="gpu")

    return parser.parse_args()


def main():
    args = parse_args()

    # load test data
    train_data, val_data, test_data = load_data(
        args.data_name,
        args.data_folder,
        val_ratio=args.val_ratio,
        random_seed=args.seed,
        rm_abnormal=args.rm_abnormal,
    )

    dataset = WrapperDataset(
        args=args,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
    )
    test_loader = dataset.get_test_loader()

    # PyTorch 2.6+ safe checkpoint loading
    torch.serialization.add_safe_globals([argparse.Namespace])

    model = LitModel.load_from_checkpoint(args.ckpt_path)

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        logger=False,
        enable_checkpointing=False,
    )

    results = trainer.test(model=model, dataloaders=test_loader)

    print("\nFinal test metrics:")
    print(results)

    save_path = os.path.splitext(args.ckpt_path)[0] + "_test_results.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved test metrics to: {save_path}")


if __name__ == "__main__":
    main()