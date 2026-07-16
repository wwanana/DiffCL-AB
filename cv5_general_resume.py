import os
import json
import math
import glob
import argparse
from typing import Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.serialization
import pytorch_lightning as pl

from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger

from lit_model import LitModel
from dataset import SeqDataset
from utils.data_utils import get_AB_pair_data, get_s1131_data

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# 避免 PyTorch 2.6+ 加载 checkpoint 时 argparse.Namespace 报错
torch.serialization.add_safe_globals([argparse.Namespace])


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1", "y"):
        return True
    if v.lower() in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    parser = argparse.ArgumentParser()

    # ========= data =========
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--data_folder", type=str, default=None)
    parser.add_argument("--data_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--rm_abnormal", type=str2bool, default=True)

    # ========= model / tokenizer =========
    parser.add_argument("--model_locate", type=str, default="./model/esm2_650m")
    parser.add_argument("--max_length", type=int, default=384)

    # ========= dataloader =========
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--drop_last", type=str2bool, default=True)

    # ========= model args =========
    parser.add_argument("--hidden_size", type=int, default=1280)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--out_dim", type=int, default=1)
    parser.add_argument("--freeze_backbone", action="store_true")

    # ========= mamba / contrastive =========
    parser.add_argument("--use_mamba_branch", type=str2bool, default=True)
    parser.add_argument("--use_contrastive", type=str2bool, default=True)
    parser.add_argument("--mamba_layers", type=int, default=2)
    parser.add_argument("--mamba_d_state", type=int, default=16)
    parser.add_argument("--mamba_d_conv", type=int, default=4)
    parser.add_argument("--mamba_expand", type=int, default=2)
    parser.add_argument("--mamba_dropout", type=float, default=0.1)
    parser.add_argument("--proj_dim", type=int, default=256)
    parser.add_argument("--contrastive_temperature", type=float, default=0.1)
    parser.add_argument("--cl1_weight", type=float, default=0.03)
    parser.add_argument("--cl2_weight", type=float, default=0.01)
    parser.add_argument("--ag_consistency_inner_weight", type=float, default=0.1)

    # ========= trainer =========
    parser.add_argument("--loss", type=str, default="mse")
    parser.add_argument("--ckpt_monitor", type=str, default="val_pearson_corr")
    parser.add_argument("--early_stop_monitor", type=str, default="val_mse")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--strategy", type=str, default="auto")
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--accumulate_grad_batches", type=int, default=4)
    parser.add_argument("--gradient_clip_val", type=float, default=5.0)
    parser.add_argument("--gradient_clip_algorithm", type=str, default="value")
    parser.add_argument("--precision", type=str, default="32")
    parser.add_argument("--seed", type=int, default=42)

    # ========= scheduler / weight decay =========
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--use_lr_scheduler", type=str2bool, default=True)
    parser.add_argument("--scheduler_patience", type=int, default=5)
    parser.add_argument("--scheduler_factor", type=float, default=0.5)
    parser.add_argument("--min_lr", type=float, default=1e-6)

    # ========= resume =========
    parser.add_argument("--resume", type=str2bool, default=True)

    args = parser.parse_args()
    seed_everything(args.seed, workers=True)
    return args


def to_list_tuple(data):
    out = []
    for x in data:
        if isinstance(x, np.ndarray):
            out.append(x.tolist())
        else:
            out.append(list(x))
    return tuple(out)


def concat_data(data1, data2):
    data1 = to_list_tuple(data1)
    data2 = to_list_tuple(data2)
    return tuple([list(a) + list(b) for a, b in zip(data1, data2)])


def load_single_csv(dataset_name, file_path, rm_abnormal=True):
    name = dataset_name.upper()
    path_upper = file_path.upper()

    if "S1131" in name or "S1131" in path_upper:
        data = get_s1131_data(file_path)
    elif "AB" in name or "AB" in path_upper:
        data = get_AB_pair_data(file_path, rm_abnormal=rm_abnormal)
    else:
        raise ValueError(
            f"Cannot infer dataset type from dataset_name={dataset_name}, file_path={file_path}"
        )

    return to_list_tuple(data)


def load_full_dataset(dataset_name, data_folder=None, data_file=None, rm_abnormal=True):
    """
    支持两种方式：

    1. 传入 data_file：
       直接对单个 csv 做五折。

    2. 传入 data_folder：
       自动寻找：
       - S1131_train.csv + S1131_test.csv
       - AB645_train.csv + AB645_test.csv
       - AB564_train.csv + AB564_test.csv
       然后合并后做五折。
    """

    if data_file is not None:
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"data_file not found: {data_file}")
        print(f"[Load] single file: {data_file}")
        return load_single_csv(dataset_name, data_file, rm_abnormal=rm_abnormal)

    if data_folder is None:
        raise ValueError("Either --data_file or --data_folder must be provided.")

    if not os.path.exists(data_folder):
        raise FileNotFoundError(f"data_folder not found: {data_folder}")

    name = dataset_name.upper()

    if "S1131" in name:
        train_file = os.path.join(data_folder, "S1131_train.csv")
        test_file = os.path.join(data_folder, "S1131_test.csv")
    else:
        train_file = os.path.join(data_folder, f"{dataset_name}_train.csv")
        test_file = os.path.join(data_folder, f"{dataset_name}_test.csv")

    # 如果严格命名找不到，则自动搜索
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        train_candidates = sorted(glob.glob(os.path.join(data_folder, "*train*.csv")))
        test_candidates = sorted(glob.glob(os.path.join(data_folder, "*test*.csv")))

        if len(train_candidates) == 0 or len(test_candidates) == 0:
            csv_candidates = sorted(glob.glob(os.path.join(data_folder, "*.csv")))
            if len(csv_candidates) == 1:
                print(f"[Load] only one csv found, use it directly: {csv_candidates[0]}")
                return load_single_csv(dataset_name, csv_candidates[0], rm_abnormal=rm_abnormal)

            raise FileNotFoundError(
                f"Could not find train/test csv in {data_folder}. "
                f"Found csv files: {csv_candidates}"
            )

        train_file = train_candidates[0]
        test_file = test_candidates[0]

    print(f"[Load] train file: {train_file}")
    print(f"[Load] test  file: {test_file}")

    train_data = load_single_csv(dataset_name, train_file, rm_abnormal=rm_abnormal)
    test_data = load_single_csv(dataset_name, test_file, rm_abnormal=rm_abnormal)

    full_data = concat_data(train_data, test_data)
    return full_data


class FoldDataModule(pl.LightningDataModule):
    def __init__(self, args, train_data, val_data, test_data):
        super().__init__()

        self.args = args
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.model_locate = args.model_locate
        self.max_length = args.max_length
        self.truncation = True if self.max_length else False
        self.drop_last = args.drop_last

        tokenizer_kwargs = {}
        if os.path.isdir(self.model_locate):
            tokenizer_kwargs["local_files_only"] = True

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_locate,
            **tokenizer_kwargs,
        )

        self.train_dataset = self._build_dataset(train_data)
        self.val_dataset = self._build_dataset(val_data)
        self.test_dataset = self._build_dataset(test_data)

    def _build_dataset(self, data):
        wt_ab, wt_ag, mt_ab, mt_ag, labels = data

        return SeqDataset(
            wt_ab_seqs=wt_ab,
            mt_ab_seqs=mt_ab,
            wt_ag_seqs=wt_ag,
            mt_ag_seqs=mt_ag,
            labels=labels,
        )

    def collate_fn(self, batch):
        wt_ab_seqs, mt_ab_seqs, wt_ag_seqs, mt_ag_seqs, labels = zip(*batch)

        wt_ab_inputs = self.tokenizer(
            wt_ab_seqs,
            padding=True,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=self.truncation,
        )

        mt_ab_inputs = self.tokenizer(
            mt_ab_seqs,
            padding=True,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=self.truncation,
        )

        wt_ag_inputs = self.tokenizer(
            wt_ag_seqs,
            padding=True,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=self.truncation,
        )

        mt_ag_inputs = self.tokenizer(
            mt_ag_seqs,
            padding=True,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=self.truncation,
        )

        labels = torch.tensor(labels, dtype=torch.float32)

        return {
            "wt_ab_inputs_ids": wt_ab_inputs["input_ids"],
            "wt_ab_inputs_mask": wt_ab_inputs["attention_mask"],
            "mut_ab_inputs_ids": mt_ab_inputs["input_ids"],
            "mt_ab_inputs_mask": mt_ab_inputs["attention_mask"],
            "wt_ag_inputs_ids": wt_ag_inputs["input_ids"],
            "wt_ag_inputs_mask": wt_ag_inputs["attention_mask"],
            "mut_ag_inputs_ids": mt_ag_inputs["input_ids"],
            "mt_ag_inputs_mask": mt_ag_inputs["attention_mask"],
            "labels": labels,
        }

    def train_dataloader(self):
        train_drop_last = self.drop_last and len(self.train_dataset) >= self.batch_size

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            drop_last=train_drop_last,
            pin_memory=True,
        )

    def val_dataloader(self):
        val_drop_last = self.drop_last and len(self.val_dataset) >= self.batch_size

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            drop_last=val_drop_last,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            drop_last=False,
            pin_memory=True,
        )


def select_by_indices(data_tuple: Tuple[List, List, List, List, List], indices):
    return tuple([[x[i] for i in indices] for x in data_tuple])


def metric_mode(metric_name: str):
    name = metric_name.lower()
    return "max" if ("corr" in name or name.endswith("r2")) else "min"


def safe_float(x):
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().item())
    return float(x)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_existing_best_ckpt(fold_dir: str):
    ckpts = [f for f in os.listdir(fold_dir) if f.endswith(".ckpt")]
    if len(ckpts) == 0:
        return ""
    ckpts = sorted(ckpts)
    return os.path.join(fold_dir, ckpts[-1])


def build_trainer(args, callbacks, logger):
    return pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=args.strategy,
        num_nodes=args.num_nodes,
        max_epochs=args.max_epochs,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        gradient_clip_algorithm=args.gradient_clip_algorithm,
        deterministic=True,
        precision=args.precision,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
    )


def run_one_fold(args, fold_id, train_data, val_data, test_data, fold_dir):
    os.makedirs(fold_dir, exist_ok=True)

    fold_result_path = os.path.join(fold_dir, "fold_result.json")
    fit_done_flag = os.path.join(fold_dir, "fit_finished.flag")

    logger = CSVLogger(save_dir=fold_dir, name="logs")

    ckpt_cb = ModelCheckpoint(
        dirpath=fold_dir,
        filename=f"fold{fold_id}" + "-{epoch:02d}-{" + args.ckpt_monitor + ":.4f}",
        monitor=args.ckpt_monitor,
        mode=metric_mode(args.ckpt_monitor),
        save_top_k=1,
        save_weights_only=True,
        verbose=True,
    )

    early_stop_cb = EarlyStopping(
        monitor=args.early_stop_monitor,
        mode=metric_mode(args.early_stop_monitor),
        patience=args.patience,
        verbose=True,
    )

    data_module = FoldDataModule(args, train_data, val_data, test_data)

    existing_ckpt = find_existing_best_ckpt(fold_dir)

    trainer = build_trainer(
        args=args,
        callbacks=[ckpt_cb, early_stop_cb],
        logger=logger,
    )

    if args.resume and os.path.exists(fit_done_flag) and existing_ckpt:
        print(f"[Resume] Fold {fold_id}: fit already finished, reuse checkpoint.")
        best_ckpt = existing_ckpt
    else:
        print(f"[Train] Fold {fold_id}: start training.")
        model = LitModel(args)
        trainer.fit(model, datamodule=data_module)

        with open(fit_done_flag, "w", encoding="utf-8") as f:
            f.write("fit finished\n")

        best_ckpt = ckpt_cb.best_model_path

        if best_ckpt is None or best_ckpt == "":
            best_ckpt = find_existing_best_ckpt(fold_dir)

    if best_ckpt is None or best_ckpt == "":
        raise RuntimeError(f"Fold {fold_id}: no checkpoint found in {fold_dir}")

    print(f"[Test] Fold {fold_id}: load checkpoint {best_ckpt}")

    with torch.serialization.safe_globals([argparse.Namespace]):
        best_model = LitModel.load_from_checkpoint(
            best_ckpt,
            args=args,
            map_location="cpu",
        )

    test_result = trainer.test(
        best_model,
        datamodule=data_module,
        verbose=False,
    )[0]

    fold_metrics = {
        "fold": fold_id,
        "best_ckpt": best_ckpt,
        "test_mse": safe_float(test_result["test_mse"]),
        "test_rmse": math.sqrt(max(safe_float(test_result["test_mse"]), 0.0)),
        "test_mae": safe_float(test_result["test_mae"]),
        "test_pearson_corr": safe_float(test_result["test_pearson_corr"]),
        "test_spearman_corr": safe_float(test_result["test_spearman_corr"]),
        "test_r2": safe_float(test_result["test_r2"]),
    }

    save_json(fold_metrics, fold_result_path)
    return fold_metrics


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 100)
    print("Args")
    print("=" * 100)
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    all_data = load_full_dataset(
        dataset_name=args.dataset_name,
        data_folder=args.data_folder,
        data_file=args.data_file,
        rm_abnormal=args.rm_abnormal,
    )

    num_samples = len(all_data[-1])
    all_indices = np.arange(num_samples)

    print("\n" + "=" * 100)
    print(f"Dataset: {args.dataset_name}")
    print(f"Loaded samples: {num_samples}")
    print("=" * 100)

    kf = KFold(
        n_splits=args.n_splits,
        shuffle=True,
        random_state=args.seed,
    )

    all_fold_results = []

    for fold_id, (trainval_idx, test_idx) in enumerate(kf.split(all_indices), start=1):
        fold_dir = os.path.join(args.output_dir, f"fold_{fold_id}")
        os.makedirs(fold_dir, exist_ok=True)

        fold_result_path = os.path.join(fold_dir, "fold_result.json")

        if args.resume and os.path.exists(fold_result_path):
            print("\n" + "=" * 100)
            print(f"[Resume] Fold {fold_id} already finished. Load result.")
            print("=" * 100)
            fold_metrics = load_json(fold_result_path)
            all_fold_results.append(fold_metrics)
            continue

        train_idx, val_idx = train_test_split(
            trainval_idx,
            test_size=args.val_ratio,
            random_state=args.seed + fold_id,
            shuffle=True,
        )

        train_data = select_by_indices(all_data, train_idx)
        val_data = select_by_indices(all_data, val_idx)
        test_data = select_by_indices(all_data, test_idx)

        print("\n" + "=" * 100)
        print(f"Fold {fold_id}/{args.n_splits}")
        print("=" * 100)
        print(f"train size: {len(train_idx)}")
        print(f"val   size: {len(val_idx)}")
        print(f"test  size: {len(test_idx)}")
        print(f"fold dir : {fold_dir}")

        fold_metrics = run_one_fold(
            args=args,
            fold_id=fold_id,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            fold_dir=fold_dir,
        )

        all_fold_results.append(fold_metrics)

        print("\nFold result:")
        print(json.dumps(fold_metrics, indent=2, ensure_ascii=False))

    results_df = pd.DataFrame(all_fold_results)
    results_df = results_df.sort_values("fold").reset_index(drop=True)

    save_csv_path = os.path.join(args.output_dir, "cv5_fold_results.csv")
    results_df.to_csv(save_csv_path, index=False)

    metric_cols = [
        "test_mse",
        "test_rmse",
        "test_mae",
        "test_pearson_corr",
        "test_spearman_corr",
        "test_r2",
    ]

    summary = {}
    for col in metric_cols:
        values = results_df[col].astype(float).values
        summary[col] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        }

    save_json_path = os.path.join(args.output_dir, "cv5_summary.json")
    save_json(summary, save_json_path)

    print("\n" + "=" * 100)
    print("Per-fold results")
    print("=" * 100)
    print(results_df[["fold"] + metric_cols].to_string(index=False))

    print("\n" + "=" * 100)
    print(f"{args.n_splits}-Fold CV summary (mean ± std)")
    print("=" * 100)

    for col in metric_cols:
        print(f"{col:<20}: {summary[col]['mean']:.6f} ± {summary[col]['std']:.6f}")

    print("\nSaved files:")
    print(save_csv_path)
    print(save_json_path)


if __name__ == "__main__":
    main()