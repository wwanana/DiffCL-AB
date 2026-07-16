import os
import glob
import pandas as pd
import numpy as np

root = "./cv5_results_S1131_dualpath_diffattn_scheduler_seed42"

metric_files = sorted(glob.glob(os.path.join(root, "fold_*", "logs", "version_*", "metrics.csv")))

if len(metric_files) == 0:
    raise FileNotFoundError(f"No metrics.csv found under {root}")

for path in metric_files:
    fold_name = path.split("/")[-4]
    df = pd.read_csv(path)

    print("=" * 100)
    print(fold_name)
    print(path)

    cols = df.columns.tolist()
    useful_cols = [c for c in cols if any(k in c for k in [
        "epoch",
        "loss_epoch",
        "train_reg_loss_epoch",
        "train_cl1_loss_epoch",
        "train_cl2_loss_epoch",
        "val_mse",
        "val_mae",
        "val_pearson_corr",
        "val_spearman_corr",
        "val_r2"
    ])]

    sub = df[useful_cols].copy()

    # Lightning 的 metrics.csv 里很多行是空的，先按 epoch 聚合
    if "epoch" not in sub.columns:
        print("No epoch column found.")
        continue

    epoch_df = sub.groupby("epoch").last().reset_index()

    print("\nLast 10 epochs:")
    print(epoch_df.tail(10).to_string(index=False))

    if "val_mse" in epoch_df.columns:
        valid = epoch_df.dropna(subset=["val_mse"])
        if len(valid) > 0:
            best_idx = valid["val_mse"].idxmin()
            best_epoch = int(valid.loc[best_idx, "epoch"])
            best_val_mse = valid.loc[best_idx, "val_mse"]
            final_val_mse = valid.iloc[-1]["val_mse"]
            final_epoch = int(valid.iloc[-1]["epoch"])

            print("\nVal MSE analysis:")
            print(f"best_epoch    : {best_epoch}")
            print(f"best_val_mse  : {best_val_mse:.6f}")
            print(f"final_epoch   : {final_epoch}")
            print(f"final_val_mse : {final_val_mse:.6f}")
            print(f"final - best  : {final_val_mse - best_val_mse:.6f}")

            last5 = valid.tail(5)["val_mse"].values
            if len(last5) >= 5:
                print(f"last5 val_mse std : {np.std(last5):.6f}")
                print(f"last5 val_mse mean: {np.mean(last5):.6f}")

    if "val_pearson_corr" in epoch_df.columns:
        valid_p = epoch_df.dropna(subset=["val_pearson_corr"])
        if len(valid_p) > 0:
            best_p_idx = valid_p["val_pearson_corr"].idxmax()
            print("\nVal Pearson analysis:")
            print(f"best_pearson_epoch : {int(valid_p.loc[best_p_idx, 'epoch'])}")
            print(f"best_val_pearson   : {valid_p.loc[best_p_idx, 'val_pearson_corr']:.6f}")
            print(f"final_val_pearson  : {valid_p.iloc[-1]['val_pearson_corr']:.6f}")