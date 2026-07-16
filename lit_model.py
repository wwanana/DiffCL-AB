import torch
from torch import nn

from model import SeqBindModel
from pytorch_lightning import LightningModule

from torchmetrics import (
    MeanSquaredError,
    SpearmanCorrCoef,
    PearsonCorrCoef,
    R2Score,
    MeanAbsoluteError,
)

from collections import OrderedDict


class LitModel(LightningModule):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.lr = args.lr

        # 对比学习损失权重
        self.cl1_weight = getattr(args, "cl1_weight", 0.03)
        self.cl2_weight = getattr(args, "cl2_weight", 0.01)

        # 主模型
        self.model = SeqBindModel(args)

        # =========================
        # Validation metrics
        # =========================
        self.valid_metrics = nn.ModuleDict(
            OrderedDict(
                [
                    ("val_mse", MeanSquaredError()),
                    ("val_spearman_corr", SpearmanCorrCoef()),
                    ("val_pearson_corr", PearsonCorrCoef()),
                    ("val_r2", R2Score()),
                    ("val_mae", MeanAbsoluteError()),
                ]
            )
        )

        # =========================
        # Test metrics
        # =========================
        self.test_metrics = nn.ModuleDict(
            OrderedDict(
                [
                    ("test_mse", MeanSquaredError()),
                    ("test_spearman_corr", SpearmanCorrCoef()),
                    ("test_pearson_corr", PearsonCorrCoef()),
                    ("test_r2", R2Score()),
                    ("test_mae", MeanAbsoluteError()),
                ]
            )
        )

        # =========================
        # Loss function
        # =========================
        if args.loss == "mse":
            self.criterion = nn.MSELoss()
        elif args.loss == "huber":
            self.criterion = nn.HuberLoss()
        else:
            raise ValueError(f"Invalid loss function: {args.loss}")

        # 不直接保存 argparse.Namespace，避免 PyTorch 2.6+ checkpoint 加载问题
        self.save_hyperparameters(ignore=["args"])

    def training_step(self, batch, batch_idx):
        labels = batch["labels"].view(-1)

        outs = self.model(
            wt_ab_inputs_ids=batch["wt_ab_inputs_ids"],
            wt_ab_inputs_mask=batch["wt_ab_inputs_mask"],
            mut_ab_inputs_ids=batch["mut_ab_inputs_ids"],
            mt_ab_inputs_mask=batch["mt_ab_inputs_mask"],
            wt_ag_inputs_ids=batch["wt_ag_inputs_ids"],
            wt_ag_inputs_mask=batch["wt_ag_inputs_mask"],
            mut_ag_inputs_ids=batch["mut_ag_inputs_ids"],
            mt_ag_inputs_mask=batch["mt_ag_inputs_mask"],
            return_aux=True,
        )

        preds = outs["pred"].view(-1)

        reg_loss = self.criterion(preds, labels)
        cl1_loss = outs["cl1_loss"]
        cl2_loss = outs["cl2_loss"]

        total_loss = (
            reg_loss
            + self.cl1_weight * cl1_loss
            + self.cl2_weight * cl2_loss
        )

        self.log(
            "train_reg_loss",
            reg_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        self.log(
            "train_cl1_loss",
            cl1_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        self.log(
            "train_cl2_loss",
            cl2_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        self.log(
            "loss",
            total_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return total_loss

    def validation_step(self, batch, batch_idx):
        labels = batch["labels"].view(-1)

        preds = self.model(
            wt_ab_inputs_ids=batch["wt_ab_inputs_ids"],
            wt_ab_inputs_mask=batch["wt_ab_inputs_mask"],
            mut_ab_inputs_ids=batch["mut_ab_inputs_ids"],
            mt_ab_inputs_mask=batch["mt_ab_inputs_mask"],
            wt_ag_inputs_ids=batch["wt_ag_inputs_ids"],
            wt_ag_inputs_mask=batch["wt_ag_inputs_mask"],
            mut_ag_inputs_ids=batch["mut_ag_inputs_ids"],
            mt_ag_inputs_mask=batch["mt_ag_inputs_mask"],
            return_aux=False,
        ).view(-1)

        val_loss = self.criterion(preds, labels)

        self.log(
            "val_loss",
            val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        # 注意：这里使用 update，而不是 metric(preds, labels)
        # 避免 R2Score 在 batch size = 1 时直接 compute 报错
        for name, metric in self.valid_metrics.items():
            metric.update(preds, labels)
            self.log(
                name,
                metric,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

    def test_step(self, batch, batch_idx):
        labels = batch["labels"].view(-1)

        preds = self.model(
            wt_ab_inputs_ids=batch["wt_ab_inputs_ids"],
            wt_ab_inputs_mask=batch["wt_ab_inputs_mask"],
            mut_ab_inputs_ids=batch["mut_ab_inputs_ids"],
            mt_ab_inputs_mask=batch["mt_ab_inputs_mask"],
            wt_ag_inputs_ids=batch["wt_ag_inputs_ids"],
            wt_ag_inputs_mask=batch["wt_ag_inputs_mask"],
            mut_ag_inputs_ids=batch["mut_ag_inputs_ids"],
            mt_ag_inputs_mask=batch["mt_ag_inputs_mask"],
            return_aux=False,
        ).view(-1)

        test_loss = self.criterion(preds, labels)

        self.log(
            "test_loss",
            test_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        # 注意：这里使用 update，而不是 metric(preds, labels)
        # 这样最后一个 batch 即使只有 1 个样本，也不会在 batch 内计算 R2
        for name, metric in self.test_metrics.items():
            metric.update(preds, labels)
            self.log(
                name,
                metric,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.lr,
            weight_decay=getattr(self.args, "weight_decay", 1e-4),
        )

        use_lr_scheduler = getattr(self.args, "use_lr_scheduler", True)

        if not use_lr_scheduler:
            return optimizer

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=getattr(self.args, "scheduler_factor", 0.5),
            patience=getattr(self.args, "scheduler_patience", 5),
            min_lr=getattr(self.args, "min_lr", 1e-6),
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_mse",
                "interval": "epoch",
                "frequency": 1,
            },
        }