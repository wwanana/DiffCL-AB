#!/usr/bin/env python3
import os

# Fix invalid OpenMP variables before importing torch.
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    try:
        ok = int(os.environ.get(name, "")) > 0
    except (TypeError, ValueError):
        ok = False
    if not ok:
        os.environ[name] = "1"

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_locate", default=None)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-5)
    return p.parse_args()


def valid_model_dir(path):
    if not path:
        return False
    p = Path(path).expanduser()
    if not p.is_dir() or not (p / "config.json").is_file():
        return False
    return (
        (p / "model.safetensors").is_file()
        or (p / "pytorch_model.bin").is_file()
        or any(p.glob("*.index.json"))
    )


def resolve_model_path(requested):
    candidates = [
        requested,
        os.environ.get("ESM2_MODEL_PATH"),
        "./model/esm2_650m",
        "/root/autodl-tmp/ProtAttBA-main/seq-identity_sig-mul/model/esm2_650m",
        "/root/autodl-tmp/ProtAttBA-main/model/esm2_650m",
    ]
    for path in candidates:
        if valid_model_dir(path):
            return str(Path(path).expanduser().resolve())

    checked = "\n".join(f"  - {x}" for x in candidates if x)
    raise FileNotFoundError(
        "No complete ESM2-650M model directory was found.\n"
        f"Checked:\n{checked}\n"
        "Use --model_locate /absolute/path, set ESM2_MODEL_PATH, "
        "or download the model into ./model/esm2_650m."
    )


def model_args(model_path, lr):
    # Includes fields used by the current and earlier project revisions.
    return SimpleNamespace(
        model_locate=model_path,
        hidden_size=1280,
        num_heads=4,
        dropout=0.1,
        out_dim=1,
        freeze_backbone=True,
        use_mamba_branch=True,
        use_contrastive=True,
        mamba_layers=2,
        mamba_d_model=1280,
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_dropout=0.1,
        proj_dim=256,
        contrastive_temperature=0.1,
        cl1_weight=0.03,
        cl2_weight=0.01,
        ag_consistency_inner_weight=0.1,
        use_diff_attention=True,
        use_state_branch=True,
        fusion_num_heads=4,
        fusion_hidden_dim=256,
        attention_pool_heads=4,
        lr=lr,
        loss="mse",
        weight_decay=1e-4,
        use_lr_scheduler=True,
        scheduler_patience=5,
        scheduler_factor=0.5,
        min_lr=1e-6,
    )


def encode(tokenizer, seqs, max_length, device):
    x = tokenizer(
        seqs,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return x["input_ids"].to(device), x["attention_mask"].to(device)


def aux_loss(outputs, name, device):
    value = outputs.get(name)
    if value is None:
        return torch.zeros((), device=device)
    return value.mean() if torch.is_tensor(value) else torch.tensor(float(value), device=device)


def main():
    a = parse_args()

    print("=" * 80)
    print("ENVIRONMENT CHECK")
    print("=" * 80)
    print("CURRENT_DIR      :", Path.cwd())
    print("OMP_NUM_THREADS  :", os.environ["OMP_NUM_THREADS"])
    print("PyTorch          :", torch.__version__)
    print("PyTorch CUDA     :", torch.version.cuda)
    print("CUDA available   :", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU is available. This project uses mamba-ssm/Triton CUDA kernels."
        )

    device = torch.device("cuda")
    print("GPU              :", torch.cuda.get_device_name(0))

    model_path = resolve_model_path(a.model_locate)
    print("MODEL_PATH       :", model_path)

    # Import only after confirming CUDA is available.
    from model import SeqBindModel

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = SeqBindModel(model_args(model_path, a.lr)).to(device)
    model.train()

    wt_ab = [
        "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWV",
        "DIQMTQSPSSLSASVGDRVTITCRASQGISNYLAWYQQKPGKAPKLLIY",
    ]
    mt_ab = [
        "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWA",
        "DIQMTQSPSSLSASVGDRVTITCRASQGISNYLAWYQQKPGKAPKLLVY",
    ]
    wt_ag = [
        "MKTIIALSYIFCLVFADYKDDDDKGGGSGGGSGGGS",
        "MNNNKDLIAYLKQATKAGVEVVNTDTPAQE",
    ]
    mt_ag = list(wt_ag)

    wt_ab_ids, wt_ab_mask = encode(tokenizer, wt_ab, a.max_length, device)
    mt_ab_ids, mt_ab_mask = encode(tokenizer, mt_ab, a.max_length, device)
    wt_ag_ids, wt_ag_mask = encode(tokenizer, wt_ag, a.max_length, device)
    mt_ag_ids, mt_ag_mask = encode(tokenizer, mt_ag, a.max_length, device)

    outputs = model(
        wt_ab_inputs_ids=wt_ab_ids,
        wt_ab_inputs_mask=wt_ab_mask,
        mut_ab_inputs_ids=mt_ab_ids,
        mt_ab_inputs_mask=mt_ab_mask,
        wt_ag_inputs_ids=wt_ag_ids,
        wt_ag_inputs_mask=wt_ag_mask,
        mut_ag_inputs_ids=mt_ag_ids,
        mt_ag_inputs_mask=mt_ag_mask,
        return_aux=True,
    )

    if isinstance(outputs, dict):
        pred = outputs["pred"].view(-1)
        cl1 = aux_loss(outputs, "cl1_loss", device)
        cl2 = aux_loss(outputs, "cl2_loss", device)
    else:
        pred = outputs.view(-1)
        cl1 = torch.zeros((), device=device)
        cl2 = torch.zeros((), device=device)

    labels = torch.tensor([0.25, -0.15], dtype=pred.dtype, device=device)
    reg = F.mse_loss(pred, labels)
    loss = reg + 0.03 * cl1 + 0.01 * cl2

    if not torch.isfinite(loss):
        raise FloatingPointError(f"Non-finite loss: {loss}")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-4)
    opt.zero_grad(set_to_none=True)
    loss.backward()

    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        raise RuntimeError("No trainable parameter received a gradient.")
    if not all(torch.isfinite(g).all().item() for g in grads):
        raise FloatingPointError("A non-finite gradient was found.")

    opt.step()
    torch.cuda.synchronize()

    print("\n" + "=" * 80)
    print("FORWARD / BACKWARD RESULT")
    print("=" * 80)
    print("prediction shape:", tuple(pred.shape))
    print("prediction      :", pred.detach().cpu().tolist())
    print("reg_loss        :", float(reg.detach().cpu()))
    print("cl1_loss        :", float(cl1.detach().cpu()))
    print("cl2_loss        :", float(cl2.detach().cpu()))
    print("total_loss      :", float(loss.detach().cpu()))
    print("gradient tensors:", len(grads))
    print("\nSUCCESS: forward, loss, backward, and optimizer step all passed.")


if __name__ == "__main__":
    main()