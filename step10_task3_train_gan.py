import os
import csv
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from joblib import load

# local imports
from step8_task3_data_module import (
    ROOT_DIR, OUT_DIR, SCALER_DIR, get_dataloaders, SEQ_LEN
)
from step9_task3_models import Discriminator, Generator


# ===== CONFIG =====
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WINDOW_MIN = 2
BATCH_SIZE = 128
EPOCHS = 50
Z_DIM = 128
LR_G = 2e-4
LR_D = 2e-4
LAMBDA_FM = 10.0    # feature-matching weight
SAVE_EVERY = 5

RUN_DIR = OUT_DIR / f"runs/window_{WINDOW_MIN}m"
SAMPLE_DIR = OUT_DIR / f"samples/window_{WINDOW_MIN}m"
LOG_PATH = RUN_DIR / "train_log.csv"
CKPT_BEST = RUN_DIR / "ckpt_best.pt"

RUN_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)


# ===== Utilities =====
def save_log_header():
    if LOG_PATH.exists():
        return
    with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch","d_loss","g_loss","fm_loss","d_real","d_fake","val_d_acc"])

def append_log(epoch, d_loss, g_loss, fm_loss, d_real, d_fake, val_acc):
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([epoch, f"{d_loss:.6f}", f"{g_loss:.6f}", f"{fm_loss:.6f}",
                    f"{d_real:.6f}", f"{d_fake:.6f}", f"{val_acc:.6f}"])

def inverse_scale(samples: torch.Tensor, scalers: Dict[str, Any]) -> np.ndarray:
    """
    samples: (N, 3, L) tensor on CPU
    returns np.ndarray float32 (N, 3, L) with sizes/iat inverse transformed; dirs unchanged
    """
    x = samples.detach().cpu().numpy().astype(np.float32)
    N, C, L = x.shape
    # inverse transform sizes (ch 0) and iats (ch 2)
    sz = x[:, 0, :].reshape(-1, 1)
    ia = x[:, 2, :].reshape(-1, 1)
    sz = scalers["sizes"].inverse_transform(sz).reshape(N, L)
    ia = scalers["iats"].inverse_transform(ia).reshape(N, L)
    x[:, 0, :] = sz
    x[:, 2, :] = ia
    return x

def postprocess_dirs(x: np.ndarray) -> np.ndarray:
    """
    Map dir channel to {-1,0,1} by rounding tanh-like output.
    If your generator already outputs raw values, clamp then round.
    """
    dirs = np.clip(x[:, 1, :], -1.0, 1.0)
    dirs = np.rint(dirs)  # -> {-1,0,1}
    x[:, 1, :] = dirs
    return x

def clamp_nonneg(x: np.ndarray) -> np.ndarray:
    # sizes & iats must be non-negative
    x[:, 0, :] = np.clip(x[:, 0, :], 0, None)
    x[:, 2, :] = np.clip(x[:, 2, :], 0, None)
    return x

def save_samples_npz_csv(step_name: str, arr: np.ndarray, max_rows: int = 10):
    """
    arr: (N, 3, L) float32, already inverse-scaled, dirs discretized, non-neg clamped
    """
    npz_path = SAMPLE_DIR / f"synth_{step_name}.npz"
    np.savez_compressed(npz_path, sizes=arr[:,0,:], dirs=arr[:,1,:], iats=arr[:,2,:])

    # CSV preview (first max_rows samples, first 16 timesteps)
    csv_path = SAMPLE_DIR / f"synth_{step_name}_preview.csv"
    T = min(arr.shape[2], 16)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["sample_id","channel","t0..t{}".format(T-1)]
        w.writerow(header)
        for i in range(min(max_rows, arr.shape[0])):
            for ch, name in enumerate(["size","dir","iat"]):
                row = [i, name] + [f"{arr[i,ch,t]:.4f}" for t in range(T)]
                w.writerow(row)

    print(f"[SAVED] {npz_path.name}, {csv_path.name}")


# ===== Losses =====
bce = nn.BCEWithLogitsLoss()

def d_step(D: Discriminator, G: Generator, x_real: torch.Tensor, z: torch.Tensor, optD):
    """
    Discriminator update: maximize log(D(x_real)) + log(1 - D(G(z)))
    """
    D.train()
    G.eval()

    optD.zero_grad(set_to_none=True)

    out_real = D(x_real)
    d_real = out_real["logit"]
    y_real = torch.ones_like(d_real)
    loss_real = bce(d_real, y_real)

    with torch.no_grad():
        x_fake = G(z)
    out_fake = D(x_fake.detach())
    d_fake = out_fake["logit"]
    y_fake = torch.zeros_like(d_fake)
    loss_fake = bce(d_fake, y_fake)

    d_loss = loss_real + loss_fake
    d_loss.backward()
    optD.step()

    return d_loss.item(), torch.sigmoid(d_real).mean().item(), torch.sigmoid(d_fake).mean().item()

def g_step(D: Discriminator, G: Generator, z: torch.Tensor, x_real: torch.Tensor, optG):
    """
    Generator update: minimize BCE(D(G(z)), 1) + λ * L1(mean(f_real) - mean(f_fake))
    where f_* are features_before_fc from D.
    """
    D.train()
    G.train()

    optG.zero_grad(set_to_none=True)

    x_fake = G(z)
    out_fake = D(x_fake)
    g_adv = bce(out_fake["logit"], torch.ones_like(out_fake["logit"]))

    with torch.no_grad():
        out_real = D(x_real)
    f_real = out_real["features_before_fc"].mean(dim=0)
    f_fake = out_fake["features_before_fc"].mean(dim=0)

    fm_loss = torch.mean(torch.abs(f_real - f_fake))
    g_loss = g_adv + LAMBDA_FM * fm_loss

    g_loss.backward()
    optG.step()

    return g_loss.item(), fm_loss.item()


# ===== Validation =====
@torch.no_grad()
def validate_d_accuracy(D: Discriminator, val_loader, G: Generator) -> float:
    D.eval()
    G.eval()
    n_correct = 0
    n_total = 0
    for xb in val_loader:
        xb = xb.to(DEVICE)
        B = xb.size(0)

        # generate fake batch
        z = torch.randn(B, Z_DIM, device=DEVICE)
        fake = G(z)

        # real predicted as real (>=0.5), fake predicted as fake (<0.5)
        prob_real = D(xb)["prob"]
        prob_fake = D(fake)["prob"]

        preds_real = (prob_real >= 0.5).sum().item()
        preds_fake = (prob_fake  <  0.5).sum().item()

        n_correct += preds_real + preds_fake
        n_total   += 2 * B

    return n_correct / max(1, n_total)



# ===== Main =====
def main():
    print(f"[INFO] Device: {DEVICE}")
    save_log_header()

    # Data
    train_loader, val_loader, scalers = get_dataloaders(window_min=WINDOW_MIN, batch_size=BATCH_SIZE)

    # Models
    D = Discriminator(seq_len=SEQ_LEN).to(DEVICE)
    G = Generator(seq_len=SEQ_LEN, z_dim=Z_DIM).to(DEVICE)

    # Optims
    optD = optim.Adam(D.parameters(), lr=LR_D, betas=(0.5, 0.999))
    optG = optim.Adam(G.parameters(), lr=LR_G, betas=(0.5, 0.999))

    best_val = 0.0  # we want D accuracy around 0.5; use closeness metric
    for epoch in range(1, EPOCHS + 1):
        D.train(); G.train()
        sum_d, sum_g, sum_fm = 0.0, 0.0, 0.0
        sum_dr, sum_df, steps = 0.0, 0.0, 0

        for xb in train_loader:
            xb = xb.to(DEVICE)
            z  = torch.randn(xb.size(0), Z_DIM, device=DEVICE)

            # 1) D step
            d_loss, d_real, d_fake = d_step(D, G, xb, z, optD)

            # 2) G step
            z2 = torch.randn(xb.size(0), Z_DIM, device=DEVICE)
            g_loss, fm_loss = g_step(D, G, z2, xb, optG)

            sum_d += d_loss
            sum_g += g_loss
            sum_fm += fm_loss
            sum_dr += d_real
            sum_df += d_fake
            steps  += 1

        # Validation: discriminator accuracy on mixed real/fake
        val_acc = validate_d_accuracy(D, val_loader, G)

        d_loss_ep = sum_d / steps
        g_loss_ep = sum_g / steps
        fm_loss_ep = sum_fm / steps
        d_real_ep = sum_dr / steps
        d_fake_ep = sum_df / steps
        print(f"[E{epoch:03d}] D={d_loss_ep:.4f}  G={g_loss_ep:.4f}  FM={fm_loss_ep:.4f}  "
              f"D(real)={d_real_ep:.3f}  D(fake)={d_fake_ep:.3f}  ValAcc={val_acc:.3f}")
        append_log(epoch, d_loss_ep, g_loss_ep, fm_loss_ep, d_real_ep, d_fake_ep, val_acc)

        # Save periodic samples
        if epoch % SAVE_EVERY == 0 or epoch == EPOCHS:
            with torch.no_grad():
                fake = G.sample(64, device=DEVICE).cpu()
            # inverse scaling, discretize dirs, clamp
            arr = inverse_scale(fake, scalers)
            arr = postprocess_dirs(arr)
            arr = clamp_nonneg(arr)
            save_samples_npz_csv(f"epoch{epoch:03d}", arr)

        # Track "best" as nearest to 0.5 discriminator accuracy
        closeness = 1.0 - abs(val_acc - 0.5) * 2.0  # 1.0 when exactly 0.5; 0 when 0 or 1
        if closeness > best_val:
            best_val = closeness
            torch.save({
                "epoch": epoch,
                "G": G.state_dict(),
                "D": D.state_dict(),
                "optG": optG.state_dict(),
                "optD": optD.state_dict(),
                "val_acc": val_acc
            }, CKPT_BEST)
            print(f"[SAVE] New best checkpoint at epoch {epoch} (ValAcc={val_acc:.3f})")

    print(f"[DONE] Training finished. Best checkpoint: {CKPT_BEST}")


if __name__ == "__main__":
    main()
