import os, sys, csv, math
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import inspect
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# ===== CONFIG =====
ROOT_DIR = Path(r"D:\UNIVERSITY DOCUMENTS\BTU COTTBUS-SENFTENBERG\COURSES\Study Project\EPIC\EPIC")
IN_DIR   = ROOT_DIR / "outputs_task2"
OUT_DIR  = ROOT_DIR / "outputs_task2" / "tsne"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOWS_MIN = [1, 2, 5]
SEQ_LEN = 256                     # must match previous steps
MAX_FLOWS_PER_SCENARIO = 600      # cap per scenario for t-SNE runtime
RND = 42

# Small tuning grids (kept modest for runtime)
PERPLEXITY_GRID = [10, 30, 50]
LR_GRID         = [100, 200, 500]
NITER_GRID      = [1000, 2000]

def load_npz_and_meta(window_min, scen_idx):
    npz_path = IN_DIR / f"flows_{window_min}m_scen{scen_idx}.npz"
    idx_csv  = IN_DIR / f"flows_{window_min}m_scen{scen_idx}_index.csv"
    if not npz_path.exists() or not idx_csv.exists():
        return None, None
    data = np.load(npz_path)
    meta = []
    with open(idx_csv, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            meta.append(row)
    return data, meta

def flatten(sizes, dirs, iats):
    # 3L vector
    return np.concatenate(
        [sizes.astype(np.float32), dirs.astype(np.float32), iats.astype(np.float32)],
        axis=0
    )

def sample_indices(n, k, seed=RND):
    if n <= k:
        return np.arange(n, dtype=int)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=k, replace=False))

def build_matrix_for_window(window_min):
    X_list = []
    y_scen = []
    for scen in range(1, 9):
        data, meta = load_npz_and_meta(window_min, scen)
        if data is None:
            continue
        n = data["sizes"].shape[0]
        idxs = sample_indices(n, MAX_FLOWS_PER_SCENARIO, seed=RND + scen)
        for i in idxs:
            v = flatten(data["sizes"][i], data["dirs"][i], data["iats"][i])
            X_list.append(v)
            y_scen.append(scen)
    if not X_list:
        return None, None
    X = np.stack(X_list, axis=0)
    y = np.array(y_scen, dtype=int)
    return X, y

def make_tsne(**kwargs):
    """Wrapper to handle both old and new sklearn versions."""
    sig = inspect.signature(TSNE.__init__)
    if "n_iter" not in sig.parameters and "max_iter" in sig.parameters:
        # rename for new sklearn
        if "n_iter" in kwargs:
            kwargs["max_iter"] = kwargs.pop("n_iter")
    return TSNE(**kwargs)

def tune_tsne(X, random_state=RND):
    best = {"kl": float("inf"), "perp": None, "lr": None, "niter": None, "embedding": None}
    for perp in PERPLEXITY_GRID:
        # Skip invalid perplexity if dataset too small
        if X.shape[0] <= 3 * perp:
            continue
        for lr in LR_GRID:
            for niter in NITER_GRID:
                tsne = make_tsne(
                    n_components=2,
                    perplexity=perp,
                    learning_rate=lr,
                    n_iter=niter,
                    init="pca",
                    random_state=random_state,
                    metric="euclidean",
                    verbose=0,
                    n_iter_without_progress=300,
                    method="barnes_hut",
                    angle=0.5,
                )
                Y = tsne.fit_transform(X)
                kl = getattr(tsne, "kl_divergence_", float("inf"))
                if kl < best["kl"]:
                    best.update({"kl": float(kl), "perp": perp, "lr": lr, "niter": niter, "embedding": Y})
    return best

def plot_tsne(Y, y, window_min, hparams):
    plt.figure(figsize=(8.5, 7))
    # simple color cycle; scenarios 1..8
    for scen in range(1, 9):
        mask = (y == scen)
        if not np.any(mask):
            continue
        plt.scatter(Y[mask, 0], Y[mask, 1], s=8, label=f"Scenario {scen}", alpha=0.75)
    plt.title(f"t-SNE (window={window_min} min)  perp={hparams['perp']} lr={hparams['lr']} it={hparams['niter']}  KL={hparams['kl']:.4f}")
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(markerscale=2, fontsize=8, ncol=2, frameon=True)
    out = OUT_DIR / f"tsne_{window_min}m.png"
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    plt.close()
    print(f"[OK] Saved: {out}")

def save_hparams(window_min, best):
    out = OUT_DIR / "tsne_hparams.csv"
    # append mode with header if new
    header = ["window_min", "perplexity", "learning_rate", "n_iter", "kl_divergence"]
    write_header = not out.exists()
    with open(out, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow([window_min, best["perp"], best["lr"], best["niter"], f"{best['kl']:.6f}"])
    print(f"[OK] Logged best hparams for {window_min}m")

def main():
    np.random.seed(RND)
    for wmin in WINDOWS_MIN:
        print(f"\n[INFO] Building matrix for window {wmin} min …")
        X, y = build_matrix_for_window(wmin)
        if X is None:
            print(f"[WARN] No data for window {wmin} min. Skipping.")
            continue

        # Standardize features (improves t-SNE stability)
        scaler = StandardScaler(with_mean=True, with_std=True)
        Xs = scaler.fit_transform(X)

        print("[INFO] Tuning t-SNE …")
        best = tune_tsne(Xs, random_state=RND + wmin)  # vary seed by window
        if best["embedding"] is None:
            print(f"[WARN] t-SNE failed to find config for window {wmin}. Skipping.")
            continue

        print(f"[OK] Best: perp={best['perp']}, lr={best['lr']}, niter={best['niter']}, KL={best['kl']:.6f}")
        save_hparams(wmin, best)
        plot_tsne(best["embedding"], y, wmin, best)

    print("\n[DONE] Task 2 Step 3: t-SNE plots + hyperparams logged.")

if __name__ == "__main__":
    main()
