import os
import csv
import glob
import math
import pickle
from pathlib import Path
from typing import Tuple, Dict, Any, List

import numpy as np
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader, random_split

# ===== CONFIG =====
ROOT_DIR = Path(r"D:\UNIVERSITY DOCUMENTS\BTU COTTBUS-SENFTENBERG\COURSES\Study Project\EPIC\EPIC")
IN_DIR   = ROOT_DIR / "outputs_task2"
OUT_DIR  = ROOT_DIR / "outputs_task3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEQ_LEN = 256
WINDOW_DEFAULT = 2
SCALER_DIR = OUT_DIR / "scalers"
SCALER_DIR.mkdir(parents=True, exist_ok=True)

# ---- helpers ----
def _npz_list(window_min: int) -> List[Path]:
    return sorted((IN_DIR).glob(f"flows_{window_min}m_scen*.npz"))

def _load_all_npz(window_min: int, seq_len: int) -> np.ndarray:
    """Load all scenarios' flows and return (N, 3, L) float32 array: [sizes, dirs, iats]."""
    files = _npz_list(window_min)
    if not files:
        raise FileNotFoundError(f"No NPZ files found for window {window_min} min in {IN_DIR}")
    chunks = []
    for f in files:
        d = np.load(f)
        sizes = d["sizes"].astype(np.float32)  # (N, L)
        dirs  = d["dirs"].astype(np.float32)   # (N, L)
        iats  = d["iats"].astype(np.float32)   # (N, L)
        assert sizes.shape[1] == seq_len, f"{f.name}: SEQ_LEN mismatch"
        X = np.stack([sizes, dirs, iats], axis=1)  # (N, 3, L)
        chunks.append(X)
    return np.concatenate(chunks, axis=0)  # (N_total, 3, L)

def _fit_or_load_scalers(X: np.ndarray, window_min: int, force_recompute: bool=False) -> Dict[str, Any]:
    """
    Fit scalers on sizes & iats across all samples/time; keep dirs unscaled.
    Returns dict with 'sizes' and 'iats' StandardScaler objects.
    """
    scaler_path = SCALER_DIR / f"feature_scaler_window_{window_min}m.joblib"
    if scaler_path.exists() and not force_recompute:
        return load(scaler_path)

    sizes = X[:, 0, :].reshape(-1, 1)  # (N*L,1)
    iats  = X[:, 2, :].reshape(-1, 1)

    sz_scaler = StandardScaler()
    ia_scaler = StandardScaler()
    sz_scaler.fit(sizes)
    ia_scaler.fit(iats)

    scalers = {"sizes": sz_scaler, "iats": ia_scaler}
    dump(scalers, scaler_path)
    return scalers

def _apply_scalers(X: np.ndarray, scalers: Dict[str, Any]) -> np.ndarray:
    Y = X.copy()
    Y[:, 0, :] = scalers["sizes"].transform(Y[:, 0, :].reshape(-1,1)).reshape(Y.shape[0], Y.shape[2])
    # dirs left as-is in channel 1 (index 1)
    Y[:, 2, :] = scalers["iats"].transform(Y[:, 2, :].reshape(-1,1)).reshape(Y.shape[0], Y.shape[2])
    return Y

class ICSFlowDataset(Dataset):
    def __init__(self, X: np.ndarray):
        """
        X: (N, 3, L), float32
        """
        self.X = X

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        # Return tensor float32
        return torch.from_numpy(self.X[idx])

def get_dataloaders(
    window_min: int = WINDOW_DEFAULT,
    batch_size: int = 128,
    val_frac: float = 0.1,
    seq_len: int = SEQ_LEN,
    seed: int = 42,
    force_recompute_scaler: bool = False,
) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
    """
    Loads NPZ flows -> scales channels (sizes & iats) -> returns train/val dataloaders and scalers.
    """
    X = _load_all_npz(window_min, seq_len)  # (N,3,L)
    scalers = _fit_or_load_scalers(X, window_min, force_recompute=force_recompute_scaler)
    Xs = _apply_scalers(X, scalers).astype(np.float32)

    ds = ICSFlowDataset(Xs)
    n_total = len(ds)
    n_val = int(round(n_total * val_frac))
    n_train = n_total - n_val
    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=g)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)
    return train_loader, val_loader, scalers

# ---- smoke test ----
if __name__ == "__main__":
    tr, va, scalers = get_dataloaders(window_min=2, batch_size=128)
    xb = next(iter(tr))  # (B, 3, L)
    print("[OK] Train batch:", tuple(xb.shape))
    print("    mean/std per channel (first batch):")
    for c, name in enumerate(["sizes(z)", "dirs(raw)", "iats(z)"]):
        ch = xb[:, c, :].numpy().ravel()
        print(f"    {name}: mean={ch.mean():.3f}, std={ch.std():.3f}")
