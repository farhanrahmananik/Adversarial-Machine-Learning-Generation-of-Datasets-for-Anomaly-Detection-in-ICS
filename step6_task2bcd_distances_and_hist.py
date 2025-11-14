import os, sys, csv, math, shutil, subprocess
from pathlib import Path
from itertools import combinations
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

# ===== CONFIG =====
ROOT_DIR = Path(r"D:\UNIVERSITY DOCUMENTS\BTU COTTBUS-SENFTENBERG\COURSES\Study Project\EPIC\EPIC")
IN_DIR   = ROOT_DIR / "outputs_task2"
OUT_DIR  = ROOT_DIR / "outputs_task2" / "plots_dist"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOWS_MIN = [1, 2, 5]
SEQ_LEN = 256               # must match step5
MAX_FLOWS_PER_SCENARIO = 300   # cap to control O(N^2) memory/time
BINS = 60

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

def flatten_flow(sample):
    """Concatenate [sizes, dirs, iats] -> 3L vector."""
    sizes = sample['sizes'].astype(np.float64)
    dirs  = sample['dirs' ].astype(np.float64)
    iats  = sample['iats' ].astype(np.float64)
    return np.concatenate([sizes, dirs, iats], axis=0)

def manual_euclidean(a: np.ndarray, b: np.ndarray) -> float:
    # No numpy.linalg; implement explicitly
    diff = a - b
    return float(math.sqrt(float(np.sum(diff * diff))))

def sample_indices(n, k):
    if n <= k:
        return np.arange(n, dtype=int)
    rng = np.random.default_rng(42)
    return np.sort(rng.choice(n, size=k, replace=False))

def manual_histogram(values, bins):
    """Return (bin_edges, counts) where counts[i] corresponds to [edge[i], edge[i+1])."""
    if len(values) == 0:
        return np.array([]), np.array([])
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if vmax == vmin:
        vmax = vmin + 1e-9
    edges = np.linspace(vmin, vmax, bins + 1)
    counts = np.zeros(bins, dtype=int)
    for v in values:
        # last edge inclusive safeguard
        if v == edges[-1]:
            counts[-1] += 1
        else:
            # find bin index
            idx = int((v - vmin) / (vmax - vmin) * bins)
            if 0 <= idx < bins:
                counts[idx] += 1
    return edges, counts

def plot_overlay_histogram(per_scen_counts, edges, window_min):
    plt.figure(figsize=(9,6))
    width = (edges[1] - edges[0]) * 0.9
    centers = (edges[:-1] + edges[1:]) / 2.0

    for scen_idx, counts in sorted(per_scen_counts.items()):
        # normalize to probability density-like curve for fair overlay
        total = counts.sum()
        if total == 0:
            continue
        heights = counts / total
        plt.plot(centers, heights, label=f"Scenario {scen_idx}")

    plt.title(f"Euclidean Distance Distributions (window = {window_min} min)")
    plt.xlabel("Euclidean distance between flows (flattened [sizes, dirs, iats])")
    plt.ylabel("Relative frequency")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=8, ncol=2)
    out = OUT_DIR / f"dist_hist_overlay_{window_min}m.png"
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"[OK] Saved: {out}")

def save_summary_csv(window_min, per_scen_dists):
    out = OUT_DIR / f"dist_summary_{window_min}m.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scenario","N","mean","std","median","p10","p90","min","max"])
        for scen_idx, vals in sorted(per_scen_dists.items()):
            if len(vals)==0:
                w.writerow([scen_idx, 0,"","","","","","",""])
                continue
            arr = np.array(vals, dtype=float)
            w.writerow([
                scen_idx, len(arr),
                f"{arr.mean():.6f}",
                f"{arr.std(ddof=1):.6f}" if len(arr)>1 else "0.000000",
                f"{np.median(arr):.6f}",
                f"{np.percentile(arr,10):.6f}",
                f"{np.percentile(arr,90):.6f}",
                f"{arr.min():.6f}",
                f"{arr.max():.6f}",
            ])
    print(f"[OK] Saved: {out}")

def main():
    np.random.seed(123)
    for wmin in WINDOWS_MIN:
        print(f"\n[INFO] Window {wmin} min: loading flows per scenario…")
        per_scen_vectors = {}  # scen -> [vectors]
        for scen in range(1, 9):
            data, meta = load_npz_and_meta(wmin, scen)
            if data is None:
                print(f"[WARN] Scenario {scen}: missing NPZ/meta for {wmin}m")
                continue
            # choose a consistent subset (cap to MAX_FLOWS_PER_SCENARIO)
            n = data["sizes"].shape[0]
            idxs = sample_indices(n, MAX_FLOWS_PER_SCENARIO)
            vectors = []
            for i in idxs:
                sample = {"sizes": data["sizes"][i], "dirs": data["dirs"][i], "iats": data["iats"][i]}
                vec = flatten_flow(sample)
                vectors.append(vec)
            if vectors:
                per_scen_vectors[scen] = np.stack(vectors, axis=0)
            else:
                per_scen_vectors[scen] = np.empty((0, 3*SEQ_LEN), dtype=np.float64)

        # compute pairwise distances WITHIN each scenario (fulfills “between every two flows” per scenario)
        print("[INFO] Computing pairwise distances (manual)…")
        per_scen_dists = {}    # scen -> list of distances
        for scen, mat in per_scen_vectors.items():
            dists = []
            m = mat.shape[0]
            # pairwise combinations
            for i in range(m):
                vi = mat[i]
                for j in range(i+1, m):
                    vj = mat[j]
                    d = manual_euclidean(vi, vj)
                    dists.append(d)
            per_scen_dists[scen] = dists
            print(f"   Scenario {scen}: flows={m}, pairs={len(dists)}")

        # manual histograms per scenario; align to common edges using global min/max
        all_vals = np.concatenate([np.array(v, dtype=float) for v in per_scen_dists.values() if len(v)>0]) if any(len(v)>0 for v in per_scen_dists.values()) else np.array([])
        if all_vals.size == 0:
            print("[WARN] No distances computed for this window.")
            continue
        vmin, vmax = float(all_vals.min()), float(all_vals.max())
        if vmax == vmin:
            vmax = vmin + 1e-9
        common_edges = np.linspace(vmin, vmax, BINS + 1)

        per_scen_counts = {}
        for scen, vals in per_scen_dists.items():
            if len(vals)==0:
                per_scen_counts[scen] = np.zeros(BINS, dtype=int)
                continue
            # manual binning using common edges
            counts = np.zeros(BINS, dtype=int)
            for v in vals:
                if v == common_edges[-1]:
                    counts[-1] += 1
                else:
                    idx = int((v - vmin) / (vmax - vmin) * BINS)
                    if 0 <= idx < BINS:
                        counts[idx] += 1
            per_scen_counts[scen] = counts

        # save summary CSV + overlay histogram plot
        save_summary_csv(wmin, per_scen_dists)
        plot_overlay_histogram(per_scen_counts, common_edges, wmin)

    print("\n[DONE] Task 2 Step 2: distances + histograms complete.")

if __name__ == "__main__":
    main()
