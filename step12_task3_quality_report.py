import argparse
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

# local
from step8_task3_data_module import IN_DIR, OUT_DIR, SEQ_LEN

REPORT_DIR = OUT_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

def load_real(window_min: int):
    xs, xd, xi = [], [], []
    for p in sorted((IN_DIR).glob(f"flows_{window_min}m_scen*.npz")):
        d = np.load(p)
        xs.append(d["sizes"]); xd.append(d["dirs"]); xi.append(d["iats"])
    Xs = np.concatenate(xs, axis=0)  # (N,L)
    Xd = np.concatenate(xd, axis=0)
    Xi = np.concatenate(xi, axis=0)
    return Xs, Xd, Xi

def load_synth(window_min: int, name: str):
    p = OUT_DIR / f"samples/window_{window_min}m/{name}.npz"
    d = np.load(p)
    return d["sizes"], d["dirs"], d["iats"]

def summarize(arr):
    a = arr.reshape(-1)
    return dict(
        N=a.size,
        mean=float(a.mean()),
        std=float(a.std(ddof=1) if a.size>1 else 0.0),
        p10=float(np.percentile(a,10)),
        p50=float(np.percentile(a,50)),
        p90=float(np.percentile(a,90)),
        min=float(a.min()),
        max=float(a.max()),
    )

def write_summary_csv(window_min, name, stats_real, stats_synth):
    out = REPORT_DIR / f"quality_window{window_min}m_{name}.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["channel","dataset","N","mean","std","p10","p50","p90","min","max"])
        for ch in ["size","dir","iat"]:
            r = stats_real[ch]; s = stats_synth[ch]
            w.writerow([ch,"real", r["N"], f"{r['mean']:.6f}", f"{r['std']:.6f}", f"{r['p10']:.6f}", f"{r['p50']:.6f}", f"{r['p90']:.6f}", f"{r['min']:.6f}", f"{r['max']:.6f}"])
            w.writerow([ch,"synth",s["N"], f"{s['mean']:.6f}", f"{s['std']:.6f}", f"{s['p10']:.6f}", f"{s['p50']:.6f}", f"{s['p90']:.6f}", f"{s['min']:.6f}", f"{s['max']:.6f}"])
    print(f"[OK] Summary CSV -> {out}")

def manual_hist_overlay(real_vals, synth_vals, title, outpath, bins=80):
    plt.figure(figsize=(8,6))
    r = real_vals.reshape(-1); s = synth_vals.reshape(-1)
    vmin = float(min(r.min(), s.min())); vmax = float(max(r.max(), s.max()))
    if vmax == vmin: vmax = vmin + 1e-9
    edges = np.linspace(vmin, vmax, bins+1)

    # counts -> normalized heights (relative frequency)
    r_counts = np.zeros(bins, dtype=float)
    s_counts = np.zeros(bins, dtype=float)
    for v in r:
        j = int((v - vmin)/(vmax - vmin)*bins)
        if j == bins: j -= 1
        r_counts[j] += 1
    for v in s:
        j = int((v - vmin)/(vmax - vmin)*bins)
        if j == bins: j -= 1
        s_counts[j] += 1
    r_heights = r_counts / max(1.0, r_counts.sum())
    s_heights = s_counts / max(1.0, s_counts.sum())
    centers = (edges[:-1] + edges[1:]) / 2.0

    plt.plot(centers, r_heights, label="real")
    plt.plot(centers, s_heights, label="synthetic")
    plt.title(title)
    plt.xlabel("value")
    plt.ylabel("relative frequency")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
    print(f"[OK] Plot -> {outpath}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=2, choices=[1,2,5])
    ap.add_argument("--name", required=True, help="synthetic file base name (without .npz), e.g., synth_demo")
    args = ap.parse_args()

    # load
    rs, rd, ri = load_real(args.window)
    ss, sd, si = load_synth(args.window, args.name)

    # stats
    stats_real = {
        "size": summarize(rs),
        "dir":  summarize(rd),
        "iat":  summarize(ri),
    }
    stats_synth = {
        "size": summarize(ss),
        "dir":  summarize(sd),
        "iat":  summarize(si),
    }
    write_summary_csv(args.window, args.name, stats_real, stats_synth)

    # plots
    manual_hist_overlay(rs, ss, f"Size distribution (window={args.window}m)", REPORT_DIR / f"size_hist_window{args.window}m_{args.name}.png")
    manual_hist_overlay(ri, si, f"IAT distribution (window={args.window}m)",  REPORT_DIR / f"iat_hist_window{args.window}m_{args.name}.png")
    # for direction, show discrete bar overlay
    # build -1,0,1 frequencies
    def dir_freq(arr):
        a = arr.reshape(-1)
        vals = [-1,0,1]
        return np.array([np.sum(a==v) for v in vals], dtype=float), vals

    r_freq, vals = dir_freq(rd)
    s_freq, _    = dir_freq(sd)
    r_freq /= max(1.0, r_freq.sum())
    s_freq /= max(1.0, s_freq.sum())

    plt.figure(figsize=(7,5))
    idx = np.arange(len(vals))
    width = 0.35
    plt.bar(idx - width/2, r_freq, width, label="real")
    plt.bar(idx + width/2, s_freq, width, label="synthetic")
    plt.xticks(idx, [str(v) for v in vals])
    plt.title(f"Direction frequencies (window={args.window}m)")
    plt.xlabel("dir value")
    plt.ylabel("relative frequency")
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    out = REPORT_DIR / f"dir_freq_window{args.window}m_{args.name}.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"[OK] Plot -> {out}")

if __name__ == "__main__":
    main()
