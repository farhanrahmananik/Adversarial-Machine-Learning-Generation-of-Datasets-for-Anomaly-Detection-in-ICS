import os
import platform
from pathlib import Path
from datetime import datetime
import subprocess

# ==== PATHS (edit ROOT_DIR if needed) ====
ROOT_DIR = Path(r"D:\UNIVERSITY DOCUMENTS\BTU COTTBUS-SENFTENBERG\COURSES\Study Project\EPIC\EPIC")
OUT2_DIR = ROOT_DIR / "outputs_task2"
README   = OUT2_DIR / "README_Task2.md"

# Subfolders we expect from Task 2 pipeline
PLOTS_DIST_DIR = OUT2_DIR / "plots_dist"
TSNE_DIR       = OUT2_DIR / "tsne"

def tshark_version():
    for exe in ("tshark", r"C:\Program Files\Wireshark\tshark.exe", r"C:\Program Files (x86)\Wireshark\tshark.exe"):
        try:
            out = subprocess.check_output([exe, "-v"], stderr=subprocess.STDOUT, text=True)
            return out.splitlines()[0].strip()
        except Exception:
            pass
    return "TShark: (not queried)"

def list_files(base: Path, suffixes=(".png", ".csv", ".npz")):
    if not base.exists():
        return []
    out = []
    for p in sorted(base.rglob("*")):
        if p.is_file() and p.suffix.lower() in suffixes:
            out.append(p)
    return out

def flows_inventory():
    """Collect flow NPZ + index CSVs produced in step5 (per scenario & window)."""
    npzs   = sorted(OUT2_DIR.glob("flows_*m_scen*.npz"))
    idxcsv = sorted(OUT2_DIR.glob("flows_*m_scen*_index.csv"))
    return npzs, idxcsv

def main():
    OUT2_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIST_DIR.mkdir(parents=True, exist_ok=True)
    TSNE_DIR.mkdir(parents=True, exist_ok=True)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    py  = f"Python {platform.python_version()} ({platform.python_implementation()})"
    ts  = tshark_version()

    # Inventories
    flow_npz, flow_idx = flows_inventory()
    dist_png_csv = list_files(PLOTS_DIST_DIR, suffixes=(".png", ".csv"))
    tsne_png = list_files(TSNE_DIR, suffixes=(".png",))
    tsne_hp  = list_files(TSNE_DIR, suffixes=(".csv",))

    lines = []
    lines.append("# Task 2 – Similarity Between Scenarios (README)")
    lines.append("")
    lines.append(f"- Generated: {now}")
    lines.append(f"- Environment: {py}")
    lines.append(f"- {ts}")
    lines.append("")
    lines.append("## Structure")
    lines.append(f"- EPIC root: `{ROOT_DIR}`")
    lines.append(f"- Task 2 outputs: `{OUT2_DIR}`")
    lines.append(f"  - Distance histograms: `{PLOTS_DIST_DIR}`")
    lines.append(f"  - t-SNE: `{TSNE_DIR}`")
    lines.append("")
    lines.append("## How to Reproduce (Windows)")
    lines.append("1. Ensure Task 1 preprocessing is done (not strictly required for Task 2).")
    lines.append("2. Install deps (inside your venv): `pip install -r requirements.txt`")
    lines.append("3. Run steps in order:")
    lines.append("   - `python step5_task2a_build_flows.py`")
    lines.append("   - `python step6_task2bcd_distances_and_hist.py`")
    lines.append("   - `python step7_task2d_tsne.py`")
    lines.append("")
    lines.append("## Outputs Index")
    lines.append("### Flow tensors (per scenario & window)")
    if flow_npz:
        for p in flow_npz:
            lines.append(f"- `{p.relative_to(ROOT_DIR)}`")
    else:
        lines.append("- (no `.npz` flow files found)")
    lines.append("")
    lines.append("### Flow indices CSV (metadata for flows)")
    if flow_idx:
        for p in flow_idx:
            lines.append(f"- `{p.relative_to(ROOT_DIR)}`")
    else:
        lines.append("- (no index CSVs found)")
    lines.append("")
    lines.append("### Distance histograms & summaries")
    if dist_png_csv:
        for p in dist_png_csv:
            lines.append(f"- `{p.relative_to(ROOT_DIR)}`")
    else:
        lines.append("- (no distance plots/summaries found)")
    lines.append("")
    lines.append("### t-SNE plots & tuned hyperparameters")
    if tsne_png:
        for p in tsne_png:
            lines.append(f"- `{p.relative_to(ROOT_DIR)}`")
    else:
        lines.append("- (no t-SNE plots found)")
    if tsne_hp:
        for p in tsne_hp:
            lines.append(f"- `{p.relative_to(ROOT_DIR)}`")
    else:
        lines.append("- (no t-SNE hyperparameter CSV found)")
    lines.append("")
    lines.append("## Notes")
    lines.append("- **step5_task2a_build_flows.py**: builds fixed-length flow tensors (sizes, dirs, iats).")
    lines.append("- **step6_task2bcd_distances_and_hist.py**: manual Euclidean distances + manual histograms per window.")
    lines.append("- **step7_task2d_tsne.py**: t-SNE with basic hyperparameter tuning; plots per window (1/2/5 min).")
    lines.append("- All histograms and t-SNE plots are saved under Task 2 output folders.")
    lines.append("")

    README.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Wrote README: {README}")

if __name__ == "__main__":
    main()
