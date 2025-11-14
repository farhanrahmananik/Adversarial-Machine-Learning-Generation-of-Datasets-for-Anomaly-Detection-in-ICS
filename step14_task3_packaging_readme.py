import os
import platform
from pathlib import Path
from datetime import datetime
import subprocess

# ==== PATHS (edit ROOT_DIR if needed) ====
ROOT_DIR = Path(r"D:\UNIVERSITY DOCUMENTS\BTU COTTBUS-SENFTENBERG\COURSES\Study Project\EPIC\EPIC")
OUT3_DIR = ROOT_DIR / "outputs_task3"
RUNS_DIR = OUT3_DIR / "runs"
SAMPLES_DIR = OUT3_DIR / "samples"
SCALERS_DIR = OUT3_DIR / "scalers"
REPORT_DIR = OUT3_DIR / "reports"
README = OUT3_DIR / "README_Task3.md"

def tshark_version():
    for exe in ("tshark", r"C:\Program Files\Wireshark\tshark.exe", r"C:\Program Files (x86)\Wireshark\tshark.exe"):
        try:
            out = subprocess.check_output([exe, "-v"], stderr=subprocess.STDOUT, text=True)
            return out.splitlines()[0].strip()
        except Exception:
            pass
    return "TShark: (not queried)"

def list_files(base: Path, suffixes=(".pt", ".joblib", ".npz", ".csv", ".png")):
    if not base.exists():
        return []
    out = []
    for p in sorted(base.rglob("*")):
        if p.is_file() and p.suffix.lower() in suffixes:
            out.append(p)
    return out

def main():
    OUT3_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    SCALERS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    py  = f"Python {platform.python_version()} ({platform.python_implementation()})"
    ts  = tshark_version()

    # inventories
    run_files     = list_files(RUNS_DIR, suffixes=(".pt", ".csv"))
    sample_files  = list_files(SAMPLES_DIR, suffixes=(".npz", ".csv"))
    scaler_files  = list_files(SCALERS_DIR, suffixes=(".joblib",))
    report_files  = list_files(REPORT_DIR, suffixes=(".csv", ".png"))

    lines = []
    lines.append("# Task 3 – GAN for Synthetic Flow Generation (README)")
    lines.append("")
    lines.append(f"- Generated: {now}")
    lines.append(f"- Environment: {py}")
    lines.append(f"- {ts}")
    lines.append("")
    lines.append("## Structure")
    lines.append(f"- EPIC root: `{ROOT_DIR}`")
    lines.append(f"- Task 3 outputs: `{OUT3_DIR}`")
    lines.append(f"  - Trained models & logs: `{RUNS_DIR}`")
    lines.append(f"  - Generated samples: `{SAMPLES_DIR}`")
    lines.append(f"  - Feature scalers: `{SCALERS_DIR}`")
    lines.append(f"  - Quality reports: `{REPORT_DIR}`")
    lines.append("")
    lines.append("## How to Reproduce (Windows)")
    lines.append("1. Activate venv and install deps: `pip install -r requirements.txt`")
    lines.append("2. **Train** (default window=2m): `python step10_task3_train_gan.py`")
    lines.append("   - Checkpoint saved to `outputs_task3/runs/window_2m/ckpt_best.pt`")
    lines.append("3. **Generate synthetic flows**:")
    lines.append("   - Example: `python step11_task3_infer_generate.py --window 2 --num 512 --name synth_window2m_512`")
    lines.append("4. **Quality report (real vs synthetic)**:")
    lines.append("   - Example: `python step12_task3_quality_report.py --window 2 --name synth_window2m_512`")
    lines.append("")
    lines.append("## Files Produced")
    lines.append("### Runs (checkpoints & train logs)")
    if run_files:
        for p in run_files:
            lines.append(f"- `{p.relative_to(ROOT_DIR)}`")
    else:
        lines.append("- (no run files found)")
    lines.append("")
    lines.append("### Samples (synthetic flows and previews)")
    if sample_files:
        for p in sample_files:
            lines.append(f"- `{p.relative_to(ROOT_DIR)}`")
    else:
        lines.append("- (no sample files found)")
    lines.append("")
    lines.append("### Scalers (per window)")
    if scaler_files:
        for p in scaler_files:
            lines.append(f"- `{p.relative_to(ROOT_DIR)}`")
    else:
        lines.append("- (no scaler files found)")
    lines.append("")
    lines.append("### Quality Reports (CSV & PNG)")
    if report_files:
        for p in report_files:
            lines.append(f"- `{p.relative_to(ROOT_DIR)}`")
    else:
        lines.append("- (no report files found)")
    lines.append("")
    lines.append("## Script Map")
    lines.append("- `step8_task3_data_module.py`: loads real flows, scales sizes & iats, provides DataLoaders.")
    lines.append("- `step9_task3_models.py`: Discriminator (9×Conv + 4×FC) and Generator (3×FC + 8×Conv with 4×Upsample).")
    lines.append("- `step10_task3_train_gan.py`: training loop with BCE + feature-matching loss on D’s pre-FC features.")
    lines.append("- `step11_task3_infer_generate.py`: loads best checkpoint, generates N flows, inverse-scales & postprocesses.")
    lines.append("- `step12_task3_quality_report.py`: real-vs-synthetic stats & plots (size/iat histograms, direction frequencies).")
    lines.append("")
    lines.append("## Notes")
    lines.append("- Direction channel is discretized to {-1,0,1} in postprocessing.")
    lines.append("- Sizes and IATs are inverse-transformed and clamped to non-negative.")
    lines.append("- For 1m/5m windows, set `--window 1/5` in the infer/report scripts and adjust training window in the trainer.")
    lines.append("")

    README.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Wrote README: {README}")

if __name__ == "__main__":
    main()
