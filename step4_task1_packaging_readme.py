import os
import sys
import platform
import subprocess
from pathlib import Path
from datetime import datetime

# === CONFIG ===
ROOT_DIR = Path(r"D:\UNIVERSITY DOCUMENTS\BTU COTTBUS-SENFTENBERG\COURSES\Study Project\EPIC\EPIC")
OUT_DIR  = ROOT_DIR / "outputs_task1"
PLOTS_DIR = OUT_DIR / "plots_task1d"
README   = OUT_DIR / "README_Task1.md"

def tshark_version():
    for exe in ("tshark", r"C:\Program Files\Wireshark\tshark.exe", r"C:\Program Files (x86)\Wireshark\tshark.exe"):
        try:
            out = subprocess.check_output([exe, "-v"], stderr=subprocess.STDOUT, text=True)
            line = out.splitlines()[0].strip()
            return line
        except Exception:
            pass
    return "TShark: NOT FOUND"

def list_files(dirpath: Path, suffix=None):
    if not dirpath.exists():
        return []
    files = []
    for p in sorted(dirpath.rglob("*")):
        if p.is_file() and (suffix is None or p.suffix.lower() == suffix.lower()):
            files.append(p)
    return files

def scenario_exists(i: int):
    p = ROOT_DIR / f"Scenario {i}" / f"Scenario_{i}.pcapng"
    return p.exists()

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    py = f"Python {platform.python_version()} ({platform.python_implementation()})"
    ts = tshark_version()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Detect scenarios available
    scen_status = []
    for i in range(1, 9):
        scen_status.append((i, "OK" if scenario_exists(i) else "MISSING"))

    # Output inventory
    csvs = list_files(OUT_DIR, ".csv")
    pngs = list_files(PLOTS_DIR, ".png")

    # Build README
    lines = []
    lines.append(f"# Task 1 Packaging (EPIC) — README\n")
    lines.append(f"- Generated: {now}\n")
    lines.append(f"- Environment: {py}\n")
    lines.append(f"- {ts}\n")
    lines.append("")
    lines.append("## Structure")
    lines.append(f"- EPIC root: `{ROOT_DIR}`")
    lines.append(f"- Outputs: `{OUT_DIR}`")
    lines.append(f"- Plots (Task 1d): `{PLOTS_DIR}`\n")
    lines.append("## How to Reproduce (Windows)")
    lines.append("1. Install Wireshark (with **TShark**). Ensure `tshark` is on PATH.")
    lines.append("2. `pip install -r requirements.txt`")
    lines.append("3. Run: `run_all.bat` (executes Steps 0–3)")
    lines.append("")
    lines.append("## Scenario Availability")
    for i, st in scen_status:
        lines.append(f"- Scenario {i}: {st}")
    lines.append("")
    lines.append("## Generated CSV index")
    if not csvs:
        lines.append("- (no CSVs found)")
    else:
        for p in csvs:
            rel = p.relative_to(ROOT_DIR)
            lines.append(f"- `{rel}`")
    lines.append("")
    lines.append("## Generated Plot index (Task 1d)")
    if not pngs:
        lines.append("- (no plots found)")
    else:
        for p in pngs:
            rel = p.relative_to(ROOT_DIR)
            lines.append(f"- `{rel}`")
    lines.append("")
    lines.append("## Notes")
    lines.append("- `step1_task1a_protocol_stats.py`: protocol counts/fractions + app×transport.")
    lines.append("- `step2_task1bc_lengths_and_interarrivals.py`: lengths & inter-arrival statistics.")
    lines.append("- `step3_task1d_manual_cdf_plots.py`: manual CDFs (header & payload).")
    lines.append("- All computations are reproducible and scenario-scoped.\n")

    README.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Wrote README: {README}")

if __name__ == "__main__":
    main()
