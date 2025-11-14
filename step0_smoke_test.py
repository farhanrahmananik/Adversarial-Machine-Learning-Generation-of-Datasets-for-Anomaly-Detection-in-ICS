import os
import sys
import shutil
import subprocess
from pathlib import Path

# ==== CONFIG: set your EPIC root folder (keep raw string r'...') ====
ROOT_DIR = Path(r"D:\UNIVERSITY DOCUMENTS\BTU COTTBUS-SENFTENBERG\COURSES\Study Project\EPIC\EPIC")

# Expected structure: EPIC\Scenario X\Scenario_X.pcapng for X in 1..8

def find_tshark() -> str:
    # 1) PATH
    p = shutil.which("tshark")
    if p:
        return p
    # 2) Common Windows install locations
    candidates = [
        r"C:\Program Files\Wireshark\tshark.exe",
        r"C:\Program Files (x86)\Wireshark\tshark.exe"
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    print("[ERROR] TShark not found. Please install Wireshark or add tshark to PATH.")
    sys.exit(1)

def scenario_files(root: Path):
    files = []
    for i in range(1, 9):
        scen_dir = root / f"Scenario {i}"
        f = scen_dir / f"Scenario_{i}.pcapng"
        files.append((i, f))
    return files

def count_packets_with_tshark(tshark_path: str, pcap_path: Path) -> int:
    # Count by streaming frame.number field lines from tshark
    # This avoids loading the whole capture into Python.
    cmd = [tshark_path, "-r", str(pcap_path), "-T", "fields", "-e", "frame.number"]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        count = 0
        for _ in proc.stdout:
            count += 1
        proc.wait()
        if proc.returncode != 0:
            err = proc.stderr.read().decode(errors="ignore")
            raise RuntimeError(f"tshark error ({proc.returncode}): {err[:500]}")
        return count
    except FileNotFoundError:
        raise RuntimeError("tshark not found. Check installation.")
    except Exception as e:
        raise

def main():
    tshark = find_tshark()
    print(f"[OK] Using TShark at: {tshark}")

    scen_files = scenario_files(ROOT_DIR)

    any_found = False
    print("\n--- Smoke Test: Packet Counts per Scenario ---")
    for idx, f in scen_files:
        if not f.exists():
            print(f"Scenario {idx}: MISSING -> {f}")
            continue
        any_found = True
        try:
            cnt = count_packets_with_tshark(tshark, f)
            print(f"Scenario {idx}: {cnt:,} packets  [{f}]")
        except Exception as e:
            print(f"Scenario {idx}: ERROR while counting -> {e}")

    if not any_found:
        print("\n[ERROR] No scenario files found. Check ROOT_DIR and folder structure.")
        sys.exit(2)

    print("\n[DONE] Smoke test finished.")

if __name__ == "__main__":
    main()
