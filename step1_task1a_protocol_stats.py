import csv
import os
import sys
import shutil
import subprocess
from pathlib import Path
from collections import Counter, defaultdict

# ====== CONFIG (edit to your path) ======
ROOT_DIR = Path(r"D:\UNIVERSITY DOCUMENTS\BTU COTTBUS-SENFTENBERG\COURSES\Study Project\EPIC\EPIC")
OUT_DIR  = ROOT_DIR / "outputs_task1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Layers considered "non-application" (remove/extend as needed)
NON_APP_LAYERS = {
    "frame", "sll", "eth", "ethertype", "vlan",
    "arp",
    "ip", "ipv6",
    "tcp", "udp", "icmp", "icmpv6", "sctp",
    "ssl", "tls", "quic",  # treat these as transport/security layers; app is above
}

TRANSPORT_LAYERS = {"tcp", "udp", "icmp", "icmpv6", "sctp"}


def find_tshark() -> str:
    p = shutil.which("tshark")
    if p:
        return p
    candidates = [
        r"C:\Program Files\Wireshark\tshark.exe",
        r"C:\Program Files (x86)\Wireshark\tshark.exe",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    print("[ERROR] TShark not found. Install Wireshark or add tshark to PATH.")
    sys.exit(1)


def scenario_files(root: Path):
    files = []
    for i in range(1, 9):
        scen_dir = root / f"Scenario {i}"
        f = scen_dir / f"Scenario_{i}.pcapng"
        files.append((i, f))
    return files


def parse_protocol_chain(proto_chain: str):
    """
    proto_chain example: 'eth:ip:tcp:modbus'
    We define:
      - transport = last transport present in the chain (from TRANSPORT_LAYERS)
      - application = last layer that is not in NON_APP_LAYERS; if none -> 'NONE'
    """
    # sometimes empty/malformed
    if not proto_chain:
        return ("NONE", "NONE")

    parts = [p.strip().lower() for p in proto_chain.split(":") if p.strip()]
    transport = "NONE"
    application = "NONE"

    # transport = last occurrence of a known transport layer
    for p in parts:
        if p in TRANSPORT_LAYERS:
            transport = p

    # application = last layer that is not in NON_APP_LAYERS
    for p in reversed(parts):
        if p not in NON_APP_LAYERS:
            application = p
            break

    return (transport, application)


def scan_file(tshark: str, pcap_path: Path):
    """
    Stream 'frame.protocols' for each packet and compute counters.
    Returns:
      total_packets, transport_counter, app_counter, app_by_transport_counter
    """
    cmd = [tshark, "-r", str(pcap_path), "-T", "fields", "-e", "frame.protocols"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="ignore")

    total = 0
    transport_ctr = Counter()
    app_ctr = Counter()
    app_by_transport = Counter()

    for line in proc.stdout:
        total += 1
        chain = line.strip()
        transport, application = parse_protocol_chain(chain)
        transport_ctr[transport] += 1
        app_ctr[application] += 1
        app_by_transport[(application, transport)] += 1

    proc.wait()
    if proc.returncode != 0:
        err = proc.stderr.read()
        raise RuntimeError(f"tshark error ({proc.returncode}): {err[:500]}")

    return total, transport_ctr, app_ctr, app_by_transport


def write_csv_protocols(scenario_idx, total, app_ctr, out_path):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "application_protocol", "count", "fraction_of_scenario"])
        for app, c in sorted(app_ctr.items(), key=lambda x: (-x[1], x[0])):
            frac = c / total if total > 0 else 0.0
            w.writerow([scenario_idx, app, c, f"{frac:.6f}"])


def write_csv_transports(scenario_idx, total, transport_ctr, out_path):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "transport_protocol", "count", "fraction_of_scenario"])
        for tr, c in sorted(transport_ctr.items(), key=lambda x: (-x[1], x[0])):
            frac = c / total if total > 0 else 0.0
            w.writerow([scenario_idx, tr, c, f"{frac:.6f}"])


def write_csv_app_by_transport(scenario_idx, total, app_by_transport, out_path):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "scenario", "application_protocol", "transport_protocol",
            "count", "fraction_of_scenario"
        ])
        for (app, tr), c in sorted(app_by_transport.items(), key=lambda x: (-x[1], x[0][0], x[0][1])):
            frac = c / total if total > 0 else 0.0
            w.writerow([scenario_idx, app, tr, c, f"{frac:.6f}"])


def append_master_totals(rows, out_path):
    # rows: list of (scenario, total_packets)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "total_packets"])
        for s, t in rows:
            w.writerow([s, t])


def main():
    tshark = find_tshark()
    scen_files = scenario_files(ROOT_DIR)

    master_totals = []

    for idx, pcap in scen_files:
        if not pcap.exists():
            print(f"[WARN] Scenario {idx} missing: {pcap}")
            continue

        print(f"[INFO] Processing Scenario {idx}: {pcap}")
        total, transport_ctr, app_ctr, app_by_transport = scan_file(tshark, pcap)

        print(f"   total packets: {total:,}")
        master_totals.append((idx, total))

        # Write per-scenario CSVs
        write_csv_protocols(idx, total, app_ctr, OUT_DIR / f"scenario_{idx}_app_protocols.csv")
        write_csv_transports(idx, total, transport_ctr, OUT_DIR / f"scenario_{idx}_transport_protocols.csv")
        write_csv_app_by_transport(idx, total, app_by_transport, OUT_DIR / f"scenario_{idx}_app_by_transport.csv")

    # Master totals
    append_master_totals(master_totals, OUT_DIR / "scenario_totals.csv")
    print(f"\n[OK] Outputs written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
