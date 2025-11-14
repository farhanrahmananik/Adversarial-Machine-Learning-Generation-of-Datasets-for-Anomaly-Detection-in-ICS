import csv
import os
import sys
import shutil
import subprocess
from pathlib import Path
from collections import defaultdict, Counter
import math
import numpy as np

# ====== CONFIG ======
ROOT_DIR = Path(r"D:\UNIVERSITY DOCUMENTS\BTU COTTBUS-SENFTENBERG\COURSES\Study Project\EPIC\EPIC")
OUT_DIR  = ROOT_DIR / "outputs_task1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NON_APP_LAYERS = {
    "frame", "sll", "eth", "ethertype", "vlan",
    "arp",
    "ip", "ipv6",
    "tcp", "udp", "icmp", "icmpv6", "sctp",
    "ssl", "tls", "quic",
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
    if not proto_chain:
        return ("NONE", "NONE")
    parts = [p.strip().lower() for p in proto_chain.split(":") if p.strip()]
    transport = "NONE"
    application = "NONE"

    for p in parts:
        if p in TRANSPORT_LAYERS:
            transport = p
    for p in reversed(parts):
        if p not in NON_APP_LAYERS:
            application = p
            break
    return (transport, application)


def to_float(x):
    try:
        return float(x)
    except:
        return 0.0


def has_app_data(application: str, tcp_len: float, udp_len: float) -> bool:
    if application == "NONE":
        return False
    # tcp.len is app payload; udp.length includes header (8 bytes)
    if tcp_len > 0:
        return True
    if udp_len > 8:
        return True
    return False


def app_payload_len(tcp_len: float, udp_len: float) -> float:
    if tcp_len > 0:
        return tcp_len
    if udp_len > 8:
        return udp_len - 8.0
    return 0.0


def stats_array(arr):
    if len(arr) == 0:
        return (0, 0.0, 0.0, 0.0)
    a = np.array(arr, dtype=float)
    return (len(a), float(a.mean()), float(a.std(ddof=1) if len(a) > 1 else 0.0), float(np.median(a)))


def write_csv(path: Path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def process_scenario(tshark: str, scen_idx: int, pcap_path: Path):
    """
    Streams required fields and computes:
      - per-app stats of lengths (only app-data packets)
      - inter-arrivals per host pair
      - inter-arrivals per host pair x app
      - unique host pair count
    """
    # Fields to extract
    fields = [
        "frame.time_epoch",  # timestamp
        "frame.len",         # total frame length
        "frame.protocols",   # protocol chain
        "ip.src","ip.dst","ipv6.src","ipv6.dst",
        "tcp.len",           # TCP app payload length
        "udp.length",        # UDP total length (header+payload)
    ]
    cmd = [tshark, "-r", str(pcap_path), "-T", "fields"]
    for f in fields:
        cmd += ["-e", f]
    cmd += ["-E", "separator=\t", "-E", "occurrence=f"]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="ignore")

    # Collectors
    # (b) per application protocol: arrays for total_len, header_len, app_payload_len
    app_total = defaultdict(list)
    app_header = defaultdict(list)
    app_payload = defaultdict(list)

    # For inter-arrival: we need ordered timestamps grouped by pair and pair×app
    pair_timestamps = defaultdict(list)         # key: (hostA, hostB) sorted tuple
    pair_app_timestamps = defaultdict(list)     # key: ((hostA, hostB), app)

    # Helper: choose src/dst ip (ipv4 prioritized, else ipv6)
    def pick_ip(v4, v6):
        return v4 if v4 else v6

    line_no = 0
    for line in proc.stdout:
        line_no += 1
        parts = line.rstrip("\n").split("\t")
        if len(parts) != len(fields):
            # Skip malformed line
            continue

        ts = to_float(parts[0])
        frame_len = to_float(parts[1])
        chain = parts[2].strip()
        ip_src = pick_ip(parts[3].strip(), parts[5].strip())
        ip_dst = pick_ip(parts[4].strip(), parts[6].strip())
        tcpL = to_float(parts[7])
        udpL = to_float(parts[8])

        transport, application = parse_protocol_chain(chain)

        # (b) only consider packets with application data
        if has_app_data(application, tcpL, udpL):
            apl = app_payload_len(tcpL, udpL)
            hdr = max(frame_len - apl, 0.0)
            # store per-app arrays
            app_total[application].append(frame_len)
            app_header[application].append(hdr)
            app_payload[application].append(apl)

        # (b) & (c): inter-arrivals per host pair (direction-agnostic)
        if ip_src and ip_dst:
            a, b = sorted([ip_src, ip_dst])
            pair_key = (a, b)
            pair_timestamps[pair_key].append(ts)
            # (c) also per app
            pair_app_key = (pair_key, application)
            pair_app_timestamps[pair_app_key].append(ts)

    proc.wait()
    if proc.returncode != 0:
        err = proc.stderr.read()
        raise RuntimeError(f"tshark error ({proc.returncode}): {err[:500]}")

    # Compute inter-arrival deltas
    def deltas_from_sorted_ts(ts_list):
        if len(ts_list) < 2:
            return []
        ts_list.sort()
        return [ts_list[i] - ts_list[i-1] for i in range(1, len(ts_list)) if ts_list[i] >= ts_list[i-1]]

    pair_interarrivals = {k: deltas_from_sorted_ts(v) for k, v in pair_timestamps.items()}
    pair_app_interarrivals = {k: deltas_from_sorted_ts(v) for k, v in pair_app_timestamps.items()}

    # Build rows
    # (b) per-app stats for total_len, header_len, payload_len
    rows_app_len = []
    for app in sorted(set(list(app_total.keys()) + list(app_header.keys()) + list(app_payload.keys()))):
        n_tot, mean_tot, std_tot, med_tot = stats_array(app_total[app])
        n_hdr, mean_hdr, std_hdr, med_hdr = stats_array(app_header[app])
        n_pay, mean_pay, std_pay, med_pay = stats_array(app_payload[app])
        rows_app_len.append([scen_idx, app,
                             n_tot, f"{mean_tot:.6f}", f"{std_tot:.6f}", f"{med_tot:.6f}",
                             n_hdr, f"{mean_hdr:.6f}", f"{std_hdr:.6f}", f"{med_hdr:.6f}",
                             n_pay, f"{mean_pay:.6f}", f"{std_pay:.6f}", f"{med_pay:.6f}"])

    # (b) inter-arrival per host pair
    rows_pair_ia = []
    for (a,b) in sorted(pair_interarrivals.keys()):
        arr = pair_interarrivals[(a,b)]
        n, mean, std, med = stats_array(arr)
        rows_pair_ia.append([scen_idx, a, b, n, f"{mean:.9f}", f"{std:.9f}", f"{med:.9f}"])

    # (c) host pair count
    unique_pair_count = len(pair_timestamps)

    # (c) inter-arrival per (host pair × app)
    rows_pair_app_ia = []
    for (pair_key, app), arr in sorted(pair_app_interarrivals.items(), key=lambda x: (x[0][0], x[0][1], x[0][1])):
        (a, b) = pair_key
        n, mean, std, med = stats_array(arr)
        rows_pair_app_ia.append([scen_idx, a, b, app, n, f"{mean:.9f}", f"{std:.9f}", f"{med:.9f}"])

    return {
        "rows_app_len": rows_app_len,
        "rows_pair_ia": rows_pair_ia,
        "rows_pair_app_ia": rows_pair_app_ia,
        "unique_pair_count": unique_pair_count
    }


def main():
    tshark = find_tshark()
    scen_files = scenario_files(ROOT_DIR)

    # Prepare output files (one per scenario)
    for idx, pcap in scen_files:
        if not pcap.exists():
            print(f"[WARN] Scenario {idx} missing: {pcap}")
            continue

        print(f"[INFO] Processing Scenario {idx}: {pcap}")
        R = process_scenario(tshark, idx, pcap)

        # (b) per-app packet length stats
        out1 = OUT_DIR / f"scenario_{idx}_app_length_stats.csv"
        write_csv(
            out1,
            header=[
                "scenario","application_protocol",
                "N_total_len","mean_total_len","std_total_len","median_total_len",
                "N_header_len","mean_header_len","std_header_len","median_header_len",
                "N_payload_len","mean_payload_len","std_payload_len","median_payload_len",
            ],
            rows=R["rows_app_len"]
        )

        # (b) inter-arrival per host pair
        out2 = OUT_DIR / f"scenario_{idx}_pair_interarrival_stats.csv"
        write_csv(
            out2,
            header=["scenario","hostA","hostB","N","mean_s","std_s","median_s"],
            rows=R["rows_pair_ia"]
        )

        # (c) inter-arrival per (host pair × app)
        out3 = OUT_DIR / f"scenario_{idx}_pair_app_interarrival_stats.csv"
        write_csv(
            out3,
            header=["scenario","hostA","hostB","application_protocol","N","mean_s","std_s","median_s"],
            rows=R["rows_pair_app_ia"]
        )

        # (c) unique host pair count (single-line CSV)
        out4 = OUT_DIR / f"scenario_{idx}_unique_host_pairs.csv"
        write_csv(
            out4,
            header=["scenario","unique_host_pairs"],
            rows=[[idx, R["unique_pair_count"]]]
        )

    print(f"\n[OK] Outputs written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
