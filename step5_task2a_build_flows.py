import os
import sys
import math
import csv
import shutil
import subprocess
from pathlib import Path
from collections import defaultdict
import numpy as np

# ===== CONFIG =====
ROOT_DIR = Path(r"D:\UNIVERSITY DOCUMENTS\BTU COTTBUS-SENFTENBERG\COURSES\Study Project\EPIC\EPIC")
OUT_DIR  = ROOT_DIR / "outputs_task2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOWS_MIN = [1, 2, 5]     # flow windows (minutes)
SEQ_LEN = 256               # L: sequence length for (sizes, dirs, iats)
ALLOW_EMPTY_FLOWS = False   # drop flows with <1 pkt if False

# Layers considered non-application (same idea as Task 1)
NON_APP_LAYERS = {
    "frame","sll","eth","ethertype","vlan",
    "arp","ip","ipv6","tcp","udp","icmp","icmpv6","sctp","ssl","tls","quic"
}
TRANSPORT_LAYERS = {"tcp","udp","icmp","icmpv6","sctp"}

# ===== HELPERS =====
def find_tshark() -> str:
    p = shutil.which("tshark")
    if p:
        return p
    for c in [r"C:\Program Files\Wireshark\tshark.exe", r"C:\Program Files (x86)\Wireshark\tshark.exe"]:
        if Path(c).exists():
            return c
    print("[ERROR] TShark not found. Install Wireshark or add to PATH.")
    sys.exit(1)

def scenario_files(root: Path):
    return [(i, root / f"Scenario {i}" / f"Scenario_{i}.pcapng") for i in range(1, 9)]

def parse_protocol_chain(proto_chain: str):
    if not proto_chain:
        return ("NONE","NONE")
    parts = [p.strip().lower() for p in proto_chain.split(":") if p.strip()]
    transport="NONE"; application="NONE"
    for p in parts:
        if p in TRANSPORT_LAYERS:
            transport=p
    for p in reversed(parts):
        if p not in NON_APP_LAYERS:
            application=p; break
    return (transport, application)

def to_float(x):
    try:
        return float(x)
    except:
        return 0.0

def pick_ip(v4, v6):
    return v4 if v4 else v6

def read_packets_stream(tshark: str, pcap_path: Path):
    """
    Streams minimal fields needed:
      - ts: frame.time_epoch
      - size: frame.len
      - chain: frame.protocols -> to detect highest app protocol
      - ip.src/dst or ipv6.src/dst
    Yields dicts with parsed fields.
    """
    fields = ["frame.time_epoch","frame.len","frame.protocols","ip.src","ip.dst","ipv6.src","ipv6.dst"]
    cmd = [tshark, "-r", str(pcap_path), "-T", "fields"]
    for f in fields:
        cmd += ["-e", f]
    cmd += ["-E", "separator=\t", "-E", "occurrence=f"]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="ignore")
    for line in proc.stdout:
        parts = line.rstrip("\n").split("\t")
        if len(parts) != len(fields):
            continue
        ts = to_float(parts[0])
        size = to_float(parts[1])
        chain = parts[2].strip()
        ip_src = pick_ip(parts[3].strip(), parts[5].strip())
        ip_dst = pick_ip(parts[4].strip(), parts[6].strip())
        _, app = parse_protocol_chain(chain)
        if not ip_src or not ip_dst or app == "NONE":
            # Only include packets we can place in (pair × app)
            continue
        # Canonical pair (direction-agnostic)
        a, b = sorted([ip_src, ip_dst])
        # Direction: +1 if min_ip -> max_ip else -1
        direction = +1 if ip_src == a else -1
        yield {"ts": ts, "size": size, "pair": (a, b), "dir": direction, "app": app}

    proc.wait()
    if proc.returncode not in (0,):
        err = proc.stderr.read()
        raise RuntimeError(f"TShark error {proc.returncode}: {err[:400]}")

def build_windows_for_group(packets, window_sec):
    """
    packets: list of dicts (ts,size,dir), already same pair×app.
    Windowing base = earliest ts for that (pair×app) group.
    Returns list of windows: each window is list of packets within [start, end).
    """
    if not packets:
        return []
    # sort by ts
    packets.sort(key=lambda x: x["ts"])
    base = packets[0]["ts"]
    windows = defaultdict(list)  # idx -> packets
    for p in packets:
        idx = int((p["ts"] - base) // window_sec)
        windows[idx].append(p)
    # materialize in order of idx
    return [windows[k] for k in sorted(windows.keys())]

def seq_from_window(pkts, L):
    """
    Build (sizes, dirs, iats) arrays of length L.
    - iat computed inside window (first packet iat=0)
    - If pkts>L -> chunk into multiple sequences (no truncation)
    - If pkts<L -> pad with zeros
    Returns list of (sizes, dirs, iats, chunk_idx, n_raw_in_chunk)
    """
    if not pkts:
        if ALLOW_EMPTY_FLOWS:
            return [(
                np.zeros(L, dtype=np.float32),
                np.zeros(L, dtype=np.int8),
                np.zeros(L, dtype=np.float32),
                0, 0
            )]
        else:
            return []

    # compute IATs within the window
    pkts.sort(key=lambda x: x["ts"])
    iats = [0.0]
    for i in range(1, len(pkts)):
        dt = pkts[i]["ts"] - pkts[i-1]["ts"]
        iats.append(dt if dt >= 0 else 0.0)

    # chunking
    chunks = []
    N = len(pkts)
    n_chunks = math.ceil(N / L)
    for ci in range(n_chunks):
        s = ci * L
        e = min(s + L, N)
        n_raw = e - s
        # slice raw
        sizes = np.array([pkts[j]["size"] for j in range(s, e)], dtype=np.float32)
        dirs  = np.array([pkts[j]["dir"]  for j in range(s, e)], dtype=np.int8)
        iatsA = np.array([iats[j]          for j in range(s, e)], dtype=np.float32)
        # pad if needed
        if n_raw < L:
            sizes = np.pad(sizes, (0, L - n_raw))
            dirs  = np.pad(dirs,  (0, L - n_raw))
            iatsA = np.pad(iatsA,(0, L - n_raw))
        chunks.append((sizes, dirs, iatsA, ci, n_raw))
    return chunks

def write_npz_and_index(window_min, scen_idx, records, meta_rows):
    # Stack arrays
    if len(records) == 0:
        print(f"[WARN] No flows for Scenario {scen_idx}, {window_min}m.")
        return
    sizes = np.stack([r[0] for r in records], axis=0)
    dirs  = np.stack([r[1] for r in records], axis=0)
    iats  = np.stack([r[2] for r in records], axis=0)

    npz_path = OUT_DIR / f"flows_{window_min}m_scen{scen_idx}.npz"
    np.savez_compressed(npz_path, sizes=sizes, dirs=dirs, iats=iats)
    print(f"[OK] Saved: {npz_path}  shapes={sizes.shape}, {dirs.shape}, {iats.shape}")

    idx_csv = OUT_DIR / f"flows_{window_min}m_scen{scen_idx}_index.csv"
    with open(idx_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "flow_id","scenario","window_min","window_start_epoch","window_end_epoch",
            "ip_a","ip_b","app_proto","chunk_idx","n_packets_raw"
        ])
        w.writerows(meta_rows)
    print(f"[OK] Saved: {idx_csv}")

def process_scenario(tshark: str, scen_idx: int, pcap_path: Path):
    print(f"[INFO] Scenario {scen_idx}: loading packets…")
    # Group packets by (pair × app)
    groups = defaultdict(list)  # key: (ip_a, ip_b, app) -> list[dict]
    global_min_ts = None
    global_max_ts = None

    for pkt in read_packets_stream(tshark, pcap_path):
        key = (pkt["pair"][0], pkt["pair"][1], pkt["app"])
        groups[key].append({"ts": pkt["ts"], "size": pkt["size"], "dir": pkt["dir"]})
        if global_min_ts is None or pkt["ts"] < global_min_ts:
            global_min_ts = pkt["ts"]
        if global_max_ts is None or pkt["ts"] > global_max_ts:
            global_max_ts = pkt["ts"]

    if not groups:
        print(f"[WARN] Scenario {scen_idx}: no valid (pair×app) packets.")
        return

    for wmin in WINDOWS_MIN:
        wsec = wmin * 60.0
        flow_records = []   # list of (sizes, dirs, iats, chunk_idx, n_raw)
        meta_rows = []
        flow_id = 0

        for (ip_a, ip_b, app), pkts in groups.items():
            windows = build_windows_for_group(pkts, wsec)
            # base for this group
            if not pkts:
                continue
            base = sorted(pkts, key=lambda x: x["ts"])[0]["ts"]
            for wi, win_pkts in enumerate(windows):
                chunks = seq_from_window(win_pkts, SEQ_LEN)
                if not chunks:
                    continue
                # window bounds
                w_start = base + wi * wsec
                w_end   = w_start + wsec
                for sizes, dirs, iats, cidx, n_raw in chunks:
                    flow_records.append((sizes, dirs, iats, cidx, n_raw))
                    meta_rows.append([
                        flow_id, scen_idx, wmin, f"{w_start:.6f}", f"{w_end:.6f}",
                        ip_a, ip_b, app, cidx, n_raw
                    ])
                    flow_id += 1

        write_npz_and_index(wmin, scen_idx, flow_records, meta_rows)

def main():
    tshark = find_tshark()
    for scen_idx, pcap in scenario_files(ROOT_DIR):
        if not pcap.exists():
            print(f"[WARN] Missing: Scenario {scen_idx} -> {pcap}")
            continue
        process_scenario(tshark, scen_idx, pcap)
    print("\n[DONE] Task 2 Step 1: flow building complete.")

if __name__ == "__main__":
    main()
