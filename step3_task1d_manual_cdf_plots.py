import os, sys, shutil, subprocess
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# ==== CONFIG ====
ROOT_DIR = Path(r"D:\UNIVERSITY DOCUMENTS\BTU COTTBUS-SENFTENBERG\COURSES\Study Project\EPIC\EPIC")
OUT_DIR  = ROOT_DIR / "outputs_task1"
PLOTS_DIR = OUT_DIR / "plots_task1d"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

NON_APP_LAYERS = {
    "frame","sll","eth","ethertype","vlan",
    "arp","ip","ipv6","tcp","udp","icmp","icmpv6","sctp","ssl","tls","quic"
}
TRANSPORT_LAYERS = {"tcp","udp","icmp","icmpv6","sctp"}

def find_tshark():
    p = shutil.which("tshark")
    if p: return p
    for c in [r"C:\Program Files\Wireshark\tshark.exe", r"C:\Program Files (x86)\Wireshark\tshark.exe"]:
        if Path(c).exists(): return c
    sys.exit("[ERROR] TShark not found")

def scenario_files(root: Path):
    return [(i, root / f"Scenario {i}" / f"Scenario_{i}.pcapng") for i in range(1, 9)]

def parse_protocol_chain(proto_chain: str):
    if not proto_chain: return ("NONE","NONE")
    parts = [p.strip().lower() for p in proto_chain.split(":") if p.strip()]
    transport="NONE"; application="NONE"
    for p in parts:
        if p in TRANSPORT_LAYERS: transport=p
    for p in reversed(parts):
        if p not in NON_APP_LAYERS:
            application=p; break
    return (transport,application)

def to_float(x):
    try: return float(x)
    except: return 0.0

def has_app_data(application, tcp_len, udp_len):
    if application=="NONE": return False
    if tcp_len>0: return True
    if udp_len>8: return True
    return False

def app_payload_len(tcp_len, udp_len):
    if tcp_len>0: return tcp_len
    if udp_len>8: return udp_len-8.0
    return 0.0

def manual_cdf(values):
    if not values: return [],[]
    v_sorted = np.sort(np.array(values,dtype=float))
    n=len(v_sorted)
    y = np.arange(1,n+1)/n
    return v_sorted, y

def plot_cdfs(data_dict, metric_name, scen_idx):
    plt.figure(figsize=(8,6))
    for app, vals in sorted(data_dict.items(), key=lambda x:x[0]):
        if len(vals)<2: continue
        x,y = manual_cdf(vals)
        plt.plot(x,y,label=f"{app} (n={len(vals)})")
    plt.xlabel(f"{metric_name} (bytes)")
    plt.ylabel("CDF")
    plt.title(f"Scenario {scen_idx} – CDF of {metric_name}")
    plt.legend(fontsize=8)
    plt.grid(True, linestyle="--", alpha=0.6)
    outpath = PLOTS_DIR / f"scenario_{scen_idx}_CDF_{metric_name.replace(' ','_')}.png"
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def process_scenario(tshark, scen_idx, pcap_path):
    fields = [
        "frame.len","frame.protocols","tcp.len","udp.length"
    ]
    cmd = [tshark,"-r",str(pcap_path),"-T","fields"]
    for f in fields: cmd += ["-e",f]
    cmd += ["-E","separator=\t","-E","occurrence=f"]

    proc = subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding="utf-8",errors="ignore")
    header_len_by_app = defaultdict(list)
    payload_len_by_app = defaultdict(list)

    for line in proc.stdout:
        parts=line.strip().split("\t")
        if len(parts)!=len(fields): continue
        frame_len=to_float(parts[0])
        chain=parts[1].strip()
        tcpL=to_float(parts[2]); udpL=to_float(parts[3])
        transport,app=parse_protocol_chain(chain)
        if has_app_data(app,tcpL,udpL):
            apl=app_payload_len(tcpL,udpL)
            hdr=max(frame_len-apl,0.0)
            header_len_by_app[app].append(hdr)
            payload_len_by_app[app].append(apl)
    proc.wait()
    if proc.returncode!=0:
        err=proc.stderr.read()
        print(f"[WARN] TShark returned {proc.returncode}: {err[:300]}")
    # plot both
    plot_cdfs(header_len_by_app,"header length",scen_idx)
    plot_cdfs(payload_len_by_app,"application payload length",scen_idx)
    print(f"[OK] Scenario {scen_idx}: CDF plots generated.")

def main():
    tshark=find_tshark()
    for idx,pcap in scenario_files(ROOT_DIR):
        if not pcap.exists():
            print(f"[WARN] Missing scenario {idx}")
            continue
        print(f"[INFO] Processing Scenario {idx}")
        process_scenario(tshark,idx,pcap)
    print(f"\n[DONE] CDF plots stored in: {PLOTS_DIR}")

if __name__=="__main__":
    main()
