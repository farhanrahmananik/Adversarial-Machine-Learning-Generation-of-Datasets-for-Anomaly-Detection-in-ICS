import argparse
from pathlib import Path
import numpy as np
import torch
from joblib import load

# local imports
from step8_task3_data_module import OUT_DIR, SCALER_DIR, SEQ_LEN
from step9_task3_models import Generator


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def inverse_scale(arr: np.ndarray, scalers) -> np.ndarray:
    # arr: (N, 3, L), float32 (z-scored sizes/iats)
    N, C, L = arr.shape
    sz = arr[:, 0, :].reshape(-1, 1)
    ia = arr[:, 2, :].reshape(-1, 1)
    sz = scalers["sizes"].inverse_transform(sz).reshape(N, L)
    ia = scalers["iats"].inverse_transform(ia).reshape(N, L)
    arr[:, 0, :] = sz
    arr[:, 2, :] = ia
    return arr

def postprocess(arr: np.ndarray) -> np.ndarray:
    # dirs -> {-1,0,1}; sizes/iats non-negative
    arr[:, 1, :] = np.rint(np.clip(arr[:, 1, :], -1.0, 1.0))
    arr[:, 0, :] = np.clip(arr[:, 0, :], 0, None)
    arr[:, 2, :] = np.clip(arr[:, 2, :], 0, None)
    return arr

def save_outputs(window_min: int, name: str, arr: np.ndarray, preview_T: int = 16, preview_N: int = 10):
    out_dir = OUT_DIR / f"samples/window_{window_min}m"
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_path = out_dir / f"{name}.npz"
    np.savez_compressed(npz_path,
                        sizes=arr[:, 0, :].astype(np.float32),
                        dirs=arr[:, 1, :].astype(np.float32),
                        iats=arr[:, 2, :].astype(np.float32))

    # small CSV preview
    csv_path = out_dir / f"{name}_preview.csv"
    T = min(preview_T, arr.shape[2])
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("sample_id,channel," + ",".join([f"t{t}" for t in range(T)]) + "\n")
        for i in range(min(preview_N, arr.shape[0])):
            for ch, nm in enumerate(["size", "dir", "iat"]):
                vals = ",".join(f"{arr[i, ch, t]:.6f}" for t in range(T))
                f.write(f"{i},{nm},{vals}\n")

    print(f"[OK] Saved: {npz_path}")
    print(f"[OK] Saved: {csv_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic ICS flows from best checkpoint.")
    parser.add_argument("--window", type=int, default=2, choices=[1,2,5], help="window minutes (1/2/5)")
    parser.add_argument("--ckpt", type=str, default=None, help="path to checkpoint .pt (defaults to best in runs)")
    parser.add_argument("--num", type=int, default=256, help="number of flows to generate")
    parser.add_argument("--zdim", type=int, default=128, help="noise dim")
    parser.add_argument("--name", type=str, default=None, help="base output name (auto if not set)")
    args = parser.parse_args()

    window_min = args.window
    run_dir = OUT_DIR / f"runs/window_{window_min}m"
    ckpt_path = Path(args.ckpt) if args.ckpt else (run_dir / "ckpt_best.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # load scalers for this window
    scalers_path = SCALER_DIR / f"feature_scaler_window_{window_min}m.joblib"
    if not scalers_path.exists():
        raise FileNotFoundError(f"Scaler not found for window {window_min}m: {scalers_path}")
    scalers = load(scalers_path)

    # build & load generator
    G = Generator(seq_len=SEQ_LEN, z_dim=args.zdim).to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE)
    G.load_state_dict(state["G"])
    G.eval()

    # sample
    with torch.no_grad():
        z = torch.randn(args.num, args.zdim, device=DEVICE)
        fake = G(z).cpu().numpy().astype(np.float32)   # (N, 3, L)

    # inverse scale + postprocess
    fake = inverse_scale(fake, scalers)
    fake = postprocess(fake)

    # name
    base_name = args.name or f"synth_window{window_min}m_{args.num}"
    save_outputs(window_min, base_name, fake)

if __name__ == "__main__":
    main()
