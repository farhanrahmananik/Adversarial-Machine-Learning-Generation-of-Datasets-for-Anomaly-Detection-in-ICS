import math
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========
# Discriminator
# Spec:
# - Input: (B, 3, L)
# - 9 Conv1d layers, LeakyReLU
# - Dropout after every conv EXCEPT the 1st and the 9th
# - Then 4 Fully-Connected layers (ReLU), final output sigmoid in [0,1)
# - We expose the tensor fed into FC1 as "features_before_fc" (for feature-matching loss)
# ==========

class Discriminator(nn.Module):
    def __init__(
        self,
        seq_len: int = 256,
        in_ch: int = 3,
        widths=(32, 64, 96, 128, 128, 160, 192, 224, 256),
        kernel_size: int = 3,
        dropout_p: float = 0.25,
        leak: float = 0.2,
        fc_sizes=(512, 256, 64, 1),
    ):
        super().__init__()
        assert len(widths) == 9, "Discriminator must have 9 conv layers per spec."
        padding = kernel_size // 2  # keep length same

        convs = []
        in_c = in_ch
        for i, out_c in enumerate(widths):
            convs.append(nn.Conv1d(in_c, out_c, kernel_size=kernel_size, padding=padding, bias=True))
            convs.append(nn.LeakyReLU(leak, inplace=True))
            # Dropout after all convs except 1st (i==0) and 9th (i==8)
            if i not in (0, 8):
                convs.append(nn.Dropout(dropout_p))
            in_c = out_c
        self.conv_stack = nn.Sequential(*convs)

        # We will flatten (B, C9, L) -> (B, C9*L) and then FC×4
        self.seq_len = seq_len
        c_last = widths[-1]
        fc_in = c_last * seq_len

        assert len(fc_sizes) == 4, "Need exactly 4 fully-connected layers per spec."
        f1, f2, f3, f4 = fc_sizes
        self.fc1 = nn.Linear(fc_in, f1)
        self.fc2 = nn.Linear(f1,   f2)
        self.fc3 = nn.Linear(f2,   f3)
        self.fc4 = nn.Linear(f3,   f4)

        # init
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x: (B, 3, L)
        returns:
          {
            'logit': (B, 1),
            'prob':  (B, 1) in [0,1),
            'features_before_fc': (B, C9*L)  # to be used in custom feature-matching loss
          }
        """
        h = self.conv_stack(x)               # (B, C9, L)
        h_flat = h.flatten(start_dim=1)      # (B, C9*L)  <-- tap here
        f1 = F.relu(self.fc1(h_flat))
        f2 = F.relu(self.fc2(f1))
        f3 = F.relu(self.fc3(f2))
        logit = self.fc4(f3)                 # (B,1)
        prob = torch.sigmoid(logit)
        return {"logit": logit, "prob": prob, "features_before_fc": h_flat}


# ==========
# Generator
# Spec:
# - Noise z -> 3 Fully Connected (ReLU)
# - Then 8 Conv1d layers (ReLU)
# - BEFORE the first 4 convs: add an Upsample layer (so 4 upsamplings total)
# - Deconvolution can be done via Upsample + Conv1d to increase temporal length
# - Output channels = 3 (size, dir, iat)
# - We will start from length L0=16 and upsample 4× (16→32→64→128→256)
# ==========

class Generator(nn.Module):
    def __init__(
        self,
        seq_len: int = 256,
        z_dim: int = 128,
        base_len: int = 16,     # L0; 16 * (2^4) = 256
        base_ch: int = 128,     # starting channel width after FCs
        kernel_size: int = 3,
        fc_sizes=(256, 512, 1024),
        conv_widths=(128, 128, 96, 96, 64, 64, 48, 32),  # 8 convs
    ):
        super().__init__()
        assert seq_len == 256, "This setup assumes seq_len=256 (adjust base_len/upsamples otherwise)."
        assert len(conv_widths) == 8, "Generator must have 8 conv layers per spec."
        self.seq_len = seq_len
        self.z_dim   = z_dim
        self.base_len = base_len
        self.kernel_size = kernel_size
        pad = kernel_size // 2

        # ---- FC block (3 FCs) ----
        self.fc1 = nn.Linear(z_dim, fc_sizes[0])
        self.fc2 = nn.Linear(fc_sizes[0], fc_sizes[1])
        # The third FC outputs a flattened (C0 * L0) tensor
        self.fc3 = nn.Linear(fc_sizes[1], base_ch * base_len)

        # ---- Upsample + Conv stack (8 convs total; upsample before first 4 convs) ----
        upsamples = []
        convs = []
        in_c = base_ch
        for i, out_c in enumerate(conv_widths):
            if i < 4:
                upsamples.append(nn.Upsample(scale_factor=2, mode="nearest"))
            else:
                upsamples.append(nn.Identity())
            convs.append(nn.Conv1d(in_c, out_c, kernel_size=kernel_size, padding=pad, bias=True))
            in_c = out_c

        self.up_blocks = nn.ModuleList(upsamples)
        self.conv_blocks = nn.ModuleList(convs)

        # final projection to 3 channels
        self.to_out = nn.Conv1d(in_c, 3, kernel_size=kernel_size, padding=pad, bias=True)

        # init
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (B, z_dim) ~ N(0,1)
        returns: fake flows (B, 3, L=256)
        """
        b = z.size(0)
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))                        # (B, C0*L0)
        h = h.view(b, -1, self.base_len)              # (B, C0, L0)

        for up, conv in zip(self.up_blocks, self.conv_blocks):
            h = up(h)                                  # upsample first 4 layers
            h = F.relu(conv(h))                        # conv + ReLU

        out = self.to_out(h)                           # (B, 3, 256)
        return out

    @torch.no_grad()
    def sample(self, n: int, device=None) -> torch.Tensor:
        """
        Convenience sampler: draws z ~ N(0,1), returns raw generator output.
        Post-processing (e.g., inverse-scaling, dir rounding) should be handled by caller.
        """
        dev = device if device is not None else next(self.parameters()).device
        z = torch.randn(n, self.z_dim, device=dev)
        return self.forward(z)


# ==========
# Smoke test
# ==========
if __name__ == "__main__":
    B, L = 8, 256
    x = torch.randn(B, 3, L)
    D = Discriminator(seq_len=L)
    out = D(x)
    print("[D] prob shape:", out["prob"].shape, "features_before_fc:", out["features_before_fc"].shape)

    G = Generator(seq_len=L, z_dim=128)
    fake = G.sample(8)
    print("[G] sample shape:", fake.shape)
