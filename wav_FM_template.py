# Dual-stream ECG-PPG self-supervised pretraining template (PyTorch)
# Requirements: torch, einops (optional)
# Fill dataset paths, preprocessing, and hyperparameters as appropriate.

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---------------------------
# Helper utilities
# ---------------------------

def patch_mask(x, patch_size=16, mask_ratio=0.25):
    # x: [B, C, T] (C=1 typically)
    B, C, T = x.shape
    n_patches = T // patch_size
    mask = torch.zeros(B, n_patches, device=x.device, dtype=torch.bool)
    n_mask = max(1, int(mask_ratio * n_patches))
    for i in range(B):
        idx = torch.randperm(n_patches)[:n_mask]
        mask[i, idx] = True
    # expand mask to time resolution
    mask_time = mask.repeat_interleave(patch_size, dim=1)[:, :T]
    return mask_time  # [B, T] boolean mask

def nt_xent_loss(z1, z2, temperature=0.1):
    # z1, z2: [B, D] - positive pairs aligned by batch index
    # standard normalized temperature-scaled cross entropy (SimCLR-style)
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    B = z1.size(0)
    representations = torch.cat([z1, z2], dim=0)  # 2B x D
    sim_matrix = torch.matmul(representations, representations.T) / temperature
    labels = torch.arange(B, device=z1.device)
    labels = torch.cat([labels, labels], dim=0)
    # mask out self-similarities
    mask = (~torch.eye(2 * B, dtype=torch.bool, device=z1.device)).float()
    exp_sim = torch.exp(sim_matrix) * mask
    # positives are off-diagonals
    positives = torch.exp(torch.sum(z1 * z2, dim=-1) / temperature)
    positives = torch.cat([positives, positives], dim=0)
    denom = exp_sim.sum(dim=1)
    loss = -torch.log(positives / denom)
    return loss.mean()

# ---------------------------
# Dataset skeleton
# ---------------------------

class ECGPPGWindowDataset(Dataset):
    """
    Provide aligned ECG and PPG waveform segments.
    Each __getitem__ returns:
      ecg: torch.Tensor [1, T]
      ppg: torch.Tensor [1, T]
      meta: dict with e.g., timestamps, patient id
    Preprocessing (resampling, filtering, channel selection) should be done offline
    or within a custom transform passed here.
    """
    def __init__(self, file_index, window_size=5000, transform=None):
        self.index = file_index  # list of (ecg_path, ppg_path, start_idx) or preloaded arrays
        self.window_size = window_size
        self.transform = transform

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        ecg_arr, ppg_arr = self.index[idx]  # replace with your retrieval
        # assume ecg_arr / ppg_arr are 1D numpy arrays or tensors with sufficient length
        # Here we assume they are already windowed to exactly window_size
        ecg = torch.as_tensor(ecg_arr, dtype=torch.float32).unsqueeze(0)  # [1, T]
        ppg = torch.as_tensor(ppg_arr, dtype=torch.float32).unsqueeze(0)
        if self.transform:
            ecg, ppg = self.transform(ecg, ppg)
        return {'ecg': ecg, 'ppg': ppg}

# ---------------------------
# Model components
# ---------------------------

class ConvFrontEnd(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=64, kernel_sizes=[15,9,5], stride=1):
        super().__init__()
        layers = []
        c = in_channels
        out = hidden_channels
        for k in kernel_sizes:
            layers.append(nn.Conv1d(c, out, kernel_size=k, stride=stride, padding=k//2))
            layers.append(nn.BatchNorm1d(out))
            layers.append(nn.GELU())
            c = out
            out = min(out * 2, 512)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: [B, C, T]
        return self.net(x)  # [B, hidden, T]

class SimpleTransformerEncoder(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=4, dim_feedforward=256, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout,
                                                   activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj = nn.Linear(d_model, d_model)  # optional projection

    def forward(self, x):
        # x: [B, T, D] (we will permute before calling)
        out = self.transformer(x)
        return self.proj(out)

class TimePooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.lin = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: [B, T, D]
        x_t = x.permute(0, 2, 1)  # [B, D, T]
        p = self.pool(x_t).squeeze(-1)  # [B, D]
        return self.lin(p)

class WaveDecoder(nn.Module):
    def __init__(self, in_dim, out_channels=1, seq_len=5000):
        super().__init__()
        # simple linear decoder that maps per-time embeddings back to waveform samples
        self.seq_len = seq_len
        self.lin = nn.Linear(in_dim, seq_len)  # maps pooled or token embeddings to full waveform
        # More sophisticated: use ConvTranspose or transformer decoder for per-time reconstruction

    def forward(self, z):
        # z: [B, D]
        recon = self.lin(z)  # [B, T]
        return recon.unsqueeze(1)  # [B, 1, T]

# ---------------------------
# Dual-stream model
# ---------------------------

class DualStreamModel(nn.Module):
    def __init__(self, seq_len=5000, d_model=128):
        super().__init__()
        # encoders
        self.ecg_front = ConvFrontEnd(in_channels=1, hidden_channels=64)
        self.ppg_front = ConvFrontEnd(in_channels=1, hidden_channels=64)
        # project conv output to d_model per time-step
        self.proj_ecg = nn.Conv1d(64, d_model, kernel_size=1)
        self.proj_ppg = nn.Conv1d(64, d_model, kernel_size=1)
        # transformer encoders (operating on tokens/time-steps)
        self.trans_ecg = SimpleTransformerEncoder(d_model=d_model, num_layers=4)
        self.trans_ppg = SimpleTransformerEncoder(d_model=d_model, num_layers=4)
        # pooling heads for contrastive
        self.pool_ecg = TimePooling(d_model)
        self.pool_ppg = TimePooling(d_model)
        # decoders for masked reconstruction (use pooled context + optional upsampling)
        self.dec_ecg = WaveDecoder(in_dim=d_model, out_channels=1, seq_len=seq_len)
        self.dec_ppg = WaveDecoder(in_dim=d_model, out_channels=1, seq_len=seq_len)

    def forward(self, ecg, ppg, mask_ecg=None, mask_ppg=None):
        # ecg, ppg: [B, 1, T]
        B, _, T = ecg.shape
        # front-end convs
        fe_ecg = self.ecg_front(ecg)[:, :64, :T]  # [B, C, T]
        fe_ppg = self.ppg_front(ppg)[:, :64, :T]
        # proj to model dim
        z_ecg = self.proj_ecg(fe_ecg).permute(0, 2, 1)  # [B, T, D]
        z_ppg = self.proj_ppg(fe_ppg).permute(0, 2, 1)
        # optional masking: for masked modeling we can zero tokens in z or replace with learnable mask token
        if mask_ecg is not None:
            z_ecg = z_ecg.masked_fill(mask_ecg.unsqueeze(-1), 0.0)
        if mask_ppg is not None:
            z_ppg = z_ppg.masked_fill(mask_ppg.unsqueeze(-1), 0.0)
        # transformer encoder
        out_ecg = self.trans_ecg(z_ecg)  # [B, T, D]
        out_ppg = self.trans_ppg(z_ppg)
        # pooled embeddings for contrastive
        pooled_ecg = self.pool_ecg(out_ecg)  # [B, D]
        pooled_ppg = self.pool_ppg(out_ppg)
        # decoders reconstruct full waveform from pooled embedding (simple)
        recon_ecg = self.dec_ecg(pooled_ecg)  # [B, 1, T]
        recon_ppg = self.dec_ppg(pooled_ppg)
        return {
            'ecg_tokens': out_ecg,
            'ppg_tokens': out_ppg,
            'pool_ecg': pooled_ecg,
            'pool_ppg': pooled_ppg,
            'recon_ecg': recon_ecg,
            'recon_ppg': recon_ppg
        }

# ---------------------------
# Training loop (example)
# ---------------------------

def train_epoch(model, dataloader, optim, device,
                mask_patch_size=16, mask_ratio=0.25,
                alpha_recon=1.0, alpha_contrast=1.0):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        ecg = batch['ecg'].to(device)  # [B,1,T]
        ppg = batch['ppg'].to(device)
        B, C, T = ecg.shape
        # make masks
        mask_e = patch_mask(ecg, patch_size=mask_patch_size, mask_ratio=mask_ratio)
        mask_p = patch_mask(ppg, patch_size=mask_patch_size, mask_ratio=mask_ratio)
        # forward with masked tokens (we pass masks for token zeroing inside model)
        out = model(ecg, ppg, mask_ecg=mask_e, mask_ppg=mask_p)
        # reconstruction targets: only compute MSE on masked positions
        recon_ecg = out['recon_ecg']  # [B,1,T]
        recon_ppg = out['recon_ppg']
        # MSE on masked timepoints
        mask_e_f = mask_e.unsqueeze(1).float()
        mask_p_f = mask_p.unsqueeze(1).float()
        # avoid dividing by zero
        eps = 1e-6
        loss_recon_ecg = ( ((recon_ecg - ecg)**2) * mask_e_f ).sum() / (mask_e_f.sum() + eps)
        loss_recon_ppg = ( ((recon_ppg - ppg)**2) * mask_p_f ).sum() / (mask_p_f.sum() + eps)
        loss_recon = loss_recon_ecg + loss_recon_ppg
        # contrastive loss on pooled embeddings
        z_ecg = out['pool_ecg']
        z_ppg = out['pool_ppg']
        loss_contrast = nt_xent_loss(z_ecg, z_ppg, temperature=0.1)
        loss = alpha_recon * loss_recon + alpha_contrast * loss_contrast
        optim.zero_grad()
        loss.backward()
        optim.step()
        total_loss += loss.item() * B
    return total_loss / len(dataloader.dataset)

# ---------------------------
# Example usage
# ---------------------------

if __name__ == "__main__":
    # Dummy index: replace with actual waveform windows
    import numpy as np
    N = 2000
    T = 4096
    dummy_index = []
    for i in range(N):
        # random noise + sinusoid as toy example
        t = np.linspace(0, 1, T)
        ecg = 0.1 * np.random.randn(T) + 0.5 * np.sin(2 * math.pi * 5 * t)  # toy
        ppg = 0.05 * np.random.randn(T) + 0.3 * np.sin(2 * math.pi * 1.2 * t + 0.2)  # toy slower
        dummy_index.append((ecg, ppg))

    dataset = ECGPPGWindowDataset(dummy_index, window_size=T)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2, drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualStreamModel(seq_len=T, d_model=128).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    for epoch in range(10):
        loss = train_epoch(model, dataloader, optimizer, device,
                           mask_patch_size=16, mask_ratio=0.25,
                           alpha_recon=1.0, alpha_contrast=1.0)
        print(f"Epoch {epoch:02d} loss: {loss:.4f}")
