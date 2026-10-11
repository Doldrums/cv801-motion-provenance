"""Baselines of Table 2, reimplemented: pose alignment, pose codes, reconstruction tokens.

* DTW over joint trajectories (10 fps, pelvis-relative joints): unmasked (a
  hidden joint reads as zero), banded (Sakoe-Chiba, 10 % of the longer side),
  and masked (frame cost over joints observed in both; fully hidden query
  frames dropped). Whole queries align globally, fragments as subsequences.
* PQ words over raw pose: per core, the centred joint positions of every
  second frame, product-quantised into 8 x 256 words, scored by the CRE.
* DPQ: the same raw-pose cores through a learned projection and 8 x 256
  codebooks, trained end to end for retrieval (supervised contrast on the
  pooled quantized cores, straight-through hard codes); scored by the CRE.
* Reconstruction tokenizer: per-core space-time encoder, EMA vector
  quantizer (512 codes), MLP decoder of the core's joints; retrieval by the
  cosine of word histograms ("codes") or of mean latents ("continuous"). The
  "+ identity" variant adds supervised contrast on the mean latent.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from motionprov.choreomatch.cre import kmeans
from motionprov.choreomatch.model import C, J, W, Block, features, windows

STRIDE = 3


# ------------------------------------------------------------------ DTW

def frames(joints, mask):
    x = (joints[:, 1:] - joints[:, :1]) * mask[..., None]
    return x[::STRIDE], mask[::STRIDE]


def frame_costs(q, qm, refs, masked):
    """q (n, 23, 3), refs list of (x (m, 23, 3), mask) -> padded costs (R, n, M), lengths (R,)."""
    device = q.device
    m_max = max(len(x) for x, _ in refs)
    X = torch.zeros(len(refs), m_max, J, 3, device=device)
    XM = torch.zeros(len(refs), m_max, J, device=device)
    lengths = torch.tensor([len(x) for x, _ in refs], device=device)
    for i, (x, m) in enumerate(refs):
        X[i, :len(x)], XM[i, :len(x)] = x, m
    d = (q[None, :, None] - X[:, None]).norm(dim=-1)                       # (R, n, M, J)
    if masked:
        w = qm[None, :, None] * XM[:, None]
        cost = (d * w).sum(-1) / w.sum(-1).clamp(min=1)
    else:
        cost = d.mean(-1)
    return cost, lengths


def dtw(cost, lengths, subsequence=False, band=None):
    """Batched DTW by anti-diagonals -> normalised alignment cost (R,)."""
    r, n, m = cost.shape
    inf = torch.tensor(float("inf"), device=cost.device)
    D = torch.full((r, n + 1, m + 1), float("inf"), device=cost.device)
    D[:, 0, 0] = 0
    if subsequence:
        D[:, 0, :] = 0
    jj = torch.arange(m, device=cost.device)
    valid_col = jj[None] < lengths[:, None]                                   # (R, M)
    for k in range(n + m - 1):
        i = torch.arange(max(0, k - m + 1), min(n, k + 1), device=cost.device)
        j = k - i
        c = cost[:, i, j]
        ok = valid_col[:, j]
        if band is not None:
            scale = lengths.float() / n
            ok = ok & ((j[None].float() - i[None].float() * scale[:, None]).abs() <= band[:, None])
        best = torch.minimum(torch.minimum(D[:, i, j], D[:, i, j + 1]), D[:, i + 1, j])
        D[:, i + 1, j + 1] = torch.where(ok, c + best, inf)
    last = D[:, n, 1:]
    if subsequence:
        end = torch.where(valid_col, last, inf).min(1).values
        return end / n
    end = last.gather(1, (lengths - 1)[:, None])[:, 0]
    return end / (n + lengths.float())


class DTWSystem:
    def __init__(self, index, device, masked=False, banded=False):
        self.device, self.masked, self.banded = device, masked, banded
        self.refs = []
        for rec in index:
            mask = np.repeat(rec["observed"][:, None], J, 1)
            x, m = frames(rec["joints"], mask)
            self.refs.append((torch.from_numpy(x).to(device), torch.from_numpy(m.astype(np.float32)).to(device)))

    @torch.no_grad()
    def __call__(self, joints, mask, kind):
        x, m = frames(joints, mask)
        if self.masked:
            keep = m.any(1)
            x, m = x[keep], m[keep]
        if len(x) < 2:
            return np.full(len(self.refs), -np.inf)
        q = torch.from_numpy(x).to(self.device)
        qm = torch.from_numpy(m.astype(np.float32)).to(self.device)
        out = []
        for s in range(0, len(self.refs), 64):
            chunk = self.refs[s:s + 64]
            cost, lengths = frame_costs(q, qm, chunk, self.masked)
            band = None
            if self.banded and kind == "whole":
                band = 0.1 * torch.maximum(lengths.float(), torch.tensor(float(len(q)), device=self.device))
            out.append(-dtw(cost, lengths, subsequence=kind == "fragment", band=band))
        return torch.cat(out).cpu().numpy()


# ------------------------------------------------------------------ PQ words over raw pose

def raw_cores(joints, mask):
    """-> (N, 345) centred-later core descriptors (every second frame), (N,) observed fraction."""
    t = len(joints)
    n = max(1, math.ceil(t / C))
    x = (joints[:, 1:] - joints[:, :1]) * mask[..., None]
    pad = n * C - t
    x = np.concatenate([x, np.zeros((pad, J, 3), np.float32)])
    m = np.concatenate([mask, np.zeros((pad, J), bool)])
    cores = x.reshape(n, C, J, 3)[:, ::2].reshape(n, -1)
    return cores, m.reshape(n, C, J).mean((1, 2)), m.reshape(n, C, J)[:, ::2].repeat(3, -1).reshape(n, -1)


class PQ:
    def __init__(self, train, device, subspaces=8, size=256):
        cores, obs, seen = zip(*(raw_cores(r["joints"], np.repeat(r["observed"][:, None], J, 1)) for r in train))
        x = torch.from_numpy(np.concatenate(cores)).to(device)
        keep = torch.from_numpy(np.concatenate(obs)).to(device) >= 0.5
        x = x[keep]
        self.mean = x.mean(0)
        x = x - self.mean
        self.split = np.array_split(np.arange(x.shape[1]), subspaces)
        self.books = [kmeans(x[:, s], size) for s in self.split]
        self.device = device

    def encode(self, joints, mask):
        cores, obs, seen = raw_cores(joints, mask)
        x = torch.from_numpy(cores).to(self.device) - self.mean
        x = x * torch.from_numpy(seen).to(self.device)          # a hidden joint sits at the mean pose
        recon = torch.cat([b[torch.cdist(x[:, s], b).argmin(1)] for s, b in zip(self.split, self.books)], 1)
        return recon, obs


class DPQ(nn.Module):
    """Retrieval-trained product quantization over raw-pose cores."""

    def __init__(self, train, device, subspaces=8, size=256, dim=256, steps=3000, seed=0, window=12, log=print):
        super().__init__()
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        cores = [raw_cores(r["joints"], np.repeat(r["observed"][:, None], J, 1)) for r in train]
        x = torch.from_numpy(np.concatenate([c[0] for c in cores])).to(device)
        self.register_buffer("mean", x.mean(0))
        self.proj = nn.Sequential(nn.Linear(x.shape[1], 512), nn.GELU(), nn.Linear(512, dim))
        self.books = nn.Parameter(torch.randn(subspaces, size, dim // subspaces) * 0.1)
        self.to(device)
        self.device = device
        labels = sorted({r["label"] for r in train})
        groups = {}
        for i, r in enumerate(train):
            groups.setdefault(r["label"], []).append(i)
        groups = [v for v in groups.values() if len({train[i]["source"] for i in v}) > 1]
        opt = torch.optim.AdamW(self.parameters(), lr=3e-4, weight_decay=0.01)
        for step in range(steps):
            picks = [(k, i) for k in rng.choice(len(groups), min(32, len(groups)), replace=False)
                     for i in rng.choice(groups[k], 2, replace=False)]
            us = []
            for _, i in picks:
                c, obs, seen = cores[i]
                s = int(rng.integers(0, max(1, len(c) - window + 1)))
                v = torch.from_numpy(c[s:s + window]).to(device)
                q = self.quantize(self.proj((v - self.mean) * torch.from_numpy(seen[s:s + window]).to(device)))
                us.append(F.normalize(F.normalize(q, dim=-1).mean(0), dim=0))
            u = torch.stack(us)
            y = torch.tensor([k for k, _ in picks], device=device)
            eye = torch.eye(len(u), dtype=torch.bool, device=device)
            logits = (u @ u.T / 0.1).masked_fill(eye, float("-inf"))
            pos = (y[:, None] == y[None]) & ~eye
            loss = -((logits - logits.logsumexp(1, keepdim=True)).masked_fill(~pos, 0).sum(1) / pos.sum(1)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % 500 == 0:
                log(f"dpq step {step} loss {float(loss):.4f}")
        self.eval()

    def quantize(self, h):
        """Soft assignment forward for gradients, hard codes in value (straight-through)."""
        h = h.view(len(h), len(self.books), -1).transpose(0, 1)               # (S, N, d)
        d = torch.cdist(h, self.books)                                         # (S, N, size)
        soft = torch.softmax(-d / 0.1, -1) @ self.books
        hard = torch.stack([b[i] for b, i in zip(self.books, d.argmin(-1))])
        return (soft + (hard - soft).detach()).transpose(0, 1).reshape(h.shape[1], -1)

    @torch.no_grad()
    def encode(self, joints, mask):
        cores, obs, seen = raw_cores(joints, mask)
        x = (torch.from_numpy(cores).to(self.device) - self.mean) * torch.from_numpy(seen).to(self.device)
        return self.quantize(self.proj(x)), obs


# ------------------------------------------------------------------ reconstruction tokenizer

class Tokenizer(nn.Module):
    def __init__(self, dim=128, latent=128, layers=4, codes=512, decay=0.99):
        super().__init__()
        self.stem = nn.Sequential(nn.Linear(12, dim), nn.GELU(), nn.Linear(dim, dim))
        self.joint = nn.Parameter(torch.randn(1, 1, J, dim) * 0.02)
        self.blocks = nn.ModuleList(Block(dim, 2) for _ in range(layers))
        self.readout = nn.Linear(C * J * dim, latent)
        self.decoder = nn.Sequential(nn.Linear(latent, 512), nn.GELU(), nn.Linear(512, C * J * 3))
        self.decay = decay
        self.register_buffer("codebook", torch.randn(codes, latent) * 0.1)
        self.register_buffer("ema_count", torch.ones(codes))
        self.register_buffer("ema_sum", self.codebook.clone())

    def latents(self, joints, mask):
        """(B, T, 24, 3), (B, T, 23) -> pre-quantisation latents (B, N, latent), core targets, core masks."""
        x, M, r = features(joints, mask)
        xw, Mw, _ = windows(x, M, r)
        b, n = xw.shape[:2]
        xw, Mw = xw.flatten(0, 1), Mw.flatten(0, 1)
        prev = F.pad(xw, (0, 0, 0, 0, 1, 0))[:, :-1]
        nxt = F.pad(xw, (0, 0, 0, 0, 0, 1))[:, 1:]
        mp, mn = F.pad(Mw, (0, 0, 1, 0))[:, :-1], F.pad(Mw, (0, 0, 0, 1))[:, 1:]
        h = Mw[..., None] * (self.stem(torch.cat([prev, xw, nxt, mp[..., None], Mw[..., None], mn[..., None]], -1))
                             + self.joint)
        for block in self.blocks:
            h = block(h, Mw)
        z = self.readout(h[:, W:W + C].reshape(len(h), -1)).view(b, n, -1)
        return z, xw[:, W:W + C].reshape(b, n, -1), Mw[:, W:W + C].reshape(b, n, C, J)

    def quantize(self, z):
        flat = z.reshape(-1, z.shape[-1])
        codes = torch.cdist(flat, self.codebook).argmin(1)
        if self.training:
            with torch.no_grad():
                onehot = F.one_hot(codes, len(self.codebook)).float()
                self.ema_count.mul_(self.decay).add_(onehot.sum(0), alpha=1 - self.decay)
                self.ema_sum.mul_(self.decay).add_(onehot.T @ flat, alpha=1 - self.decay)
                n = self.ema_count.sum()
                count = (self.ema_count + 1e-5) / (n + len(self.codebook) * 1e-5) * n
                self.codebook.copy_(self.ema_sum / count[:, None])
        q = self.codebook[codes].view_as(z)
        return z + (q - z).detach(), codes.view(z.shape[:-1])


def train_tokenizer(train, device, steps=4000, identity=False, seed=0, clip=120, batch_size=32, log=print):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model = Tokenizer().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    labels = sorted({r["label"] for r in train})
    for step in range(steps):
        picks = rng.integers(0, len(train), batch_size)
        joints, masks, ys = [], [], []
        for p in picks:
            r = train[p]
            s = int(rng.integers(0, max(1, len(r["joints"]) - clip + 1)))
            j = r["joints"][s:s + clip]
            if len(j) < clip:
                j = np.concatenate([j, np.repeat(j[-1:], clip - len(j), 0)])
            joints.append(j)
            ys.append(labels.index(r["label"]))
        joints = torch.from_numpy(np.stack(joints)).to(device)
        mask = torch.ones(joints.shape[:2] + (J,), dtype=torch.bool, device=device)
        z, target, cm = model.latents(joints, mask)
        q, _ = model.quantize(z)
        recon = model.decoder(q)
        loss = F.mse_loss(recon, target) + 0.25 * F.mse_loss(z, q.detach())
        if identity:
            u = F.normalize(z.mean(1), dim=-1)
            y = torch.tensor(ys, device=device)
            logits = (u @ u.T / 0.1).masked_fill(torch.eye(len(u), dtype=torch.bool, device=device), float("-inf"))
            pos = (y[:, None] == y[None]) & ~torch.eye(len(u), dtype=torch.bool, device=device)
            lp = logits - logits.logsumexp(1, keepdim=True)
            has = pos.any(1)
            if has.any():
                loss = loss + (-(lp.masked_fill(~pos, 0).sum(1) / pos.sum(1).clamp(min=1))[has]).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 500 == 0:
            log(f"tokenizer{'+id' if identity else ''} step {step} loss {float(loss):.4f}")
    model.eval()
    return model


@torch.no_grad()
def tokenizer_vectors(model, joints, mask, device):
    """-> word histogram (codes) and mean latent (continuous) over observed cores, codes used."""
    j = torch.from_numpy(joints[None]).to(device)
    m = torch.from_numpy(mask[None]).to(device)
    z, _, cm = model.latents(j, m)
    _, codes = model.quantize(z)
    seen = cm[0].float().mean((1, 2)) >= 0.5
    if not seen.any():
        seen = torch.ones_like(seen)
    hist = torch.bincount(codes[0][seen], minlength=len(model.codebook)).float()
    return F.normalize(hist, dim=0), F.normalize(z[0][seen].mean(0), dim=0), codes[0][seen]
