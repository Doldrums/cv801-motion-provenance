"""The ChoreoMatch encoder (Section 4.2, Appendix D).

Input per recording: pelvis-relative joints x (T, 23, 3) with a mask M (T, 23)
and root motion r (T, 4) = (v_x, v_z, yaw rate, availability). A recording is
cut into cores of C frames; each core is read with W frames of context on
either side (C + 2W = 28).

* looking:  per-joint stem h0 = M (g([x_{t-1:t+1}; M_{t-1:t+1}]) + e_j), then L
            blocks of masked time attention (same joint, |s - t| <= delta) and
            masked body attention (same frame); hidden joints give no keys and
            are reset to zero after every block.
* writing:  the ordered anchor concatenates frames, then joints (Eq. 3), and adds
            gated part, root and interaction corrections (Eq. 4).
* reading:  the query branch lets hidden parts query visible ones, never the
            other way round, and adds a zero-initialised correction (Eqs. 5, 6).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from motionprov.choreomatch.data import FPS, PARTS

C, W = 10, 9                      # core frames, context frames per side
J = 23
PART_IDS = [[j - 1 for j in ids] for ids in PARTS.values()]     # into the 23 pelvis-relative joints
K = len(PART_IDS)
LEFT_HIP, RIGHT_HIP = 0, 1        # joints 1, 2 of SMPL in the 23-joint order


# ------------------------------------------------------------------- features

def features(joints, observed):
    """(B, T, 24, 3) joints, (B, T, 23) joint mask -> x (B, T, 23, 3), M, r (B, T, 4)."""
    pelvis = joints[:, :, 0]
    x = (joints[:, :, 1:] - pelvis[:, :, None]) * observed[..., None]
    seen = observed.any(-1)
    have = seen.clone()
    have[:, 1:-1] = seen[:, :-2] & seen[:, 1:-1] & seen[:, 2:]
    have[:, [0, -1]] = False
    vel = torch.zeros_like(pelvis)
    vel[:, 1:-1] = (pelvis[:, 2:] - pelvis[:, :-2]) * FPS / 2
    hips = joints[:, :, 1 + LEFT_HIP] - joints[:, :, 1 + RIGHT_HIP]
    yaw = torch.atan2(hips[..., 2], hips[..., 0])
    turn = torch.zeros_like(yaw)
    turn[:, 1:-1] = torch.remainder(yaw[:, 2:] - yaw[:, :-2] + math.pi, 2 * math.pi) - math.pi
    turn = turn * FPS / 2
    hips_seen = observed[..., LEFT_HIP] & observed[..., RIGHT_HIP]
    have = have & hips_seen
    r = torch.stack([vel[..., 0], vel[..., 2], turn, have.float()], -1) * have[..., None]
    return x, observed.float(), r


def windows(x, M, r):
    """Cut (B, T, ...) into cores with context -> (B, N, C + 2W, ...) and core masks; T padded to N * C."""
    b, t = x.shape[:2]
    n = max(1, math.ceil(t / C))
    pad = (W, n * C - t + W)
    xp = F.pad(x, (0, 0, 0, 0) + pad)
    Mp = F.pad(M, (0, 0) + pad)
    rp = F.pad(r, (0, 0) + pad)
    idx = torch.arange(n, device=x.device)[:, None] * C + torch.arange(C + 2 * W, device=x.device)
    return xp[:, idx], Mp[:, idx], rp[:, idx]


# ------------------------------------------------------------------- attention

class MaskedAttention(nn.Module):
    """Attention whose keys are limited by a boolean mask; a row with no allowed key outputs zero."""

    def __init__(self, dim, heads=4):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, h, allowed, kv=None):
        """h (S, L, D) queries; allowed (S, L, Lk) bool; kv (S, Lk, D) keys/values (default h)."""
        s, l, d = h.shape
        kv = h if kv is None else kv
        q = F.linear(h, self.qkv.weight[:d], self.qkv.bias[:d])
        k, v = F.linear(kv, self.qkv.weight[d:], self.qkv.bias[d:]).chunk(2, -1)
        split = lambda z: z.view(z.shape[0], z.shape[1], self.heads, -1).transpose(1, 2)
        q, k, v = split(q), split(k), split(v)
        logits = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
        logits = logits.masked_fill(~allowed[:, None], float("-inf"))
        w = torch.nan_to_num(logits.softmax(-1), nan=0.0)
        o = (w @ v).transpose(1, 2).reshape(s, l, d)
        return self.out(o) * allowed.any(-1, keepdim=True)


def ffn(dim, mult=4):
    return nn.Sequential(nn.Linear(dim, dim * mult), nn.GELU(), nn.Linear(dim * mult, dim))


class Block(nn.Module):
    def __init__(self, dim, delta):
        super().__init__()
        self.delta = delta
        self.n1, self.n2, self.n3 = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.time, self.body, self.ffn = MaskedAttention(dim), MaskedAttention(dim), ffn(dim)

    def forward(self, h, M):
        n, t, j, d = h.shape
        obs = M > 0
        band = (torch.arange(t, device=h.device)[:, None] - torch.arange(t, device=h.device)).abs() <= self.delta
        ht = h.permute(0, 2, 1, 3).reshape(n * j, t, d)
        keys = obs.permute(0, 2, 1).reshape(n * j, 1, t)
        ht = ht + self.time(self.n1(ht), band[None] & keys)
        h = ht.view(n, j, t, d).permute(0, 2, 1, 3)
        hb = h.reshape(n * t, j, d)
        keys = obs.reshape(n * t, 1, j).expand(-1, j, -1)
        hb = hb + self.body(self.n2(hb), keys)
        hb = hb + self.ffn(self.n3(hb))
        return hb.view(n, t, j, d) * M[..., None]


# ------------------------------------------------------------------- encoder

class Anchor(nn.Module):
    """Ordered readout (Eq. 3) plus gated part, root and interaction corrections (Eq. 4)."""

    def __init__(self, dim, out, du=32, dp=64):
        super().__init__()
        self.wt = nn.Linear(C * dim, du)
        self.wj = nn.Linear(J * du, out)
        self.norm = nn.LayerNorm(out)
        self.part_in = nn.Linear(du, dp)
        self.part_norm = nn.LayerNorm(dp)
        self.part_attn = MaskedAttention(dp)
        self.part_out = nn.Sequential(nn.LayerNorm(K * dp), nn.Linear(K * dp, out))
        self.root_in = nn.Sequential(nn.Linear(C * 4, dp), nn.LayerNorm(dp), nn.GELU(), nn.Linear(dp, dp))
        self.root_out = nn.Sequential(nn.LayerNorm(dp), nn.Linear(dp, out))
        self.cross = MaskedAttention(dp)
        self.int_out = nn.Sequential(nn.Linear(dp, out), nn.LayerNorm(out))
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.beta = nn.Parameter(torch.tensor(0.1))
        self.gamma = nn.Parameter(torch.tensor(0.1))

    def forward(self, H, Mc, rc):
        """H (N, C, J, D) core states, Mc (N, C, J), rc (N, C, 4) -> (N, out)."""
        n = H.shape[0]
        U = F.gelu(self.wt(H.permute(0, 2, 1, 3).reshape(n, J, -1)))                   # (N, J, du)
        z_body = self.norm(self.wj(U.reshape(n, -1)))
        joint_seen = Mc.amax(1)                                                          # (N, J)
        parts, part_seen = [], []
        for ids in PART_IDS:
            w = joint_seen[:, ids]
            parts.append((U[:, ids] * w[..., None]).sum(1) / w.sum(1, keepdim=True).clamp(min=1))
            part_seen.append(w.amax(1) > 0)
        P = self.part_in(torch.stack(parts, 1))                                          # (N, K, dp)
        seen = torch.stack(part_seen, 1)                                                 # (N, K)
        P = P + self.part_attn(self.part_norm(P), seen[:, None, :].expand(-1, K, -1))
        P = P * seen[..., None]
        d_parts = self.part_out(P.reshape(n, -1))
        q = rc[..., 3].mean(1, keepdim=True)                                             # observed root fraction
        root = self.root_in(rc.reshape(n, -1))
        d_root = self.root_out(root)
        inter = self.cross(root[:, None], seen[:, None, :], kv=P)[:, 0]
        d_int = self.int_out(inter)
        support = ((joint_seen.sum(1, keepdim=True) > 0) & (q > 0) & seen.any(1, keepdim=True)).float()
        return z_body + self.alpha * d_parts + self.beta * q * d_root + self.gamma * q * support * d_int


class QueryBranch(nn.Module):
    """Hidden parts may query visible ones but never serve as evidence (Eqs. 5, 6)."""

    def __init__(self, dim, out, dq=128):
        super().__init__()
        self.part_in = nn.Linear(dim + 1, dq)
        self.root_in = nn.Linear(4, dq)
        self.kind = nn.Parameter(torch.zeros(1, 1, K + 1, dq))
        self.time = nn.Parameter(torch.zeros(1, C, 1, dq))
        nn.init.normal_(self.kind, std=0.02)
        nn.init.normal_(self.time, std=0.02)
        self.steps = nn.ModuleList()
        for _ in range(2):                                   # within a frame, then within a token type
            self.steps.append(nn.ModuleDict({"n1": nn.LayerNorm(dq), "attn": MaskedAttention(dq),
                                             "n2": nn.LayerNorm(dq), "ffn": ffn(dq)}))
        self.w0 = nn.Linear((K + 1) * dq, out)
        nn.init.zeros_(self.w0.weight)
        nn.init.zeros_(self.w0.bias)

    def forward(self, H, Mc, rc):
        n = H.shape[0]
        tokens, evidence = [], []
        for ids in PART_IDS:
            m = Mc[:, :, ids]                                                        # (N, C, |P|)
            frac = m.mean(-1, keepdim=True)
            mean = (H[:, :, ids] * m[..., None]).sum(2) / m.sum(-1, keepdim=True).clamp(min=1)
            tokens.append(self.part_in(torch.cat([mean, frac], -1)))
            evidence.append(frac[..., 0] > 0)
        tokens.append(self.root_in(rc))
        evidence.append(rc[..., 3] > 0)
        V = torch.stack(tokens, 2) + self.kind + self.time                            # (N, C, K+1, dq)
        E = torch.stack(evidence, 2)                                                 # (N, C, K+1)
        d = V.shape[-1]
        for axis, step in enumerate(self.steps):
            if axis == 0:
                v, e = V.reshape(n * C, K + 1, d), E.reshape(n * C, 1, K + 1).expand(-1, K + 1, -1)
            else:
                v = V.permute(0, 2, 1, 3).reshape(n * (K + 1), C, d)
                e = E.permute(0, 2, 1).reshape(n * (K + 1), 1, C).expand(-1, C, -1)
            v2 = v + step["attn"](step["n1"](v), e)
            v = v2 + step["ffn"](step["n2"](v2))                                      # permission mask is all ones
            V = v.view(n, C, K + 1, d) if axis == 0 else v.view(n, K + 1, C, d).permute(0, 2, 1, 3)
        return self.w0(V.mean(1).reshape(n, -1))


class Encoder(nn.Module):
    def __init__(self, dim=128, out=128, layers=4, delta=2, query=True):
        super().__init__()
        self.stem = nn.Sequential(nn.Linear(12, dim), nn.GELU(), nn.Linear(dim, dim))
        self.joint = nn.Parameter(torch.zeros(1, 1, J, dim))
        nn.init.normal_(self.joint, std=0.02)
        self.blocks = nn.ModuleList(Block(dim, delta) for _ in range(layers))
        self.anchor = Anchor(dim, out)
        self.query = QueryBranch(dim, out) if query else None
        self.register_buffer("stride", torch.tensor(1))   # input frame stride; a core spans C * stride frames
        self.register_load_state_dict_pre_hook(
            lambda module, state, prefix, *_: state.setdefault(prefix + "stride", torch.tensor(1)))

    def anchor_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith("query.")]

    def forward(self, x, M, r, use_query=True, chunk=2048):
        """x (B, T, 23, 3), M (B, T, 23), r (B, T, 4) -> z (B, N, out), core observed fraction (B, N)."""
        xw, Mw, rw = windows(x, M, r)
        b, n = xw.shape[:2]
        xw, Mw, rw = xw.flatten(0, 1), Mw.flatten(0, 1), rw.flatten(0, 1)
        out = [self._cores(xw[i:i + chunk], Mw[i:i + chunk], rw[i:i + chunk], use_query)
               for i in range(0, len(xw), chunk)]
        z = torch.cat(out).view(b, n, -1)
        observed = Mw[:, W:W + C].mean((1, 2)).view(b, n)
        return z, observed

    def _cores(self, x, M, r, use_query):
        prev = F.pad(x, (0, 0, 0, 0, 1, 0))[:, :-1]
        nxt = F.pad(x, (0, 0, 0, 0, 0, 1))[:, 1:]
        mp = F.pad(M, (0, 0, 1, 0))[:, :-1]
        mn = F.pad(M, (0, 0, 0, 1))[:, 1:]
        inp = torch.cat([prev, x, nxt, mp[..., None], M[..., None], mn[..., None]], -1)
        h = M[..., None] * (self.stem(inp) + self.joint)
        for block in self.blocks:
            h = block(h, M)
        H, Mc, rc = h[:, W:W + C], M[:, W:W + C], r[:, W:W + C]
        z = self.anchor(H, Mc, rc)
        if use_query and self.query is not None:
            z = z + self.query(H, Mc, rc)
        return z
