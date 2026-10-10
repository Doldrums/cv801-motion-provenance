"""The Choreography Retrieval Engine (Section 4.3, 4.4).

* residual quantization: R codebooks fitted by k-means, each on the residual
  the previous stages left (Eq. 8), on frozen training descriptors;
* scoring (Eq. 9): the query slides over the reference; at each lag only cores
  sufficiently observed in both vote, each with the cosine of the two cores'
  reconstructions (geometry) or with 1[codes equal] (equality); the best lag wins;
* rejection (Eq. 10): the best reference from another recording is accepted
  when its score clears a threshold calibrated for a target false acceptance.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

OBSERVED = 0.5            # a core votes when at least this fraction of its joint-frames is observed
MIN_OVERLAP = 0.5         # a lag needs votes from at least this fraction of the shorter observed side


def kmeans(x, k, iters=30, seed=0):
    g = torch.Generator(device=x.device).manual_seed(seed)
    centres = x[torch.randperm(len(x), generator=g, device=x.device)[:k]].clone()
    for _ in range(iters):
        assign = torch.cdist(x, centres).argmin(1)
        sums = torch.zeros_like(centres).index_add_(0, assign, x)
        counts = torch.bincount(assign, minlength=k).float()[:, None]
        empty = counts[:, 0] == 0
        centres = torch.where(empty[:, None], centres, sums / counts.clamp(min=1))
        if empty.any():                                   # re-seed empty cells on the worst-fit points
            far = torch.cdist(x, centres).min(1).values.topk(int(empty.sum())).indices
            centres[empty] = x[far]
    return centres


class RVQ:
    def __init__(self, codebooks):
        self.codebooks = codebooks                        # list of (k, d)

    @classmethod
    def fit(cls, z, stages=4, size=256, seed=0):
        residual, books = z.clone(), []
        for s in range(stages):
            book = kmeans(residual, size, seed=seed + s)
            residual = residual - book[torch.cdist(residual, book).argmin(1)]
            books.append(book)
        return cls(books)

    def encode(self, z):
        residual, codes = z.clone(), []
        recon = torch.zeros_like(z)
        for book in self.codebooks:
            a = torch.cdist(residual, book).argmin(1)
            codes.append(a)
            recon = recon + book[a]
            residual = residual - book[a]
        return torch.stack(codes, -1), recon

    def state(self):
        return [b.cpu() for b in self.codebooks]


def pad(seqs, device):
    """List of (n_i, d) -> (R, Nmax, d), lengths."""
    lengths = torch.tensor([len(s) for s in seqs], device=device)
    out = torch.zeros(len(seqs), int(lengths.max()), seqs[0].shape[-1], device=device, dtype=seqs[0].dtype)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = s
    return out, lengths


class Index:
    """References as padded core sequences; `score` gives S(Q, X) and the best lag for every X."""

    def __init__(self, items, device, mode="geometry"):
        """items: list of dicts with 'vectors' (n, d) or 'codes' (n, R), and 'observed' (n,) fractions."""
        self.mode = mode
        key = "codes" if mode == "equality" else "vectors"
        self.vectors, self.lengths = pad([torch.as_tensor(it[key], device=device) for it in items], device)
        if mode != "equality":
            self.vectors = F.normalize(self.vectors.float(), dim=-1)
        obs, _ = pad([torch.as_tensor(it["observed"], device=device)[:, None] for it in items], device)
        self.observed = obs[..., 0] >= OBSERVED
        self.observed &= torch.arange(self.vectors.shape[1], device=device)[None] < self.lengths[:, None]

    def score(self, query, observed):
        """query (n, d) or (n, R); observed (n,) fractions -> scores (R,), best lags (R,)."""
        device = self.vectors.device
        q = torch.as_tensor(query, device=device)
        qo = torch.as_tensor(observed, device=device) >= OBSERVED
        n, nx = len(q), self.vectors.shape[1]
        if self.mode == "equality":
            sim = (q[None, :, None] == self.vectors[:, None]).all(-1).float()                 # (R, n, Nx)
        else:
            sim = torch.einsum("qd,rxd->rqx", F.normalize(q.float(), dim=-1), self.vectors)
        valid = qo[None, :, None] & self.observed[:, None]
        lag = (torch.arange(nx, device=device)[None] - torch.arange(n, device=device)[:, None] + n - 1).flatten()
        sums = torch.zeros(len(sim), n + nx - 1, device=device).index_add_(1, lag, (sim * valid).flatten(1))
        counts = torch.zeros_like(sums).index_add_(1, lag, valid.flatten(1).float())
        need = MIN_OVERLAP * torch.minimum(qo.sum().float(), self.observed.sum(1).float())
        ok = (counts >= need[:, None].clamp(min=1)) & (counts > 0)
        mean = torch.where(ok, sums / counts.clamp(min=1), torch.full_like(sums, float("-inf")))
        best, at = mean.max(1)
        return best, at - (n - 1)


def ranking(scores, seed=0):
    """Descending order; ties are broken at random (fixed seed), never by index position."""
    shuffle = np.random.default_rng(seed).permutation(len(scores))
    return shuffle[np.argsort(-scores[shuffle], kind="stable")]


def average_precision(scores, relevant):
    """AP over the full relevant set; a relevant reference without a finite score counts as a miss."""
    total = int(relevant.sum())
    if total == 0:
        return float("nan")
    order = ranking(scores)
    rel = relevant[order] & np.isfinite(scores[order])
    hits = np.cumsum(rel)
    return float((hits[rel] / (np.flatnonzero(rel) + 1)).sum() / total)


def reciprocal_rank(scores, relevant):
    order = ranking(scores)
    rel = relevant[order] & np.isfinite(scores[order])
    return float(1 / (np.flatnonzero(rel)[0] + 1)) if rel.any() else 0.0


def chance_ap(relevant, total):
    """Expected AP of a random ranking (Appendix A.4)."""
    m = np.arange(1, total + 1)
    return float(np.mean((1 + (m - 1) * (relevant - 1) / max(total - 1, 1)) / m))
