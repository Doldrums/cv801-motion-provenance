"""Staged identity training (Algorithm 1) and codebook fitting.

Each batch draws two performances (other sources) of each of P choreographies
from the AIST++ train split, crops a clip from each, hides part of it with the
occlusion prior and rotates / rescales it. The loss is supervised contrast on
the clip's normalised mean descriptor, consistency between each core and the
same core of a transformed copy, and uniformity. No reconstruction target.
Stage 1 trains the anchor; stage 2 the query branch with the anchor frozen.
Afterwards residual codebooks are fitted on the frozen training descriptors.

    python -m motionprov.choreomatch.train --corpus aist.pkl --output OUT
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from motionprov.choreomatch.cre import RVQ, Index
from motionprov.choreomatch.data import PARTS
from motionprov.choreomatch.model import C, Encoder, features
from motionprov.choreomatch import protocol

CLIP = 120


# ------------------------------------------------------------------ batches

def occlusion(t, rng):
    """Occlusion prior: (T, 23) joint mask."""
    mask = np.ones((t, 23), bool)
    kind = rng.random()
    if kind < 0.3:
        return mask
    if kind < 0.6:
        part = list(PARTS.values())[rng.integers(len(PARTS))]
        mask[:, [j - 1 for j in part]] = False
    elif kind < 0.8:
        gap, period = int(rng.integers(4, 13)), 40
        mask[(np.arange(t) + rng.integers(period)) % period < gap] = False
    else:
        mask &= rng.random((t, 23)) > 0.2
        for _ in range(int(rng.integers(1, 4))):
            s = int(rng.integers(t))
            mask[s:s + int(rng.integers(5, 30))] = False
    return mask


def transform(rng, yaw_range=np.pi, tilt=np.radians(10)):
    """Random yaw (within +-yaw_range), small pitch / roll and scale about the origin, as one (3, 3) matrix."""
    yaw, pitch, roll = rng.uniform(-yaw_range, yaw_range), rng.uniform(-tilt, tilt), rng.uniform(-tilt, tilt)
    cy, sy, cp, sp, cr, sr = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return (rz @ rx @ ry * rng.uniform(0.8, 1.2)).astype(np.float32)


def retime(joints, observed, frames):
    """Resample a segment to `frames` frames: joints by linear interpolation, observed where both ends are."""
    t = np.linspace(0, len(joints) - 1, frames)
    lo = np.floor(t).astype(int)
    hi = np.minimum(lo + 1, len(joints) - 1)
    w = (t - lo)[:, None, None].astype(np.float32)
    return joints[lo] * (1 - w) + joints[hi] * w, observed[lo] & observed[hi]


def session(rec):
    """Dancer and music of an AIST++ recording (gLH_sBM_cAll_d16_mLH0_ch01 -> d16_mLH0); None elsewhere."""
    parts = rec["id"].split("_")
    return "_".join(parts[3:5]) if len(parts) >= 6 else None


def hard_picks(by_label, labels_per_batch, rng):
    """Labels drawn genre by genre; the first performance of each label shares, where it can, the dancer
    and music of the other labels' first performances, so that same-dancer, same-tempo negatives meet."""
    genres = {}
    for k, recs in enumerate(by_label):
        genres.setdefault(recs[0]["label"][:3], []).append(k)
    chosen = []
    for g in rng.permutation(list(genres)):
        chosen += list(rng.permutation(genres[g]))
        if len(chosen) >= labels_per_batch:
            break
    chosen = chosen[:labels_per_batch]
    sessions = {}
    for k in chosen:
        for i, r in enumerate(by_label[k]):
            sessions.setdefault(by_label[k][0]["label"][:3], {}).setdefault(session(r), []).append((k, i))
    picks = []
    for k in chosen:
        recs = by_label[k]
        shared = [s for s, v in sessions[recs[0]["label"][:3]].items() if s and len({kk for kk, _ in v}) > 1]
        first = None
        if shared:
            key = shared[int(rng.integers(len(shared)))]
            first = next((i for kk, i in sessions[recs[0]["label"][:3]][key] if kk == k), None)
        if first is None:
            first = int(rng.integers(len(recs)))
        others = [i for i in range(len(recs)) if recs[i]["source"] != recs[first]["source"]] or \
                 [i for i in range(len(recs)) if i != first]
        picks.append((k, [first, others[int(rng.integers(len(others)))]]))
    return picks


def aligned_pair(a, b, clip, rate, rng):
    """The same stretch of the music in two AIST++ performances of one basic choreography, both played at
    one tempo: the recordings start with the music and span the same beats, so time maps linearly."""
    scale = len(b["joints"]) / len(a["joints"])
    span = min(len(a["joints"]), int(round(clip * rate)))
    s = int(rng.integers(0, len(a["joints"]) - span + 1))
    sb, spanb = int(round(s * scale)), max(2, int(round(span * scale)))
    sb = min(sb, len(b["joints"]) - spanb)
    out = []
    for r, start, n in ((a, s, span), (b, sb, spanb)):
        out.append(retime(r["joints"][start:start + n], r["observed"][start:start + n], clip))
    return out


def batch(by_label, labels_per_batch, rng, device, clip=CLIP, yaw_range=np.pi, speed=1.0, hard=False,
          aligned=False):
    if hard:
        picks = hard_picks(by_label, labels_per_batch, rng)
    else:
        picks = [(k, rng.choice(len(by_label[k]), 2, replace=False))
                 for k in rng.choice(len(by_label), labels_per_batch, replace=False)]
    joints, masks, ys, dancers = [], [], [], []
    for k, pair in picks:
        recs = by_label[k]
        if aligned:
            rate = float(np.exp(rng.uniform(-np.log(speed), np.log(speed))))
            views = aligned_pair(recs[pair[0]], recs[pair[1]], clip, rate, rng)
        for n, i in enumerate(pair):
            r = recs[i]
            rate = float(np.exp(rng.uniform(-np.log(speed), np.log(speed))))     # tempo of the copy
            span = int(round(clip * rate))
            s = int(rng.integers(0, max(1, len(r["joints"]) - span + 1)))
            j, o = r["joints"][s:s + span], r["observed"][s:s + span]
            if aligned:
                j, o = views[n]
            elif speed > 1:
                j, o = retime(j, o, max(2, min(clip, int(round(len(j) / rate)))))
            m = occlusion(len(j), rng) & o[:, None]
            if len(j) < clip:
                j = np.concatenate([j, np.repeat(j[-1:], clip - len(j), 0)])
                m = np.concatenate([m, np.zeros((clip - len(m), 23), bool)])
            joints.append(j)
            masks.append(m)
            ys.append(k)
            dancers.append(DANCERS.get(r["id"].split("_")[3], -1) if r["id"].count("_") >= 5 else -1)
    joints = torch.from_numpy(np.stack(joints)).to(device)
    masks = torch.from_numpy(np.stack(masks)).to(device)
    rot = torch.from_numpy(np.stack([transform(rng, yaw_range) for _ in ys])).to(device)
    rot2 = torch.from_numpy(np.stack([transform(rng, yaw_range) for _ in ys])).to(device)
    return joints, masks, torch.tensor(ys, device=device), rot, rot2, torch.tensor(dancers, device=device)


# ------------------------------------------------------------------ losses

DANCERS = {f"d{i:02d}": i - 1 for i in range(1, 31)}       # AIST++ performers d01..d30


class Reverse(torch.autograd.Function):
    """Identity forward, negated and scaled gradient backward (gradient reversal)."""

    @staticmethod
    def forward(ctx, x, weight):
        ctx.weight = weight
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.weight * grad, None


def pooled(z, observed):
    """Normalised mean descriptor over observed cores (kappa)."""
    k = (observed >= 0.5).float()[..., None]
    k = torch.where(k.sum(1, keepdim=True) > 0, k, torch.ones_like(k))
    return F.normalize((F.normalize(z, dim=-1) * k).sum(1), dim=-1)


def supcon(u, y, tau, dancer=None, weight=1.0):
    """Supervised contrastive loss; negatives danced by the anchor's own performer count `weight` times."""
    logits = u @ u.T / tau
    eye = torch.eye(len(u), dtype=torch.bool, device=u.device)
    if dancer is not None and weight != 1.0:
        own = (dancer[:, None] == dancer[None]) & (dancer[:, None] >= 0) & (y[:, None] != y[None])
        logits = logits + own * math.log(weight)
    logits = logits.masked_fill(eye, float("-inf"))
    log_prob = logits - logits.logsumexp(1, keepdim=True)
    pos = (y[:, None] == y[None]) & ~eye
    return -(log_prob.masked_fill(~pos, 0).sum(1) / pos.sum(1).clamp(min=1)).mean()


def core_contrast(z, observed, tau):
    """Cores at the same moment of two aligned performances (rows 2p, 2p + 1) are positives; every core of
    the other pairs' second performance is a negative; other moments of the same pair are left out."""
    a, b = F.normalize(z[0::2], dim=-1), F.normalize(z[1::2], dim=-1)
    p, n = a.shape[:2]
    va, vb = observed[0::2] >= 0.5, observed[1::2] >= 0.5
    logits = a.reshape(p * n, -1) @ b.reshape(p * n, -1).T / tau
    pair = torch.arange(p, device=z.device).repeat_interleave(n)
    eye = torch.eye(p * n, dtype=torch.bool, device=z.device)
    allowed = ((pair[:, None] != pair[None]) | eye) & vb.reshape(-1)[None]
    logits = logits.masked_fill(~allowed, float("-inf"))
    rows = va.reshape(-1) & vb.reshape(-1)
    loss = -(logits.diagonal() - logits.logsumexp(1))[rows]
    return loss.mean() if rows.any() else logits.sum() * 0


def core_targets(joints, mask):
    """Pelvis-relative joints of every core -> (B, N, C * 23 * 3) targets and (B, N, C * 23 * 3) weights."""
    b, t = joints.shape[:2]
    n = math.ceil(t / C)
    x = (joints[:, :, 1:] - joints[:, :, :1]) * mask[..., None]
    w = mask[..., None].expand_as(x).float()
    pad = (0, 0, 0, 0, 0, n * C - t)
    return F.pad(x, pad).reshape(b, n, -1), F.pad(w, pad).reshape(b, n, -1)


def quantize(z, books):
    """Residual quantization of normalised cores, straight-through -> codes-valued z, codebook + commitment loss."""
    zn = F.normalize(z, dim=-1)
    residual, recon = zn, torch.zeros_like(zn)
    for book in books:
        q = book[torch.cdist(residual.reshape(-1, residual.shape[-1]), book).argmin(1)].view_as(residual)
        recon = recon + q
        residual = residual - q
    loss = F.mse_loss(recon, zn.detach()) + 0.25 * F.mse_loss(zn, recon.detach())
    return zn + (recon - zn).detach(), loss


def uniformity(u):
    d = torch.cdist(u, u).pow(2)
    off = ~torch.eye(len(u), dtype=torch.bool, device=u.device)
    return torch.log(torch.exp(-2 * d[off]).mean())


def encode(model, joints, mask, use_query=True):
    stride = int(model.stride)
    x, M, r = features(joints[:, ::stride], mask[:, ::stride])
    return model(x, M, r, use_query=use_query)


# ------------------------------------------------------------------ evaluation helpers

@torch.no_grad()
def descriptors(model, rec, condition, kind, device, use_query=True):
    joints, mask = protocol.query_view(rec, condition, kind)
    j = torch.from_numpy(joints[None]).to(device)
    m = torch.from_numpy(mask[None]).to(device)
    z, obs = encode(model, j, m, use_query)
    return F.normalize(z[0], dim=-1), obs[0]


def validation_map(model, records, device, use_query=True, conditions=protocol.CONDITIONS):
    model.eval()
    index_items = []
    for r in records:
        z, o = descriptors(model, r, "clean", "whole", device, use_query)
        index_items.append({"vectors": z, "observed": o})
    index = Index(index_items, device)
    cache = {}

    def system(q, condition, kind):
        if (q["id"], condition) not in cache:
            z, o = descriptors(model, q, condition, kind, device, use_query)
            cache[(q["id"], condition)] = index.score(z, o)[0].cpu().numpy()
        return cache[(q["id"], condition)]

    rows = protocol.evaluate(system, records, records, conditions, ("whole",))
    model.train()
    return float(np.mean([x["ap"] for r in rows.values() for x in r]))


# ------------------------------------------------------------------ main

def stage(model, params, by_label, val, args, use_query, steps, log, output, name, device, rng):
    if args.adversary:
        params = list(params) + list(args.dancer_head.parameters())
    if args.aux_decoder and not use_query:
        params = list(params) + list(args.decoder.parameters())
    if args.joint_codebooks and not use_query:
        params = list(params) + [args.books]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / 300) * 0.5 * (1 + math.cos(math.pi * min(1, s / max(steps, 1)))))
    best = -1.0
    for step in range(steps + 1):
        joints, mask, y, rot, rot2, dancer = batch(by_label, args.labels_per_batch, rng, device, args.clip,
                                           np.radians(args.yaw), args.speed, args.hard, args.core > 0)
        with torch.autocast(device.type, torch.bfloat16):
            z, obs = encode(model, joints @ rot.transpose(1, 2)[:, None], mask, use_query)
            z2, _ = encode(model, joints @ rot2.transpose(1, 2)[:, None], mask, use_query)
        z, z2 = z.float(), z2.float()
        aux = torch.zeros((), device=device)
        if args.aux_decoder and not use_query:       # Table 3: reconstruct each core's pose from its descriptor
            target, weight = core_targets(joints @ rot.transpose(1, 2)[:, None], mask)
            aux = args.aux_decoder * (((args.decoder(z) - target) ** 2) * weight).sum() / weight.sum().clamp(min=1)
        if args.joint_codebooks and not use_query:   # Table 3: codebooks trained with the encoder, straight through
            if step == 0 and not args.books_ready:
                with torch.no_grad():
                    args.books.copy_(torch.stack(RVQ.fit(F.normalize(z.reshape(-1, z.shape[-1]), dim=-1),
                                                         args.codebooks, args.codebook_size).codebooks))
                args.books_ready = True
            z, vq = quantize(z, args.books)
            aux = aux + vq
        u = pooled(z, obs)
        seen = (obs >= 0.5).float()
        cons = ((1 - F.cosine_similarity(z, z2, dim=-1)) * seen).sum() / seen.sum().clamp(min=1)
        loss = supcon(u, y, args.tau, dancer, args.dancer_neg) + args.lambda_cons * cons + args.lambda_uni * uniformity(u) + aux
        if args.core > 0:
            loss = loss + args.core * core_contrast(z, obs, args.tau)
        known = dancer >= 0
        if args.adversary and known.any():                # forget the performer: the head cannot name them
            logits = args.dancer_head(Reverse.apply(u[known], args.adversary))
            loss = loss + F.cross_entropy(logits, dancer[known])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if step % args.eval_every == 0:
            m = validation_map(model, val, device, use_query)
            row = {"stage": name, "step": step, "loss": float(loss), "cons": float(cons), "val_mAP": m}
            log.append(row)
            print(json.dumps(row), flush=True)
            if m > best or not args.select:          # validation inside training: keep the last step
                best = m
                torch.save(model.state_dict(), output / f"{name}.pt")
                if args.joint_codebooks and not use_query:
                    torch.save(args.books.detach().cpu(), output / "joint_books.pt")
    model.load_state_dict(torch.load(output / f"{name}.pt", map_location=device))
    return best


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--anchor-steps", type=int, default=6000)
    parser.add_argument("--query-steps", type=int, default=3000)
    parser.add_argument("--labels-per-batch", type=int, default=32)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--lambda-cons", type=float, default=0.5)
    parser.add_argument("--lambda-uni", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--codebooks", type=int, default=4)
    parser.add_argument("--codebook-size", type=int, default=256)
    parser.add_argument("--clip", type=int, default=CLIP, help="training clip length, frames")
    parser.add_argument("--yaw", type=float, default=180, help="yaw augmentation range, +- degrees")
    parser.add_argument("--speed", type=float, default=1.0, help="tempo augmentation: rate in [1/speed, speed]")
    parser.add_argument("--adversary", type=float, default=0.0, help="gradient-reversal weight of a dancer classifier")
    parser.add_argument("--stride", type=int, default=1, help="encoder input frame stride (2: 15 fps, cores of 20 frames)")
    parser.add_argument("--init", help="anchor checkpoint to start from; skips the anchor stage")
    parser.add_argument("--core", type=float, default=0.0, help="weight of the aligned cross-performer core contrast")
    parser.add_argument("--hard", action="store_true", help="genre-grouped batches with same-session negatives")
    parser.add_argument("--dancer-neg", type=float, default=1.0,
                        help="weight of negatives danced by the anchor's own performer in SupCon")
    parser.add_argument("--aux-decoder", type=float, default=0.0,
                        help="weight of an auxiliary pose decoder on the anchor's cores (zero-initialised)")
    parser.add_argument("--joint-codebooks", action="store_true",
                        help="train the residual codebooks with the anchor (straight-through) instead of fitting after")
    parser.add_argument("--train-splits", default="train", help="comma-separated splits to train on")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    records = pickle.load(open(args.corpus, "rb"))
    train = [r for r in records if r["split"] in args.train_splits.split(",")]
    args.select = "validation" not in args.train_splits.split(",")
    val = [r for r in records if r["split"] == "validation"]
    groups = {}
    for r in train:
        groups.setdefault(r["label"], []).append(r)
    by_label = [v for v in groups.values() if len({r["source"] for r in v}) > 1]
    args.labels_per_batch = min(args.labels_per_batch, len(by_label))
    print(f"train {len(train)} recordings, {len(by_label)} labels with two sources; validation {len(val)}")

    model = Encoder().to(device)
    model.stride.fill_(args.stride)
    args.dancer_head = nn.Linear(128, len(DANCERS)).to(device)
    args.decoder = nn.Sequential(nn.Linear(128, 512), nn.GELU(), nn.Linear(512, C * 23 * 3)).to(device)
    nn.init.zeros_(args.decoder[-1].weight)
    nn.init.zeros_(args.decoder[-1].bias)
    args.books = nn.Parameter(torch.zeros(args.codebooks, args.codebook_size, 128, device=device))
    args.books_ready = False
    log = []
    if args.init:                                          # a trained anchor: only the query stage runs
        model.load_state_dict(torch.load(args.init, map_location=device), strict=False)
        model.stride.fill_(args.stride)
        best_anchor = validation_map(model, val, device, use_query=False)
        torch.save(model.state_dict(), output / "anchor.pt")
    else:
        best_anchor = stage(model, model.anchor_parameters(), by_label, val, args, False, args.anchor_steps,
                            log, output, "anchor", device, rng)
    for p in model.anchor_parameters():
        p.requires_grad_(False)
    best_query = stage(model, list(model.query.parameters()), by_label, val, args, True, args.query_steps,
                       log, output, "query", device, rng)

    # Codebooks on the frozen descriptors of the clean training recordings.
    model.eval()
    zs = []
    for r in train:
        z, o = descriptors(model, r, "clean", "whole", device)
        zs.append(z[o >= 0.5])
    z = torch.cat(zs)
    books = {"residual": RVQ.fit(z, args.codebooks, args.codebook_size, args.seed).state(),
             "single": RVQ.fit(z, 1, 1024, args.seed).state()}
    if args.joint_codebooks:                               # the codebooks learned with the selected anchor
        books["residual"] = list(torch.load(output / "joint_books.pt").unbind(0))
    torch.save(books, output / "codebooks.pt")
    torch.save(model.state_dict(), output / "model.pt")
    summary = {"train_descriptors": len(z), "best_val_anchor": best_anchor, "best_val_query": best_query,
               "config": {k: v for k, v in vars(args).items() if k not in ("dancer_head", "decoder", "books", "books_ready")}}
    (output / "train_log.json").write_text(json.dumps(log, indent=1))
    (output / "train_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
