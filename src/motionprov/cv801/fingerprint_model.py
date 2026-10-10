"""A small trained fingerprint on top of frozen CoMotion: refiner, per-part evidence, visibility.

The proposal's model, kept small enough to train in minutes on one GPU:

* refiner r_theta: a temporal transformer over CoMotion frames (+ per-part 2D
  confidence) that predicts a residual on the joints. L_geo pulls the refined
  joints to the AIST++ fit, L_dyn their velocity and acceleration; both after
  one similarity alignment per clip, since camera and world frames differ.
* encoder: one token per (frame, body part), a transformer over all of them,
  then a per-part attention pool -> z_r, and a visibility weight v_r from that
  part's 2D confidence and its features.
* score: s(q, c) = sum_r v_r(q) v_r(c) cos(z_r(q), z_r(c)) / sum_r v_r(q) v_r(c),
  so a part that was not seen on either side carries little weight.
* L_fp: multi-positive InfoNCE on s, positives = same choreography from another
  take (another camera of the same take is neither positive nor negative,
  as in the retrieval protocol). Geometry and dynamics train first, L_fp after.

Training sees clean views and pixel-occluded copies of the train takes, so
hidden parts that CoMotion filled in appear with their own (low) confidence.
The train classes are disjoint from the confirmation classes.

    python -m motionprov.cv801.fingerprint_model --split SPLIT.json --reference REF_CACHE \
        --train clean=CACHE,SCORES --train occluded_arms=CACHE,SCORES ... \
        --validation clean=CACHE,SCORES --evaluate comotion_c01=CACHE,SCORES ... --output OUT
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from motionprov.core.datasets.local_pose_cache import load_pose_cache
from motionprov.core.evaluation.retrieval_evaluation import retrieval_metrics
from motionprov.core.models.local_extensions import PART_JOINT_IDS
from motionprov.cv801.diagnostic import load_views, skeleton
from motionprov.cv801.visibility import PART_KEYPOINTS, STRIDE

PARTS = tuple(PART_JOINT_IDS)
MAX_PART = max(len(ids) for ids in PART_JOINT_IDS.values())
NECK, PELVIS, LEFT_HIP, RIGHT_HIP = 12, 0, 1, 2
CLIP = 120              # frames at 15 fps = 8 s
EVAL_HOP = 60


# ----------------------------------------------------------------------------- data

def canonical(joints):
    """Rotate a clip (T, 24, 3) so the mean body up is +y and the mean hip line is +x."""
    up = (joints[:, NECK] - joints[:, PELVIS]).mean(0)
    up /= np.linalg.norm(up) + 1e-8
    hip = (joints[:, LEFT_HIP] - joints[:, RIGHT_HIP]).mean(0)
    x = hip - (hip @ up) * up
    x /= np.linalg.norm(x) + 1e-8
    z = np.cross(x, up)
    return joints @ np.stack([x, up, z]).T


def part_confidence(scores_dir: Path, video, frames):
    """(T, 5) per-part DWPose confidence on these frames, or None if not extracted."""
    path = scores_dir / f"{video}.npz"
    if not path.exists():
        return None
    data = np.load(path)
    lookup = dict(zip(data["frames"].tolist(), data["scores"]))
    out = np.ones((len(frames), len(PARTS)), np.float32)
    for i, f in enumerate(frames.tolist()):
        s = lookup.get(f)
        if s is not None:
            out[i] = [s[list(PART_KEYPOINTS[p])].mean() for p in PARTS]
    return np.clip(out, 0, 1)


def load_items(spec, reference, labels):
    """NAME=CACHE,SCORES -> list of dicts at 15 fps with joints, confidence, reference joints."""
    name, paths = spec.split("=", 1)
    cache, scores = paths.split(",", 1)
    items, missing = [], 0
    for video, (joints, frames, label, take) in load_views(cache).items():
        keep = frames % STRIDE == 0
        joints, frames = joints[keep], frames[keep]
        conf = part_confidence(Path(scores), video, frames)
        if conf is None:
            missing += 1
            conf = np.ones((len(frames), len(PARTS)), np.float32)
        ref = np.full_like(joints, np.nan)
        if take in reference:
            ref_joints, ref_frames = reference[take]
            _, pi, ri = np.intersect1d(frames, ref_frames, return_indices=True)
            ref[pi] = ref_joints[ri]
        items.append({"id": video, "variant": name, "label": labels.get(take, label), "take": take,
                      "joints": joints.astype(np.float32), "conf": conf, "ref": ref.astype(np.float32)})
    print(f"{name}: {len(items)} views from {cache}" + (f" ({missing} without 2D confidence)" if missing else ""))
    return name, items


def crop(item, length, start=None):
    n = len(item["joints"])
    if start is None:
        start = random.randint(0, max(0, n - length))
    sl = slice(start, start + length)
    joints, conf, ref = item["joints"][sl], item["conf"][sl], item["ref"][sl]
    pad = length - len(joints)
    mask = np.ones(length, bool)
    if pad > 0:
        mask[len(joints):] = False
        joints = np.concatenate([joints, np.repeat(joints[-1:], pad, 0)])
        conf = np.concatenate([conf, np.zeros((pad, len(PARTS)), np.float32)])
        ref = np.concatenate([ref, np.full((pad, 24, 3), np.nan, np.float32)])
    return canonical(joints).astype(np.float32), conf, ref, mask


MIRROR = [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20, 23, 22]
MIRROR_PARTS = [0, 2, 1, 4, 3]       # torso, arms and legs swapped


def augment(item):
    """Training-time variety: replay speed 0.8-1.25x, a left-right mirror half the time, ~1 cm joint noise."""
    n = len(item["joints"])
    t = np.arange(0, n - 1, np.exp(random.uniform(np.log(0.8), np.log(1.25))))
    i, w = t.astype(int), t % 1

    def lerp(x):
        ww = w.reshape(-1, *[1] * (x.ndim - 1))
        return (1 - ww) * x[i] + ww * x[i + 1]

    joints, conf, ref = lerp(item["joints"]), lerp(item["conf"]), lerp(item["ref"])
    if random.random() < 0.5:
        joints, conf, ref = joints[:, MIRROR], conf[:, MIRROR_PARTS], ref[:, MIRROR]
        joints[..., 0] *= -1
        ref[..., 0] *= -1
    joints = joints + np.random.normal(0, 0.01, joints.shape)
    return item | {"joints": joints.astype(np.float32), "conf": conf.astype(np.float32), "ref": ref.astype(np.float32)}


def batch(items, length, augmented=False):
    out = [crop(augment(it) if augmented else it, length) for it in items]
    return [torch.from_numpy(np.stack(x)) for x in zip(*out)]


# ---------------------------------------------------------------------------- model

class Refiner(nn.Module):
    def __init__(self, dim=128, layers=3):
        super().__init__()
        self.inp = nn.Linear(24 * 3 + len(PARTS), dim)
        self.time = nn.Parameter(torch.zeros(1, CLIP, dim))
        self.body = nn.TransformerEncoder(nn.TransformerEncoderLayer(dim, 4, dim * 4, 0.1, batch_first=True,
                                                                     norm_first=True), layers)
        self.out = nn.Linear(dim, 24 * 3)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, joints, conf, mask):
        b, t = joints.shape[:2]
        h = self.inp(torch.cat([joints.flatten(2), conf], -1)) + self.time[:, :t]
        h = self.body(h, src_key_padding_mask=~mask)
        return joints + self.out(h).view(b, t, 24, 3)


class PartEncoder(nn.Module):
    def __init__(self, dim=128, layers=4, out=64):
        super().__init__()
        self.ids = [list(ids) + [ids[-1]] * (MAX_PART - len(ids)) for ids in PART_JOINT_IDS.values()]
        self.inp = nn.ModuleList(nn.Linear(MAX_PART * 6 + 1, dim) for _ in PARTS)
        self.part = nn.Parameter(torch.zeros(1, 1, len(PARTS), dim))
        self.time = nn.Parameter(torch.zeros(1, CLIP, 1, dim))
        nn.init.normal_(self.part, std=0.02)
        nn.init.normal_(self.time, std=0.02)
        self.body = nn.TransformerEncoder(nn.TransformerEncoderLayer(dim, 4, dim * 4, 0.1, batch_first=True,
                                                                     norm_first=True), layers)
        self.pool = nn.Linear(dim, 1)
        self.head = nn.Linear(dim, out)
        self.vis = nn.Sequential(nn.Linear(dim + 2, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, joints, conf, mask, use_visibility=True):
        b, t = joints.shape[:2]
        vel = torch.cat([torch.zeros_like(joints[:, :1]), joints[:, 1:] - joints[:, :-1]], 1)
        tokens = []
        for r, ids in enumerate(self.ids):
            feat = torch.cat([joints[:, :, ids].flatten(2), vel[:, :, ids].flatten(2), conf[:, :, r:r + 1]], -1)
            tokens.append(self.inp[r](feat))
        h = torch.stack(tokens, 2) + self.part + self.time[:, :t]          # (B, T, P, D)
        pad = (~mask)[:, :, None].expand(b, t, len(PARTS)).reshape(b, t * len(PARTS))
        h = self.body(h.view(b, t * len(PARTS), -1), src_key_padding_mask=pad).view(b, t, len(PARTS), -1)
        logits = self.pool(h).squeeze(-1).masked_fill(~mask[:, :, None], -1e4)
        pooled = (logits.softmax(1)[..., None] * h).sum(1)                 # (B, P, D)
        z = F.normalize(self.head(pooled), dim=-1)
        if not use_visibility:
            return z, torch.ones(b, len(PARTS), device=z.device)
        m = mask[:, :, None].float()
        c_mean = (conf * m).sum(1) / m.sum(1).clamp(min=1)
        c_min = conf.masked_fill(~mask[:, :, None], 1).amin(1)
        v = torch.sigmoid(self.vis(torch.cat([pooled, c_mean[..., None], c_min[..., None]], -1))).squeeze(-1)
        return z, v


def similarity(zq, vq, zc, vc):
    """Visibility-weighted mean of per-part cosines: (Q, C)."""
    cos = torch.einsum("qpd,cpd->qcp", zq, zc)
    w = vq[:, None] * vc[None]
    return (w * cos).sum(-1) / w.sum(-1).clamp(min=1e-6)


def aligned_reference(pred, ref, mask):
    """One similarity transform per clip taking ref onto pred (no gradient through it)."""
    out = torch.full_like(ref, float("nan"))
    for i in range(len(pred)):
        ok = mask[i] & ~torch.isnan(ref[i]).any(-1).any(-1)
        if ok.sum() < 8:
            continue
        p, r = pred[i][ok].detach().reshape(-1, 3), ref[i][ok].reshape(-1, 3)
        pm, rm = p.mean(0), r.mean(0)
        u, s, vt = torch.linalg.svd((r - rm).T @ (p - pm))
        d = torch.sign(torch.linalg.det(u @ vt))
        s = s.clone(); s[-1] *= d; u = u.clone(); u[:, -1] *= d
        rot = u @ vt
        scale = s.sum() / ((r - rm) ** 2).sum()
        out[i] = scale * (ref[i] - rm) @ rot + pm
    return out


def geometry_losses(refined, ref, mask):
    target = aligned_reference(refined, ref, mask)
    ok = ~torch.isnan(target).any(-1)                                       # (B, T, 24)
    if not ok.any():
        zero = refined.sum() * 0
        return zero, zero
    geo = (refined - target.nan_to_num()).norm(dim=-1)[ok].mean()
    v_ok = ok[:, 1:] & ok[:, :-1]
    a_ok = v_ok[:, 1:] & v_ok[:, :-1]
    dv = (refined.diff(dim=1) - target.nan_to_num().diff(dim=1)).norm(dim=-1)
    da = (refined.diff(n=2, dim=1) - target.nan_to_num().diff(n=2, dim=1)).norm(dim=-1)
    dyn = dv[v_ok].mean() + da[a_ok].mean()
    return geo, dyn


def infonce(scores, labels, takes, temperature):
    """Multi-positive InfoNCE; same-take pairs are excluded from both sides."""
    same_take = takes[:, None] == takes[None]
    pos = (labels[:, None] == labels[None]) & ~same_take
    logits = (scores / temperature).masked_fill(same_take, -1e4)
    log_prob = logits - logits.logsumexp(1, keepdim=True)
    has = pos.any(1)
    return -(log_prob * pos).sum(1)[has].div(pos.sum(1)[has]).mean()


class Model(nn.Module):
    def __init__(self, refine=True, visibility=True):
        super().__init__()
        self.refiner = Refiner() if refine else None
        self.encoder = PartEncoder()
        self.visibility = visibility

    def forward(self, joints, conf, mask):
        refined = self.refiner(joints, conf, mask) if self.refiner is not None else joints
        z, v = self.encoder(refined, conf, mask, self.visibility)
        return refined, z, v


# ------------------------------------------------------------------------ train/eval

def sample_batch(by_label, labels_per_batch, per_label):
    chosen = random.sample(list(by_label), min(labels_per_batch, len(by_label)))
    out = []
    for label in chosen:
        takes = by_label[label]
        for take in random.sample(list(takes), min(per_label, len(takes))):
            out.append(random.choice(takes[take]))
    return out


@torch.no_grad()
def embed(model, items, device):
    """Full-sequence embedding: confidence-weighted average of window embeddings."""
    model.eval()
    zs, vs = [], []
    for it in items:
        n = len(it["joints"])
        starts = list(range(0, max(1, n - CLIP + 1), EVAL_HOP)) or [0]
        if n > CLIP and starts[-1] + CLIP < n:
            starts.append(n - CLIP)
        windows = [crop(it, CLIP, s) for s in starts]
        joints, conf, _, mask = [torch.from_numpy(np.stack(x)).to(device) for x in zip(*windows)]
        _, z, v = model(joints, conf, mask)
        w = v[..., None]
        zs.append(F.normalize((w * z).sum(0) / w.sum(0).clamp(min=1e-6), dim=-1))
        vs.append(v.mean(0))
    return torch.stack(zs), torch.stack(vs)


def evaluate(model, items, device):
    z, v = embed(model, items, device)
    scores = similarity(z, v, z, v).double().cpu().numpy()
    ids = [it["id"] for it in items]
    metrics = retrieval_metrics(scores, np.ones(scores.shape, bool), ids,
                                {it["id"]: it["label"] for it in items}, {it["id"]: it["take"] for it in items})
    return metrics, scores, v.cpu().numpy()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--train", action="append", required=True, metavar="NAME=CACHE,SCORES")
    parser.add_argument("--validation", action="append", required=True, metavar="NAME=CACHE,SCORES")
    parser.add_argument("--evaluate", action="append", default=[], metavar="NAME=CACHE,SCORES")
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--warmup-geometry", type=int, default=800, help="steps with L_geo + L_dyn only")
    parser.add_argument("--no-refiner", action="store_true")
    parser.add_argument("--no-visibility", action="store_true")
    parser.add_argument("--augment", action="store_true", help="speed, mirror and noise augmentation")
    parser.add_argument("--lambda-geo", type=float, default=1.0)
    parser.add_argument("--lambda-dyn", type=float, default=1.0)
    parser.add_argument("--lambda-fp", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--labels-per-batch", type=int, default=24)
    parser.add_argument("--per-label", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    split = json.loads(Path(args.split).read_text())
    keep = {t for part in ("train", "validation", "confirmation") for t in split["videos"][part]}
    reference = {v: skeleton(r) for v, r in load_pose_cache(args.reference).items() if v in keep}
    labels = split["labels"]
    train = [it for spec in args.train for it in load_items(spec, reference, labels)[1]]
    validation = [it for spec in args.validation for it in load_items(spec, reference, labels)[1]]
    by_label = {}
    for it in train:
        by_label.setdefault(it["label"], {}).setdefault(it["take"], []).append(it)
    by_label = {k: v for k, v in by_label.items() if len(v) > 1}      # a class needs two takes to give a positive
    print(f"train: {len(train)} views, {len(by_label)} classes with positives")

    device = torch.device(args.device)
    model = Model(refine=not args.no_refiner, visibility=not args.no_visibility).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * min(1, s / args.steps))))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    log, best = [], (-1.0, None)
    for step in range(args.steps + 1):
        model.train()
        items = sample_batch(by_label, args.labels_per_batch, args.per_label)
        joints, conf, ref, mask = [x.to(device) for x in batch(items, CLIP, args.augment)]
        refined, z, v = model(joints, conf, mask)
        geo, dyn = geometry_losses(refined, ref, mask) if model.refiner is not None else (refined.sum() * 0,) * 2
        fp_on = model.refiner is None or step >= args.warmup_geometry
        lab = torch.tensor([hash(it["label"]) for it in items], device=device)
        take = torch.tensor([hash(it["take"]) for it in items], device=device)
        fp = infonce(similarity(z, v, z, v), lab, take, args.temperature) if fp_on else geo * 0
        loss = args.lambda_geo * geo + args.lambda_dyn * dyn + args.lambda_fp * fp
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 250 == 0:
            metrics, _, _ = evaluate(model, validation, device)
            row = {"step": step, "geo_cm": float(geo) * 100, "dyn": float(dyn), "fp": float(fp),
                   "val_mAP": metrics["mAP"], "val_Hit@1": metrics["Hit@1"]}
            log.append(row)
            print(json.dumps(row), flush=True)
            if fp_on and metrics["mAP"] > best[0]:
                best = (metrics["mAP"], step)
                torch.save(model.state_dict(), output / "model.pt")
    (output / "train_log.json").write_text(json.dumps(log, indent=1))

    model.load_state_dict(torch.load(output / "model.pt", map_location=device))
    results = {"best_step": best[1], "val_mAP": best[0], "config": vars(args), "conditions": {}}
    queries = {}
    for spec in args.evaluate:
        name, items = load_items(spec, reference, labels)
        metrics, scores, v = evaluate(model, items, device)
        ids = [it["id"] for it in items]
        results["conditions"][name] = {
            key: metrics[key] for key in ("n_queries", "n_corpus", "mAP", "Hit@1", "R@5", "MRR", "mAP_random_floor")
        } | {"mAP_class_ci95": metrics["bootstrap"]["class_cluster_ci95"],
             "visibility_by_part": dict(zip(PARTS, v.mean(0).round(3).tolist()))}
        queries[name] = {"ids": ids, "scores": scores.round(5).tolist(), "visibility": v.round(3).tolist()}
        print(name, json.dumps(results["conditions"][name]), flush=True)
    (output / "summary.json").write_text(json.dumps(results, indent=2))
    (output / "scores.json").write_text(json.dumps(queries))


if __name__ == "__main__":
    main()
