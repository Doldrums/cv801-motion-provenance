"""Evaluation protocol (Section 5.1, Appendix B).

Every recording with a relevant reference (same label, other source) is a
query against the clean whole recordings of its split. A query is the whole
recording or a 4 s fragment (start on a 1 s grid, set by hashing the
recording id), under three observation conditions: clean; hidden body part
(one arm, collarbone to hand, side by hash, for the whole recording); periodic
gaps (whole body hidden 8 of every 40 frames, phase by hash). Masks are placed
before the fragment is cut. For rejection, the same query is scored as
unknown: all references of its label are left out.

A system turns a query into scores over the index; the protocol does the rest.
"""

from __future__ import annotations

import hashlib

import numpy as np

from motionprov.choreomatch.cre import average_precision, chance_ap, reciprocal_rank
from motionprov.choreomatch.data import FPS, PARTS

CONDITIONS = ("clean", "part", "gaps")
KINDS = ("whole", "fragment")
FRAGMENT = 4 * FPS
GAP, PERIOD = 8, 40
TARGET_FA = 0.05


def hashed(rid, salt, n):
    return int(hashlib.md5(f"{rid}:{salt}".encode()).hexdigest(), 16) % n


def fragment_start(rec, kind):
    t = len(rec["joints"])
    if kind != "fragment" or t <= FRAGMENT:
        return 0
    starts = np.arange(0, t - FRAGMENT + 1, FPS)
    return int(starts[hashed(rec["id"], "start", len(starts))])


def query_view(rec, condition, kind):
    """-> joints (T, 24, 3), joint mask (T, 23) for one query variant."""
    joints, t = rec["joints"], len(rec["joints"])
    mask = np.repeat(rec["observed"][:, None], 23, 1)
    if condition == "part":
        side = ("left_arm", "right_arm")[hashed(rec["id"], "side", 2)]
        mask[:, [j - 1 for j in PARTS[side]]] = False
    elif condition == "gaps":
        phase = hashed(rec["id"], "phase", PERIOD)
        mask[(np.arange(t) + phase) % PERIOD < GAP] = False
    if kind == "fragment" and t > FRAGMENT:
        s = fragment_start(rec, kind)
        joints, mask = joints[s:s + FRAGMENT], mask[s:s + FRAGMENT]
    return joints, mask


def queries(records, index):
    """Records of the query set that have at least one relevant reference in the index."""
    return [q for q in records if any(x["label"] == q["label"] and x["source"] != q["source"] for x in index)]


def evaluate(system, records, index, conditions=CONDITIONS, kinds=KINDS):
    """system(rec, condition, kind) -> scores over index (len(index),) -> per variant rows."""
    labels = np.array([x["label"] for x in index])
    sources = np.array([x["source"] for x in index])
    rows = {}
    for condition in conditions:
        for kind in kinds:
            out = []
            for q in queries(records, index):
                s = np.asarray(system(q, condition, kind), dtype=np.float64)
                other = sources != q["source"]
                s = np.where(other, s, -np.inf)
                rel = (labels == q["label"]) & other
                n_other = int(other.sum())
                known = int(np.argmax(s)) if np.isfinite(s).any() else -1
                unknown = np.where(labels != q["label"], s, -np.inf)
                out.append({"id": q["id"], "label": q["label"],
                            "ap": average_precision(s[other], rel[other]), "rr": reciprocal_rank(s[other], rel[other]),
                            "chance": chance_ap(int(rel.sum()), n_other),
                            "known_score": float(s[known]) if known >= 0 else float("-inf"),
                            "known_correct": bool(known >= 0 and labels[known] == q["label"]),
                            "unknown_score": float(unknown.max())})
            rows[(condition, kind)] = out
    return rows


def summary(rows, threshold=None, boot=2000, seed=0):
    """Per variant: mAP (95 % CI, label-cluster bootstrap), MRR, chance; CA / FA at a threshold."""
    out = {}
    for (condition, kind), r in rows.items():
        ap = np.array([x["ap"] for x in r])
        entry = {"n": len(r), "mAP": float(ap.mean()), "MRR": float(np.mean([x["rr"] for x in r])),
                 "chance": float(np.mean([x["chance"] for x in r])), "mAP_ci95": bootstrap(r, "ap", boot, seed)}
        if threshold is not None:
            entry |= acceptance(r, threshold)
        out[f"{condition}/{kind}"] = entry
    return out


def bootstrap(r, key, boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    by_label = {}
    for x in r:
        by_label.setdefault(x["label"], []).append(x[key])
    groups = list(by_label.values())
    stats = []
    for _ in range(boot):
        pick = rng.integers(0, len(groups), len(groups))
        vals = np.concatenate([groups[i] for i in pick])
        stats.append(vals.mean())
    return np.percentile(stats, [2.5, 97.5]).round(4).tolist()


def paired_difference(ra, rb, key="ap", boot=2000, seed=0):
    """Mean difference b - a over the same queries, with a label-cluster bootstrap interval."""
    a = {x["id"]: x[key] for x in ra}
    diff = [{"label": x["label"], "d": x[key] - a[x["id"]]} for x in rb if x["id"] in a]
    return {"delta": float(np.mean([x["d"] for x in diff])), "ci95": bootstrap(diff, "d", boot, seed)}


def calibrate(rows, target=TARGET_FA):
    """Threshold giving `target` false acceptance on unknown queries, pooled over variants."""
    unknown = np.array([x["unknown_score"] for r in rows.values() for x in r])
    unknown = unknown[np.isfinite(unknown)]
    return float(np.quantile(unknown, 1 - target)) if len(unknown) else float("inf")


def acceptance(r, threshold):
    ca = np.mean([x["known_correct"] and x["known_score"] >= threshold for x in r])
    fa = np.mean([x["unknown_score"] >= threshold for x in r])
    return {"CA": float(ca), "FA": float(fa)}


def acceptance_curve(rows, points=200):
    """CA against FA as the threshold sweeps, pooled over variants (Fig. 11b)."""
    pooled = [x for r in rows.values() for x in r]
    scores = np.array([x["unknown_score"] for x in pooled] + [x["known_score"] for x in pooled])
    scores = scores[np.isfinite(scores)]
    curve = []
    for t in np.quantile(scores, np.linspace(0, 1, points)):
        a = acceptance(pooled, t)
        curve.append([a["FA"], a["CA"]])
    return sorted(curve)
