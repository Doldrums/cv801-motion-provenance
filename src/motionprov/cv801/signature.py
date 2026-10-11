"""Path-signature matcher: a tempo-free, part-wise fingerprint of a skeleton sequence.

The signature of a path is the list of its iterated integrals. It depends on the order in which the
path moves, not on how fast (re-parametrisation invariance), which is the property DTW buys with an
alignment and kNN pose statistics do not have at all. Here each body part (torso, two arms, two legs)
is a path of two joints in a per-frame body frame (hips and spine, as in ``limb_edit``), scaled by
the neck distance; its truncated signature (levels 1-3, each level L2-normalised) is the part code, and
the score is the mean over parts of the cosine between part codes, so a part can be compared alone.

Tests, on the same views and queries as the trained model:
* retrieval (mAP, Hit@1) on clean views and with arms or legs painted out, next to kNN and DTW and
  their fusion (each score matrix z-scored, then summed);
* tempo: every view re-sampled 1.2x faster; share of views whose sped-up copy scores above every view
  of every other dance (kNN, DTW, signature);
* limb replacement as in ``limb_edit`` (same take beats other dances, edited gallery).

    python signature.py --split SPLIT --restrict scores.json --cache clean=C --cache occluded_arms=C2 ... --output OUT.json
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

from motionprov.cv801.limb_edit import EDITS, body_frames, edit, metrics

CAMERA = re.compile(r"_c\d\d_")
PARTS = {"torso": (12, 15), "left_arm": (18, 20), "right_arm": (19, 21), "left_leg": (4, 7), "right_leg": (5, 8)}
STRIDE = 3                      # 30 fps -> 10 fps, as for DTW


def part_paths(joints):
    """(T, 24, 3) -> {part: (T', 6)} body-frame coordinates, scaled by the mean neck distance."""
    frames = body_frames(joints)
    scale = np.linalg.norm(joints[:, 12], axis=1).mean() + 1e-9
    out = {}
    for name, ids in PARTS.items():
        x = np.einsum("tji,tkj->tki", frames, joints[:, list(ids)]) / scale
        out[name] = x.reshape(len(x), -1)[::STRIDE]
    return out


def signature(path, depth=3):
    """Truncated signature of a piecewise-linear path by Chen's relation: levels 1..depth, flattened."""
    d = path.shape[1]
    s1, s2, s3 = np.zeros(d), np.zeros((d, d)), np.zeros((d, d, d))
    for delta in np.diff(path, axis=0):
        dd = np.outer(delta, delta) / 2
        if depth >= 3:
            s3 += np.einsum("ij,k->ijk", s2, delta) + np.einsum("i,jk->ijk", s1, dd) + np.einsum("i,j,k->ijk", delta, delta, delta) / 6
        s2 += np.outer(s1, delta) + dd
        s1 += delta
    levels = [s1, s2.ravel()] + ([s3.ravel()] if depth >= 3 else [])
    return [lv / (np.linalg.norm(lv) + 1e-12) for lv in levels]


def codes(joints, depth=3):
    return {name: np.concatenate(signature(p, depth)) for name, p in part_paths(joints).items()}


def part_scores(skeletons, depth=3):
    """(parts, N, N) cosine of part codes (levels weighted equally)."""
    c = [codes(s, depth) for s in skeletons]
    out = {}
    for name in PARTS:
        x = np.stack([ci[name] for ci in c])
        x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
        out[name] = x @ x.T
    return out


def signature_scores(skeletons, depth=3, parts=tuple(PARTS)):
    ps = part_scores(skeletons, depth)
    return np.mean([ps[p] for p in parts], axis=0)


def zscore(m):
    off = m[~np.eye(len(m), dtype=bool)]
    return (m - off.mean()) / (off.std() + 1e-12)


def speed_up(joints, factor):
    at = np.arange(0, len(joints) - 1, factor)
    low = np.floor(at).astype(int)
    w = (at - low)[:, None, None]
    return (joints[low] * (1 - w) + joints[np.minimum(low + 1, len(joints) - 1)] * w).astype(joints.dtype)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True)
    parser.add_argument("--restrict", required=True, help="scores.json whose views are the test set")
    parser.add_argument("--cache", action="append", required=True, metavar="CONDITION=CACHE")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    from motionprov.core.evaluation.retrieval_evaluation import retrieval_metrics
    from motionprov.cv801.diagnostic import dtw_scores, knn_scores, load_views
    labels = json.loads(Path(args.split).read_text())["labels"]
    saved = json.loads(Path(args.restrict).read_text())
    result = {}
    for spec in args.cache:
        cond, cache = spec.split("=", 1)
        keep = set(saved[cond]["ids"])
        views = load_views(cache)
        ids = sorted(v for v in views if v in keep)
        joints = [views[v][0] for v in ids]
        label = {v: views[v][2] for v in ids}
        group = {v: views[v][3] for v in ids}
        mats = {"knn": knn_scores(joints), "dtw": dtw_scores(joints, args.device)}
        for depth in (2, 3):
            mats[f"sig{depth}"] = signature_scores(joints, depth)
        for part in ("left_arm", "left_leg"):
            mats[f"sig3_{part}_only"] = signature_scores(joints, 3, (part,))
        z = {k: zscore(m) for k, m in mats.items()}
        mats["dtw+knn"] = z["dtw"] + z["knn"]
        mats["dtw+knn+sig3"] = z["dtw"] + z["knn"] + z["sig3"]
        result[cond] = {}
        for name, m in mats.items():
            r = retrieval_metrics(np.asarray(m, dtype=np.float64), np.ones(m.shape, bool), ids, label, group)
            result[cond][name] = {k: float(r[k]) for k in ("mAP", "Hit@1", "mAP_random_floor", "n_queries")}
            print(cond, f"{name:22s} mAP {r['mAP']:.3f}  Hit@1 {r['Hit@1']:.3f}", flush=True)
        if cond != "clean":
            continue
        # tempo: a view and the same view 1.2x faster, against every view of every other dance
        take = [CAMERA.sub("_cAll_", v) for v in ids]
        lab = [labels[t] for t in take]
        fast = [speed_up(j, 1.2) for j in joints]
        n = len(ids)
        both = joints + fast
        tempo = {}
        for name, m in (("knn", knn_scores(both)), ("dtw", dtw_scores(both, args.device)), ("sig3", signature_scores(both, 3))):
            hit = [float(m[i, n + i] > max(m[i, n + k] for k in range(n) if lab[k] != lab[i])) for i in range(n)]
            tempo[name] = float(np.mean(hit))
            print("tempo 1.2x: own sped-up copy beats all other dances", name, f"{tempo[name]:.3f}", flush=True)
        result["tempo_1.2x"] = tempo
        # limb replacement
        take_id = take
        edits = {}
        for ename in ("left_arm", "both_arms", "left_leg"):
            rng = np.random.default_rng(0)
            edited = []
            for i in range(n):
                donors = [k for k in range(n) if lab[k] != lab[i]]
                edited.append(edit(joints[i], EDITS[ename], "swap", joints[int(rng.choice(donors))]))
            both = joints + edited
            edits[f"swap/{ename}"] = {}
            for name, m in (("knn", knn_scores(both)), ("dtw", dtw_scores(both, args.device)), ("sig3", signature_scores(both, 3))):
                mm = metrics(m, n, lab, take_id)
                edits[f"swap/{ename}"][name] = {k: mm[k] for k in ("take_beats_other_dances_original", "take_beats_other_dances_edited", "p1_edited_gallery")}
                print("swap", ename, name, {k: round(v, 2) for k, v in edits[f"swap/{ename}"][name].items()}, flush=True)
        result["limb_swap"] = edits
    Path(args.output).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    sys.exit(main())
