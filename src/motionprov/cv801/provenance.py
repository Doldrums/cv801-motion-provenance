"""Motion provenance: a transfer of one existing recording, or a new performance?

A person re-performing a choreography does it their own way: timing,
amplitude, small details. A generator driven by a recording carries that one
recording's motion over almost exactly. So "did this really happen?" becomes
"is this motion lifted from recording X?", which a fingerprint can answer when
X is in the index, like Content ID for motion.

For every source view s (a clean view of one camera) the test compares
* the copy:              a generated video driven by s, against s;
* the re-performances:   every other take of the same choreography from the
                         same camera (other dancers, other repetitions), against s.
Both use the fixed matchers of the diagnostic. Reported per generator and
matcher: ROC AUC of copy vs re-performance scores, and how often the copy
scores above every re-performance of its source.

    python -m motionprov.cv801.provenance --clean COMOTION_CACHE \
        --generated mimicmotion=GEN_CACHE [--generated ...] --camera 01 [--frames 150] [--mirror] --output OUT
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from motionprov.cv801.diagnostic import dtw_scores, knn_scores, load_views
from motionprov.cv801.prepare import CAMERA

# SMPL joint order with every left joint swapped for its right counterpart.
MIRROR = [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20, 23, 22]


def mirrored(joints):
    """Left-right mirror of (T, 24, 3) camera-frame joints: x negated, left and right joints swapped."""
    out = joints[:, MIRROR].copy()
    out[..., 0] *= -1
    return out


def provenance(clean, generated, device, mirror=False):
    """-> per method: summary and per-source rows (copy score, re-performance scores).

    With ``mirror``, every pair scores the better of the query as is and the
    query mirrored, for real and generated queries alike.
    """
    cids, gids = sorted(clean), sorted(v for v in generated if v in clean)
    joints = [clean[v][0] for v in cids] + [generated[v][0] for v in gids]
    label = {v: clean[v][2] for v in cids}
    out = {}
    n = len(joints)
    for method, score in (("knn", knn_scores), ("dtw", lambda j: dtw_scores(j, device))):
        if mirror:
            both = score(joints + [mirrored(j) for j in joints])
            scores = np.maximum(both[:n, :n], both[n:, :n])
        else:
            scores = score(joints)
        rows, copies, others = {}, [], []
        for k, g in enumerate(gids):
            i = cids.index(g)
            copy = float(scores[len(cids) + k, i])
            reperf = {v: float(scores[j, i]) for j, v in enumerate(cids) if v != g and label[v] == label[g]}
            rows[g] = {"copy": copy, "reperformances": reperf,
                       "copy_above_all": bool(reperf) and copy > max(reperf.values())}
            copies.append(copy)
            others.extend(reperf.values())
        with_reperf = [r for r in rows.values() if r["reperformances"]]
        y = np.r_[np.ones(len(copies)), np.zeros(len(others))]
        out[method] = {
            "summary": {"copies": len(copies), "reperformances": len(others),
                        "auc_copy_vs_reperformance": float(roc_auc_score(y, np.r_[copies, others])),
                        "copy_above_all_rate": float(np.mean([r["copy_above_all"] for r in with_reperf])),
                        "sources_with_reperformance": len(with_reperf),
                        "median_copy": float(np.median(copies)), "median_reperformance": float(np.median(others))},
            "open_set": open_set(scores, cids, gids),
            "sources": rows,
        }
    return out


def open_set(scores, cids, gids):
    """No source given: each video's matches in the whole index, real videos excluding themselves.

    A copy is flagged when a statistic of its matches clears a threshold set on
    the real videos (each against every other indexed take), so the
    false-positive rate is how often a genuine performance would be called a copy.
    Statistics: the best score, and the gap from the best to the second best
    (a copy has one recording far above the rest, a re-performance several alike).
    """
    n = len(cids)
    real = scores[:n, :n].copy()
    np.fill_diagonal(real, -np.inf)
    copy = scores[n:, :n]
    found = np.array([cids[j] == g for j, g in zip(copy.argmax(1), gids)])
    out = {"source_is_top1": float(found.mean())}
    y = np.r_[np.ones(len(gids)), np.zeros(n)]
    def gap(s):
        top = np.sort(s, 1)
        return top[:, -1] - top[:, -2]

    for name, stat in (("best", lambda s: s.max(1)), ("gap", gap)):
        real_stat, copy_stat = stat(real), stat(copy)
        out[f"{name}_auc"] = float(roc_auc_score(y, np.r_[copy_stat, real_stat]))
        for fpr in (0.01, 0.05):
            threshold = np.quantile(real_stat, 1 - fpr)
            out[f"{name}_identified_at_fpr{fpr:g}"] = float(np.mean((copy_stat > threshold) & found))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", required=True)
    parser.add_argument("--generated", action="append", required=True, metavar="NAME=CACHE")
    parser.add_argument("--camera", default="01")
    parser.add_argument("--frames", type=int, help="keep only frames below this index (copies of a clip's start)")
    parser.add_argument("--mirror", action="store_true", help="score each pair also with the query mirrored")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    def head(views):
        if args.frames is None:
            return views
        keep = {v: (j[f < args.frames], f[f < args.frames], *rest) for v, (j, f, *rest) in views.items()}
        return {v: x for v, x in keep.items() if len(x[0]) > 1}

    clean = head({v: x for v, x in load_views(args.clean).items() if CAMERA.search(v).group(1) == args.camera})
    result = {}
    for spec in args.generated:
        name, cache = spec.split("=", 1)
        result[name] = provenance(clean, head(load_views(cache)), args.device, args.mirror)
        for method, r in result[name].items():
            print(name, method, json.dumps(r["summary"]), json.dumps(r["open_set"]))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "provenance.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
