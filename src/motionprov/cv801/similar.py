"""Can the fingerprint tell near-identical dances apart? Hard against easy negatives.

The proposal's question is whether the representation keeps the small detail that
separates one motion from a near-identical one. AIST++ gives a natural split of the
negatives of a query: a dance of the same genre (same style, same kind of moves, often
the same music family) is a hard negative, a dance of another genre is an easy one.
For every query the candidates are its true matches plus one pool of negatives, and we
report mAP and precision@1 inside that pool next to what a random ranking of the same
pool would reach. Views of one take (other cameras) are neither positive nor negative.

Score matrices come from a trained model (``scores.json`` of ``fingerprint_model``) or from the
fixed matchers (``--cache CONDITION=CoMotion cache``, the views of that cache).
    python -m motionprov.cv801.similar --split SPLIT --scores full=scores.json --output OUT.json
    python -m motionprov.cv801.similar --split SPLIT --cache clean=CACHE --output OUT.json
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

CAMERA = re.compile(r"_c\d\d_")
POOLS = ("all", "hard", "easy")


def average_precision(row, positive):
    order = np.argsort(-row)
    hits = positive[order]
    return float((np.cumsum(hits) / np.arange(1, len(hits) + 1))[hits].mean())


def evaluate(scores, ids, labels, rng=None):
    """Per pool: mAP and precision@1 over the queries that have both kinds of candidate."""
    scores = np.asarray(scores, dtype=float)
    take = [CAMERA.sub("_cAll_", v) for v in ids]
    label = np.array([labels[t] for t in take])
    genre = np.array([v.split("_")[0] for v in ids])
    out = {p: {"ap": [], "p1": [], "chance_p1": []} for p in POOLS}
    for i in range(len(ids)):
        valid = np.array([j for j in range(len(ids)) if take[j] != take[i]])
        positive = label[valid] == label[i]
        pools = {"all": np.ones(len(valid), bool),
                 "hard": positive | (genre[valid] == genre[i]),
                 "easy": positive | (genre[valid] != genre[i])}
        for name, keep in pools.items():
            pos = positive[keep]
            if not pos.any() or pos.all():
                continue
            row = scores[i, valid][keep]
            out[name]["ap"].append(average_precision(row, pos))
            out[name]["p1"].append(float(pos[row.argmax()]))
            out[name]["chance_p1"].append(float(pos.mean()))
    result = {}
    for name, v in out.items():
        n = len(v["ap"])
        result[name] = {"queries": n, "mAP": float(np.mean(v["ap"])), "precision_at_1": float(np.mean(v["p1"])),
                        "chance_precision_at_1": float(np.mean(v["chance_p1"]))}
    if rng is not None:
        random = [evaluate(rng.normal(size=scores.shape), ids, labels) for _ in range(10)]
        for name in POOLS:
            result[name]["chance_mAP"] = float(np.mean([r[name]["mAP"] for r in random]))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True)
    parser.add_argument("--scores", action="append", default=[], metavar="MODEL=scores.json")
    parser.add_argument("--cache", action="append", default=[], metavar="CONDITION=CACHE")
    parser.add_argument("--restrict", help="scores.json whose views are the test set (others are dropped)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    labels = json.loads(Path(args.split).read_text())["labels"]
    rng = np.random.default_rng(0)
    result = {}
    for spec in args.scores:
        name, path = spec.split("=", 1)
        for condition, v in json.loads(Path(path).read_text()).items():
            if condition != "wan":
                result[f"{name}/{condition}"] = evaluate(v["scores"], v["ids"], labels, rng)
    if args.cache:
        from motionprov.cv801.diagnostic import dtw_scores, knn_scores, load_views
        for spec in args.cache:
            condition, cache = spec.split("=", 1)
            views = load_views(cache)
            keep = {v for c in json.loads(Path(args.restrict).read_text()).values() for v in c["ids"]} if args.restrict else None
            ids = sorted(v for v in views if CAMERA.sub("_cAll_", v) in labels and (keep is None or v in keep))
            joints = [views[v][0] for v in ids]
            result[f"knn/{condition}"] = evaluate(knn_scores(joints), ids, labels, rng)
            result[f"dtw/{condition}"] = evaluate(dtw_scores(joints, args.device), ids, labels, rng)
    for key, r in result.items():
        print(f"{key:28s}", "  ".join(f"{p}: mAP {r[p]['mAP']:.3f} (chance {r[p]['chance_mAP']:.3f}) p@1 {r[p]['precision_at_1']:.2f}"
                                       for p in ("hard", "easy")), flush=True)
    Path(args.output).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
