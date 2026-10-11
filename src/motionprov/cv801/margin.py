"""Margin, abstention and margin drop under occlusion, from saved fingerprint scores.

The proposal asks for the margin of a query, delta(q) = s(q, c+) - max s(q, c-): the
best true match against the best look-alike. delta > 0 means the deciding detail
survived and the right dance ranks first. Three readouts come from the score
matrices that ``fingerprint_model`` already wrote (no new model run):

* margin: delta per query, and how often it is positive (precision at 1);
* abstention: answer only when the system is confident. Confidence is the top score,
  or the gap between the top score and the best other dance. We report precision at
  1 over the most confident fractions of queries, and the AUC of the confidence for
  telling right answers from wrong ones;
* occlusion test: delta_clean(q) - delta_occluded(q) when arms or legs are painted
  out of the video, on the same queries. The clean matrix is restricted to the
  occluded views (camera c01), so query and gallery sets match.

Views of one take (other cameras) are neither positive nor negative.
    python -m motionprov.cv801.margin --scores full=in/full.json --split in/split.json --output out.json
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

CAMERA = re.compile(r"_c\d\d_")
COVERAGE = (1.0, 0.75, 0.5, 0.25)


def take_of(view):
    return CAMERA.sub("_cAll_", view)


def auc(positive, negative):
    """Mann-Whitney AUC: chance that a right answer is more confident than a wrong one."""
    if not len(positive) or not len(negative):
        return None
    scores = np.r_[positive, negative]
    ranks = scores.argsort().argsort() + 1.0
    for value in np.unique(scores):
        tie = scores == value
        ranks[tie] = ranks[tie].mean()
    return float((ranks[:len(positive)].sum() - len(positive) * (len(positive) + 1) / 2)
                 / (len(positive) * len(negative)))


def margins(scores, ids, labels):
    scores = np.asarray(scores, dtype=float)
    take = [take_of(v) for v in ids]
    label = [labels[t] for t in take]
    out = {}
    for i, view in enumerate(ids):
        valid = np.array([j for j in range(len(ids)) if take[j] != take[i]])
        same = np.array([label[j] == label[i] for j in valid])
        if same.all() or not same.any():
            continue
        row = scores[i, valid]
        top = int(row.argmax())
        other = [k for k in range(len(valid)) if label[valid[k]] != label[valid[top]]]
        out[view] = {"delta": float(row[same].max() - row[~same].max()),
                     "top": float(row[top]), "gap": float(row[top] - row[other].max())}
    return out


def summarise(m):
    delta = np.array([v["delta"] for v in m.values()])
    correct = delta > 0
    result = {"queries": len(m), "precision_at_1": float(correct.mean()), "mean_margin": float(delta.mean())}
    for name in ("top", "gap"):
        conf = np.array([v[name] for v in m.values()])
        order = np.argsort(-conf)
        result[name] = {"auc_right_vs_wrong": auc(conf[correct], conf[~correct]),
                        "precision_at_1_by_coverage": {
                            str(c): float(correct[order[:max(1, round(c * len(m)))]].mean()) for c in COVERAGE}}
    return result


def drop(clean, occluded, rng):
    shared = [v for v in occluded if v in clean]
    a = np.array([clean[v]["delta"] for v in shared])
    b = np.array([occluded[v]["delta"] for v in shared])
    d = a - b
    boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    return {"queries": len(shared), "precision_at_1_clean": float((a > 0).mean()),
            "precision_at_1_occluded": float((b > 0).mean()),
            "mean_margin_drop": float(d.mean()),
            "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "right_to_wrong": int(((a > 0) & (b <= 0)).sum()), "wrong_to_right": int(((a <= 0) & (b > 0)).sum())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", action="append", required=True, metavar="MODEL=scores.json")
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    labels = json.loads(Path(args.split).read_text())["labels"]
    rng = np.random.default_rng(0)
    result = {}
    for spec in args.scores:
        name, path = spec.split("=", 1)
        saved = json.loads(Path(path).read_text())
        per = {c: margins(v["scores"], v["ids"], labels) for c, v in saved.items() if c != "wan"}
        entry = {c: summarise(m) for c, m in per.items()}
        clean_ids = saved["clean"]["ids"]
        clean_matrix = np.asarray(saved["clean"]["scores"], dtype=float)
        for c in ("occluded_arms", "occluded_legs", "mimicmotion"):
            if c not in saved:
                continue
            ids = saved[c]["ids"]
            index = [clean_ids.index(v) for v in ids]
            same_view = margins(clean_matrix[np.ix_(index, index)], ids, labels)
            entry[c]["clean_same_views"] = summarise(same_view)
            entry[c]["margin_drop"] = drop(same_view, per[c], rng)
        result[name] = entry
        print(name, flush=True)
        for c, e in entry.items():
            line = f"  {c:14s} p@1 {e['precision_at_1']:.3f}  margin {e['mean_margin']:+.3f}"
            if "margin_drop" in e:
                m = e["margin_drop"]
                line += f"  | vs clean {m['precision_at_1_clean']:.3f}  drop {m['mean_margin_drop']:+.3f} {m['ci95']}"
            print(line, flush=True)
    Path(args.output).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
