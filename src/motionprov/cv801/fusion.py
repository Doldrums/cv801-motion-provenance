"""Does the trained fingerprint add to the fixed matchers? Late fusion of scores.

Each score matrix is standardised per query row (mean 0, std 1 over the
corpus) and summed with equal weight, so nothing is tuned on the test
conditions. The control is the fusion of the fixed matchers alone (DTW + kNN):
the trained model helps only if adding it beats that.

    python -m motionprov.cv801.fusion --model MODEL_OUT --split SPLIT.json \
        --cache clean=CACHE --cache occluded_arms=CACHE ... --output OUT
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from motionprov.core.evaluation.retrieval_evaluation import retrieval_metrics
from motionprov.cv801.diagnostic import dtw_scores, knn_scores, load_views


def standardised(scores):
    return (scores - scores.mean(1, keepdims=True)) / scores.std(1, keepdims=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="fingerprint_model output dir with scores.json")
    parser.add_argument("--split", required=True)
    parser.add_argument("--cache", action="append", required=True, metavar="CONDITION=CACHE")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    labels = json.loads(Path(args.split).read_text())["labels"]
    model = json.loads((Path(args.model) / "scores.json").read_text())
    result = {}
    for spec in args.cache:
        condition, cache = spec.split("=", 1)
        views, ids = load_views(cache), model[condition]["ids"]
        joints = [views[v][0] for v in ids]
        m = standardised(np.array(model[condition]["scores"]))
        d = standardised(dtw_scores(joints, args.device))
        k = standardised(knn_scores(joints))
        label = {v: labels.get(views[v][3], views[v][2]) for v in ids}
        take = {v: views[v][3] for v in ids}
        mean_ap = lambda s: float(retrieval_metrics(s.astype(np.float64), np.ones(s.shape, bool), ids, label, take)["mAP"])
        result[condition] = {"model": mean_ap(m), "dtw": mean_ap(d), "knn": mean_ap(k), "dtw+knn": mean_ap(d + k),
                             "model+dtw": mean_ap(m + d), "model+dtw+knn": mean_ap(m + d + k)}
        print(condition, json.dumps({key: round(v, 3) for key, v in result[condition].items()}), flush=True)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "fusion.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
