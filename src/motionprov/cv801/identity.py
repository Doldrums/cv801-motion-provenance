"""Whose body moves in a generated copy: the new face's, or the source dancer's?

A motion-transfer generator paints a new person over the source's pose and
silhouette. If the copy keeps the source dancer's body proportions, the
copy carries a second trace of its origin, and a claimed identity can be
checked against that person's known proportions.

Per view: the median length of each SMPL bone over the clip, divided by the
sum, so camera distance and monocular scale drop out. Reported
* real:  does a real view sit nearer to the same dancer's other takes than to
         other dancers (AUC), i.e. do proportions identify a person at all;
* copy:  distance of each copy to its source view, to the source dancer's
         other takes and to other dancers; how often the copy's nearest dancer
         is the source dancer.

    python -m motionprov.cv801.identity --clean CACHE --generated wan=CACHE [--camera 01] --output OUT
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from motionprov.cv801.diagnostic import load_views
from motionprov.cv801.prepare import CAMERA

PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
DANCER = re.compile(r"_d(\d+)_")


def proportions(joints):
    """(T, 24, 3) -> (23,) median bone lengths, summing to one."""
    bones = np.linalg.norm(joints[:, 1:] - joints[:, PARENTS[1:]], axis=-1)
    length = np.median(bones, 0)
    return length / length.sum()


def nearest_dancer(p, views, exclude=()):
    """The dancer whose nearest take is closest to p."""
    best = {}
    for v, q in views.items():
        if v in exclude:
            continue
        d, dancer = float(np.abs(p - q).sum()), DANCER.search(v).group(1)
        best[dancer] = min(best.get(dancer, np.inf), d)
    return min(best, key=best.get)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", required=True)
    parser.add_argument("--generated", action="append", required=True, metavar="NAME=CACHE")
    parser.add_argument("--camera", default="01")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    clean = {v: proportions(x[0]) for v, x in load_views(args.clean).items()
             if CAMERA.search(v).group(1) == args.camera}
    dancer = {v: DANCER.search(v).group(1) for v in clean}
    dist = lambda a, b: float(np.abs(a - b).sum())

    same, other, hits = [], [], []
    for v, p in clean.items():
        for u, q in clean.items():
            if u != v:
                (same if dancer[u] == dancer[v] else other).append(dist(p, q))
        hits.append(nearest_dancer(p, clean, exclude={v}) == dancer[v])
    result = {"real": {"views": len(clean), "dancers": len(set(dancer.values())),
                       "auc_same_vs_other_dancer": float(roc_auc_score(np.r_[np.ones(len(same)), np.zeros(len(other))],
                                                                       -np.r_[same, other])),
                       "nearest_dancer_is_self": float(np.mean(hits)),
                       "median_same": float(np.median(same)), "median_other": float(np.median(other))}}
    for spec in args.generated:
        name, cache = spec.split("=", 1)
        copies = {v: proportions(x[0]) for v, x in load_views(cache).items() if v in clean}
        to_source, to_dancer, to_other, hits = [], [], [], []
        for v, p in copies.items():
            to_source.append(dist(p, clean[v]))
            for u, q in clean.items():
                if u != v:
                    (to_dancer if dancer[u] == dancer[v] else to_other).append(dist(p, q))
            hits.append(nearest_dancer(p, clean) == dancer[v])
        y = lambda a, b: np.r_[np.ones(len(a)), np.zeros(len(b))]
        result[name] = {"copies": len(copies),
                        "nearest_dancer_is_source_dancer": float(np.mean(hits)),
                        "auc_source_dancer_vs_other": float(roc_auc_score(y(to_dancer, to_other), -np.r_[to_dancer, to_other])),
                        "median_to_source_view": float(np.median(to_source)),
                        "median_to_source_dancer": float(np.median(to_dancer)),
                        "median_to_other_dancers": float(np.median(to_other))}
    print(json.dumps(result, indent=1))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "identity.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
