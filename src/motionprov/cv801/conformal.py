"""Is the "1 false alarm per 100 real clips" threshold of the open-set alarm a guarantee?

``provenance.open_set`` flags a copy when the best match in the library of real clips is above the
99th percentile of the same statistic over the real clips themselves. With n real clips that
quantile is the largest one or two values: it carries no guarantee and varies from library to library.
Split-conformal calibration gives one: for a new genuine clip with statistic t (exchangeable with
the calibration clips), the p-value p = (1 + #{calibration s >= t}) / (n + 1) is super-uniform, so
"flag when p <= alpha" has false-alarm rate <= alpha, for every n. It also shows its limit: p >= 1 / (n + 1),
so alpha = 0.01 needs n >= 99 real clips.

The experiment repeats a random calibration / test split of the real clips many times and reports the
realised false-alarm rate of both rules (mean and 5-95 % range over splits), then flags the generated copies
with the conformal rule. Statistic: best score of a video against all other real clips.

    python conformal.py --clean CACHE --generated NAME=CACHE --output OUT.json [--camera 01 --frames 150]
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

CAMERA = re.compile(r"_c(\d\d)_")
ALPHAS = (0.01, 0.05, 0.1)


def real_statistic(scores, n):
    real = scores[:n, :n].copy()
    np.fill_diagonal(real, -np.inf)
    return real.max(1)


def conformal_p(calibration, t):
    return (1 + np.sum(calibration[None, :] >= np.asarray(t)[:, None], axis=1)) / (len(calibration) + 1)


def split_rates(stat, rng, repeats=2000, share=0.5):
    n = len(stat)
    out = {a: {"empirical": [], "conformal": []} for a in ALPHAS}
    for _ in range(repeats):
        order = rng.permutation(n)
        k = int(round(n * share))
        calibration, test = stat[order[:k]], stat[order[k:]]
        for a in ALPHAS:
            out[a]["empirical"].append(float(np.mean(test > np.quantile(calibration, 1 - a))))
            out[a]["conformal"].append(float(np.mean(conformal_p(calibration, test) <= a)))
    summary = {}
    for a in ALPHAS:
        summary[str(a)] = {}
        for rule, v in out[a].items():
            summary[str(a)][rule] = {"mean": float(np.mean(v)), "p05": float(np.quantile(v, 0.05)),
                                     "p95": float(np.quantile(v, 0.95)),
                                     "share_of_splits_above_alpha": float(np.mean(np.array(v) > a))}
    return summary, k


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", required=True)
    parser.add_argument("--generated", action="append", required=True, metavar="NAME=CACHE")
    parser.add_argument("--camera", default="01")
    parser.add_argument("--frames", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    from motionprov.cv801.diagnostic import dtw_scores, knn_scores, load_views
    from motionprov.cv801.provenance import open_set

    def head(views):
        if args.frames is None:
            return views
        keep = {v: (j[f < args.frames], f[f < args.frames], *rest) for v, (j, f, *rest) in views.items()}
        return {v: x for v, x in keep.items() if len(x[0]) > 1}

    clean = head({v: x for v, x in load_views(args.clean).items() if CAMERA.search(v).group(1) == args.camera})
    cids = sorted(clean)
    n = len(cids)
    result = {"n_real": n, "min_p": 1 / (n + 1), "n_needed_for_alpha_0.01": 99}
    print(f"real clips {n}; smallest possible p-value {1 / (n + 1):.4f} (alpha 0.01 needs 99 clips)", flush=True)
    for spec in args.generated:
        name, cache = spec.split("=", 1)
        generated = head(load_views(cache))
        gids = sorted(v for v in generated if v in clean)
        joints = [clean[v][0] for v in cids] + [generated[v][0] for v in gids]
        result[name] = {"copies": len(gids)}
        for method, score in (("knn", knn_scores), ("dtw", lambda j: dtw_scores(j, args.device))):
            scores = score(joints)
            stat = real_statistic(scores, n)
            copy_stat = scores[n:, :n].max(1)
            found = np.array([cids[j] == g for j, g in zip(scores[n:, :n].argmax(1), gids)])
            rates, k = split_rates(stat, np.random.default_rng(0))
            entry = {"open_set_empirical": open_set(scores, cids, gids), "split_rates": rates, "calibration_size": k,
                     "min_p_with_calibration_half": 1 / (k + 1)}
            p = conformal_p(stat, copy_stat)
            entry["conformal_all_real"] = {str(a): {"flagged": float(np.mean(p <= a)),
                                                    "flagged_and_source_found": float(np.mean((p <= a) & found)),
                                                    "feasible": bool(1 / (n + 1) <= a)} for a in ALPHAS}
            result[name][method] = entry
            print(name, method, "empirical open set", json.dumps(entry["open_set_empirical"]), flush=True)
            print("  conformal flagged", json.dumps(entry["conformal_all_real"]), flush=True)
            for a in ALPHAS:
                r = rates[str(a)]
                print(f"  alpha {a}: empirical FPR {r['empirical']['mean']:.3f} [{r['empirical']['p05']:.3f}, {r['empirical']['p95']:.3f}], "
                      f"conformal {r['conformal']['mean']:.3f} [{r['conformal']['p05']:.3f}, {r['conformal']['p95']:.3f}]", flush=True)
    Path(args.output).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
