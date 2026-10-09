"""Reference vs CoMotion skeletons under one fixed retrieval method.

The question from the proposal review: do retrieval failures come from pose
recovery, or from how motion is represented and compared? Here the matcher is
held fixed and only the pose source changes:

* ``reference``      AIST++ fitted SMPL motion (the annotation, not a video)
* ``comotion_cXX``   CoMotion from one camera view of the same takes
* ``comotion_all``   every view at once; views of one take share a source
                     group, so a query never retrieves its own take

Both matchers read only inter-joint distances, so they are invariant to the
camera frame and need no world/camera alignment. ``knn`` is the pose-statistics
descriptor of ``baselines.knn_baseline``; ``dtw`` aligns per-frame distance
vectors with the exact DTW of ``baselines.dtw_baseline``. Pose error is
per-frame Procrustes-aligned error against the reference, split by body part.

    python -m motionprov.cv801.diagnostic --reference REF_CACHE \
        --comotion COMOTION_CACHE --split SPLIT.json --output OUT \
        [--extra occluded_arms=ARMS_CACHE ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from motionprov.core.baselines.dtw_baseline import dtw_to_corpus
from motionprov.core.baselines.knn_baseline import descriptor
from motionprov.core.datasets.local_pose_cache import load_pose_cache
from motionprov.core.evaluation.retrieval_evaluation import retrieval_metrics
from motionprov.core.models.local_extensions import PART_JOINT_IDS
from motionprov.cv801.prepare import CAMERA

DTW_STRIDE = 3          # 30 fps -> 10 fps for the quadratic DTW
TOP_K = 5


def skeleton(record):
    """PoseRecord -> (T, 24, 3) root-centred joints on fully observed frames."""
    keep = record.frame_valid & record.observed.all(1)
    joints = np.concatenate([np.zeros((len(record.positions), 1, 3), np.float32),
                             record.positions], axis=1)
    return joints[keep], record.frame_index[keep]


def knn_scores(skeletons):
    feats = np.stack([descriptor(torch.from_numpy(s).permute(2, 0, 1).contiguous())
                      for s in skeletons])
    feats /= np.linalg.norm(feats, axis=1, keepdims=True)
    return feats @ feats.T


def distance_frames(joints):
    """(T, 24, 3) -> (T', 276) pairwise distances, scaled by their sequence mean."""
    x = torch.from_numpy(joints[::DTW_STRIDE])
    d = torch.cdist(x, x)
    iu = torch.triu_indices(24, 24, offset=1)
    d = d[:, iu[0], iu[1]]
    return d / d.mean().clamp(min=1e-8)


def dtw_scores(skeletons, device):
    feats = [distance_frames(s) for s in skeletons]
    lens = torch.tensor([len(f) for f in feats])
    padded = torch.stack([torch.cat([f, f[-1:].expand(int(lens.max()) - len(f), -1)])
                          for f in feats]).to(device)
    return -np.stack([dtw_to_corpus(f, padded, lens, device) for f in feats])


def margins(scores, labels, groups):
    """Best positive minus best negative score per query (cross-source only)."""
    out = []
    for i in range(len(labels)):
        other = groups != groups[i]
        pos, neg = other & (labels == labels[i]), other & (labels != labels[i])
        if pos.any() and neg.any():
            out.append(float(scores[i, pos].max() - scores[i, neg].max()))
        else:
            out.append(None)
    return out


def top_k(scores, ids, groups, i):
    other = np.flatnonzero(groups != groups[i])
    best = other[np.argsort(-scores[i, other], kind="stable")[:TOP_K]]
    return [{"id": ids[j], "score": float(scores[i, j])} for j in best]


def evaluate(items, device):
    """items: id -> (joints, label, group). Returns metrics and per-query rows."""
    ids = sorted(items)
    joints = [items[v][0] for v in ids]
    labels = np.asarray([items[v][1] for v in ids])
    groups = np.asarray([items[v][2] for v in ids])
    result = {}
    for method, scores in (("knn", knn_scores(joints)), ("dtw", dtw_scores(joints, device))):
        metrics = retrieval_metrics(scores.astype(np.float64), np.ones(scores.shape, bool), ids,
                                    dict(zip(ids, labels)), dict(zip(ids, groups)))
        delta = margins(scores, labels, groups)
        known = [m for m in delta if m is not None]
        result[method] = {
            "summary": {key: metrics[key] for key in
                        ("n_queries", "n_corpus", "mAP", "Hit@1", "R@5", "MRR", "mAP_random_floor")}
            | {"mAP_class_ci95": metrics["bootstrap"]["class_cluster_ci95"],
               "margin_positive_rate": float(np.mean([m > 0 for m in known])),
               "margin_median": float(np.median(known))},
            "queries": {ids[i]: {"label": str(labels[i]), "margin": delta[i],
                                 "top": top_k(scores, ids, groups, i)} for i in range(len(ids))},
        }
    return result


def procrustes_align(pred, ref):
    """Per-frame similarity alignment of pred onto ref, in ref's frame: (T, J, 3)."""
    centre = ref.mean(1, keepdims=True)
    p = pred - pred.mean(1, keepdims=True)
    r = ref - centre
    u, s, vt = np.linalg.svd(np.einsum("tji,tjk->tik", p, r))
    sign = np.sign(np.linalg.det(u @ vt))
    s[:, -1] *= sign
    u[:, :, -1] *= sign[:, None]
    rot = u @ vt
    scale = s.sum(1) / (p ** 2).sum((1, 2))
    return scale[:, None, None] * p @ rot + centre


def procrustes_errors(pred, ref):
    """Per-frame similarity-aligned error of pred against ref: (T, 24)."""
    return np.linalg.norm(procrustes_align(pred, ref) - ref, axis=2)


def pose_errors(views, reference):
    """Per-view, per-part PA error (reference units, ~metres) on shared frames."""
    out = {}
    for view, (joints, frames, _, take) in views.items():
        ref_joints, ref_frames = reference[take]
        common, pi, ri = np.intersect1d(frames, ref_frames, return_indices=True)
        if not len(common):
            continue
        err = procrustes_errors(joints[pi], ref_joints[ri])
        parts = {name: err[:, list(ids)].mean(1) for name, ids in PART_JOINT_IDS.items()}
        out[view] = {"frames": common.tolist(),
                     "coverage": len(common) / len(ref_frames),
                     "mean": {name: float(v.mean()) for name, v in parts.items()},
                     "series": {name: np.round(v, 4).tolist() for name, v in parts.items()}}
    return out


def load_views(cache):
    """CoMotion cache -> view id -> (joints, frames, choreography, take)."""
    views = {}
    for video, record in load_pose_cache(cache).items():
        joints, frames = skeleton(record)
        if len(joints) > 1:
            views[video] = (joints, frames, record.metadata["choreography_id"],
                            record.metadata["source_group_id"])
    return views


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--comotion", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--partition", default="confirmation")
    parser.add_argument("--extra", action="append", default=[], metavar="NAME=CACHE",
                        help="another CoMotion cache evaluated as its own condition, e.g. occluded views")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    split = json.loads(Path(args.split).read_text())
    takes = set(split["videos"][args.partition])

    reference = {}
    for video, record in load_pose_cache(args.reference).items():
        if video in takes:
            reference[video] = skeleton(record)
    views = load_views(args.comotion)

    conditions = {"reference": {v: (j, split["labels"][v], v) for v, (j, _) in reference.items()},
                  "comotion_all": {v: (j, label, take) for v, (j, _, label, take) in views.items()}}
    for camera in sorted({CAMERA.search(v).group(1) for v in views}):
        conditions[f"comotion_c{camera}"] = {
            v: (j, label, take) for v, (j, _, label, take) in views.items()
            if CAMERA.search(v).group(1) == camera}
    extra = {}
    for spec in args.extra:
        name, cache = spec.split("=", 1)
        extra[name] = load_views(cache)
        conditions[name] = {v: (j, label, take) for v, (j, _, label, take) in extra[name].items()}

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    results = {name: evaluate(items, args.device) for name, items in conditions.items()}
    errors = pose_errors(views, reference)
    summary = {name: {method: r["summary"] for method, r in res.items()}
               for name, res in results.items()}
    summary["pose_error_by_part"] = {
        name: float(np.mean([e["mean"][name] for e in errors.values()])) for name in PART_JOINT_IDS}
    for name, extra_views in extra.items():
        extra_errors = pose_errors(extra_views, reference)
        summary.setdefault("pose_error_by_part_extra", {})[name] = {
            part: float(np.mean([e["mean"][part] for e in extra_errors.values()])) for part in PART_JOINT_IDS}
    summary["views"] = {"reference_takes": len(reference), "comotion_views": len(views),
                        "views_with_error": len(errors)}
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    (output / "queries.json").write_text(json.dumps(
        {name: {method: r["queries"] for method, r in res.items()} for name, res in results.items()}))
    (output / "pose_errors.json").write_text(json.dumps(errors))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
