"""kNN identity-retrieval baseline over pose-statistics descriptors.

The "no learning at all" competitor: each video is a fixed-size descriptor of
inter-joint distance statistics - mean and std over time of the 276 pairwise
joint distances, plus the mean absolute frame-to-frame change of each
distance (a velocity signature). Distances are invariant to rigid translation
and rotation of the supplied 3D joints; this does not remove camera-dependent
pose-estimation errors. The comparison with motion words changes several
representation choices and does not isolate the causal effect of quantization.
Cosine similarity, evaluated under the
SAME protocol and metric code as the motion-word method (val-only queries,
full corpus, shared tie policy) - see metrics.ranking_metrics.

    python -m motionprov.core.baselines.knn_baseline \
        --videos-csv out/videos.csv --output-dir out --tag knn_tiktok \
        --val-from-checkpoint sweep/small_w30/best_by_rec.pt
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from ..datasets.dataset import eligible_videos, load_eligible_pose, val_video_split
from ..evaluation.metrics import ranking_metrics, source_grouped_queries
from ..pose.preprocessing import process_comotion_data


def descriptor(sk: torch.Tensor, parts: str = "full") -> np.ndarray:
    """(3, T, 24) skeleton -> pose-statistics descriptor.

    parts: 'full' = mean/std/velocity (828-d); 'static' = mean/std only -
    containing pose variation and limb-proportion cues; 'vel' = absolute
    frame-to-frame distance changes. Scalar normalization does not remove
    relative body proportions from either branch. Comparing these branches
    can probe possible confounds, but does not establish performer leakage.
    """
    if (not isinstance(sk, torch.Tensor) or sk.ndim != 3 or sk.shape[0] != 3
            or sk.shape[1] < 1 or sk.shape[2] != 24 or not sk.is_floating_point()
            or not torch.isfinite(sk).all() or parts not in {"full", "static", "vel"}):
        raise ValueError("descriptor requires finite nonempty (3,T,24) skeleton and valid parts")
    x = sk.permute(1, 2, 0)                     # (T, 24, 3)
    d = torch.cdist(x, x)                       # (T, 24, 24)
    iu = torch.triu_indices(d.shape[1], d.shape[2], offset=1)
    d = d[:, iu[0], iu[1]]                      # (T, 276)
    scale = d.mean().clamp(min=1e-8)            # body-size normalization
    d = d / scale
    vel = (d[1:] - d[:-1]).abs()
    vel_m = vel.mean(0) if len(vel) else torch.zeros(d.shape[1])
    std = d.std(0) if len(d) > 1 else torch.zeros_like(d[0])
    blocks = {"full": [d.mean(0), std, vel_m],
              "static": [d.mean(0), std],
              "vel": [vel_m]}
    return torch.cat(blocks[parts]).numpy()


def load_descriptors(csv_path: Path, comotion_dir: Path, smpl,
                     max_frames: int, parts: str = "full"):
    if type(max_frames) is not int or max_frames < 1:
        raise ValueError("max_frames must be a positive integer")
    import pandas as pd
    df = pd.read_csv(csv_path)
    feats, labels = {}, {}
    for _, row in df.iterrows():
        vid = Path(row["video_path"]).stem
        data = load_eligible_pose(comotion_dir, vid)
        if data is None:
            continue
        T = min(int(data["frame_idx"].max().item()) + 1, max_frames)
        sk = process_comotion_data(data, smpl, 0, T)  # (3, T, 24)
        feats[vid] = descriptor(sk, parts)
        labels[vid] = row["dance_id"]
    return feats, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-csv", required=True)
    ap.add_argument("--output-dir", default="out")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--tag", default="knn")
    ap.add_argument("--max-frames", type=int, default=900)
    ap.add_argument("--features", choices=["full", "static", "vel"],
                    default="full",
                    help="Compare distance mean/std ('static'), absolute "
                         "changes ('vel'), or both; neither removes all body-shape cues.")
    ap.add_argument("--split", choices=["val", "all"], default="val")
    ap.add_argument("--val-from-checkpoint", type=Path, default=None,
                    help="Read the held-out video list from this checkpoint's "
                         "meta, guaranteeing the SAME query set as the "
                         "motion-word evaluation it is compared against.")
    args = ap.parse_args()

    from ..comotion.utils.smpl_kinematics import SMPLKinematics
    smpl = SMPLKinematics()
    comotion_dir = Path(args.output_dir) / "comotion"

    feats, labels_by_vid = load_descriptors(
        Path(args.videos_csv), comotion_dir, smpl, args.max_frames,
        args.features)
    vids = sorted(feats)
    labels = np.array([labels_by_vid[v] for v in vids])
    X = np.stack([feats[v] for v in vids])
    X = X / np.linalg.norm(X, axis=1, keepdims=True).clip(min=1e-12)
    print(f"{len(vids)} videos, descriptor dim {X.shape[1]}")

    query_videos = None
    split_source = "all"
    if args.split == "val":
        if args.val_from_checkpoint is not None:
            meta = torch.load(args.val_from_checkpoint, map_location="cpu",
                              weights_only=False).get("meta", {})
            if not meta.get("val_videos"):
                raise ValueError(f"{args.val_from_checkpoint} has no "
                                 f"val_videos meta")
            query_videos = set(meta["val_videos"])
            split_source = "checkpoint_meta"
        else:
            query_videos = val_video_split(
                eligible_videos(args.videos_csv, comotion_dir))
            split_source = "recomputed"

    queries, groups = source_grouped_queries(vids, labels, query_videos)

    t0 = time.time()
    sims = X @ X.T
    np.fill_diagonal(sims, -np.inf)
    wall = time.time() - t0

    res = ranking_metrics(sims, labels, queries, query_videos=vids, groups=groups)
    res.update({
        "method": "knn_pose_statistics", "features": args.features,
        "descriptor": "mean/std of 276 inter-joint distances + mean |delta|, "
                      "body-scale normalized, cosine",
        "max_frames": args.max_frames, "split": args.split,
        "split_source": split_source,
        "num_queries": len(queries), "corpus_size": len(vids),
        "total_wall_seconds": round(wall, 4),
    })
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"identity_{args.tag}.json"
    path.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res[k] for k in
                      ["mAP", "MRR", "Hit@1", "Hit@3", "num_queries"]},
                     indent=2))
    print(f"-> {path}")


if __name__ == "__main__":
    main()
