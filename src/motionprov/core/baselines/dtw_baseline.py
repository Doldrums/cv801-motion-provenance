"""DTW identity-retrieval baseline over raw joint trajectories.

The primary competitor (after the Keogh letter): exact dynamic time warping
over root-centered, scale-normalized skeleton sequences, distances normalized
by sqrt(len_q * len_c) so different lengths are comparable. Evaluated under
the SAME protocol and metric code as the motion-word method (val-only
queries, full corpus, shared tie policy) - see metrics.ranking_metrics.

This legacy whole-sequence endpoint uses Euclidean local costs and variable
lengths. benchmarks.ucr_dtw instead resamples to a common length and uses
squared costs and a band. Its timings must NOT be attached to this endpoint's
quality. Bounds preserve a distance only when preprocessing, cost, path
constraints and tie policy are identical. Neither endpoint is subsequence
search or missing-observation-aware retrieval.

    python -m motionprov.core.baselines.dtw_baseline \
        --videos-csv out/videos.csv --output-dir out --tag dtw_tiktok

The DP is vectorized over anti-diagonals and batched over the whole corpus:
cells on one anti-diagonal (i + j = d) depend only on the two previous
diagonals, so each of the Tq+Tc diagonals is a single tensor op over
(corpus, cells) - no per-cell Python loops.
"""

import argparse
import json
import time
from numbers import Integral
from pathlib import Path

import numpy as np
import torch

from ..datasets.dataset import eligible_videos, load_eligible_pose, val_video_split
from ..evaluation.metrics import ranking_metrics, source_grouped_queries
from ..pose.preprocessing import process_comotion_data
from ..utils import get_device


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _numpy_sequence(value, name):
    value = np.asarray(value)
    if (value.ndim != 2 or not all(value.shape)
            or value.dtype.kind not in "fiu" or not np.isfinite(value).all()):
        raise ValueError(f"{name} must be a finite nonempty real (T, D) array")
    return value.astype(np.float64, copy=False)


def load_features(csv_path: Path, comotion_dir: Path, smpl, fps_down: int,
                  max_frames: int):
    """Legacy root-centered coordinates divided by one per-sequence std.

    This is scalar scale normalization, not per-coordinate z-normalization.
    The preprocessing hold policy is retained, without an observation mask;
    imputed frames contribute to both normalization and DTW. Frame stride is
    not a conversion of heterogeneous source frame rates to a common clock.
    """
    fps_down = _positive_integer(fps_down, "fps_down")
    max_frames = _positive_integer(max_frames, "max_frames")
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
        x = sk.permute(1, 2, 0).reshape(T, -1)[::fps_down]  # (T', 72)
        if not torch.isfinite(x).all():
            raise ValueError(f"nonfinite skeleton features for {vid}")
        std = x.std()
        if std > 0:
            x = x / std
        feats[vid] = x.float()
        labels[vid] = row["dance_id"]
    return feats, labels


def dtw_to_corpus(q: torch.Tensor, corpus_padded: torch.Tensor,
                  lens: torch.Tensor, device) -> np.ndarray:
    """Exact normalized DTW from one query to every (padded) corpus sequence.

    Padding (repeated last frame) is unread by construction: the warping path
    to cell (Tq, len_c) is monotone and can never visit j > len_c.
    """
    for name, value, ndim in (("q", q, 2), ("corpus_padded", corpus_padded, 3)):
        if (not isinstance(value, torch.Tensor) or value.ndim != ndim
                or not all(value.shape) or value.dtype not in (torch.float32, torch.float64)
                or not torch.isfinite(value).all()):
            raise ValueError(f"{name} must be a finite nonempty float32/float64 tensor")
    if q.shape[1] != corpus_padded.shape[2] or q.dtype != corpus_padded.dtype:
        raise ValueError("query and corpus feature dimensions and dtypes must match")
    B, Tmax, _ = corpus_padded.shape
    if (not isinstance(lens, torch.Tensor) or lens.shape != (B,)
            or lens.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
            or not ((lens > 0) & (lens <= Tmax)).all()):
        raise ValueError("lens must contain one integer in [1, Tmax] per corpus item")
    Tq = q.shape[0]
    q = q.to(device)
    if q.device != corpus_padded.device:
        raise ValueError("corpus_padded must already be on the requested device")
    cost = torch.cdist(q[None].expand(B, -1, -1), corpus_padded)  # (B, Tq, Tmax)
    if not torch.isfinite(cost).all():
        raise ValueError("Euclidean costs overflowed")
    D = torch.full((B, Tq + 1, Tmax + 1), float("inf"), device=device, dtype=q.dtype)
    D[:, 0, 0] = 0.0
    for d in range(2, Tq + Tmax + 1):
        i0, i1 = max(1, d - Tmax), min(Tq, d - 1)
        if i0 > i1:
            continue
        ii = torch.arange(i0, i1 + 1, device=device)
        jj = d - ii
        step = torch.minimum(torch.minimum(D[:, ii - 1, jj], D[:, ii, jj - 1]),
                             D[:, ii - 1, jj - 1])
        D[:, ii, jj] = cost[:, ii - 1, jj - 1] + step
    final = D[torch.arange(B, device=device), Tq, lens.to(device=device, dtype=torch.long)]
    if not torch.isfinite(final).all():
        raise ValueError("accumulated Euclidean costs overflowed")
    norm = (float(Tq) * lens.to(device=device, dtype=q.dtype)).sqrt()
    return (final / norm).cpu().numpy()


def dtw_pair_numpy(q: np.ndarray, x: np.ndarray) -> float:
    """Reference implementation (plain numpy) used by tests."""
    q, x = _numpy_sequence(q, "q"), _numpy_sequence(x, "x")
    if q.shape[1] != x.shape[1]:
        raise ValueError("query and candidate feature dimensions must match")
    Tq, Tc = len(q), len(x)
    with np.errstate(over="ignore", invalid="ignore"):
        cost = np.sqrt(((q[:, None, :] - x[None, :, :]) ** 2).sum(-1))
    if not np.isfinite(cost).all():
        raise ValueError("Euclidean costs overflowed")
    D = np.full((Tq + 1, Tc + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, Tq + 1):
        for j in range(1, Tc + 1):
            D[i, j] = cost[i - 1, j - 1] + min(D[i - 1, j], D[i, j - 1],
                                               D[i - 1, j - 1])
    if not np.isfinite(D[Tq, Tc]):
        raise ValueError("accumulated Euclidean costs overflowed")
    return float(D[Tq, Tc] / np.sqrt(Tq * Tc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-csv", required=True)
    ap.add_argument("--output-dir", default="out")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--tag", default="dtw")
    ap.add_argument("--fps-down", type=int, default=3,
                    help="Temporal downsample factor (30fps -> 10fps).")
    ap.add_argument("--max-frames", type=int, default=900)
    ap.add_argument("--split", choices=["val", "all"], default="val")
    ap.add_argument("--val-from-checkpoint", type=Path, default=None,
                    help="Read the held-out video list from this checkpoint's "
                         "meta, guaranteeing the SAME query set as the "
                         "motion-word evaluation it is compared against.")
    args = ap.parse_args()

    from ..comotion.utils.smpl_kinematics import SMPLKinematics
    smpl = SMPLKinematics()
    comotion_dir = Path(args.output_dir) / "comotion"
    device = get_device("auto")

    feats, labels_by_vid = load_features(
        Path(args.videos_csv), comotion_dir, smpl, args.fps_down, args.max_frames)
    vids = sorted(feats)
    labels = np.array([labels_by_vid[v] for v in vids])
    lens = torch.tensor([len(feats[v]) for v in vids])
    print(f"{len(vids)} sequences, feature dim {feats[vids[0]].shape[1]}, "
          f"median length {int(lens.float().median())}")

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

    # Pad the corpus once by repeating each sequence's last frame.
    Tmax = int(lens.max())
    corpus = torch.stack([
        torch.cat([feats[v], feats[v][-1:].expand(Tmax - len(feats[v]), -1)])
        if len(feats[v]) < Tmax else feats[v]
        for v in vids
    ]).to(device)

    sims = np.zeros((len(vids), len(vids)))
    t0 = time.time()
    for n, i in enumerate(queries):
        d = dtw_to_corpus(feats[vids[i]], corpus, lens, device)
        sims[i] = -d
        sims[i, i] = -np.inf
        if (n + 1) % 10 == 0 or n == len(queries) - 1:
            print(f"query {n + 1}/{len(queries)}  ({time.time() - t0:.0f}s)",
                  flush=True)
    wall = time.time() - t0

    res = ranking_metrics(sims, labels, queries, query_videos=vids, groups=groups)
    res.update({
        "method": "dtw_raw_trajectories",
        "fps_down": args.fps_down,
        "max_frames": args.max_frames, "split": args.split,
        "split_source": split_source,
        "normalization": "sqrt(len_q*len_c), per-sequence std scaling",
        "total_wall_seconds": round(wall, 1),
        "endpoint": "legacy_whole_sequence_euclidean_dtw_v1",
        "note": "Legacy whole-sequence accuracy endpoint; UCR benchmark "
                "uses different resampling, squared costs and band constraints. "
                "Its latency is not comparable to this endpoint's quality.",
    })
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"identity_{args.tag}.json"
    out_path.write_text(json.dumps(res, indent=2))
    print(f"\nmAP {res['mAP']:.4f}  MRR {res['MRR']:.4f}  "
          f"Hit@1 {res['Hit@1']:.2f}  Hit@3 {res['Hit@3']:.2f}  "
          f"(floor {res['mAP_random_floor']:.3f})  -> {out_path}")


if __name__ == "__main__":
    main()
