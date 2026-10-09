"""Does 2D keypoint confidence tell observed body parts from filled-in ones?

The proposal weights each body part's evidence by a visibility v_r taken from
2D confidence. That is only sound if low confidence marks the frames where
the 3D pose is wrong. Two steps:

* ``extract`` (inside the MimicMotion environment, which ships DWPose): per
  video, every STRIDE-th frame, DWPose body-keypoint scores of the most
  confident person -> one npz per video.
* ``analyze`` (project environment): per part and variant, mean confidence,
  and how well low confidence picks out frames where CoMotion's per-part
  error against the AIST++ reference is large (ROC AUC; 0.5 = no signal).

    MimicMotion/.venv/bin/python -m motionprov.cv801.visibility extract --mimicmotion MimicMotion \
        --videos clean=WORK/videos --videos mimicmotion=GEN/videos ... --camera 01 --output OUT
    python -m motionprov.cv801.visibility analyze --scores OUT --reference REF_CACHE --split SPLIT.json \
        --cache clean=COMOTION_CACHE --cache mimicmotion=GEN_CACHE ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

STRIDE = 2
# OpenPose-18 body order used by DWPose; left/right are the person's own. The neck (1)
# is left out: DWPose sets its score to 0/1 from the shoulders.
PART_KEYPOINTS = {
    "torso": (2, 5, 8, 11),
    "left_arm": (6, 7),
    "right_arm": (3, 4),
    "left_leg": (12, 13),
    "right_leg": (9, 10),
}
LARGE_ERROR_QUANTILE = 0.9      # "large" = above this quantile of the clean videos' error for that part


def extract(args):
    os.chdir(args.mimicmotion)
    sys.path.insert(0, os.getcwd())
    import cv2
    import onnxruntime
    onnxruntime.preload_dlls()      # CUDA/cuDNN from the torch wheels; otherwise DWPose falls back to CPU
    from mimicmotion.dwpose.wholebody import Wholebody
    model = Wholebody("models/DWPose/yolox_l.onnx", "models/DWPose/dw-ll_ucoco_384.onnx", "cuda")
    output = Path(args.output).resolve()
    for spec in args.videos:
        name, directory = spec.split("=", 1)
        (output / name).mkdir(parents=True, exist_ok=True)
        for video in sorted(Path(directory).glob(f"*_c{args.camera}_*.mp4")):
            target = output / name / f"{video.stem}.npz"
            if target.exists():
                continue
            capture = cv2.VideoCapture(str(video))
            frames, scores, k = [], [], 0
            while True:
                ok, image = capture.read()
                if not ok:
                    break
                if k % STRIDE == 0:
                    try:
                        _, score = model(image)
                    except (ValueError, IndexError):     # no person detected
                        score = np.zeros((0, 18), np.float32)
                    best = score[:, :18].mean(1).argmax() if len(score) else None
                    frames.append(k)
                    scores.append(score[best, :18] if best is not None else np.zeros(18, np.float32))
                k += 1
            capture.release()
            np.savez_compressed(target, frames=np.asarray(frames), scores=np.asarray(scores, np.float32))
            print(name, video.stem, len(frames), flush=True)


def analyze(args):
    from sklearn.metrics import roc_auc_score
    from motionprov.core.datasets.local_pose_cache import load_pose_cache
    from motionprov.cv801.diagnostic import load_views, procrustes_errors, skeleton
    from motionprov.core.models.local_extensions import PART_JOINT_IDS

    split = json.loads(Path(args.split).read_text())
    takes = set(split["videos"]["confirmation"])
    reference = {v: skeleton(r) for v, r in load_pose_cache(args.reference).items() if v in takes}
    scores_dir = Path(args.scores)
    rows = {}   # variant -> part -> list of (confidence, error)
    for spec in args.cache:
        name, cache = spec.split("=", 1)
        rows[name] = {p: [] for p in PART_JOINT_IDS}
        for video, (joints, frames, _, take) in load_views(cache).items():
            path = scores_dir / name / f"{video}.npz"
            if not path.exists() or take not in reference:
                continue
            data = np.load(path)
            conf = dict(zip(data["frames"].tolist(), data["scores"]))
            ref_joints, ref_frames = reference[take]
            common, pi, ri = np.intersect1d(frames, ref_frames, return_indices=True)
            err = procrustes_errors(joints[pi], ref_joints[ri])
            for f, e in zip(common.tolist(), err):
                if f not in conf:
                    continue
                for part, ids in PART_JOINT_IDS.items():
                    rows[name][part].append((float(conf[f][list(PART_KEYPOINTS[part])].mean()),
                                             float(e[list(ids)].mean())))
    clean = args.baseline
    out = {}
    for name, parts in rows.items():
        out[name] = {}
        for part, pairs in parts.items():
            if not pairs:
                continue
            c, e = np.asarray(pairs).T
            threshold = np.quantile(np.asarray(rows[clean][part])[:, 1], LARGE_ERROR_QUANTILE)
            large = e > threshold
            auc = roc_auc_score(large, -c) if 0 < large.sum() < len(large) else None
            out[name][part] = {"frames": int(len(c)), "confidence": float(c.mean()), "error_cm": float(e.mean() * 100),
                               "large_error_rate": float(large.mean()), "auc_low_conf_vs_large_error": auc}
    Path(args.output).write_text(json.dumps(out, indent=2))
    for name, parts in out.items():
        print(name)
        for part, r in parts.items():
            auc = "–" if r["auc_low_conf_vs_large_error"] is None else f"{r['auc_low_conf_vs_large_error']:.2f}"
            print(f"  {part:10s} conf {r['confidence']:.2f}  error {r['error_cm']:.1f} cm  "
                  f"large {r['large_error_rate']:.2f}  AUC {auc}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--mimicmotion", required=True)
    e.add_argument("--videos", action="append", required=True, metavar="NAME=DIR")
    e.add_argument("--camera", default="01")
    e.add_argument("--output", required=True)
    a = sub.add_parser("analyze")
    a.add_argument("--scores", required=True)
    a.add_argument("--reference", required=True)
    a.add_argument("--split", required=True)
    a.add_argument("--cache", action="append", required=True, metavar="NAME=CACHE")
    a.add_argument("--baseline", default="clean")
    a.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    extract(args) if args.command == "extract" else analyze(args)


if __name__ == "__main__":
    main()
