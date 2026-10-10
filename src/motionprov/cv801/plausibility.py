"""Physical plausibility of recovered motion: real, occluded and generated video.

If a generator fills in motion, the recovered skeleton may show it as
physics a body does not have. Three cheap per-view statistics on CoMotion
output (camera frame, metres), all at 15 fps so the 15 fps generator output
re-timed by frame repetition is compared on equal terms:

* foot_skate   horizontal speed (m/s) of a foot while it is in ground contact
               (within CONTACT_M of its lowest height in a 1 s window)
* jerk         mean norm of the third difference of root-relative joints (m/s^3)

These are not a detector: a better generator removes them. They describe
what this generator leaves behind. Bone-length drift is not among them:
CoMotion holds body shape fixed per track, so bones cannot drift.

Paired (default): AIST views present in every source. ``--unpaired``: every
clip of each source, e.g. real TikTok clips against text-to-video clips, with
each later source's AUC against the first per statistic. A source at another
frame rate (``NAME=DIR@24``) is linearly resampled to 15 fps between
consecutive frames; the resampling smooths, so it can only hide excess jerk.

    python -m motionprov.cv801.plausibility --source clean=WORK/comotion \
        --source occluded_arms=OCC/arms/comotion --source mimicmotion=GEN/comotion --output OUT
    python -m motionprov.cv801.plausibility --unpaired --source real=TIKTOK/comotion \
        --source t2v=T2V/comotion@24 --output OUT
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from motionprov.cv801.harmony4d import comotion_joints
from motionprov.cv801.occlude import main_track

STRIDE = 2              # 30 -> 15 fps
FPS = 30 / STRIDE
FEET = (10, 11)
CONTACT_M = 0.03
WINDOW = int(FPS)


def track(pt: Path, decoder, fps=30):
    """Main track as (T, 24, 3) on consecutive frames at 15 fps."""
    frame_idx, track_ids, joints = comotion_joints(pt, decoder)
    keep = track_ids == main_track(frame_idx, track_ids)
    frame_idx, joints = frame_idx[keep], joints[keep]
    order = np.argsort(frame_idx)
    frame_idx, joints = frame_idx[order], joints[order]
    if fps == 30:
        pick = frame_idx % STRIDE == 0
        return frame_idx[pick] // STRIDE, joints[pick]
    out_idx, out = [], []
    for k in range(int(np.ceil(frame_idx[-1] / fps * FPS)) + 1):
        t = k / FPS * fps
        i = np.searchsorted(frame_idx, t, side="right") - 1
        if i < 0 or i + 1 >= len(frame_idx) or frame_idx[i + 1] - frame_idx[i] != 1:
            if i >= 0 and frame_idx[i] == t:
                out_idx.append(k); out.append(joints[i])
            continue
        w = t - frame_idx[i]
        out_idx.append(k); out.append((1 - w) * joints[i] + w * joints[i + 1])
    return np.array(out_idx), np.array(out)


def statistics(frames, joints):
    # Only runs of consecutive frames enter the finite differences.
    step = np.diff(frames) == 1
    rel = joints - joints[:, :1]
    third = rel[3:] - 3 * rel[2:-1] + 3 * rel[1:-2] - rel[:-3]
    valid3 = step[2:] & step[1:-1] & step[:-2]
    jerk = float(np.linalg.norm(third[valid3], axis=-1).mean() * FPS ** 3) if valid3.any() else None
    # Camera frame: image y points down, so height is -y.
    skate = []
    for foot in FEET:
        height = -joints[:, foot, 1]
        low = np.array([height[max(0, i - WINDOW):i + WINDOW + 1].min() for i in range(len(height))])
        contact = (height - low < CONTACT_M)
        both = contact[1:] & contact[:-1] & step
        speed = np.linalg.norm(np.diff(joints[:, foot][:, [0, 2]], axis=0), axis=-1) * FPS
        skate.extend(speed[both].tolist())
    return {"frames": int(len(frames)), "jerk": jerk,
            "foot_skate": float(np.median(skate)) if skate else None}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True, metavar="NAME=COMOTION_DIR")
    parser.add_argument("--camera", default="01")
    parser.add_argument("--unpaired", action="store_true", help="every clip of each source, no camera filter")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    from motionprov.core.comotion.utils.smpl_kinematics import SMPLKinematics
    decoder = SMPLKinematics()
    sources, fps = {}, {}
    for spec in args.source:
        name, path = spec.split("=", 1)
        path, _, rate = path.partition("@")
        sources[name], fps[name] = path, int(rate or 30)
    if args.unpaired:
        stems = {name: sorted(p.stem for p in Path(d).glob("*.pt")) for name, d in sources.items()}
    else:
        # Compare only views present in every source, so the sets are paired.
        common = set.intersection(*({p.stem for p in Path(d).glob(f"*_c{args.camera}_*.pt")} for d in sources.values()))
        stems = {name: sorted(common) for name in sources}
    per_view = {}
    for name, d in sources.items():
        per_view[name] = {}
        for s in stems[name]:
            frames, joints = track(Path(d) / f"{s}.pt", decoder, fps[name])
            if len(frames) > 3:
                per_view[name][s] = statistics(frames, joints)
    summary = {"views": {name: len(rows) for name, rows in per_view.items()}}
    first = next(iter(per_view))
    for name, rows in per_view.items():
        summary[name] = {}
        for key in ("foot_skate", "jerk"):
            values = [r[key] for r in rows.values() if r[key] is not None]
            summary[name][key] = float(np.median(values))
            summary[name][key + "_iqr"] = np.percentile(values, [25, 75]).round(4).tolist()
            if args.unpaired and name != first:
                base = [r[key] for r in per_view[first].values() if r[key] is not None]
                summary[name][key + "_auc_vs_" + first] = float(
                    roc_auc_score(np.r_[np.ones(len(values)), np.zeros(len(base))], np.r_[values, base]))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "plausibility.json").write_text(json.dumps({"summary": summary, "views": per_view}, indent=1))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
