"""Skeleton corpora on one footing: 30 fps SMPL joints in metres, y up, with a per-frame mask.

* AIST++ (dev split): the released SMPL fits (60 fps -> 30 fps), labelled by
  (genre, situation, choreography) and partitioned by the repo's class split.
* TikTok (zero-shot test): CoMotion's main track on the Jafarian & Park clips,
  labelled by the trend of their source compilation. Camera frame (y down)
  is turned to y up; frames without a detection stay in the sequence, unobserved.

A recording is {id, label, source, split, joints (T, 24, 3), observed (T,)}.

    python -m motionprov.choreomatch.data aist --motions AIST/motions --split SPLIT.json --output aist.pkl
    python -m motionprov.choreomatch.data tiktok --dataset TIKTOK --comotion DIR --output tiktok.pkl
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch

PARTS = {                     # SMPL joints (pelvis = 0 excluded) by body part
    "torso": (3, 6, 9, 12, 15),
    "left_arm": (13, 16, 18, 20, 22),
    "right_arm": (14, 17, 19, 21, 23),
    "left_leg": (1, 4, 7, 10),
    "right_leg": (2, 5, 8, 11),
}
FPS = 30


def smpl_decoder():
    from motionprov.core.comotion.utils.smpl_kinematics import SMPLKinematics
    return SMPLKinematics()


def aist(motions: Path, split: dict, decoder):
    partition = {v: p for p, videos in split["videos"].items() for v in videos}
    out = []
    for pkl in sorted(motions.glob("*.pkl")):
        if pkl.stem not in partition:
            continue
        m = pickle.load(open(pkl, "rb"))
        pose = torch.from_numpy(m["smpl_poses"][::2]).float()
        trans = torch.from_numpy(m["smpl_trans"][::2] / m["smpl_scaling"][0]).float()
        with torch.no_grad():
            joints = decoder(torch.zeros(len(pose), 10), pose, trans).numpy()
        out.append({"id": pkl.stem, "label": split["labels"][pkl.stem], "source": split["source_groups"][pkl.stem],
                    "split": partition[pkl.stem], "joints": joints.astype(np.float32),
                    "observed": np.ones(len(joints), bool)})
    return out


def tiktok(dataset: Path, comotion: Path, decoder):
    from motionprov.cv801.harmony4d import comotion_joints
    from motionprov.cv801.occlude import main_track
    from motionprov.cv801.tiktok import dance_labels
    labels, out = dance_labels(dataset), []
    for pt in sorted(comotion.glob("*.pt")):
        if pt.stem not in labels:
            continue
        frame_idx, track_ids, joints = comotion_joints(pt, decoder)
        if not len(frame_idx):
            continue
        keep = track_ids == main_track(frame_idx, track_ids)
        frame_idx, joints = frame_idx[keep], joints[keep]
        n = int(frame_idx.max()) + 1
        dense, observed = np.zeros((n, 24, 3), np.float32), np.zeros(n, bool)
        dense[frame_idx], observed[frame_idx] = joints, True
        dense[..., 1:] *= -1                        # camera (y down, z forward) -> y up
        out.append({"id": pt.stem, "label": labels[pt.stem], "source": pt.stem, "split": "test",
                    "joints": dense, "observed": observed})
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="corpus", required=True)
    a = sub.add_parser("aist")
    a.add_argument("--motions", required=True)
    a.add_argument("--split", required=True)
    a.add_argument("--output", required=True)
    t = sub.add_parser("tiktok")
    t.add_argument("--dataset", required=True)
    t.add_argument("--comotion", required=True)
    t.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    decoder = smpl_decoder()
    if args.corpus == "aist":
        records = aist(Path(args.motions), json.loads(Path(args.split).read_text()), decoder)
    else:
        records = tiktok(Path(args.dataset), Path(args.comotion), decoder)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(records, open(args.output, "wb"))
    labels = {r["label"] for r in records}
    print(f"{len(records)} recordings, {len(labels)} labels, {sum(len(r['joints']) for r in records)} frames")


if __name__ == "__main__":
    main()
