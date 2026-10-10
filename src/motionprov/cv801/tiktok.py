"""Choreography retrieval on real viral dances: the TikTok dataset (Jafarian & Park, CVPR 2021).

The dataset's raw YouTube compilations group its 340 clips by dance: one
trend, many people each filming it at home. That gives what AIST++ gives in
a studio, re-performances of one choreography by different people, on
phone-shot, close-framed, cluttered real content. Compilations named by
month or "Mix" hold several dances and are left out.

* ``videos``: frame folders of the kept groups -> 30 fps mp4 + videos.csv
  (run ``toolkit.process_comotion`` on it).
* ``evaluate``: the diagnostic's fixed matchers (kNN, DTW) on CoMotion's main
  track; a positive is another clip of the same dance, every clip its own source.
* ``sources``: one source clip per well-populated dance, for generated copies.
* ``provenance``: copies of those sources against the source and against the
  same dance by other people, on the copies' first frames (see provenance.py).

    python -m motionprov.cv801.tiktok videos --dataset TIKTOK --output OUT
    python -m motionprov.cv801.tiktok evaluate --dataset TIKTOK --comotion OUT/comotion --output OUT/eval
    python -m motionprov.cv801.tiktok sources --dataset TIKTOK --comotion OUT/comotion --output sources.txt
    python -m motionprov.cv801.tiktok provenance --dataset TIKTOK --comotion OUT/comotion \
        --generated wan=GEN/comotion --output OUT/provenance
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from motionprov.cv801.diagnostic import evaluate as retrieval
from motionprov.cv801.harmony4d import comotion_joints
from motionprov.cv801.occlude import main_track
from motionprov.cv801.provenance import provenance as copy_scores

MIXED = {"Dec2020", "Oct2020", "Mix1", "september2020"}
MIN_FRAMES = 60
COPY_FRAMES = 150
MIN_DANCERS = 7


def dance_labels(dataset: Path):
    """clip id -> dance name, from the raw compilations' seq_<first>_<last> folders."""
    labels = {}
    for seq in sorted((dataset / "TikTok_Raw_Videos/TikTok_Raw_Videos").glob("seq_*")):
        name = (seq / "dance_name.txt").read_text().strip()
        if name in MIXED:
            continue
        first, last = (int(x) for x in seq.name.split("_")[1:])
        labels.update({f"{i:05d}": name for i in range(first, last + 1)})
    return labels


def videos(args):
    dataset, output = Path(args.dataset), Path(args.output)
    (output / "videos").mkdir(parents=True, exist_ok=True)
    rows = ["dance_id,video_url,video_path"]
    for clip in sorted(dance_labels(dataset)):
        frames = dataset / "TikTok_dataset/TikTok_dataset" / clip / "images"
        target = output / "videos" / f"{clip}.mp4"
        if not frames.is_dir():
            continue
        if not target.exists():
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-framerate", "30", "-i", str(frames / "%04d.png"),
                            "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", str(target)], check=True)
        rows.append(f"{clip},,{target}")
    (output / "videos.csv").write_text("\n".join(rows) + "\n")
    print(len(rows) - 1, "clips")


def main_joints(comotion, labels, frames=None):
    """clip id -> (root-centred main-track joints, frame indices), clips of known dances only."""
    from motionprov.core.comotion.utils.smpl_kinematics import SMPLKinematics
    decoder, out = SMPLKinematics(), {}
    for pt in sorted(Path(comotion).glob("*.pt")):
        if pt.stem not in labels:
            continue
        frame_idx, track_ids, joints = comotion_joints(pt, decoder)
        if not len(frame_idx):
            continue
        keep = track_ids == main_track(frame_idx, track_ids)
        order = np.argsort(frame_idx[keep], kind="stable")
        joints, frame_idx = joints[keep][order], frame_idx[keep][order]
        if frames is not None:
            joints, frame_idx = joints[frame_idx < frames], frame_idx[frame_idx < frames]
        if len(joints) >= MIN_FRAMES:
            out[pt.stem] = ((joints - joints[:, :1]).astype(np.float32), frame_idx)
    return out


def evaluate(args):
    labels = dance_labels(Path(args.dataset))
    items = {c: (j, labels[c], c) for c, (j, _) in main_joints(args.comotion, labels).items()}
    result = retrieval(items, args.device)
    counts = {}
    for _, label, _ in items.values():
        counts[label] = counts.get(label, 0) + 1
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    summary = {"clips": len(items), "dances": counts} | {m: r["summary"] for m, r in result.items()}
    (output / "summary.json").write_text(json.dumps(summary, indent=1))
    (output / "queries.json").write_text(json.dumps({m: r["queries"] for m, r in result.items()}))
    print(json.dumps(summary, indent=1))


def sources(args):
    """The first clip with a full copy's worth of frames, per dance with at least MIN_DANCERS clips."""
    labels = dance_labels(Path(args.dataset))
    tracked = main_joints(args.comotion, labels)
    dances, chosen = {}, {}
    for clip in tracked:
        dances.setdefault(labels[clip], []).append(clip)
    for dance, clips in sorted(dances.items()):
        full = [c for c in clips if tracked[c][1].max() >= COPY_FRAMES - 1]
        if len(clips) >= MIN_DANCERS and full:
            chosen[dance] = full[0]
    Path(args.output).write_text("".join(f"{c}\n" for c in chosen.values()))
    print(json.dumps(chosen, indent=1))


def provenance(args):
    labels = dance_labels(Path(args.dataset))
    clean = {c: (j, f, labels[c], c) for c, (j, f) in main_joints(args.comotion, labels, COPY_FRAMES).items()}
    result = {}
    for spec in args.generated:
        name, cache = spec.split("=", 1)
        copies = {c: (j, f, labels[c], c) for c, (j, f) in main_joints(cache, labels, COPY_FRAMES).items()}
        result[name] = copy_scores(clean, copies, args.device, args.mirror)
        for method, r in result[name].items():
            print(name, method, json.dumps(r["summary"]), json.dumps(r["open_set"]))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "provenance.json").write_text(json.dumps(result, indent=1))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    v = sub.add_parser("videos")
    v.add_argument("--dataset", required=True)
    v.add_argument("--output", required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("--dataset", required=True)
    e.add_argument("--comotion", required=True)
    e.add_argument("--output", required=True)
    s = sub.add_parser("sources")
    s.add_argument("--dataset", required=True)
    s.add_argument("--comotion", required=True)
    s.add_argument("--output", required=True)
    p = sub.add_parser("provenance")
    p.add_argument("--dataset", required=True)
    p.add_argument("--comotion", required=True)
    p.add_argument("--generated", action="append", required=True, metavar="NAME=COMOTION_DIR")
    p.add_argument("--mirror", action="store_true", help="score each pair also with the query mirrored")
    p.add_argument("--output", required=True)
    for command in (e, p):
        command.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    {"videos": videos, "evaluate": evaluate, "sources": sources, "provenance": provenance}[args.command](args)


if __name__ == "__main__":
    main()
