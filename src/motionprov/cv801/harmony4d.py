"""CoMotion against Harmony4D ground truth: two people, contact, occlusion.

AIST++ is one dancer in a studio; there CoMotion and the reference fit gave
the same retrieval. Harmony4D (hugging, grappling, sword, ballroom, MMA) is
where a single-view estimator should break, and it has multi-view SMPL fits
for both people. There is no choreography identity here, so this measures
pose quality only: per-part error, missed frames, and error against how close
the two people are.

    python -m motionprov.cv801.harmony4d extract --zips ZIPS --output WORK
    # run toolkit.process_comotion on WORK/videos.csv
    python -m motionprov.cv801.harmony4d evaluate --work WORK --output OUT
"""

from __future__ import annotations

import argparse
import json
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from motionprov.core.models.local_extensions import PART_JOINT_IDS
from motionprov.cv801.diagnostic import procrustes_align

CAMERAS_PER_SEQUENCE = 4
FPS = 30                # nominal: only the frame order matters here
CONTACT_BINS = (0.0, 0.05, 0.15, 0.4, np.inf)   # closest joint pair, metres


def sequences(archive: zipfile.ZipFile):
    """-> {sequence path: sorted exo camera names}."""
    found = {}
    for name in archive.namelist():
        parts = name.split("/")
        if len(parts) > 4 and parts[2] == "exo" and parts[4] == "images":
            found.setdefault("/".join(parts[:2]), set()).add(parts[3])
    return {seq: sorted(cams) for seq, cams in found.items()}


def spread(cams):
    """Evenly spaced cameras around the arena."""
    step = len(cams) / CAMERAS_PER_SEQUENCE
    return sorted({cams[int(i * step)] for i in range(CAMERAS_PER_SEQUENCE)})


def extract(zips: Path, output: Path):
    rows = []
    for path in sorted(zips.glob("*.zip")):
        with zipfile.ZipFile(path) as archive:
            for seq, cams in sequences(archive).items():
                chosen = spread(cams)
                prefixes = [f"{seq}/processed_data/smpl/"] + [f"{seq}/exo/{c}/images/" for c in chosen]
                members = [n for n in archive.namelist() if n.startswith(tuple(prefixes))]
                archive.extractall(output / "raw", members)
                for cam in chosen:
                    video_id = f"{seq.replace('/', '__')}__{cam}"
                    target = output / "videos" / f"{video_id}.mp4"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if not target.exists():
                        frames = output / "raw" / seq / "exo" / cam / "images"
                        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y",
                                        "-framerate", str(FPS), "-pattern_type", "glob",
                                        "-i", str(frames / "*.jpg"), "-vf", "scale=1920:-2",
                                        "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
                                        "-pix_fmt", "yuv420p", str(target)], check=True)
                    rows.append(f"{seq.split('/')[1]},,videos/{video_id}.mp4")
                print(f"{seq}: {len(cams)} exo cameras, kept {chosen}")
    (output / "videos.csv").write_text("dance_id,video_url,video_path\n" + "\n".join(rows) + "\n")


def ground_truth(seq_dir: Path):
    """-> frame numbers (as in image names) and {person: (F, 24, 3)} world joints."""
    files = sorted((seq_dir / "processed_data" / "smpl").glob("*.npy"))
    numbers, people = [], {}
    for f in files:
        data = np.load(f, allow_pickle=True).item()
        numbers.append(int(f.stem))
        for person, fit in data.items():
            people.setdefault(person, []).append(np.asarray(fit["joints"][:24], np.float32))
    return np.asarray(numbers), {p: np.stack(j) for p, j in people.items()}


def comotion_joints(pt: Path, decoder):
    """-> rows' frame indices, track ids and (N, 24, 3) camera-frame joints."""
    data = torch.load(pt, map_location="cpu", weights_only=False)
    with torch.no_grad():
        joints = decoder(data["betas"], data["pose"], data["trans"])
    return data["frame_idx"].numpy(), data["id"].numpy(), joints.numpy()


def closest_pair(people):
    """Closest joint pair between the two people per frame: a direct proxy for contact."""
    a, b = (people[n] for n in sorted(people)[:2])
    return np.linalg.norm(a[:, :, None] - b[:, None], axis=-1).min((1, 2))


def associate(frame_idx, joints, image_numbers, numbers, people):
    """-> [(video frame k, ground-truth frame g, {person: (24,) aligned error, (24, 3) aligned joints})].

    Oracle one-to-one association per frame: each person takes a distinct
    detection, the assignment minimising aligned error. This bounds pose
    quality, not tracking identity; a person left without one is missed.
    """
    gt_index = {n: i for i, n in enumerate(numbers)}
    names = sorted(people)
    out = []
    for k, number in enumerate(image_numbers):
        g = gt_index.get(number)
        if g is None:
            continue
        rows = np.flatnonzero(frame_idx == k)
        assigned = {}
        if len(rows):
            aligned = {n: procrustes_align(joints[rows], np.repeat(people[n][g][None], len(rows), 0))
                       for n in names}
            errs = {n: np.linalg.norm(aligned[n] - people[n][g], axis=2) for n in names}
            cost = np.stack([errs[n].mean(1) for n in names])
            for p, r in zip(*linear_sum_assignment(cost)):
                assigned[names[p]] = (errs[names[p]][r], aligned[names[p]][r])
        out.append((k, g, assigned))
    return out


def evaluate_view(pt: Path, seq_dir: Path, image_numbers, decoder):
    numbers, people = ground_truth(seq_dir)
    frame_idx, track_ids, joints = comotion_joints(pt, decoder)
    names = sorted(people)
    closeness = closest_pair(people)
    per_frame = {n: [] for n in names}
    for _, g, assigned in associate(frame_idx, joints, image_numbers, numbers, people):
        for n in names:
            per_frame[n].append((closeness[g], assigned[n][0] if n in assigned else None))
    out = {"track_ids": int(len(np.unique(track_ids))), "people": {}}
    for n, frames in per_frame.items():
        seen = [(c, e) for c, e in frames if e is not None]
        parts = {p: float(np.mean([e[list(ids)].mean() for _, e in seen])) if seen else None
                 for p, ids in PART_JOINT_IDS.items()}
        binned = []
        for lo, hi in zip(CONTACT_BINS[:-1], CONTACT_BINS[1:]):
            inside = [(c, e) for c, e in frames if lo <= c < hi]
            hit = [e for _, e in inside if e is not None]
            binned.append({"closest_m": [lo, None if np.isinf(hi) else hi], "frames": len(inside),
                           "missed": len(inside) - len(hit),
                           "error": float(np.mean([e[1:].mean() for e in hit])) if hit else None})
        out["people"][n] = {"frames": len(frames), "missed": len(frames) - len(seen),
                            "error_by_part": parts, "by_closeness": binned}
    return out


def evaluate(work: Path, output: Path):
    from motionprov.core.comotion.utils.smpl_kinematics import SMPLKinematics
    decoder = SMPLKinematics()
    results = {}
    for pt in sorted((work / "comotion").glob("*.pt")):
        group, seq, cam = pt.stem.split("__")
        seq_dir = work / "raw" / group / seq
        images = sorted((seq_dir / "exo" / cam / "images").glob("*.jpg"))
        results[pt.stem] = evaluate_view(pt, seq_dir, [int(p.stem) for p in images], decoder)
        print(pt.stem, json.dumps({n: p["error_by_part"] | {"missed": p["missed"]}
                                   for n, p in results[pt.stem]["people"].items()}))
    output.mkdir(parents=True, exist_ok=True)
    (output / "harmony4d.json").write_text(json.dumps(results, indent=1))
    people = [p for r in results.values() for p in r["people"].values()]
    summary = {"views": len(results),
               "missed_rate": sum(p["missed"] for p in people) / sum(p["frames"] for p in people),
               "error_by_part": {part: float(np.mean([p["error_by_part"][part] for p in people
                                                      if p["error_by_part"][part] is not None]))
                                 for part in PART_JOINT_IDS},
               "by_closeness": []}
    for i, (lo, hi) in enumerate(zip(CONTACT_BINS[:-1], CONTACT_BINS[1:])):
        cells = [p["by_closeness"][i] for p in people]
        frames, missed = sum(c["frames"] for c in cells), sum(c["missed"] for c in cells)
        errors = [(c["error"], c["frames"] - c["missed"]) for c in cells if c["error"] is not None]
        summary["by_closeness"].append({
            "closest_m": [lo, None if np.isinf(hi) else hi], "frames": frames,
            "missed_rate": missed / frames if frames else None,
            "error": (sum(e * w for e, w in errors) / sum(w for _, w in errors)) if errors else None})
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--zips", required=True)
    e.add_argument("--output", required=True)
    v = sub.add_parser("evaluate")
    v.add_argument("--work", required=True)
    v.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "extract":
        extract(Path(args.zips), Path(args.output))
    else:
        evaluate(Path(args.work), Path(args.output))


if __name__ == "__main__":
    main()
