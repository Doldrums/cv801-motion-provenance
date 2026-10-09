"""Pixel-level occlusion intervention on AIST++ views.

The skeleton matters, if at all, where the body is hidden. Here a body part is
painted out of the video itself (grey capsules over its bones, every frame),
then CoMotion is run again on the edited video. Holding the take and the
matcher fixed, the change in retrieval and per-part error is the effect of
losing that part of the image.

Where to paint comes from the clean CoMotion run of the same video, projected
with the same default intrinsics CoMotion assumes (focal 2 * max side, centred
principal point). The mask is a function of the clean video only.

    python -m motionprov.cv801.occlude --videos WORK/videos --comotion WORK/comotion \
        --camera 01 --part arms --output OCC/arms
    # run toolkit.process_comotion on OCC/arms/videos.csv
"""

from __future__ import annotations

import argparse
import subprocess
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from motionprov.cv801.harmony4d import comotion_joints

BONES = {
    "arms": [(16, 18), (18, 20), (20, 22), (17, 19), (19, 21), (21, 23)],
    "legs": [(1, 4), (4, 7), (7, 10), (2, 5), (5, 8), (8, 11)],
}
NECK, PELVIS = 12, 0
WIDTH = 0.4             # capsule thickness as a fraction of the pelvis-neck length in pixels
HOLD = 15               # frames a mask is kept when the clean run has no detection
GREY = (128, 128, 128)


def project(joints, width, height):
    """Camera-frame joints -> pixels under CoMotion's default intrinsics."""
    focal = 2 * max(width, height)
    z = np.clip(joints[..., 2:3], 1e-3, None)
    return joints[..., :2] / z * focal + np.array([width / 2, height / 2])


def main_track(frame_idx, track_ids):
    """The dancer: the track present in most frames (AIST++ is one person)."""
    ids, counts = np.unique(track_ids, return_counts=True)
    return ids[counts.argmax()]


def occlude_video(video: Path, pt: Path, part: str, output: Path, decoder):
    capture = cv2.VideoCapture(str(video))
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_idx, track_ids, joints = comotion_joints(pt, decoder)
    keep = track_ids == main_track(frame_idx, track_ids)
    pixels = dict(zip(frame_idx[keep].tolist(), project(joints[keep], width, height)))
    writer = subprocess.Popen(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-crf", "18",
         "-preset", "veryfast", "-pix_fmt", "yuv420p", str(output)], stdin=subprocess.PIPE)
    last, since, k, painted = None, HOLD + 1, 0, 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if k in pixels:
            last, since = pixels[k], 0
        else:
            since += 1
        if last is not None and since <= HOLD:
            thick = max(12, int(WIDTH * np.linalg.norm(last[NECK] - last[PELVIS])))
            for a, b in BONES[part]:
                pa, pb = tuple(last[a].astype(int)), tuple(last[b].astype(int))
                cv2.line(frame, pa, pb, GREY, thick, cv2.LINE_AA)
                cv2.circle(frame, pa, thick // 2, GREY, -1, cv2.LINE_AA)
                cv2.circle(frame, pb, thick // 2, GREY, -1, cv2.LINE_AA)
            painted += 1
        writer.stdin.write(frame.tobytes())
        k += 1
    writer.stdin.close()
    writer.wait()
    capture.release()
    return video.stem, k, painted


def _job(video, comotion: Path, part: str, output: Path):
    from motionprov.core.comotion.utils.smpl_kinematics import SMPLKinematics
    target = output / "videos" / video.name
    if target.exists():
        return video.stem, None, None
    return occlude_video(video, comotion / f"{video.stem}.pt", part, target, SMPLKinematics())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", required=True)
    parser.add_argument("--comotion", required=True)
    parser.add_argument("--camera", default="01")
    parser.add_argument("--part", choices=sorted(BONES), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args(argv)
    comotion, output = Path(args.comotion), Path(args.output)
    (output / "videos").mkdir(parents=True, exist_ok=True)
    videos = [v for v in sorted(Path(args.videos).glob(f"*_c{args.camera}_*.mp4"))
              if (comotion / f"{v.stem}.pt").exists()]
    with Pool(args.workers) as pool:
        for stem, frames, painted in pool.imap_unordered(
                partial(_job, comotion=comotion, part=args.part, output=output), videos):
            print(stem, frames, painted, flush=True)
    rows = [f"{v.stem},,videos/{v.name}" for v in videos]
    (output / "videos.csv").write_text("dance_id,video_url,video_path\n" + "\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
