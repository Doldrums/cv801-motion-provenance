"""Vertical close framing of AIST++ views, the way a dance clip is posted.

The studio cameras see the whole floor, so the dancer is a third of the frame
height and the face a few dozen pixels: too little for a generator to put a
recognisable face on. Here every view is cropped to 9:16 around the dancer
(the clean CoMotion track, projected as in ``occlude``), with a fixed crop
size per clip and a smoothed centre, then scaled to SIZE.

    python -m motionprov.cv801.reframe --videos WORK/videos --comotion WORK/comotion \
        --list SOURCES.txt --frames 150 --output OUT
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np

from motionprov.cv801.harmony4d import comotion_joints
from motionprov.cv801.occlude import main_track, project

SIZE = (480, 832)       # output width, height (9:16)
MARGIN = 1.35           # crop height as a multiple of the tallest projected body in the clip
SMOOTH = 15             # frames in the moving average of the crop centre


def crop_boxes(pixels, n, width, height):
    """frame -> (x0, y0, w, h) of a fixed-size 9:16 box following the body centre."""
    frames = sorted(f for f in pixels if f < n)
    tall = max(np.ptp(pixels[f][:, 1]) for f in frames)
    h = min(height, int(MARGIN * tall))
    w = int(h * SIZE[0] / SIZE[1])
    centre = np.array([[*pixels[f].min(0) / 2 + pixels[f].max(0) / 2] for f in frames])
    track = np.stack([np.interp(np.arange(n), frames, centre[:, i]) for i in range(2)], 1)
    kernel = np.ones(SMOOTH) / SMOOTH
    track = np.stack([np.convolve(np.pad(track[:, i], SMOOTH // 2, mode="edge"), kernel, "valid")[:n]
                      for i in range(2)], 1)
    x0 = np.clip(track[:, 0] - w / 2, 0, width - w).astype(int)
    y0 = np.clip(track[:, 1] - h / 2, 0, height - h).astype(int)
    return x0, y0, w, h


def reframe(video: Path, pt: Path, frames: int, output: Path, decoder):
    capture = cv2.VideoCapture(str(video))
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    n = min(frames, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    frame_idx, track_ids, joints = comotion_joints(pt, decoder)
    keep = track_ids == main_track(frame_idx, track_ids)
    pixels = dict(zip(frame_idx[keep].tolist(), project(joints[keep], width, height)))
    x0, y0, w, h = crop_boxes(pixels, n, width, height)
    writer = subprocess.Popen(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{SIZE[0]}x{SIZE[1]}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-crf", "16",
         "-preset", "veryfast", "-pix_fmt", "yuv420p", str(output)], stdin=subprocess.PIPE)
    for k in range(n):
        ok, frame = capture.read()
        if not ok:
            break
        crop = frame[y0[k]:y0[k] + h, x0[k]:x0[k] + w]
        writer.stdin.write(cv2.resize(crop, SIZE, interpolation=cv2.INTER_LANCZOS4).tobytes())
    capture.release()
    writer.stdin.close()
    writer.wait()
    return h


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", required=True)
    parser.add_argument("--comotion", required=True)
    parser.add_argument("--list", required=True, help="one video id per line")
    parser.add_argument("--frames", type=int, default=150)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    from motionprov.core.comotion.utils.smpl_kinematics import SMPLKinematics
    decoder = SMPLKinematics()
    for video in Path(args.list).read_text().split():
        h = reframe(Path(args.videos) / f"{video}.mp4", Path(args.comotion) / f"{video}.pt", args.frames,
                    output / f"{video}.mp4", decoder)
        print(video, "crop height", h, flush=True)


if __name__ == "__main__":
    main()
