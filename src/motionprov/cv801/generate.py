"""Motion transfer: re-render each AIST++ view with a different dancer (MimicMotion).

A generated copy of a dance keeps the source motion only as far as the
generator carries it over; the rest it fills in. Here the source is known
exactly, so per body part one can see what survived generation.

Each source view drives MimicMotion (SVD + DWPose conditioning) with a frame
of a *different* AIST++ dancer as the reference image: dancers are pooled
from the same camera, and take d is rendered as the next dancer in sorted
order. MimicMotion samples every second frame (15 fps); the output is
re-timed to 30 fps by frame repetition so frame indices match the source.

Runs inside a MimicMotion checkout's environment:

    MimicMotion/.venv/bin/python -m motionprov.cv801.generate --mimicmotion MimicMotion \
        --videos WORK/videos --camera 01 --output GEN/mimicmotion [--shard 0/3]
    # run toolkit.process_comotion on GEN/mimicmotion/videos.csv
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

DANCER = re.compile(r"_(d\d\d)_")
STEPS, TILE, OVERLAP, GUIDANCE, RESOLUTION, STRIDE, SEED = 25, 72, 6, 2.0, 576, 2, 42


def reference_frames(videos, output: Path):
    """One still per dancer (2 s into their first take) -> {dancer: jpg}."""
    stills = {}
    for video in videos:
        dancer = DANCER.search(video.name).group(1)
        target = output / "references" / f"{dancer}.jpg"
        if dancer not in stills:
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-ss", "2", "-i", str(video),
                                "-frames:v", "1", "-q:v", "2", str(target)], check=True)
            stills[dancer] = target
    return stills


def pairs(videos, stills):
    dancers = sorted(stills)
    for video in videos:
        own = DANCER.search(video.name).group(1)
        yield video, stills[dancers[(dancers.index(own) + 1) % len(dancers)]]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimicmotion", required=True)
    parser.add_argument("--videos", required=True)
    parser.add_argument("--camera", default="01")
    parser.add_argument("--output", required=True)
    parser.add_argument("--shard", default="0/1", help="i/n: this worker takes every n-th view")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    videos = sorted(Path(args.videos).resolve().glob(f"*_c{args.camera}_*.mp4"))
    stills = reference_frames(videos, output)
    jobs = list(pairs(videos, stills))[:args.limit]
    (output / "videos.csv").write_text("dance_id,video_url,video_path\n" + "".join(
        f"{v.stem},,videos/{v.name}\n" for v, _ in jobs))
    (output / "pairs.csv").write_text("video_id,reference\n" + "".join(f"{v.stem},{r.stem}\n" for v, r in jobs))
    index, count = map(int, args.shard.split("/"))
    jobs = jobs[index::count]

    # MimicMotion resolves its weights and modules relative to its checkout.
    os.chdir(args.mimicmotion)
    sys.path.insert(0, os.getcwd())
    import torch
    from omegaconf import OmegaConf
    from mimicmotion.utils.geglu_patch import patch_geglu_inplace
    patch_geglu_inplace()
    from inference import preprocess, run_pipeline
    from mimicmotion.utils.loader import create_pipeline
    from mimicmotion.utils.utils import save_to_mp4

    torch.set_default_dtype(torch.float16)
    device = torch.device("cuda")
    config = OmegaConf.create({"base_model_path": "models/svd-xt", "ckpt_path": "models/MimicMotion_1-1.pth"})
    # The loader passes several names to torch's safe_globals, which takes one list;
    # without the attribute it falls back to a plain weights_only load.
    if hasattr(torch.serialization, "safe_globals"):
        delattr(torch.serialization, "safe_globals")
    pipeline = create_pipeline(config, device)
    task = OmegaConf.create({"num_frames": TILE, "frames_overlap": OVERLAP, "num_inference_steps": STEPS,
                             "noise_aug_strength": 0, "guidance_scale": GUIDANCE, "seed": SEED})
    (output / "videos").mkdir(parents=True, exist_ok=True)
    for video, still in jobs:
        target = output / "videos" / video.name
        if target.exists():
            continue
        with torch.no_grad():
            pose, image = preprocess(str(video), str(still), resolution=RESOLUTION, sample_stride=STRIDE)
            frames = run_pipeline(pipeline, image, pose, device, task)
        raw = output / "raw" / video.name
        raw.parent.mkdir(parents=True, exist_ok=True)
        save_to_mp4(frames, str(raw), fps=30 // STRIDE)
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(raw), "-r", "30",
                        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(target)], check=True)
        print(video.stem, still.stem, len(frames), flush=True)


if __name__ == "__main__":
    main()
