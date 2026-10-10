"""Edits a copier can make to hide where a generated video's motion came from.

Each attack re-encodes every video of a directory with one ffmpeg edit; the
results go through CoMotion and provenance.py like the unedited copies.

* mirror     horizontal flip (left and right limbs swap)
* faster     played 1.2x faster, 30 fps kept
* slower     played 1.2x slower, 30 fps kept
* trim       first second cut
* zoom       centre crop to 75 %, scaled back
* lowq       half resolution and heavy compression, scaled back

    python -m motionprov.cv801.attacks --videos GEN/videos --output OUT
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

ATTACKS = {
    "mirror": ["-vf", "hflip"],
    "faster": ["-vf", "setpts=PTS/1.2", "-r", "30"],
    "slower": ["-vf", "setpts=PTS*1.2", "-r", "30"],
    "trim": ["-ss", "1"],
    "zoom": ["-vf", "crop=trunc(iw*0.375)*2:trunc(ih*0.375)*2,scale=trunc(iw/0.75/2)*2:trunc(ih/0.75/2)*2"],
    "lowq": ["-vf", "scale=trunc(iw/4)*2:trunc(ih/4)*2,scale=trunc(iw*2/2)*2:trunc(ih*2/2)*2", "-crf", "38"],
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    videos = sorted(Path(args.videos).glob("*.mp4"))
    for name, edit in ATTACKS.items():
        out = Path(args.output) / name
        (out / "videos").mkdir(parents=True, exist_ok=True)
        rows = ["dance_id,video_url,video_path"]
        for video in videos:
            target = out / "videos" / video.name
            if not target.exists():
                subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(video), *edit, "-an",
                                "-c:v", "libx264", "-pix_fmt", "yuv420p", *(["-crf", "16"] if "-crf" not in edit else []),
                                str(target)], check=True)
            rows.append(f"{video.stem},,{target}")
        (out / "videos.csv").write_text("\n".join(rows) + "\n")
        print(name, len(videos))


if __name__ == "__main__":
    main()
