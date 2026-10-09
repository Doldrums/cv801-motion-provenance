"""Turn CoMotion outputs for AIST++ views into a PoseRecord cache.

Each view keeps its camera in the video ID; its source group and performance
are the camera-free take (``cAll``), so two views of one take are one source.
The records go through the same exporter that built the reference cache.

    python -m motionprov.cv801.prepare --split SPLIT.json \
        --comotion-dir WORK/comotion --output-dir WORK/pose_cache_comotion
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import tempfile
from pathlib import Path

from tools.data.export_local_pose_cache import main as export_cache

CAMERA = re.compile(r"_c(0[1-9])_")


def take_id(video_id: str) -> str:
    """``gBR_sBM_c01_d04_mBR0_ch03`` -> ``gBR_sBM_cAll_d04_mBR0_ch03``."""
    if not CAMERA.search(video_id):
        raise ValueError(f"{video_id}: no camera token")
    return CAMERA.sub("_cAll_", video_id, count=1)


def manifest_rows(split: dict, comotion_dir: Path, partition: str) -> list[dict]:
    takes = set(split["videos"][partition])
    rows = []
    for path in sorted(comotion_dir.glob("*.pt")):
        take = take_id(path.stem)
        if take in takes:
            rows.append({"video_id": path.stem, "choreography_id": split["labels"][take],
                         "source_group_id": take, "performance_id": take, "fps": "30"})
    if not rows:
        raise ValueError(f"No CoMotion outputs match the {partition} partition")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True)
    parser.add_argument("--comotion-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--partition", default="confirmation")
    args = parser.parse_args(argv)
    split = json.loads(Path(args.split).read_text())
    rows = manifest_rows(split, Path(args.comotion_dir), args.partition)
    with tempfile.NamedTemporaryFile("w", suffix=".csv", newline="", delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} views from {len({r['source_group_id'] for r in rows})} takes")
    export_cache(["--videos-csv", handle.name, "--comotion-dir", args.comotion_dir,
                  "--output-dir", args.output_dir, "--fps", "30"])


if __name__ == "__main__":
    main()
