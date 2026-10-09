"""Inspection stand for Harmony4D: one camera view, both people, CoMotion against the fit.

Each person's CoMotion detection is aligned onto the multi-view fit frame by
frame (the same oracle association as `harmony4d evaluate`) and drawn over it,
so a broken arm in a clinch is visible next to the video. Views are listed
from the evaluation output; poses are decoded on first request and cached.

    python -m motionprov.cv801.harmony_viewer --work WORK --results OUT   # serves 127.0.0.1:8082
"""

from __future__ import annotations

import argparse
import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from motionprov.core.models.local_extensions import PART_JOINT_IDS
from motionprov.cv801.harmony4d import FPS, associate, closest_pair, comotion_joints, ground_truth

HERE = Path(__file__).resolve().parent
VIEW_ID = re.compile(r"^[A-Za-z0-9_]+$")
NECK, PELVIS = 12, 0


def upright(people):
    """Rotation taking the mean pelvis-to-neck direction to +y."""
    up = np.concatenate([p[:, NECK] - p[:, PELVIS] for p in people.values()]).mean(0)
    up /= np.linalg.norm(up)
    y = np.array([0.0, 1.0, 0.0])
    v, c = np.cross(up, y), float(up @ y)
    if np.linalg.norm(v) < 1e-8:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k / (1 + c)


def rounded(a):
    return np.round(a * 1000).astype(int).tolist()   # millimetres keep the payload small


def view_payload(work: Path, view: str, decoder):
    group, seq, cam = view.split("__")
    seq_dir = work / "raw" / group / seq
    images = sorted((seq_dir / "exo" / cam / "images").glob("*.jpg"))
    numbers, people = ground_truth(seq_dir)
    frame_idx, _, joints = comotion_joints(work / "comotion" / f"{view}.pt", decoder)
    rot = upright(people)
    closeness = closest_pair(people)
    names = sorted(people)
    frames, close = [], []
    per = {n: {"gt": [], "est": [], "error": {p: [] for p in PART_JOINT_IDS}} for n in names}
    for k, g, assigned in associate(frame_idx, joints, [int(p.stem) for p in images], numbers, people):
        frames.append(k)
        close.append(round(float(closeness[g]), 3))
        for n in names:
            per[n]["gt"].append(rounded(people[n][g] @ rot.T))
            err, est = assigned.get(n, (None, None))
            per[n]["est"].append(None if est is None else rounded(est @ rot.T))
            for part, ids in PART_JOINT_IDS.items():
                per[n]["error"][part].append(None if err is None else round(float(err[list(ids)].mean()), 4))
    return {"view": view, "fps": FPS, "frames": frames, "closeness": close, "people": per}


def build_app(work: Path, results: Path) -> FastAPI:
    from motionprov.core.comotion.utils.smpl_kinematics import SMPLKinematics
    app = FastAPI(title="CV801 Harmony4D stand")
    evaluated = json.loads((results / "harmony4d.json").read_text())
    summary = json.loads((results / "summary.json").read_text())
    decoder = SMPLKinematics()

    def checked(view):
        if not VIEW_ID.match(view) or view not in evaluated:
            raise HTTPException(404)
        return view

    @lru_cache(maxsize=16)
    def payload(view):
        return view_payload(work, view, decoder)

    @app.get("/")
    def index():
        return FileResponse(HERE / "harmony_viewer.html")

    @app.get("/api/summary")
    def get_summary():
        return summary

    @app.get("/api/views")
    def get_views():
        rows = []
        for view, result in evaluated.items():
            people = result["people"].values()
            limbs = [p["error_by_part"][part] for p in people for part in PART_JOINT_IDS
                     if part != "torso" and p["error_by_part"][part] is not None]
            contact = sum(p["by_closeness"][0]["frames"] for p in people) / max(1, sum(p["frames"] for p in people))
            rows.append({"view": view, "activity": view.split("__")[0].split("_", 1)[1],
                         "limb_error": float(np.mean(limbs)) if limbs else None,
                         "contact_share": contact,
                         "missed": sum(p["missed"] for p in people)})
        return sorted(rows, key=lambda r: -(r["limb_error"] or 0))

    @app.get("/api/view/{view}")
    def get_view(view: str):
        return payload(checked(view))

    @app.get("/video/{view}.mp4")
    def get_video(view: str):
        path = work / "videos" / f"{checked(view)}.mp4"
        if not path.is_file():
            raise HTTPException(404)
        return FileResponse(path, media_type="video/mp4")

    return app


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--port", type=int, default=8082)
    args = parser.parse_args(argv)
    import uvicorn
    uvicorn.run(build_app(Path(args.work), Path(args.results)), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
