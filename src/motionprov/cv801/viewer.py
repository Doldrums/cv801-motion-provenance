"""Inspection stand: one AIST++ take, its video, both skeletons and retrieval.

Reads only finished artefacts: the pose caches, the transcoded videos and the
diagnostic output. Besides the clean camera views, a view can come in
variants (``--variant``): the same take with a body part painted out, or
re-rendered with another dancer by a generator. A variant brings its own
videos and CoMotion cache; its per-part error is computed once at start-up.
For generated variants the stand also ranks the clean views of the same
camera against each copy, to show whether the copy points at its source.

    python -m motionprov.cv801.viewer --reference REF_CACHE --comotion COMOTION_CACHE \
        --videos WORK/videos --results OUT --split SPLIT.json \
        [--variant occluded_arms=OCC/arms/videos,OCC/arms/pose_cache ...] \
        [--plausibility PLAUS.json] [--provenance PROVENANCE.json]   # serves 127.0.0.1:8081
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from motionprov.core.apps.demo.pose_preview import preview_payload
from motionprov.core.datasets.local_pose_cache import load_pose_cache
from motionprov.cv801.diagnostic import knn_scores, pose_errors, skeleton
from motionprov.cv801.prepare import CAMERA, take_id

HERE = Path(__file__).resolve().parent
PLAYER = HERE.parents[1] / "src/motionprov.core/apps/demo/pose_player.js"
VIDEO_ID = re.compile(r"^[A-Za-z0-9_]+$")
CLEAN = "comotion"
GENERATED = ("mimicmotion", "wan")
TOP_K = 5


def views_of(cache):
    out = {}
    for video, record in cache.items():
        joints, frames = skeleton(record)
        if len(joints) > 1:
            out[video] = (joints, frames, record.metadata["choreography_id"], record.metadata["source_group_id"])
    return out


def attribution(clean_views, generated_views):
    """Generated view -> top clean views of the same camera by kNN score (nothing excluded)."""
    out = {}
    for video, (joints, *_) in generated_views.items():
        camera = CAMERA.search(video).group(1)
        gallery = sorted(v for v in clean_views if CAMERA.search(v).group(1) == camera)
        scores = knn_scores([clean_views[v][0] for v in gallery] + [joints])[-1, :-1]
        order = np.argsort(-scores, kind="stable")
        rank = int(np.flatnonzero(np.asarray(gallery)[order] == video)[0]) + 1 if video in gallery else None
        out[video] = {"source_rank": rank, "gallery": len(gallery),
                      "top": [{"id": gallery[i], "score": float(scores[i])} for i in order[:TOP_K]]}
    return out


def build_app(reference, comotion, videos: Path, results: Path, split: dict,
              variants: dict | None = None, plausibility: dict | None = None,
              provenance: dict | None = None) -> FastAPI:
    app = FastAPI(title="CV801 inspection stand")
    queries = json.loads((results / "queries.json").read_text())
    summary = json.loads((results / "summary.json").read_text())
    takes = sorted(reference)
    caches = {CLEAN: comotion} | {name: cache for name, (_, cache) in (variants or {}).items()}
    video_dirs = {CLEAN: videos} | {name: d for name, (d, _) in (variants or {}).items()}
    errors = {CLEAN: json.loads((results / "pose_errors.json").read_text())}
    reference_joints = {v: skeleton(r) for v, r in reference.items()}
    clean_views = views_of(comotion)
    attributions = {}
    for name, cache in caches.items():
        if name == CLEAN:
            continue
        extra = views_of(cache)
        errors[name] = pose_errors(extra, reference_joints)
        if name in GENERATED:
            attributions.update(attribution(clean_views, extra))

    def checked(video_id):
        if not VIDEO_ID.match(video_id):
            raise HTTPException(404)
        return video_id

    def variant(name):
        if name not in caches:
            raise HTTPException(404)
        return name

    @app.get("/")
    def index():
        return FileResponse(HERE / "viewer.html")

    @app.get("/assets/pose-player.js")
    def player():
        return FileResponse(PLAYER, media_type="text/javascript")

    @app.get("/api/summary")
    def get_summary():
        return summary

    @app.get("/api/takes")
    def get_takes():
        views = {}
        for video in comotion:
            views.setdefault(take_id(video), []).append(video)
        return [{"take": take, "label": split["labels"][take], "views": sorted(views.get(take, [])),
                 "variants": {v: sorted(name for name, cache in caches.items() if name != CLEAN and v in cache)
                              for v in views.get(take, [])},
                 "margin": queries["reference"]["knn"].get(take, {}).get("margin")} for take in takes]

    @app.get("/api/pose/{source}/{video_id}")
    def get_pose(source: str, video_id: str):
        cache = reference if source == "reference" else caches.get(source)
        record = cache.get(checked(video_id)) if cache else None
        if record is None:
            raise HTTPException(404)
        # CoMotion is in camera coordinates, where image y points down.
        return preview_payload(record, display_up_axis="y" if source == "reference" else "-y")

    @app.get("/api/errors/{name}/{video_id}")
    def get_errors(name: str, video_id: str):
        row = errors[variant(name)].get(checked(video_id))
        if row is None:
            raise HTTPException(404)
        return row

    @app.get("/api/query/{condition}/{method}/{video_id}")
    def get_query(condition: str, method: str, video_id: str):
        row = queries.get(condition, {}).get(method, {}).get(checked(video_id))
        if row is None:
            raise HTTPException(404)
        return row

    @app.get("/api/attribution/{video_id}")
    def get_attribution(video_id: str):
        row = attributions.get(checked(video_id))
        if row is None:
            raise HTTPException(404)
        return row

    @app.get("/api/plausibility/{video_id}")
    def get_plausibility(video_id: str):
        views = (plausibility or {}).get("views", {})
        return {name: rows[checked(video_id)] for name, rows in views.items() if video_id in rows}

    @app.get("/api/provenance")
    def get_provenance_summary():
        return {name: {method: r["summary"] for method, r in methods.items()}
                for name, methods in (provenance or {}).items()}

    @app.get("/api/provenance/{video_id}")
    def get_provenance(video_id: str):
        checked(video_id)
        return {name: {method: r["sources"][video_id] for method, r in methods.items() if video_id in r["sources"]}
                for name, methods in (provenance or {}).items()}

    @app.get("/video/{name}/{video_id}.mp4")
    def get_video(name: str, video_id: str):
        path = video_dirs[variant(name)] / f"{checked(video_id)}.mp4"
        if not CAMERA.search(video_id) or not path.is_file():
            raise HTTPException(404)
        return FileResponse(path, media_type="video/mp4")

    return app


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "comotion", "videos", "results", "split"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--variant", action="append", default=[], metavar="NAME=VIDEOS,CACHE")
    parser.add_argument("--plausibility")
    parser.add_argument("--provenance")
    parser.add_argument("--port", type=int, default=8081)
    args = parser.parse_args(argv)
    split = json.loads(Path(args.split).read_text())
    takes = set(split["videos"]["confirmation"])
    reference = {v: r for v, r in load_pose_cache(args.reference).items() if v in takes}
    variants = {}
    for spec in args.variant:
        name, paths = spec.split("=", 1)
        video_dir, cache = paths.split(",", 1)
        variants[name] = (Path(video_dir), load_pose_cache(cache))
    plausibility = json.loads(Path(args.plausibility).read_text()) if args.plausibility else None
    provenance = json.loads(Path(args.provenance).read_text()) if args.provenance else None
    app = build_app(reference, load_pose_cache(args.comotion), Path(args.videos), Path(args.results), split,
                    variants, plausibility, provenance)
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
