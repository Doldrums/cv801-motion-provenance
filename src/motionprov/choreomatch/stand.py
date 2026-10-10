"""ChoreoMatch stand: search a query variant against an index and see why it matched.

Serves 127.0.0.1:8090 (view through an ssh tunnel). Reads the bundles and
results written by evaluate.py; scoring runs live through the CRE on CPU.

    python -m motionprov.choreomatch.stand --results EVAL_OUT [--port 8090]
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from motionprov.choreomatch.cre import Index, OBSERVED
from motionprov.choreomatch.model import C

FIELDS = {"continuous": "vectors", "codes": "recon", "equality": "codes"}
PLAYBACK_STRIDE = 2                 # 30 -> 15 fps for the browser


class Corpus:
    def __init__(self, bundle):
        self.records = bundle["records"]
        self.by_id = {r["id"]: r for r in self.records}
        self.views = bundle["views"]
        self.thresholds = bundle["thresholds"]
        self.tempos = bundle.get("tempos", [1.0])
        self.indexes = {}
        for mode, field in FIELDS.items():
            key, dtype = ("codes", np.int64) if mode == "equality" else ("vectors", np.float32)
            items = [{key: torch.from_numpy(self.view(r["id"])[field].astype(dtype)),
                      "observed": torch.from_numpy(self.view(r["id"])["observed"].astype(np.float32))}
                     for r in self.records]
            self.indexes[mode] = Index(items, "cpu", "equality" if mode == "equality" else "geometry")

    def view(self, rid, condition="clean", kind="whole", rate=1.0):
        return self.views[f"{rid}|{condition}|{kind}|{rate}"]

    def query(self, rid, condition, kind, mode):
        """Best score over lags and query tempos -> scores, lags, tempos (one per reference)."""
        field = FIELDS[mode]
        best = None
        for rate in self.tempos:
            v = self.view(rid, condition, kind, rate)
            q = torch.from_numpy(v[field].astype(np.int64 if mode == "equality" else np.float32))
            s, lag = (t.numpy() for t in self.indexes[mode].score(q, torch.from_numpy(v["observed"].astype(np.float32))))
            if best is None:
                best = [s, lag, np.full(len(s), rate)]
            else:
                better = s > best[0]
                best = [np.where(better, s, best[0]), np.where(better, lag, best[1]), np.where(better, rate, best[2])]
        return best


def votes(corpus, rid, ref, condition, kind, mode, lag, rate):
    q, x = corpus.view(rid, condition, kind, rate), corpus.view(ref)
    field = FIELDS[mode]
    out = []
    for i in range(len(q["observed"])):
        j = i + lag
        if not 0 <= j < len(x["observed"]):
            continue
        if mode == "equality":
            sim = float((q["codes"][i] == x["codes"][j]).all())
        else:
            a = F.normalize(torch.from_numpy(q[field][i].astype(np.float32)), dim=0)
            b = F.normalize(torch.from_numpy(x[field][j].astype(np.float32)), dim=0)
            sim = float(a @ b)
        voted = bool(q["observed"][i] >= OBSERVED and x["observed"][j] >= OBSERVED)
        out.append({"q": i, "x": j, "sim": round(sim, 4), "voted": voted,
                    "q_code": q["codes"][i].tolist(), "x_code": x["codes"][j].tolist()})
    return out


def skeleton(v):
    j = v["joints"][::PLAYBACK_STRIDE].astype(np.float32)
    j = j - j[:, :1]                                      # pelvis-centred for display
    return {"joints": np.round(j, 3).tolist(), "mask": v["mask"][::PLAYBACK_STRIDE].tolist()}


def create_app(results: Path):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse
    corpora = {c: Corpus(pickle.load(open(results / f"stand_{c}.pkl", "rb")))
               for c in ("aist", "tiktok") if (results / f"stand_{c}.pkl").exists()}
    summary = json.loads((results / "results.json").read_text())
    app = FastAPI(title="ChoreoMatch stand")

    def corpus(name):
        if name not in corpora:
            raise HTTPException(404, f"unknown corpus {name}")
        return corpora[name]

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (Path(__file__).parent / "stand.html").read_text()

    @app.get("/api/results")
    def get_results():
        return summary

    @app.get("/api/records/{name}")
    def records(name: str):
        return corpus(name).records

    @app.get("/api/search/{name}")
    def search(name: str, id: str, condition: str = "clean", kind: str = "whole", mode: str = "codes", top: int = 15):
        cp = corpus(name)
        if id not in cp.by_id:
            raise HTTPException(404, id)
        scores, lags, rates = cp.query(id, condition, kind, mode)
        me = cp.by_id[id]
        other = np.array([r["source"] != me["source"] for r in cp.records])
        s = np.where(other, scores, -np.inf)
        order = np.argsort(-s, kind="stable")[:top]
        thr_key = {"continuous": "ChoreoMatch, continuous", "codes": "ChoreoMatch, codes (CRE)"}.get(mode)
        threshold = cp.thresholds.get(thr_key) if thr_key else None
        best = int(order[0])
        accepted = bool(np.isfinite(s[best]) and threshold is not None and s[best] >= threshold)
        relevant = [r["label"] == me["label"] and r["source"] != me["source"] for r in cp.records]
        return {"query": me, "threshold": threshold, "accepted": accepted,
                "prediction": cp.records[best]["label"] if accepted else "UNKNOWN",
                "n_relevant": int(sum(relevant)),
                "results": [{"id": cp.records[i]["id"], "label": cp.records[i]["label"],
                             "score": float(s[i]) if np.isfinite(s[i]) else None, "lag": int(lags[i]),
                             "tempo": float(rates[i]),
                             "relevant": relevant[i]} for i in order]}

    @app.get("/api/align/{name}")
    def align(name: str, id: str, ref: str, lag: int, condition: str = "clean", kind: str = "whole",
              mode: str = "codes", tempo: float = 1.0):
        cp = corpus(name)
        q, x = cp.view(id, condition, kind, tempo), cp.view(ref)
        return {"core": C, "stride": PLAYBACK_STRIDE, "lag": lag, "start": int(q["start"]), "tempo": tempo,
                "query": skeleton(q), "reference": skeleton(x),
                "votes": votes(cp, id, ref, condition, kind, mode, lag, tempo)}

    return app


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args(argv)
    import uvicorn
    uvicorn.run(create_app(Path(args.results)), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
