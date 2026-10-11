"""Tables 2 and 3 and Fig. 11 on AIST++ (confirmation split) and TikTok (zero-shot), plus the stand's bundle.

Every system is scored by one protocol (protocol.py). Rejection thresholds are
calibrated on the AIST++ validation split at 5 % false acceptance and kept
fixed for both test corpora.

    python -m motionprov.choreomatch.evaluate --aist aist.pkl --tiktok tiktok.pkl --model MODEL_DIR --output OUT
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from motionprov.choreomatch import baselines, protocol
from motionprov.choreomatch.cre import RVQ, Index
from motionprov.choreomatch.model import Encoder
from motionprov.choreomatch.train import encode, retime

# Our extension of the CRE: the query is also played at these tempos and the best one wins (AIST++
# performs one basic choreography to music of 80-130 BPM; a lag alone cannot align such pairs).
TEMPOS = (0.7, 0.8, 0.9, 1.0, 1.12, 1.25, 1.4)


def at_tempo(joints, mask, rate):
    """Play a query `rate` times faster: resample joints linearly and the mask by nearest frame."""
    if rate == 1.0:
        return joints, mask
    n = max(20, int(round(len(joints) / rate)))
    j, _ = retime(joints, np.ones(len(joints), bool), n)
    return j.astype(np.float32), mask[np.round(np.linspace(0, len(joints) - 1, n)).astype(int)]


class ChoreoMatch:
    """Encodes every query variant once; scores through the CRE in several modes."""

    def __init__(self, model, books, device):
        self.model, self.device = model, device
        self.rvq = RVQ([b.to(device) for b in books["residual"]])
        self.single = RVQ([b.to(device) for b in books["single"]])
        self.cache = {}

    @torch.no_grad()
    def view(self, rec, condition, kind, use_query=True, rate=1.0):
        key = (rec["id"], condition, kind, use_query, rate)
        if key not in self.cache:
            joints, mask = at_tempo(*protocol.query_view(rec, condition, kind), rate)
            z, obs = encode(self.model, torch.from_numpy(joints[None]).to(self.device),
                            torch.from_numpy(mask[None]).to(self.device), use_query)
            z = F.normalize(z[0], dim=-1)
            codes, recon = self.rvq.encode(z)
            single_codes, single = self.single.encode(z)
            self.cache[key] = {"vectors": z, "observed": obs[0], "codes": codes, "recon": recon,
                               "single": single, "single_codes": single_codes}
        return self.cache[key]

    def index(self, records, mode, use_query=True):
        field = {"continuous": "vectors", "codes": "recon", "equality": "codes", "single": "single"}[mode]
        items = []
        for r in records:
            v = self.view(r, "clean", "whole", use_query)
            items.append({("codes" if mode == "equality" else "vectors"): v[field], "observed": v["observed"]})
        return Index(items, self.device, "equality" if mode == "equality" else "geometry")

    def system(self, records, mode, use_query=True, tempos=TEMPOS):
        index = self.index(records, mode, use_query)
        field = {"continuous": "vectors", "codes": "recon", "equality": "codes", "single": "single"}[mode]

        def score(q, condition, kind):
            best = None
            for rate in tempos:
                v = self.view(q, condition, kind, use_query, rate)
                s = index.score(v[field], v["observed"])[0].cpu().numpy()
                best = s if best is None else np.maximum(best, s)
            return best
        return score


def view_system(fn):
    """Wrap fn(joints, mask, kind) -> scores into a protocol system."""
    def score(q, condition, kind):
        joints, mask = protocol.query_view(q, condition, kind)
        return fn(joints, mask, kind)
    return score


def pq_system(pq, records, device):
    items = []
    for r in records:
        recon, obs = pq.encode(r["joints"], np.repeat(r["observed"][:, None], 23, 1))
        items.append({"vectors": recon, "observed": obs})
    index = Index(items, device)
    return view_system(lambda j, m, kind: index.score(*pq.encode(j, m))[0].cpu().numpy())


def tokenizer_system(tok, records, device, field):
    refs = []
    for r in records:
        hist, mean, _ = baselines.tokenizer_vectors(tok, r["joints"], np.repeat(r["observed"][:, None], 23, 1), device)
        refs.append(hist if field == "codes" else mean)
    refs = torch.stack(refs)

    def fn(j, m, kind):
        hist, mean, _ = baselines.tokenizer_vectors(tok, j, m, device)
        return (refs @ (hist if field == "codes" else mean)).cpu().numpy()
    return view_system(fn)


def two_stage(first, second, k=10):
    """Shortlist the top k references by `first`, re-rank them by `second`; the rest keep `first`'s order."""
    def score(q, condition, kind):
        a = standardised(first(q, condition, kind))
        b = standardised(second(q, condition, kind))
        out = a - 1e3
        top = np.argsort(-a)[:k]
        top = top[np.isfinite(a[top])]
        out[top] = np.where(np.isfinite(b[top]), b[top], -100.0) + 1e3
        return out
    return score


def standardised(s):
    finite = np.isfinite(s)
    out = np.full(len(s), -np.inf)
    out[finite] = (s[finite] - s[finite].mean()) / (s[finite].std() + 1e-9)
    return out


def fused(a, b, weight=0.5):
    """Our extension: sum of per-query standardised scores of two systems."""
    return lambda q, condition, kind: standardised(a(q, condition, kind)) + weight * standardised(b(q, condition, kind))


def headline(summary):
    """AIST++ number of Table 2: mean mAP over the three conditions, whole recordings; MRR alike."""
    whole = [summary[f"{c}/whole"] for c in protocol.CONDITIONS]
    return {"mAP": float(np.mean([w["mAP"] for w in whole])), "MRR": float(np.mean([w["MRR"] for w in whole])),
            "chance": float(np.mean([w["chance"] for w in whole]))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aist", required=True)
    parser.add_argument("--tiktok", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer-steps", type=int, default=4000)
    parser.add_argument("--limit", type=int, help="smoke test: keep this many recordings per corpus")
    parser.add_argument("--only", help="system names separated by |: score just these, merge into OUTPUT/results.json")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    only = set(args.only.split("|")) if args.only else None
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    aist = pickle.load(open(args.aist, "rb"))
    corpora = {"aist": [r for r in aist if r["split"] == "confirmation"],
               "tiktok": [r for r in pickle.load(open(args.tiktok, "rb")) if r["observed"].sum() >= 60]}
    train = [r for r in aist if r["split"] == "train"]
    val = [r for r in aist if r["split"] == "validation"]
    if args.limit:
        corpora = {k: v[:args.limit] for k, v in corpora.items()}
        val = val[:args.limit]

    model = Encoder().to(device)
    model.load_state_dict(torch.load(Path(args.model) / "model.pt", map_location=device))
    model.eval()
    cm = ChoreoMatch(model, torch.load(Path(args.model) / "codebooks.pt"), device)
    trained = {}

    def fit(name):
        """Baseline models are trained on the AIST++ train split once, when a requested system needs them."""
        if name not in trained:
            trained[name] = {"pq": lambda: baselines.PQ(train, device),
                             "dpq": lambda: baselines.DPQ(train, device),
                             "tok": lambda: baselines.train_tokenizer(train, device, args.tokenizer_steps),
                             "tok_id": lambda: baselines.train_tokenizer(train, device, args.tokenizer_steps,
                                                                         identity=True)}[name]()
        return trained[name]

    def systems(records):
        """name -> builder of the scoring function; builders share indexes through `built`."""
        built = {}

        def once(key, make):
            if key not in built:
                built[key] = make()
            return built[key]

        dtw = lambda: once("dtw", lambda: view_system(baselines.DTWSystem(records, device)))
        dtw_masked = lambda: once("dtw_masked", lambda: view_system(baselines.DTWSystem(records, device, masked=True)))
        continuous = lambda: once("continuous", lambda: cm.system(records, "continuous"))
        pq = lambda: once("pq", lambda: pq_system(fit("pq"), records, device))
        dpq = lambda: once("dpq", lambda: pq_system(fit("dpq"), records, device))
        return {
            "ChoreoMatch continuous + skeleton DTW (fusion)": lambda: fused(continuous(), dtw_masked()),
            "ChoreoMatch, continuous": continuous,
            "ChoreoMatch, codes (CRE)": lambda: cm.system(records, "codes"),
            "ChoreoMatch, codes by ID equality": lambda: cm.system(records, "equality"),
            "ChoreoMatch, single codebook": lambda: cm.system(records, "single"),
            "ChoreoMatch anchor only, continuous": lambda: cm.system(records, "continuous", use_query=False),
            "ChoreoMatch, continuous, lag only": lambda: cm.system(records, "continuous", tempos=(1.0,)),
            "ChoreoMatch, codes, lag only": lambda: cm.system(records, "codes", tempos=(1.0,)),
            "DTW, joint trajectories": dtw,
            "DTW, banded": lambda: view_system(baselines.DTWSystem(records, device, banded=True)),
            "Skeleton DTW, masked": dtw_masked,
            "PQ words over raw pose (8x256)": pq,
            "DPQ, retrieval-trained codebooks (8x256)": dpq,
            "Two-stage: PQ shortlist + DTW": lambda: two_stage(pq(), dtw()),
            "Two-stage: DPQ shortlist + DTW": lambda: two_stage(dpq(), dtw()),
            "Two-stage: ordered shortlist + DTW": lambda: two_stage(continuous(), dtw()),
            "Reconstruction tokenizer, continuous": lambda: tokenizer_system(fit("tok"), records, device, "continuous"),
            "Reconstruction tokenizer, codes": lambda: tokenizer_system(fit("tok"), records, device, "codes"),
            "Reconstruction tokenizer + identity, codes": lambda: tokenizer_system(fit("tok_id"), records, device,
                                                                                   "codes"),
        }

    pretrained_names = ("MotionBERT, mean-pooled", "PQ words over MotionBERT", "TMR motion encoder", "TMR, window mean",
                        "PQ words over TMR latents", "MoMask, code histograms", "MoMask, mean embeddings",
                        "MoMask, DTW over embeddings")

    def with_pretrained(records, builders):
        """The authors' pretrained encoders (pretrained.py) run only when asked for: they need their repositories."""
        if only and only & set(pretrained_names):
            from motionprov.choreomatch import pretrained
            fns = pretrained.systems(records, train, device)
            builders |= {n: (lambda fn=fns[n]: view_system(fn)) for n in pretrained_names}
        return builders

    def wanted(builders):
        return {n: b for n, b in builders.items() if only is None or n in only}

    calibrated = ("ChoreoMatch, continuous", "ChoreoMatch, codes (CRE)", "Skeleton DTW, masked")
    thresholds = {}
    for name, build in wanted(with_pretrained(val, systems(val))).items():
        if name in calibrated:
            thresholds[name] = protocol.calibrate(protocol.evaluate(build(), val, val))
    print("thresholds", json.dumps(thresholds), flush=True)

    results, rows_all = {}, {}
    for corpus, records in corpora.items():
        results[corpus], rows_all[corpus] = {}, {}
        for name, build in wanted(with_pretrained(records, systems(records))).items():
            rows = protocol.evaluate(build(), records, records)
            s = protocol.summary(rows, thresholds.get(name))
            results[corpus][name] = {"headline": headline(s), "variants": s}
            if name in thresholds:
                results[corpus][name]["curve"] = protocol.acceptance_curve(rows)
            rows_all[corpus][name] = rows
            print(corpus, name, json.dumps(results[corpus][name]["headline"]), flush=True)

    if only:                                   # merge the requested rows into the full run and stop
        path = output / "results.json"
        full = json.loads(path.read_text())
        full["thresholds"].update(thresholds)
        for corpus in results:
            full["results"][corpus].update(results[corpus])
        path.write_text(json.dumps(full, indent=1))
        path = output / "queries.json"
        per_query = json.loads(path.read_text())
        for corpus, rows in rows_all.items():
            per_query[corpus].update({n: {f"{k[0]}/{k[1]}": v for k, v in r.items()} for n, r in rows.items()})
        path.write_text(json.dumps(per_query))
        print("merged", sorted(only))
        return

    ablations = {}
    for corpus, rows in rows_all.items():
        pairs = {
            "reconstruction tokens -> ChoreoMatch codes": ("Reconstruction tokenizer, codes", "ChoreoMatch, codes (CRE)", None),
            "+ query branch, part hidden": ("ChoreoMatch anchor only, continuous", "ChoreoMatch, continuous", "part"),
            "+ query branch, clean": ("ChoreoMatch anchor only, continuous", "ChoreoMatch, continuous", "clean"),
            "ID equality -> reconstruction cosine, part hidden": ("ChoreoMatch, codes by ID equality", "ChoreoMatch, codes (CRE)", "part"),
            "single codebook -> residual, part hidden": ("ChoreoMatch, single codebook", "ChoreoMatch, codes (CRE)", "part"),
            "skeleton DTW (masked) -> ChoreoMatch codes": ("Skeleton DTW, masked", "ChoreoMatch, codes (CRE)", None),
            "+ tempo search, codes": ("ChoreoMatch, codes, lag only", "ChoreoMatch, codes (CRE)", None),
        }
        ablations[corpus] = {}
        for label, (a, b, cond) in pairs.items():
            conds = [cond] if cond else list(protocol.CONDITIONS)
            ra = [x | {"id": f"{c}|{x['id']}"} for c in conds for x in rows[a][(c, "whole")]]   # one pair per condition
            rb = [x | {"id": f"{c}|{x['id']}"} for c in conds for x in rows[b][(c, "whole")]]
            ablations[corpus][label] = {"from": float(np.mean([x["ap"] for x in ra])),
                                        "to": float(np.mean([x["ap"] for x in rb]))} | protocol.paired_difference(ra, rb)
    (output / "results.json").write_text(json.dumps({"thresholds": thresholds, "results": results,
                                                     "ablations": ablations}, indent=1))
    per_query = {c: {n: {f"{k[0]}/{k[1]}": v for k, v in r.items()} for n, r in rows.items()} for c, rows in rows_all.items()}
    (output / "queries.json").write_text(json.dumps(per_query))

    # Stand bundle: skeletons, ChoreoMatch descriptors and codes of every query variant.
    for corpus, records in corpora.items():
        bundle = {"records": [{k: r[k] for k in ("id", "label", "source")} | {"frames": len(r["joints"])} for r in records],
                  "thresholds": thresholds, "views": {}}
        bundle["tempos"] = list(TEMPOS)
        for r in records:
            for c in protocol.CONDITIONS:
                for kind in protocol.KINDS:
                    for rate in TEMPOS:
                        v = cm.view(r, c, kind, rate=rate)
                        joints, mask = at_tempo(*protocol.query_view(r, c, kind), rate)
                        bundle["views"][f"{r['id']}|{c}|{kind}|{rate}"] = {
                            "vectors": v["vectors"].cpu().numpy().astype(np.float16),
                            "recon": v["recon"].cpu().numpy().astype(np.float16),
                            "codes": v["codes"].cpu().numpy().astype(np.uint8),
                            "observed": v["observed"].cpu().numpy().astype(np.float16),
                            "joints": joints.astype(np.float16), "mask": mask,
                            "start": protocol.fragment_start(r, kind)}
        pickle.dump(bundle, open(output / f"stand_{corpus}.pkl", "wb"))
    print("done")


if __name__ == "__main__":
    main()
