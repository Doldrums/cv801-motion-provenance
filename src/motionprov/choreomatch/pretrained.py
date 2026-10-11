"""Pretrained motion encoders of Table 2: TMR, MoMask and MotionBERT, with the authors' weights.

The code and weights come from the authors' repositories, cloned under
PRETRAINED (Mathux/TMR, EricGuo5513/momask-codes, Walter0807/MotionBERT); every
checkpoint is read with weights_only=True. Our skeletons (SMPL24, metres, y up,
30 fps) are converted to each model's input:

* TMR and MoMask read Guo et al.'s 263 HumanML3D features: the 22 body joints
  resampled to 20 fps, through TMR's joints_to_guofeats, normalised with each
  model's own statistics.
* MotionBERT reads 2D H36M-17 keypoints: SMPL joints mapped to H36M, turned to
  face the camera, projected orthographically and scaled to [-1, 1] over the
  clip; hidden joints get confidence 0. Clips longer than 243 frames are cut.

Neither HumanML3D model can be told that a joint is missing. Hidden joints are
therefore filled in: interpolated in time where the joint is seen elsewhere,
otherwise held at their parent plus the mean offset of the AIST++ train split.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from motionprov.choreomatch.baselines import dtw as baselines_dtw
from motionprov.choreomatch.cre import Index, kmeans, pad
from motionprov.choreomatch.train import retime

ROOT = Path(os.environ.get("PRETRAINED", "pretrained"))
PARENT = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
WINDOW, HOP = 80, 40                       # TMR windows at 20 fps: 4 s, half overlap
# H36M-17 from SMPL: pelvis, R hip/knee/ankle, L hip/knee/ankle, spine, thorax, nose, head (extrapolated),
# L shoulder/elbow/wrist, R shoulder/elbow/wrist.
H36M = [0, 2, 5, 8, 1, 4, 7, 6, 12, 15, -1, 16, 18, 20, 17, 19, 21]


def _package(name, path):
    """Import a directory as a package under a private name (TMR's `src` would clash with ours)."""
    spec = importlib.util.spec_from_file_location(name, path / "__init__.py", submodule_search_locations=[str(path)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def rest_offsets(train):
    """Mean child - parent offset of every joint over the observed train frames (y up, metres)."""
    j = np.concatenate([r["joints"][r["observed"]] for r in train])
    return np.stack([np.zeros(3)] + [(j[:, k] - j[:, PARENT[k]]).mean(0) for k in range(1, 24)]).astype(np.float32)


def fill(joints, mask, rest):
    """Joints (T, 24, 3) with a (T, 23) mask -> every joint placed; the pelvis follows frames with any joint."""
    j = joints.astype(np.float32).copy()
    seen = np.concatenate([mask.any(1, keepdims=True), mask], 1)
    t = np.arange(len(j))
    for k in range(24):
        s = seen[:, k]
        if s.all():
            continue
        if s.any():
            for d in range(3):
                j[~s, k, d] = np.interp(t[~s], t[s], j[s, k, d])
        else:
            j[:, k] = j[:, PARENT[k]] + rest[k]
    return j


class HumanML3D:
    """Guo et al.'s 263 features from our joints."""

    def __init__(self, train):
        guofeats = _package("tmr_guofeats", ROOT / "TMR/src/guofeats")
        self.convert = guofeats.joints_to_guofeats
        self.rest = rest_offsets(train)

    def __call__(self, joints, mask):
        j = fill(joints, mask, self.rest)
        j, _ = retime(j, np.ones(len(j), bool), max(8, int(round(len(j) * 20 / 30))))
        return np.asarray(self.convert(j[:, :22]), np.float32)          # (T20 - 1, 263)


class TMR:
    def __init__(self, features, device):
        actor = _module("tmr_actor", ROOT / "TMR/src/model/actor.py")
        base = ROOT / "TMR/models/tmr_humanml3d_guoh3dfeats"
        self.encoder = actor.ACTORStyleEncoder(nfeats=263, vae=True, latent_dim=256, ff_size=1024, num_layers=6,
                                               num_heads=4, dropout=0.1, activation="gelu")
        self.encoder.load_state_dict(torch.load(base / "last_weights/motion_encoder.pt", map_location="cpu",
                                                weights_only=True))
        self.encoder.to(device).eval()
        stats = ROOT / "TMR/stats/humanml3d/guoh3dfeats"
        self.mean = torch.load(stats / "mean.pt", weights_only=True).to(device)
        self.std = torch.load(stats / "std.pt", weights_only=True).to(device)
        self.features, self.device = features, device

    @torch.no_grad()
    def encode(self, x):
        """(B, T, 263) normalised features -> (B, 256) mean latents."""
        mask = torch.ones(x.shape[:2], dtype=torch.bool, device=self.device)
        return self.encoder({"x": x, "mask": mask})[:, 0]

    @torch.no_grad()
    def __call__(self, joints, mask):
        """-> whole-sequence latent, per-window latents (N, 256)."""
        x = (torch.from_numpy(self.features(joints, mask)).to(self.device) - self.mean) / (self.std + 1e-12)
        whole = F.normalize(self.encode(x[None])[0], dim=0)
        starts = list(range(0, max(1, len(x) - WINDOW + 1), HOP))
        windows = torch.stack([F.pad(x[s:s + WINDOW], (0, 0, 0, max(0, WINDOW - len(x[s:s + WINDOW]))))
                               for s in starts])
        return whole, F.normalize(self.encode(windows), dim=-1)


class MoMask:
    def __init__(self, features, device):
        sys.path.insert(0, str(ROOT / "momask-codes"))
        from models.vq.model import RVQVAE
        sys.path.pop(0)
        base = ROOT / "momask-codes/checkpoints/t2m/rvq_nq6_dc512_nc512_noshare_qdp0.2"
        args = SimpleNamespace(num_quantizers=6, shared_codebook=False, quantize_dropout_prob=0.2, mu=0.99)
        self.model = RVQVAE(args, 263, 512, 512, 512, down_t=2, stride_t=2, width=512, depth=3,
                            dilation_growth_rate=3, activation="relu", norm=None)
        state = torch.load(base / "model/net_best_fid.tar", map_location="cpu", weights_only=True)
        self.model.load_state_dict(state["vq_model" if "vq_model" in state else "net"])
        self.model.to(device).eval()
        self.mean = torch.from_numpy(np.load(base / "meta/mean.npy")).float().to(device)
        self.std = torch.from_numpy(np.load(base / "meta/std.npy")).float().to(device)
        self.features, self.device = features, device

    @torch.no_grad()
    def __call__(self, joints, mask):
        """-> latents (T/4, 512) before quantisation, base-layer codes (T/4,)."""
        x = (torch.from_numpy(self.features(joints, mask)).to(self.device) - self.mean) / self.std
        x = x[:max(4, len(x) // 4 * 4)]
        z = self.model.encoder(self.model.preprocess(x[None]))                 # (1, 512, T/4)
        codes, _ = self.model.quantizer.quantize(z, return_latent=True)        # (1, T/4, 6)
        return z[0].T, codes[0, :, 0]


class MotionBERT:
    def __init__(self, device):
        sys.path.insert(0, str(ROOT / "MotionBERT"))
        from lib.model.DSTformer import DSTformer
        sys.path.pop(0)
        self.model = DSTformer(dim_in=3, dim_out=3, dim_feat=512, dim_rep=512, depth=5, num_heads=8, mlp_ratio=2,
                               num_joints=17, maxlen=243, att_fuse=True)
        state = torch.load(ROOT / "MotionBERT/checkpoint/pretrain/MB_release/latest_epoch.bin", map_location="cpu",
                           weights_only=True)["model_pos"]
        self.model.load_state_dict({k.removeprefix("module."): v for k, v in state.items()})
        self.model.to(device).eval()
        self.device = device

    @staticmethod
    def keypoints(joints, mask):
        """(T, 24, 3), (T, 23) -> (T, 17, 3): image x, image y in [-1, 1], confidence."""
        seen = np.concatenate([mask.any(1, keepdims=True), mask], 1)
        lr = (joints[:, 1] - joints[:, 2]).mean(0)
        yaw = np.arctan2(lr[2], lr[0])                                         # turn the hips onto +x
        c, s = np.cos(yaw), np.sin(yaw)
        j = joints @ np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]], np.float32).T
        top = j[:, 15] + 0.6 * (j[:, 15] - j[:, 12])
        k = np.stack([top if i < 0 else j[:, i] for i in H36M], 1)
        conf = np.stack([seen[:, 15] if i < 0 else seen[:, i] for i in H36M], 1).astype(np.float32)
        xy = np.stack([k[..., 0], -k[..., 1]], -1)
        valid = conf > 0
        if valid.any():
            lo, hi = xy[valid].min(0), xy[valid].max(0)
            xy = (xy - (lo + hi) / 2) / ((hi - lo).max() / 2 + 1e-6)
        return np.concatenate([xy * conf[..., None], conf[..., None]], -1).astype(np.float32)

    @torch.no_grad()
    def __call__(self, joints, mask):
        """-> per-frame representation (T, 512), mean over the joints."""
        k = torch.from_numpy(self.keypoints(joints, mask)).to(self.device)
        out = [self.model.get_representation(k[s:s + 243][None])[0].mean(1) for s in range(0, len(k), 243)]
        return torch.cat(out)


# ------------------------------------------------------------------ systems of Table 2

class VectorPQ:
    """Product quantization (8 x 256) of per-core vectors, fitted on the train split."""

    def __init__(self, vectors, device, subspaces=8, size=256):
        x = torch.cat(vectors).float().to(device)
        self.split = np.array_split(np.arange(x.shape[1]), subspaces)
        self.books = [kmeans(x[:, s], size) for s in self.split]

    def __call__(self, v):
        v = v.float()
        return torch.cat([b[torch.cdist(v[:, s], b).argmin(1)] for s, b in zip(self.split, self.books)], 1)


_MODELS = {}


def _model(name, train, device):
    if name not in _MODELS:
        if name in ("tmr", "momask") and "features" not in _MODELS:
            _MODELS["features"] = HumanML3D(train)
        _MODELS[name] = {"tmr": lambda: TMR(_MODELS["features"], device),
                         "momask": lambda: MoMask(_MODELS["features"], device),
                         "motionbert": lambda: MotionBERT(device)}[name]()
    return _MODELS[name]


def core_mean(frames, mask, c=10):
    """Per-frame vectors (T, d) and a (T, 23) mask -> per-core means (N, d), observed fractions (N,)."""
    t = len(frames)
    n = max(1, int(np.ceil(t / c)))
    seen = torch.as_tensor(mask.mean(1), device=frames.device).float()
    f = F.pad(frames, (0, 0, 0, n * c - t)).view(n, c, -1)
    s = F.pad(seen, (0, n * c - t)).view(n, c)
    return f.mean(1), s.mean(1)


def systems(records, train, device):
    """name -> fn(joints, mask, kind) -> scores over `records`, for the pretrained encoders of Table 2."""
    full = lambda r: (r["joints"], np.repeat(r["observed"][:, None], 23, 1))
    cache = {}

    def run(name, joints, mask):
        return _model(name, train, device)(joints, mask)

    def ref(name, r):
        """Encoder output of a whole, clean reference (or train) recording, computed once."""
        if (name, r["id"]) not in cache:
            cache[(name, r["id"])] = run(name, *full(r))
        return cache[(name, r["id"])]

    def refs(name, fn):
        return [fn(ref(name, r)) for r in records]

    def cosine(name, fn):
        def score(joints, mask, kind):
            R = torch.stack(refs(name, fn))
            return (R @ fn(run(name, joints, mask))).cpu().numpy()
        return score

    def words(name, cores):
        """PQ words over per-core vectors, scored by the CRE."""
        state = {}

        def score(joints, mask, kind):
            if "index" not in state:
                pq = VectorPQ([cores(ref(name, r), full(r)[1])[0] for r in train], device)
                items = []
                for r in records:
                    v, o = cores(ref(name, r), full(r)[1])
                    items.append({"vectors": pq(v), "observed": o})
                state["pq"], state["index"] = pq, Index(items, device)
            v, o = cores(run(name, joints, mask), mask)
            return state["index"].score(state["pq"](v), o)[0].cpu().numpy()
        return score

    def mb_cores(rep, mask):
        return core_mean(rep, mask)

    def tmr_cores(out, mask):
        return out[1], torch.ones(len(out[1]), device=device)

    def momask_dtw(joints, mask, kind):
        seqs = refs("momask", lambda o: F.normalize(o[0], dim=-1))
        q = F.normalize(run("momask", joints, mask)[0], dim=-1)
        X, lengths = pad(seqs, device)
        cost = 1 - torch.einsum("nd,rmd->rnm", q, X)
        return (-baselines_dtw(cost, lengths, subsequence=kind == "fragment")).cpu().numpy()

    def histogram(o):
        return F.normalize(torch.bincount(o[1], minlength=512).float(), dim=0)

    return {
        "MotionBERT, mean-pooled": cosine("motionbert", lambda rep: F.normalize(rep.mean(0), dim=0)),
        "PQ words over MotionBERT": words("motionbert", mb_cores),
        "TMR motion encoder": cosine("tmr", lambda o: o[0]),
        "TMR, window mean": cosine("tmr", lambda o: F.normalize(o[1].mean(0), dim=0)),
        "PQ words over TMR latents": words("tmr", tmr_cores),
        "MoMask, code histograms": cosine("momask", histogram),
        "MoMask, mean embeddings": cosine("momask", lambda o: F.normalize(o[0].mean(0), dim=0)),
        "MoMask, DTW over embeddings": momask_dtw,
    }
