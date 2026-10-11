"""Table 3, codebook refinement: refit the residual codebooks of a trained model with its encoder frozen.

The fitted centroids are moved by gradient descent on the reconstruction of the
training cores and, with --ranking, on supervised contrast between the pooled
reconstructions of the training recordings. The encoder is copied unchanged.

    python -m motionprov.choreomatch.refine --corpus aist.pkl --model MODEL_DIR --output OUT [--ranking 1]
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pickle
import torch
import torch.nn.functional as F

from motionprov.choreomatch.model import Encoder
from motionprov.choreomatch.train import descriptors, supcon


def reconstruct(z, books):
    """Hard residual assignment; the gradient reaches the chosen centroids."""
    residual, recon = z, torch.zeros_like(z)
    for book in books:
        q = book[torch.cdist(residual, book).argmin(1)]
        recon = recon + q
        residual = residual - q
    return recon


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--ranking", type=float, default=0.0, help="weight of the ranking loss")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    src, out = Path(args.model), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    model = Encoder().to(device)
    model.load_state_dict(torch.load(src / "model.pt", map_location=device))
    model.eval()
    train = [r for r in pickle.load(open(args.corpus, "rb")) if r["split"] == "train"]
    zs, ys = [], []
    labels = sorted({r["label"] for r in train})
    for r in train:
        z, o = descriptors(model, r, "clean", "whole", device)
        zs.append(z[o >= 0.5])
        ys.append(labels.index(r["label"]))
    flat = torch.cat(zs)
    y = torch.tensor(ys, device=device)

    books = torch.load(src / "codebooks.pt")
    params = [torch.nn.Parameter(b.to(device).clone()) for b in books["residual"]]
    opt = torch.optim.Adam(params, lr=1e-3)
    for step in range(args.steps + 1):
        loss = F.mse_loss(reconstruct(flat, params), flat)
        if args.ranking:
            u = torch.stack([F.normalize(F.normalize(reconstruct(z, params), dim=-1).mean(0), dim=0) for z in zs])
            loss = loss + args.ranking * supcon(u, y, 0.1)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0:
            print(f"refine step {step} loss {float(loss):.5f}", flush=True)
    books["residual"] = [p.detach().cpu() for p in params]
    torch.save(books, out / "codebooks.pt")
    shutil.copy(src / "model.pt", out / "model.pt")


if __name__ == "__main__":
    main()
