"""Does a fixed matcher notice when one limb of a skeleton is replaced?

The proposal asks whether the representation keeps the small detail that separates
a motion from a near-identical one. Same-genre dances differ in everything, so they
do not isolate a detail. Here we make the near-identical motion ourselves: a recorded
view keeps its body and everything but one limb, and that limb gets the motion of
another dance (``swap``: the donor's limb offsets, expressed in the body frame, time-
resampled and rescaled to the limb length) or stops moving (``freeze``: its mean offset).
Offsets are taken in a per-frame body frame (hips and spine), so a limb keeps following
the body when the dancer turns.

For a query q (one camera) and its take seen from another camera (original o, edited e):
* ``noticed``: share of (q, take) pairs with s(q, o) > s(q, e), ties half. 0.5 = the
  matcher cannot tell them apart, 1 = it always prefers the original.
* ``take_beats_other_dances_*``: share of those pairs where the same take (original, or edited) scores
  above every view of every other dance: whether the edit makes the dance look like another one.
* ``p1_edited_gallery``: precision@1 of originals as queries against a gallery in which
  every view is edited, next to ``p1_original_gallery`` for the unedited gallery (views
  of the query's own take excluded).

    python -m motionprov.cv801.limb_edit --split SPLIT --restrict scores.json --cache CACHE --output OUT.json
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

CAMERA = re.compile(r"_c\d\d_")
# SMPL: (root joint, chain). Pelvis is joint 0, centred at the origin.
LIMBS = {"left_arm": (16, [18, 20, 22]), "right_arm": (17, [19, 21, 23]),
         "left_leg": (1, [4, 7, 10]), "right_leg": (2, [5, 8, 11])}
EDITS = {"left_arm": ["left_arm"], "right_arm": ["right_arm"], "both_arms": ["left_arm", "right_arm"],
         "left_leg": ["left_leg"], "both_legs": ["left_leg", "right_leg"]}


def body_frames(joints):
    """(T, 24, 3) -> (T, 3, 3) with columns across the hips, up the spine and forward."""
    across = joints[:, 2] - joints[:, 1]
    across /= np.linalg.norm(across, axis=1, keepdims=True) + 1e-9
    up = joints[:, 3] - joints[:, 0]
    forward = np.cross(across, up)
    forward /= np.linalg.norm(forward, axis=1, keepdims=True) + 1e-9
    up = np.cross(forward, across)
    return np.stack([across, up, forward], axis=2)


def offsets(joints, root, chain):
    frames = body_frames(joints)
    return np.einsum("tji,tkj->tki", frames, joints[:, chain] - joints[:, [root]])


def resample(x, length):
    at = np.linspace(0, len(x) - 1, length)
    low = np.floor(at).astype(int)
    high = np.minimum(low + 1, len(x) - 1)
    w = (at - low)[:, None, None]
    return x[low] * (1 - w) + x[high] * w


def edit(joints, limbs, kind, donor=None):
    out = joints.copy()
    frames = body_frames(joints)
    for limb in limbs:
        root, chain = LIMBS[limb]
        own = offsets(joints, root, chain)
        if kind == "freeze":
            new = np.broadcast_to(own.mean(0, keepdims=True), own.shape)
        else:
            theirs = resample(offsets(donor, root, chain), len(joints))
            new = theirs * (np.linalg.norm(own[:, 0], axis=1).mean() / (np.linalg.norm(theirs[:, 0], axis=1).mean() + 1e-9))
        out[:, chain] = joints[:, [root]] + np.einsum("tij,tkj->tki", frames, new)
    return out


def metrics(scores, n, label, take):
    """scores: (2n, 2n); rows/cols 0..n-1 originals, n..2n-1 edited copies of the same views."""
    noticed, p1_edit, p1_orig, beats_edit, beats_orig = [], [], [], [], []
    for i in range(n):
        same_take = [j for j in range(n) if take[j] == take[i] and j != i]
        other = [k for k in range(n) if label[k] != label[i]]
        for j in same_take:
            d = scores[i, j] - scores[i, n + j]
            noticed.append(1.0 if d > 0 else 0.5 if d == 0 else 0.0)
            beats_orig.append(float(scores[i, j] > scores[i, other].max()))
            beats_edit.append(float(scores[i, n + j] > scores[i, other].max()))
        valid = np.array([j for j in range(n) if take[j] != take[i]])
        if not any(label[j] == label[i] for j in valid):
            continue
        p1_orig.append(float(label[valid[scores[i, valid].argmax()]] == label[i]))
        p1_edit.append(float(label[valid[scores[i, n + valid].argmax()]] == label[i]))
    return {"noticed": float(np.mean(noticed)), "p1_original_gallery": float(np.mean(p1_orig)),
            "p1_edited_gallery": float(np.mean(p1_edit)),
            "take_beats_other_dances_original": float(np.mean(beats_orig)),
            "take_beats_other_dances_edited": float(np.mean(beats_edit)), "pairs": len(noticed), "queries": len(p1_orig)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True)
    parser.add_argument("--restrict", required=True, help="scores.json whose clean views are the test set")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--draws", type=int, default=2, help="random donor assignments per swap edit")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    from motionprov.cv801.diagnostic import dtw_scores, knn_scores, load_views
    labels = json.loads(Path(args.split).read_text())["labels"]
    keep = set(json.loads(Path(args.restrict).read_text())["clean"]["ids"])
    views = load_views(args.cache)
    ids = sorted(v for v in views if v in keep)
    joints = [views[v][0] for v in ids]
    take = [CAMERA.sub("_cAll_", v) for v in ids]
    label = [labels[t] for t in take]
    n = len(ids)
    result = {}
    for name, limbs in EDITS.items():
        for kind in ("swap", "freeze"):
            runs = []
            for draw in range(args.draws if kind == "swap" else 1):
                rng = np.random.default_rng(draw)
                edited = []
                for i in range(n):
                    donors = [k for k in range(n) if label[k] != label[i]]
                    donor = joints[int(rng.choice(donors))] if kind == "swap" else None
                    edited.append(edit(joints[i], limbs, kind, donor))
                both = joints + edited
                runs.append({"knn": metrics(knn_scores(both), n, label, take),
                             "dtw": metrics(dtw_scores(both, args.device), n, label, take)})
            result[f"{kind}/{name}"] = {m: {k: float(np.mean([r[m][k] for r in runs])) for k in runs[0][m]} for m in ("knn", "dtw")}
            print(f"{kind}/{name:10s}", "  ".join(
                f"{m}: noticed {result[f'{kind}/{name}'][m]['noticed']:.2f}, p@1 {result[f'{kind}/{name}'][m]['p1_original_gallery']:.2f}"
                f" -> {result[f'{kind}/{name}'][m]['p1_edited_gallery']:.2f}, same take beats other dances {result[f'{kind}/{name}'][m]['take_beats_other_dances_original']:.2f}"
                f" -> {result[f'{kind}/{name}'][m]['take_beats_other_dances_edited']:.2f}" for m in ("knn", "dtw")), flush=True)
    Path(args.output).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
