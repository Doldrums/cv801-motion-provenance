# Motion provenance from skeletons

Course project for CV801 (Advanced Computer Vision, MBZUAI).
Mentor: Mohammed Irfan K. Author: Arina Kharlamova.

The project builds a fingerprint of a motion from a skeleton, and uses it for two questions:

1. Is a skeleton recovered from video good enough to find the same motion in a database?
2. Can an AI-generated video be recognised, with and without a library of real clips?

## Contents

| Path | What it is |
|---|---|
| `src/motionprov/cv801/` | experiments: pose error, limb replacement, trained model, fusion, abstention, copy detection, conformal threshold |
| `src/motionprov/choreomatch/` | matchers (kNN, DTW, trained), evaluation, and the web interface (`stand.py`, `stand.html`) |
| `src/motionprov/core/` | the small part of the shared skeleton and retrieval code that the experiments import |
| `docs/` | progress report slides and video |

## Not included

Datasets (AIST++, NTU RGB+D, the social media dataset), pose caches, model checkpoints and the pose model assets are not in this repository, and the code does not run without them. This is a snapshot of one part of the work, not a complete release.
