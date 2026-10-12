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

## Main results (AIST++, 3 training runs unless noted)

- Pose error: 4.0 cm torso, 5.4 legs, 5.8 arms; 7.5 hidden legs, 10.8 hidden arms.
- Replacing one arm: the correct take stays on top for 0.99 of queries unedited, 0.30 (kNN) and 0.73 (DTW) after replacing the left arm.
- Trained model alone: mAP 0.59. DTW + kNN: 0.70. DTW + kNN + trained model: 0.77.
- AI-generated video: AUC 0.53 to 0.76 without a library of real clips, 0.96 to 0.997 for copies with a library.
- Conformal threshold: 4.5 to 4.8 % false alarms at a 5 % target, about 89 % of copies caught.
