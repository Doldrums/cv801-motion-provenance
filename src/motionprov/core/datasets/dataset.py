import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler, WeightedRandomSampler

from motionprov.core.pose.preprocessing import _ready_joint_mask, process_comotion_data
from motionprov.core.video_ids import base_video_id, expand_source_ids


class PKVideoBatchSampler(Sampler):
    """Class-balanced batches for supervised-contrastive training: each batch
    holds P choreographies x K distinct source clips, one chunk per source.
    Synthetic body variants cannot provide another positive for their source.
    Only choreographies with >= K sources among the given indices are eligible;
    the rest are skipped. Yields floor(sum eligible chunks / (P*K)) batches/epoch.
    """

    def __init__(self, video_of, label_of, indices, p_classes, k_per_class,
                 seed=0):
        self.P, self.K = p_classes, k_per_class
        self.seed = seed
        self.epoch = 0
        # class -> {source -> [all original/derived chunk indices]}
        by_class = defaultdict(lambda: defaultdict(list))
        for i in indices:
            by_class[label_of[i]][base_video_id(video_of[i])].append(i)
        # keep only classes with at least K independent sources
        self.by_class = {c: v for c, v in by_class.items() if len(v) >= k_per_class}
        self.classes = sorted(self.by_class)
        n_chunks = sum(len(ch) for v in self.by_class.values() for ch in v.values())
        self.num_batches = max(1, n_chunks // (p_classes * k_per_class))
        if len(self.classes) < p_classes:
            raise ValueError(
                f"P x K sampler needs >= {p_classes} classes with >= "
                f"{k_per_class} source clips each; only {len(self.classes)} qualify. "
                f"Lower --pk-classes / --pk-per-class.")

    def set_epoch(self, e):
        self.epoch = e

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        for _ in range(self.num_batches):
            chosen = rng.sample(self.classes, self.P)
            batch = []
            for c in chosen:
                vids = rng.sample(list(self.by_class[c]), self.K)
                for v in vids:
                    batch.append(rng.choice(self.by_class[c][v]))
            yield batch


class CoMotionSMPLDataset(Dataset):
    def __init__(
        self,
        csv_path: Path,
        comotion_dir: Path,
        min_length: int = 30,
        split: str = 'train',
        chunk_size: int = None,
        patch_size: int = 30
    ):
        self.csv_path = Path(csv_path)
        self.comotion_dir = Path(comotion_dir)
        self.min_length = min_length
        self.chunk_size = chunk_size
        self.patch_size = patch_size

        # Validate chunk_size is divisible by patch_size
        if chunk_size is not None and chunk_size % patch_size != 0:
            raise ValueError(
                f"chunk_size ({chunk_size}) must be divisible by patch_size ({patch_size})")

        df = pd.read_csv(csv_path)
        self.dance_to_idx = {
            dance: idx for idx, dance in enumerate(sorted(df['dance_id'].unique()))
        }
        self.idx_to_dance = {
            idx: dance for dance,
            idx in self.dance_to_idx.items()
        }

        self.samples = []
        for _, row in df.iterrows():
            dance_id = row['dance_id']
            video_path = row['video_path']
            video_id = Path(video_path).stem

            # Shared eligibility filter - keep in lockstep with the val-split
            # recompute (see load_eligible_pose).
            data = load_eligible_pose(self.comotion_dir, video_id, min_length)
            if data is None:
                continue
            # frames, not detection rows - see load_eligible_pose
            T = int(data['frame_idx'].max().item()) + 1

            label = self.dance_to_idx[dance_id]

            if self.chunk_size is not None and T > self.chunk_size:
                num_chunks = T // self.chunk_size
                for i in range(num_chunks):
                    start = i * self.chunk_size
                    end = start + self.chunk_size
                    self.samples.append((video_id, label, start, end))

                # Non-overlapping tail: the old (T - chunk_size, T) window
                # shared up to chunk_size - remainder frames with the previous
                # chunk, double-weighting near-identical content in the
                # gradient, sampler counts and val_rec. A short variable-length
                # tail is fine - collate pads and the loss masks the padding.
                remainder = T % self.chunk_size
                if remainder >= min_length:
                    self.samples.append(
                        (video_id, label, num_chunks * self.chunk_size, T))
            else:
                self.samples.append((video_id, label, 0, T))

        random.shuffle(self.samples)

        self._smpl_decoder = None

    @property
    def smpl_decoder(self):
        if self._smpl_decoder is None:
            from motionprov.core.comotion.utils.smpl_kinematics import SMPLKinematics
            self._smpl_decoder = SMPLKinematics()
        return self._smpl_decoder

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        video_id, label, start, end = self.samples[idx]

        pt_path = self.comotion_dir / f"{video_id}.pt"
        data = _source_pose_policy(torch.load(pt_path, weights_only=False), video_id)

        # The decoder is lazy, but reading the property here built it anyway -
        # so an NTU-only checkout with ready joints3d and no SMPL assets failed
        # before reaching the branch that does not need them.
        decoder = None if 'joints3d' in data else self.smpl_decoder
        skeleton, observed = process_comotion_data(
            data, decoder, start, end, return_mask=True)

        return {
            'skeleton': skeleton,
            'observed': observed,
            'label': label,
            'video_id': video_id
        }

    def get_num_classes(self) -> int:
        return len(self.dance_to_idx)


def val_video_split(videos) -> set[str]:
    """Split unique source clips 80/20, returning every held-out corpus ID.

    Single source of truth for the train/val boundary: training and every
    evaluation must derive the split from this function, or metrics silently
    mix train videos into the query set.
    """
    videos = sorted(set(videos))
    sources = sorted({base_video_id(v) for v in videos})
    if len(sources) < 2:
        raise ValueError(
            f"need at least 2 source clips for a leak-free train/val "
            f"split, got {len(sources)}. Process at least one more video into "
            f"the comotion dir (each video must have a comotion/<id>.pt with "
            f">=30 frames), or point --videos-csv at a larger set.")
    rng = random.Random(42)
    shuffled = sources[:]
    rng.shuffle(shuffled)
    n_val = min(max(1, int(round(0.2 * len(shuffled)))), len(shuffled) - 1)
    return expand_source_ids(videos, shuffled[:n_val])


def _source_pose_policy(data, video_id):
    """Tag bare legacy NTU input; never guess missingness for generic joints."""
    if (set(data) == {'joints3d', 'frame_idx'}
            and re.fullmatch(r'S\d{3}C\d{3}P\d{3}R\d{3}A\d{3}', str(video_id))):
        return dict(data, joints3d_observation_policy='zero_tuple_missing')
    return data


def load_eligible_pose(comotion_dir: Path, video_id: str,
                       min_length: int = 30):
    """THE eligibility filter, single source of truth: returns the loaded
    pose dict if the video can be trained on (pose file exists and covers at
    least min_length frames), else None. Both the dataset and the val-split
    recompute in evaluate_identity go through this function - the seed-42
    split membership silently depends on the filter being identical in both
    places, so there is exactly one implementation."""
    pt_path = Path(comotion_dir) / f"{video_id}.pt"
    if not pt_path.exists():
        return None
    data = torch.load(pt_path, weights_only=False)
    # Historical NTU conversion has no mask field: its exact two-field schema
    # and canonical source ID identify Kinect zero-XYZ missing sentinels. Keep
    # that inference here, where source identity is available, rather than
    # treating legitimate origins in every generic ready-joints array as gaps.
    data = _source_pose_policy(data, video_id)
    fi = data['frame_idx']
    # Zero-detection files must be skipped, not crash every pipeline.
    if fi.numel() == 0:
        return None
    # Frame count is max(frame_idx)+1, NOT len(frame_idx): CoMotion emits one
    # row per frame PER PERSON, so multi-person videos have more detections
    # than frames (measured: 10/230 corpus files, worst 1854 rows over 676
    # real frames). But max+1 alone would admit sparse-detection videos whose
    # gaps process_comotion_data fills with fabricated rest pose - so
    # eligibility ALSO requires enough uniquely-detected frames and majority
    # coverage of the span. (Current corpus coverage is ~100%; this guards
    # future scraped data.)
    n_frames = int(fi.max().item()) + 1
    observed_fi = fi
    if 'joints3d' in data:
        present = _ready_joint_mask(data)
        if fi.ndim != 1 or len(fi) != len(present):
            raise ValueError('Ready joints require one frame_idx per source row')
        observed_fi = fi[present[:, 1:].any(-1).to(fi.device)]
    n_detected = int(torch.unique(observed_fi).numel())
    if n_frames < min_length or n_detected < min_length:
        return None
    if n_detected / n_frames < 0.5:
        return None
    return data


def eligible_videos(csv_path: Path, comotion_dir: Path, min_length: int = 30) -> list[str]:
    """Video ids the dataset would actually train on - the population
    val_video_split is applied to."""
    df = pd.read_csv(csv_path)
    return [vid for _, row in df.iterrows()
            if (vid := Path(row['video_path']).stem)
            and load_eligible_pose(comotion_dir, vid, min_length) is not None]


def collate_fn(batch: list[dict]) -> dict:
    max_T = max(item['skeleton'].shape[1] for item in batch)

    skeletons = []
    labels = []
    video_ids = []
    lengths = []
    observations = []

    for item in batch:
        skeleton = item['skeleton']
        C, T, V = skeleton.shape
        lengths.append(T)
        observed = item.get('observed', torch.ones(T, dtype=torch.bool))
        if observed.ndim == 1:
            observed = observed[:, None].expand(T, V)
        if observed.shape != (T, V):
            raise ValueError(f"observed must have shape {(T, V)} or {(T,)}")
        observed = observed.to(dtype=torch.bool)

        if T < max_T:
            # Pad by FREEZING the last pose, not with zeros. 170/230 videos are
            # shorter than the 900-frame chunk, so zero padding was ~20% of all
            # frames - and every zero patch mapped to a single codebook entry
            # (measured: code 298 at 20.5% usage, matching the 20.2% padding
            # share exactly). A held pose is a plausible motion; a zero
            # skeleton is not.
            pad = skeleton[:, -1:, :].expand(C, max_T - T, V)
            skeleton = torch.cat([skeleton, pad], dim=1)
            observed = torch.cat([observed, torch.zeros(
                max_T - T, V, dtype=torch.bool, device=observed.device)], dim=0)

        skeletons.append(skeleton)
        observations.append(observed)
        labels.append(item['label'])
        video_ids.append(item['video_id'])

    return {
        'skeleton': torch.stack(skeletons),
        'observed': torch.stack(observations),
        'label': torch.tensor(labels),
        'video_id': video_ids,
        # True (unpadded) frame counts, so the loss can ignore the frozen-pose
        # padding: a 60-frame clip batched with a 900-frame chunk is 93%
        # synthetic zero-velocity motion, which otherwise dominates both the
        # training gradient and the val_rec used for checkpoint selection.
        'length': torch.tensor(lengths)
    }


def create_dataloaders(
    csv_path: Path,
    comotion_dir: Path,
    batch_size: int = 16,
    num_workers: int = 4,
    min_length: int = 30,
    pin_memory: bool = False,
    use_stratified_sampling: bool = True,
    chunk_size: int = None,
    patch_size: int = 30,
    val_videos_override: set = None,
    pk_classes: int = 0,
    pk_per_class: int = 2,
    pk_exclude_val_classes: bool = False,
):
    dataset = CoMotionSMPLDataset(
        csv_path=csv_path,
        comotion_dir=comotion_dir,
        min_length=min_length,
        chunk_size=chunk_size,
        patch_size=patch_size
    )

    all_labels = [dataset.samples[i][1] for i in range(len(dataset))]

    # Split by SOURCE, never by chunk or synthetic body variant.
    # Long clips are cut into several chunks;
    # splitting the chunk list at random puts pieces of the same video on both
    # sides, which leaks the validation set into training and makes any
    # retrieval number measured on it meaningless.
    video_of = [dataset.samples[i][0] for i in range(len(dataset))]
    videos = sorted(set(video_of))
    if val_videos_override is not None:
        # Externally imposed split (e.g. a joint-corpus run holding out the
        # union of the per-corpus val splits, so per-corpus evaluations of
        # the joint checkpoint keep their exact query sets). Ids that are
        # not in this corpus are ignored; an empty side is a config error.
        val_videos = expand_source_ids(videos, val_videos_override)
        sources = {base_video_id(v) for v in videos}
        ignored = sum(base_video_id(v) not in sources for v in set(val_videos_override))
        if ignored:
            print(f"val override: {ignored} ids not in this corpus, ignored")
        if not val_videos or len(val_videos) >= len(videos):
            raise ValueError(
                f"--val-videos-file leaves {len(val_videos)} val / "
                f"{len(videos) - len(val_videos)} train videos; both sides "
                f"must be non-empty")
    else:
        val_videos = val_video_split(videos)

    indices = list(range(len(dataset)))
    train_indices = [i for i in indices if video_of[i] not in val_videos]
    val_indices = [i for i in indices if video_of[i] in val_videos]

    print(f"Split by source (all variants grouped): {len(videos) - len(val_videos)} train / "
          f"{len(val_videos)} val videos "
          f"({len(train_indices)} / {len(val_indices)} chunks)")

    train_labels = [all_labels[i] for i in train_indices]
    # A class whose few videos all land in val contributes no training chunks;
    # its val metrics then measure an untrained class. Unavoidable under a
    # leak-free video split with singleton classes, but never silent.
    missing = set(all_labels) - set(train_labels)
    if missing:
        names = sorted(dataset.idx_to_dance[m] for m in missing)
        print(f"Warning: {len(missing)} class(es) have no training chunks "
              f"(all their videos fell into val): {names}")

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    if pk_classes > 0:
        # Class-balanced batches for supervised contrastive: index by
        # SUBSET-LOCAL positions (0..len(train_indices)-1), since the loader
        # wraps a Subset. batch_size is then P*K and set by the sampler.
        local_video = [video_of[gi] for gi in train_indices]
        local_label = [all_labels[gi] for gi in train_indices]
        local_idx = list(range(len(train_indices)))
        if pk_exclude_val_classes:
            # Open-set arm: drop every class that can act as a query (a val
            # video of a class with >= 2 videos) from TRAINING entirely, so
            # the encoder never sees the query choreographies. Stricter than
            # excluding them from positives only.
            label_of_video = {}
            for gi in indices:
                label_of_video.setdefault(base_video_id(video_of[gi]), all_labels[gi])
            vids_per_class = Counter(label_of_video.values())
            query_classes = {label_of_video[base_video_id(v)] for v in val_videos
                             if vids_per_class[label_of_video[base_video_id(v)]] >= 2}
            local_idx = [i for i in local_idx
                         if local_label[i] not in query_classes]
            print(f"open-set PK: {len(query_classes)} query classes "
                  f"excluded; {len(local_idx)}/{len(train_indices)} train "
                  f"chunks remain")
        bsamp = PKVideoBatchSampler(
            local_video, local_label, local_idx,
            pk_classes, pk_per_class)
        print(f"P x K sampler: {pk_classes} classes x {pk_per_class} videos "
              f"= batch {pk_classes * pk_per_class}, {len(bsamp)} batches/epoch")
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_sampler=bsamp,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory
        )
    elif use_stratified_sampling:
        train_label_counts = Counter(train_labels)
        class_weights = {
            label: 1.0 / count for label,
            count in train_label_counts.items()
        }
        sample_weights = [class_weights[label] for label in train_labels]

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory
        )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, dataset.get_num_classes(), sorted(val_videos)
