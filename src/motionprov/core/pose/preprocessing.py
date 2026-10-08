from typing import List, Optional, Tuple

import torch

# Identifies trajectory selection, gap filling and observation-mask semantics.
PREPROCESSING_VERSION = 2


def _ready_joint_mask(data):
    """Admit ready SMPL24 joints before centering, with explicit missingness.

    Generic ready joints may have genuine zero coordinates (including a root
    at the origin). Only the explicitly identified legacy NTU source policy
    treats raw zero XYZ tuples as missing. An explicit observation mask is
    authoritative about zero values; finiteness and root availability always
    constrain the usable coordinates. No missing coordinate is interpolated.
    """
    joints = data['joints3d']
    if (not torch.is_tensor(joints) or joints.ndim != 3
            or joints.shape[1:] != (24, 3) or not joints.is_floating_point()):
        raise ValueError('Ready joints require floating [T,24,3] SMPL24 coordinates')
    policy = data.get('joints3d_observation_policy', 'finite_or_explicit')
    if policy not in ('finite_or_explicit', 'zero_tuple_missing'):
        raise ValueError('Unknown ready-joints observation policy')
    present = torch.isfinite(joints).all(-1)
    if 'observed' in data:
        observed = torch.as_tensor(data['observed'], device=joints.device)
        if observed.shape == joints.shape[:1]:
            observed = observed[:, None].expand(-1, 24)
        if observed.dtype != torch.bool or observed.shape != joints.shape[:2]:
            raise ValueError('Ready joints observed must be bool[T] or bool[T,24]')
        present &= observed
    elif policy == 'zero_tuple_missing':
        present &= (joints != 0).any(-1)
    if 'frame_valid' in data:
        frame_valid = torch.as_tensor(data['frame_valid'], device=joints.device)
        if frame_valid.dtype != torch.bool or frame_valid.shape != joints.shape[:1]:
            raise ValueError('Ready joints frame_valid must be bool[T]')
        present &= frame_valid[:, None]
    return present & present[:, :1]


def preprocess_smpl_to_skeleton(
    smpl_decoder,
    pose: torch.Tensor,
    trans: torch.Tensor,
    betas: torch.Tensor
) -> torch.Tensor:
    joints_3d = smpl_decoder(betas, pose, trans)  # (T, V, 3)

    root = joints_3d[:, 0:1, :]
    joints_3d = joints_3d - root

    skeleton = joints_3d.permute(2, 0, 1)  # (3, T, 24)

    return skeleton


def create_sliding_windows(
    sequence_length: int,
    window_size: int,
    stride: int,
    min_length: int = 30
) -> List[Tuple[int, int]]:
    if sequence_length < min_length:
        return []

    windows = []

    for start in range(0, max(1, sequence_length - min_length + 1), stride):
        end = min(start + window_size, sequence_length)
        if end - start >= min_length:
            windows.append((start, end))

    return windows


def aggregate_overlapping_predictions(
    predictions: List[Tuple[int, int, List[int]]],
    total_length: int
) -> List[int]:
    from collections import Counter

    frame_votes = [[] for _ in range(total_length)]

    for start, _end, words in predictions:
        for i, word in enumerate(words):
            frame_idx = start + i
            if frame_idx < total_length:
                frame_votes[frame_idx].append(word)

    aggregated = []
    for votes in frame_votes:
        if votes:
            counter = Counter(votes)
            most_common = counter.most_common(1)[0][0]
            aggregated.append(most_common)
        else:
            aggregated.append(-1)

    for i in range(len(aggregated)):
        if aggregated[i] == -1:
            if i > 0 and aggregated[i-1] != -1:
                aggregated[i] = aggregated[i-1]
            elif i < len(aggregated) - 1:
                for j in range(i+1, len(aggregated)):
                    if aggregated[j] != -1:
                        aggregated[i] = aggregated[j]
                        break

    return aggregated


def _rotation_distance(aa: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Угол между вращениями, заданными axis-angle. (N,3) против (3,) -> (N,).

    Евклидова разность axis-angle не является метрикой на вращениях: у ±π
    почти совпадающие повороты оказываются максимально далёкими.
    """
    def to_mat(v):
        th = v.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        k = v / th
        K = torch.zeros(*v.shape[:-1], 3, 3, dtype=v.dtype, device=v.device)
        K[..., 0, 1], K[..., 0, 2] = -k[..., 2], k[..., 1]
        K[..., 1, 0], K[..., 1, 2] = k[..., 2], -k[..., 0]
        K[..., 2, 0], K[..., 2, 1] = -k[..., 1], k[..., 0]
        eye = torch.eye(3, dtype=v.dtype, device=v.device).expand_as(K)
        th = th.unsqueeze(-1)
        return eye + torch.sin(th) * K + (1 - torch.cos(th)) * (K @ K)

    A, B = to_mat(aa), to_mat(ref.unsqueeze(0))
    tr = (A @ B.transpose(-1, -2)).diagonal(dim1=-2, dim2=-1).sum(-1)
    return torch.arccos(((tr - 1) / 2).clamp(-1.0, 1.0))


def process_comotion_data(
    data: dict,
    smpl_decoder,
    start: int,
    end: Optional[int] = None,
    person_idx: int = 0,
    gap_policy: str = 'hold',
    return_mask: bool = False
):
    """CoMotion detections -> root-centered skeleton (3, T, V).

    Two things this function used to get wrong, both found in review.

    A frame with no detection was filled with zero pose, zero trans and zero
    betas, which SMPL decodes into a full neutral stance - a fabricated pose
    that then entered velocity, reconstruction, quantization and retrieval as
    if it had been observed. On the local crosscam corpus 8.87% of frames took
    that branch, and one clip had 421 consecutive fabricated frames. The
    default policy is now 'hold': carry the last observed pose forward, so a
    gap contributes no motion instead of a violent jump to neutral. Pass
    gap_policy='neutral' to reproduce numbers computed under the old behaviour.

    The performer was chosen by ROW POSITION within each frame, ignoring the
    stored track id, so a permutation of rows or the disappearance of the main
    person could switch the selected trajectory mid-clip. Selection now uses
    a continuity heuristic: the dominant track seeds it, that id is preferred
    whenever it is present, and otherwise the detection closest to the last
    accepted pose is taken.

    Following the dominant id alone would have been worse than the bug. On the
    local crosscam corpus the tracker churns ids - a median of 2 per clip, up
    to 9 - and the dominant id covers a median of 77% of frames against 96%
    for any detection at all, with 31% of clips under half. These saved track
    statistics do not establish physical performer identity. Continuity can
    retain a re-acquisition, but can also switch people; identity is not
    authenticated by the pose-distance or track-id heuristic.

    With return_mask=True the observation mask is returned alongside, so a
    caller can exclude fabricated frames rather than trust them.
    """
    if gap_policy not in ('hold', 'neutral'):
        raise ValueError(f"gap_policy must be 'hold' or 'neutral', got {gap_policy!r}")

    # Ready joints need no SMPL decoder. Source admission explicitly identifies
    # legacy NTU zero sentinels; generic coordinate origins remain legitimate.
    if 'joints3d' in data:
        j = data['joints3d']
        present = _ready_joint_mask(data)
        if end is None:
            end = j.shape[0]
        j = j[start:end].clone()
        present = present[start:end]
        j = torch.where(present[..., None], j, 0.)
        j = j - j[:, 0:1, :]
        j = torch.where(present[..., None], j, 0.)
        out = j.permute(2, 0, 1).contiguous()
        if return_mask:
            # Preserve the old frame-only shape whenever all joint slots share
            # validity; partial observations use the already supported [T,V].
            mask = present[:, 0] if bool((present == present[:, :1]).all()) else present
            return out, mask
        return out

    frame_indices = torch.as_tensor(data['frame_idx'])
    clip_length = int(frame_indices.max().item()) + 1 if frame_indices.numel() else 0
    if end is None:
        end = clip_length

    ids = data.get('id')
    track = None
    if ids is not None:
        ids = ids if torch.is_tensor(ids) else torch.as_tensor(ids)
        # Трек выбирается по ВСЕМУ клипу, а не по запрошенному куску: иначе
        # два выреза одного ролика могут пойти за разными людьми, и целое не
        # совпадает со своими же кусками (замерено: расхождение на кадрах
        # 253-255 реального crosscam-клипа).
        if ids.numel():
            vals, counts = torch.unique(ids, return_counts=True)
            track = vals[counts.argmax()]

    # Select one trajectory before cropping. A crop that starts after an ID
    # change still needs the preceding pose for continuity, and a crop inside
    # a gap must hold that preceding pose rather than borrow a future one.
    # Keep row indices here: only the requested interval is SMPL-decoded.
    selected = torch.full((max(clip_length, end),), -1, dtype=torch.long,
                          device=frame_indices.device)
    last_pose = last_trans = None
    # Stable grouping preserves the existing row preference within a frame
    # without scanning every detection again for every missing frame.
    order = torch.argsort(frame_indices, stable=True)
    frames, counts = torch.unique_consecutive(frame_indices[order], return_counts=True)
    for frame, cand in zip(frames.tolist(), order.split(counts.tolist())):
        pick = None
        if ids is not None and track is not None:
            same = cand[ids[cand] == track]
            if same.numel():
                pick = int(same[0])
        if pick is None:
            if cand.numel() == 1 or last_pose is None:
                # Accept the only detection, without proving performer identity;
                # with no prior pose, fall back to the caller's row preference.
                pick = int(cand[min(person_idx, cand.numel() - 1)])
            else:
                # Continue by pose proximity, which does not prove body identity.
                # Первые три числа позы — axis-angle глобального поворота, и
                # евклидова разность между ними ломается на границе ±π: два
                # почти одинаковых поворота дают расстояние около 2π. Поэтому
                # ориентация сравнивается через угол между вращениями, а
                # остальная поза — как есть.
                d = ((data['pose'][cand, 3:] - last_pose[3:]) ** 2).sum(1) \
                    + ((data['trans'][cand] - last_trans) ** 2).sum(1) \
                    + _rotation_distance(data['pose'][cand, :3],
                                         last_pose[:3]) ** 2
                pick = int(cand[int(d.argmin())])

        selected[frame] = pick
        last_pose, last_trans = data['pose'][pick], data['trans'][pick]

    observed = selected >= 0
    if gap_policy == 'hold' and observed.any():
        timeline = torch.arange(len(selected), device=selected.device)
        previous = torch.where(observed, timeline, -1).cummax(dim=0).values
        # Before the first detection, preserve the full-video policy of
        # borrowing the first observed pose, even when it lies after the crop.
        first = int(observed.nonzero(as_tuple=False)[0, 0])
        selected = selected[previous.clamp(min=first)]

    selected = selected[start:end]
    observed = observed[start:end]
    valid = selected >= 0

    def gather(key):
        source = data[key]
        result = source.new_zeros((len(selected), *source.shape[1:]))
        result[valid] = source[selected[valid]]
        return result

    pose, trans, betas = gather('pose'), gather('trans'), gather('betas')

    skeleton = preprocess_smpl_to_skeleton(smpl_decoder, pose, trans, betas)
    if return_mask:
        return skeleton, observed
    return skeleton
