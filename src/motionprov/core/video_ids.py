"""One place that knows how a derived clip is named.

Synthetic body variants are written as `<video>__b2`: the same pose sequence
decoded onto a different SMPL body. They carry a new string id and the same
dance label, which is how one slipped past the split - `meta.val_videos` holds
exact names, so `val_video__b2` stayed in the training windows while `val_video`
was held out, and in the gallery it looked like a second performance of the same
choreography. A minimal PQ run built that way reported mAP 1.0 whose only
positive was the query's own synthetic copy.

Everything that splits, trains or scores must group by the SOURCE clip.
"""

DERIVED_SEP = "__"


def base_video_id(video_id: str) -> str:
    """The source clip a (possibly derived) id came from."""
    return video_id.split(DERIVED_SEP, 1)[0]


def base_ids(video_ids) -> list[str]:
    return [base_video_id(v) for v in video_ids]


def expand_source_ids(video_ids, selected_ids) -> set[str]:
    """Select actual corpus IDs by source, including all available variants."""
    selected = set(base_ids(selected_ids))
    return {v for v in video_ids if base_video_id(v) in selected}


def is_derived(video_id: str) -> bool:
    return DERIVED_SEP in video_id
