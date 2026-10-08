"""Retrieval metrics for choreography *identity* retrieval.

The previous evaluation reported a "Mean Score" built from hand-chosen rank
weights {1.0, 0.5, 0.25}; CVPR reviewers called it heuristic and they were
right - it is comparable to nothing in the literature. This module reports mAP,
R@k and MRR instead.

A note on which metrics are meaningful on the current corpus: classes hold at
most 4 videos (median 3), so with leave-one-out there are at most 3 relevant
items per query. R@5 and R@10 are therefore near-degenerate - they measure the
class-size ceiling, not the system. Report R@1 and mAP; R@5/R@10 only become
informative once classes grow.
"""

from collections import Counter, defaultdict

import numpy as np

from motionprov.core.video_ids import base_video_id, expand_source_ids


def average_precision(relevant: list[bool]) -> float:
    """AP over a ranked list. `relevant[i]` is True if rank i+1 is a true match."""
    n_rel = sum(relevant)
    if n_rel == 0:
        return 0.0
    hits, score = 0, 0.0
    for i, r in enumerate(relevant, start=1):
        if r:
            hits += 1
            score += hits / i
    return score / n_rel


def reciprocal_rank(relevant: list[bool]) -> float:
    for i, r in enumerate(relevant, start=1):
        if r:
            return 1.0 / i
    return 0.0


def recall_at_k(relevant: list[bool], k: int, n_relevant_total: int) -> float:
    """Fraction of all relevant items retrieved within the top k.

    The denominator is the total number of relevant items, not min(k, n_rel):
    with the latter, R@1 collapses onto Hit@1 and the numbers are inflated for
    every k < n_rel.
    """
    if n_relevant_total == 0:
        return 0.0
    return sum(relevant[:k]) / n_relevant_total


def hit_at_k(relevant: list[bool], k: int) -> float:
    """1.0 if any relevant item appears in the top k. This is what the old
    'match rate' measured, kept so old numbers remain comparable."""
    return 1.0 if any(relevant[:k]) else 0.0


def histogram(tokens: list[int], num_words: int) -> np.ndarray:
    """L2-normalised bag-of-words over the motion vocabulary."""
    h = np.bincount(np.asarray(tokens, dtype=np.int64), minlength=num_words).astype(np.float64)
    n = np.linalg.norm(h)
    return h / n if n > 0 else h


def codebook_stats(all_tokens: dict[str, list[int]], num_words: int) -> dict:
    """Codebook health measured AT INFERENCE, with the codebook frozen.

    This differs from the usage counter reported during training, which is
    inflated by dead-code revival: revival reassigns unused codes to random
    latent patches, so they count as "used" without the encoder ever choosing
    them. The honest number is how many distinct codes the frozen model
    actually emits over the corpus.
    """
    counts = Counter()
    for toks in all_tokens.values():
        counts.update(toks)
    used = len(counts)
    total = sum(counts.values())
    p = np.array([c / total for c in counts.values()]) if total else np.array([1.0])
    entropy = float(-(p * np.log(p)).sum())
    return {
        "codes_used": used,
        "codes_total": num_words,
        "usage_frac": used / num_words,
        "usage_entropy": entropy,
        "usage_entropy_max": float(np.log(num_words)),
        "usage_entropy_norm": entropy / float(np.log(num_words)),
        # share of all emitted tokens taken by the single most common code
        "top1_share": float(max(counts.values()) / total) if total else 1.0,
    }


def source_grouped_queries(video_ids, labels, query_videos=None):
    """Select positive-bearing source-disjoint queries for legacy baselines.

    The shared video-id reader defines synthetic aliases. A held-out source
    includes its available variants; copies never create a second performance.
    Return the selected row indices and the groups to pass to ranking_metrics.
    """
    if len(video_ids) != len(labels):
        raise ValueError("Video IDs and labels must have the same length")
    sources = [base_video_id(v) for v in video_ids]
    sources_by_class = defaultdict(set)
    for source, label in zip(sources, labels):
        sources_by_class[label].add(source)
    selected = None if query_videos is None else expand_source_ids(video_ids, query_videos)
    queries = [
        i
        for i, video in enumerate(video_ids)
        if len(sources_by_class[labels[i]]) >= 2 and (selected is None or video in selected)
    ]
    return queries, sources


def ranking_metrics(
    sims: np.ndarray,
    labels: np.ndarray,
    queries: list[int],
    query_videos=None,
    ks: tuple[int, ...] = (1, 3, 5),
    groups=None,
) -> dict:
    """Identity-retrieval metrics from a full similarity matrix.

    Shared by the motion-word evaluation and every baseline (DTW, embedding
    kNN, ...), so numbers are comparable by construction: same tie policy
    (seeded random permutation), same self-exclusion, same random floor.
    `sims` is (N, N), higher = more similar; rows are queries.

    `groups`, when given, is one id per corpus item naming the SOURCE the item
    came from - a synthetic body variant shares its source clip's id. Every
    candidate sharing the query's group is dropped from the ranking and from
    the relevant count. Setting those similarities to -inf is not enough, for
    the same reason it is not enough for the query itself: the item still
    carries the query's label and is counted as a further correct match at the
    tail of the list, which inflates AP.
    """
    counts = Counter(labels.tolist())
    n = len(labels)
    if not queries:
        # np.mean([]) would silently write NaN into results JSON
        out = {"n_queries": 0, "n_corpus": n, "n_classes_queried": 0, "mAP": 0.0, "MRR": 0.0, "mAP_random_floor": 0.0}
        for k in ks:
            out[f"R@{k}"] = 0.0
            out[f"Hit@{k}"] = 0.0
        return out
    aps, rrs = [], []
    hits = {k: [] for k in ks}
    recs = {k: [] for k in ks}
    tie_rng = np.random.default_rng(1)
    tie_break = tie_rng.permutation(n)
    groups = None if groups is None else np.asarray(groups)
    for i in queries:
        order = np.lexsort((tie_break, -sims[i]))
        # Drop the query itself from the ranking. Masking its similarity is
        # not enough: it carries the query's own label and would otherwise be
        # counted as a further correct match at the tail of the list.
        if groups is None:
            order = order[order != i]
            n_rel = int(counts[labels[i]] - 1)
        else:
            same_source = groups == groups[i]
            order = order[~same_source[order]]
            n_rel = int(((labels == labels[i]) & ~same_source).sum())
        rel = (labels[order] == labels[i]).tolist()
        aps.append(average_precision(rel))
        rrs.append(reciprocal_rank(rel))
        for k in ks:
            hits[k].append(hit_at_k(rel, k))
            recs[k].append(recall_at_k(rel, k, n_rel))

    out = {
        "n_queries": len(queries),
        "n_corpus": n,
        "n_classes_queried": len({labels[i] for i in queries}),
        "mAP": float(np.mean(aps)),
        "MRR": float(np.mean(rrs)),
        # Per-query APs make paired bootstrap CIs computable from the
        # artifact alone - with 25-78 queries, differences below ~0.05 are
        # routinely inside noise and must be checked, not eyeballed.
        "per_query_AP": [round(float(a), 6) for a in aps],
    }
    # Query ids alongside the APs: different scripts iterate videos in
    # different orders (csv insertion vs sorted), and a positional pairing
    # of per_query_AP across families silently subtracts APs of different
    # videos. With ids stored, paired_ci aligns by id.
    if query_videos is not None:
        out["query_videos"] = [str(query_videos[i]) for i in queries]
    for k in ks:
        out[f"R@{k}"] = float(np.mean(recs[k]))
        out[f"Hit@{k}"] = float(np.mean(hits[k]))

    # Expected mAP of a random ranking, estimated with 200 permutations per
    # query. A single draw (the previous estimator) has per-corpus noise the
    # size of real effects and depends on index ordering: identical corpora
    # produced floors from 0.022 to 0.056 against a true 0.029.
    # Пол считается на ТОЙ ЖЕ допустимой галерее, что и метрика: иначе он
    # исключает только сам запрос, а AP уже исключает всю группу-источник, и
    # два числа относятся к разным задачам (замер на фикстуре: сообщалось
    # 0.8004 при корректных 0.75).
    rng = np.random.default_rng(0)
    rnd = []
    for i in queries:
        rel = labels == labels[i]
        if groups is None:
            keep = np.ones(n, bool)
            keep[i] = False
        else:
            keep = groups != groups[i]
        rel_keep = rel[keep]
        m_keep = int(keep.sum())
        if m_keep == 0:
            continue
        for _ in range(200):
            rnd.append(average_precision(rel_keep[rng.permutation(m_keep)].tolist()))
    out["mAP_random_floor"] = float(np.mean(rnd)) if rnd else 0.0
    return out


def evaluate_identity_retrieval(
    tokens_by_video: dict[str, list[int]],
    label_by_video: dict[str, str],
    num_words: int = 512,
    ks: tuple[int, ...] = (1, 3, 5),
    query_videos: set[str] | None = None,
) -> dict:
    """Leave-one-out identity retrieval over bag-of-words token histograms.

    Every video whose class has another independent source is used as a query.
    All variants of the query's own source are excluded from the gallery and
    positive denominator; remaining singleton classes can act as distractors.

    `query_videos` restricts which videos may act as queries (the corpus is
    always the full token set). Pass the held-out split here: with queries
    drawn from videos the model trained on, the numbers measure memorisation,
    not retrieval.
    """
    vids = [v for v in tokens_by_video if v in label_by_video]
    if not vids:
        raise ValueError(
            "no video produced motion words - check that --output-dir points "
            "at a directory containing comotion/*.pt for the videos in the CSV"
        )
    labels = np.array([label_by_video[v] for v in vids])
    H = np.stack([histogram(tokens_by_video[v], num_words) for v in vids])

    sims = H @ H.T
    np.fill_diagonal(sims, -np.inf)  # never retrieve the query itself

    sources = [base_video_id(v) for v in vids]
    sources_by_class = defaultdict(set)
    for source, label in zip(sources, labels):
        sources_by_class[label].add(source)
    if query_videos is not None:
        query_videos = expand_source_ids(vids, query_videos)
    queries = [
        i
        for i, v in enumerate(vids)
        if len(sources_by_class[labels[i]]) >= 2 and (query_videos is None or v in query_videos)
    ]

    if not queries:
        # every class is a singleton: nothing can be retrieved. Return a fully
        # populated dict so callers that print fields do not KeyError after
        # having already paid for motion-word extraction.
        out = {
            "n_queries": 0,
            "n_corpus": len(vids),
            "n_classes_queried": 0,
            "mAP": 0.0,
            "MRR": 0.0,
            "mAP_random_floor": 0.0,
        }
        for k in ks:
            out[f"R@{k}"] = 0.0
            out[f"Hit@{k}"] = 0.0
        out.update(codebook_stats(tokens_by_video, num_words))
        return out

    out = ranking_metrics(sims, labels, queries, query_videos=vids, ks=ks, groups=sources)
    out.update(codebook_stats(tokens_by_video, num_words))
    return out
