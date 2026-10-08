"""Shared frozen-retrieval scoring, AP floors and deterministic bootstrap summaries.

These functions consume supplied arrays/records. They perform no experiment
admission, artifact reads, fitting or deployment selection.
"""
from __future__ import annotations

from numbers import Integral

import numpy as np

from motionprov.core.pose.local_records import CodeRecord


class _VectorScorer:
    def __init__(self, query, gallery, *, normalize=True):
        self.query = np.asarray(query, dtype=np.float64)
        self.gallery = np.asarray(gallery, dtype=np.float64)
        if normalize:
            self.query = self.query / np.linalg.norm(self.query, axis=1, keepdims=True).clip(1e-12)
            self.gallery = self.gallery / np.linalg.norm(self.gallery, axis=1, keepdims=True).clip(1e-12)

    def pair_scores(self, query_codes, gallery_codes, kernel='centroid_cosine'):
        return self.query[query_codes[:, 0]] @ self.gallery[gallery_codes[:, 0]].T


def _row_record(record):
    if len(record.codes) > 65536:
        raise ValueError('Float reference row address exceeds uint16; split this long record explicitly')
    arrays = {name: getattr(record, name) for name in CodeRecord.ARRAY_FIELDS if name != 'codes'}
    metadata = dict(record.metadata)
    metadata.pop('num_words', None)
    metadata['num_books'] = 1
    return CodeRecord(codes=np.arange(len(record.codes), dtype=np.uint16)[:, None],
                      metadata=metadata, **arrays)


def analytic_random_ap(n_candidates, n_relevant_retrievable, n_relevant_total):
    """Expected AP for random order of eligible candidates, fixed total positives."""
    n, r, total = n_candidates, n_relevant_retrievable, n_relevant_total
    if (any(isinstance(v, bool) or not isinstance(v, Integral) or v < 0 for v in (n, r, total))
            or r > n or r > total):
        raise ValueError('Random AP requires nonnegative counts and attainable positive counts')
    if not n or not r or not total:
        return 0.0
    if n == 1:
        return r/total
    harmonic = sum(1/j for j in range(1, n+1))
    return float((r/n) * (harmonic+(r-1)*(n-harmonic)/(n-1)) / total)


def bootstrap_summary(values, classes, *, draws=2000, seed=20260914):
    values = np.asarray(values, dtype=np.float64)
    classes = np.asarray(classes)
    if (values.ndim != 1 or classes.shape != values.shape or not len(values)
            or not np.isfinite(values).all() or type(draws) is not int or draws < 1):
        raise ValueError('Bootstrap requires aligned nonempty finite values/classes and positive draws')
    rng = np.random.default_rng(seed)
    query_means = values[rng.integers(0, len(values), (draws, len(values)))].mean(1)
    unique = np.unique(classes)
    groups = [values[classes == label] for label in unique]
    totals = np.array([g.sum() for g in groups])
    sizes = np.array([len(g) for g in groups])
    selected = rng.integers(0, len(groups), (draws, len(groups)))
    class_means = totals[selected].sum(1)/sizes[selected].sum(1)
    return {'draws': draws, 'seed': seed, 'query_ci95': np.quantile(query_means, [.025, .975]).tolist(),
            'class_cluster_ci95': np.quantile(class_means, [.025, .975]).tolist(),
            'n_queries': len(values), 'n_classes': len(groups),
            'scope': 'conditional on frozen trained seeds; not population equivalence evidence'}


def retrieval_metrics(scores, valid, ids, labels, source_groups):
    """Seeded ranking identical to ranking_metrics when every pair is eligible.

    Invalid pairs are omitted from the list but remain in the total positive
    denominator. They never gain random-tail AP merely by having score -inf.
    """
    ids = tuple(ids)
    scores, valid = np.asarray(scores), np.asarray(valid)
    if (not ids or len(set(ids)) != len(ids)
            or scores.shape != (len(ids), len(ids)) or valid.shape != scores.shape
            or valid.dtype != np.bool_ or scores.dtype.kind != 'f'
            or not np.isfinite(scores[valid]).all()
            or not set(ids) <= labels.keys() or not set(ids) <= source_groups.keys()):
        raise ValueError('Unique IDs, square scores/bool validity, labels and finite eligible scores required')
    labels = np.asarray([labels[v] for v in ids])
    groups = np.asarray([source_groups[v] for v in ids])
    tie = np.random.default_rng(1).permutation(len(ids))
    rows, singleton = [], []
    for i, video in enumerate(ids):
        admissible = groups != groups[i]
        relevant_total = int(((labels == labels[i]) & admissible).sum())
        if not relevant_total:
            singleton.append(video)
            continue
        candidates = np.flatnonzero(admissible & valid[i])
        order = candidates[np.lexsort((tie[candidates], -scores[i, candidates]))]
        relevant = labels[order] == labels[i]
        precision = np.cumsum(relevant)/np.arange(1, len(relevant)+1)
        ap = float(precision[relevant].sum()/relevant_total)
        correct = np.flatnonzero(relevant)
        floor = analytic_random_ap(len(order), int(relevant.sum()), relevant_total)
        rows.append({'video_id': video, 'choreography_id': str(labels[i]), 'AP': ap,
                     'RR': 1/float(correct[0]+1) if len(correct) else 0.0,
                     'eligible_candidates': len(order), 'positive_candidates_total': relevant_total,
                     'positive_candidates_eligible': int(relevant.sum()), 'random_AP': floor,
                     'hit1': bool(len(relevant) and relevant[0]),
                     'recall5': float(relevant[:5].sum()/relevant_total),
                     'status': 'ranked' if len(order) else 'insufficient_evidence'})
    if not rows:
        raise ValueError('Partition contains no queries with a cross-source positive')
    aps = [r['AP'] for r in rows]
    return {'n_queries': len(rows), 'n_corpus': len(ids),
            'n_classes_queried': len({r['choreography_id'] for r in rows}),
            'mAP': float(np.mean(aps)), 'MRR': float(np.mean([r['RR'] for r in rows])),
            'Hit@1': float(np.mean([r['hit1'] for r in rows])),
            'R@5': float(np.mean([r['recall5'] for r in rows])),
            'mAP_random_floor': float(np.mean([r['random_AP'] for r in rows])),
            'random_floor_method': 'analytic eligible-candidate permutation; same positive denominator',
            'query_videos': [r['video_id'] for r in rows], 'per_query_AP': aps, 'per_query': rows,
            'singletons_without_available_positive': singleton,
            'insufficient_evidence_queries': sum(r['status'] != 'ranked' for r in rows),
            'bootstrap': bootstrap_summary(aps, [r['choreography_id'] for r in rows])}
