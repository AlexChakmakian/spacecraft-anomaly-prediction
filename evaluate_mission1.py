"""Event-level scoring of Mission 1 anomaly prediction at a controlled false-alarm rate.

A model gives one score per grid time (higher = anomaly more likely); valid times with score >= threshold alarm.
Alarms are counted per 30-minute clock block, so an alarm left on costs 48 false alarms a day. For horizon h a
block is a hit if any of its alarms is within h minutes before an anomaly onset, otherwise it is one false alarm.
An anomaly is caught if any alarm hits it; its lead time is how long before onset the earliest hitting alarm came.
"""

import numpy as np
import pandas as pd

from clean_mission1 import FREQ
from label_mission1 import HORIZONS

STEP = pd.Timedelta(FREQ)
BLOCK = pd.Timedelta(minutes=30)
FAR_TARGETS = [0.1, 0.5, 1, 2]  # false alarms per day


def _period(scores, labels, period):
    lo, hi = labels.index.searchsorted(period[0]), labels.index.searchsorted(period[1])
    return np.asarray(scores)[lo:hi], labels.iloc[lo:hi]


def score_period(scores, labels, period, h, threshold):
    s, lab = _period(scores, labels, period)
    idx = np.flatnonzero(lab.valid.to_numpy() & (s >= threshold))
    minutes = lab.minutes_to_onset.to_numpy()[idx]
    hit = minutes <= h
    blocks, block = np.unique(idx // int(BLOCK / STEP), return_inverse=True)
    block_hit = np.bincount(block, weights=hit, minlength=len(blocks)) > 0
    ids = lab.next_id.to_numpy()[idx][hit]
    return {'false_alarms': int((~block_hit).sum()),
            'alarm_blocks': len(blocks),
            'days': lab.valid.sum() * STEP / pd.Timedelta(days=1),
            'lead': pd.Series(minutes[hit], index=ids).groupby(level=0).max()}


def threshold_for_far(scores, labels, period, h, target):
    """Lowest threshold whose false alarms per day in `period` stay at or below `target` (binary search)."""
    s, lab = _period(scores, labels, period)
    s = np.sort(s[lab.valid.to_numpy() & ~np.isnan(s)])
    at = lambda k: s[k] if k < len(s) else np.inf
    lo, hi = 0, len(s)
    while lo < hi:
        mid = (lo + hi) // 2
        r = score_period(scores, labels, period, h, at(mid))
        if r['false_alarms'] <= target * r['days']:
            hi = mid
        else:
            lo = mid + 1
    return at(lo)


def evaluate(scores, labels, events, fold, far_targets=FAR_TARGETS, horizons=HORIZONS):
    """Pick thresholds on the fold's calibration year, then score its test block.
    Returns one row per (horizon, target, test anomaly) and one summary row per (horizon, target)."""
    test = events[(events.Category == 'Anomaly') & events.onset.between(*fold['test'], inclusive='left')]
    per_event, summary = [], []
    for h in horizons:
        eligible = test[test[f'valid_pos_{h}'] > 0]
        for far in far_targets:
            threshold = threshold_for_far(scores, labels, fold['calib'], h, far)
            r = score_period(scores, labels, fold['test'], h, threshold)
            lead = r['lead'].reindex(eligible.index)
            per_event.append(pd.DataFrame({'fold': fold['name'], 'horizon': h, 'far_target': far,
                                           'Class': eligible.Class, 'caught': lead.notna(), 'lead_min': lead}))
            summary.append({'fold': fold['name'], 'horizon': h, 'far_target': far, 'threshold': threshold,
                            **{k: r[k] for k in ('false_alarms', 'alarm_blocks', 'days')}})
    return pd.concat(per_event), pd.DataFrame(summary)


def bootstrap_ci(x, n=2000, seed=0):
    means = np.random.default_rng(seed).choice(np.asarray(x, float), (n, len(x))).mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def pool(per_event, summary, by=()):
    """Pool all folds: recall with a 95% bootstrap CI over events, median lead, and test false alarms per day."""
    keys = ['horizon', 'far_target', *by]
    g = per_event.groupby(keys)
    out = g.agg(events=('caught', 'size'), recall=('caught', 'mean'), median_lead_min=('lead_min', 'median'))
    out[['recall_lo', 'recall_hi']] = g.caught.apply(bootstrap_ci).tolist()
    s = summary.groupby(['horizon', 'far_target'])[['false_alarms', 'alarm_blocks', 'days']].sum()
    rates = pd.DataFrame({'test_far_per_day': s.false_alarms / s.days,
                          'precision': 1 - s.false_alarms / s.alarm_blocks.where(s.alarm_blocks > 0)})
    return out.join(rates)
