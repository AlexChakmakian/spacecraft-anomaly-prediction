"""Window features for ESA Mission 1: per-channel summaries of the last 5 and 60 minutes at every 30s grid time.

Row t uses only telemetry in [t - WINDOW, t]. No standardization: tree models are unaffected by per-channel scaling.
"""

import os

import numpy as np
import pandas as pd

from clean_mission1 import FREQ, GRID_END, GRID_START, OUT_DIR, load_cleaned
from label_mission1 import WINDOW, load_labels

STEP = pd.Timedelta(FREQ)
SHORT = pd.Timedelta(minutes=5)
NEG_RATE = 0.01  # share of valid negative times kept for training
TRAIN_PATH = f'{OUT_DIR}/train_features.parquet'


def window_features(x):
    """x: cleaned telemetry (time x channel). Rows closer than WINDOW to the start of x come out NaN."""
    k_long, k_short = int(WINDOW / STEP) + 1, int(SHORT / STEP) + 1
    long, short = x.rolling(k_long), x.rolling(k_short)
    out = pd.concat({
        'last': x,
        'd5': x - x.shift(k_short - 1),
        'dev60': x - long.mean(),
        'std5': short.std(),
        'std60': long.std(),
        'range60': long.max() - long.min(),
    }, axis=1).astype('float32')
    out.columns = [f'{ch}_{feat}' for feat, ch in out.columns]
    return out


def iter_features(start, end):
    """Yield features for times in [start, end), one calendar month at a time."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    x = load_cleaned(max(start - WINDOW, pd.Timestamp(GRID_START)), end - STEP)
    edges = [start, *pd.date_range(start, end, freq='MS', inclusive='neither'), end]
    for a, b in zip(edges[:-1], edges[1:]):
        f = window_features(x.loc[a - WINDOW:b - STEP])
        yield f.loc[a:]


def build_training_table(labels):
    """All valid times with an anomaly within 30 min (covers every horizon) plus NEG_RATE of the other valid times."""
    rand = np.random.default_rng(0).random(len(labels)) < NEG_RATE
    keep = labels.valid & (labels.y_30 | rand)
    parts = []
    for year in range(pd.Timestamp(GRID_START).year, pd.Timestamp(GRID_END).year):
        for f in iter_features(f'{year}-01-01', f'{year + 1}-01-01'):
            parts.append(f[keep.reindex(f.index).to_numpy()])
        print(f'{year}: {sum(map(len, parts)):,} rows', flush=True)
    X = pd.concat(parts)
    X.to_parquet(TRAIN_PATH)
    return X


def load_training_table(labels=None):
    if os.path.exists(TRAIN_PATH):
        return pd.read_parquet(TRAIN_PATH)
    return build_training_table(load_labels() if labels is None else labels)


if __name__ == '__main__':
    build_training_table(load_labels())
