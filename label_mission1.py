"""Label every 30s grid time of ESA Mission 1 for 5/15/30 min anomaly prediction, and define time-based folds.

At time t the model may use telemetry in [t - WINDOW, t]. The label for horizon h is 1 if an anomaly
begins in (t, t + h]. A time is valid only if its input window is complete and touches no labeled event.
"""

import os

import numpy as np
import pandas as pd

from clean_mission1 import DATA_DIR, FREQ, GRID_END, GRID_START, OUT_DIR

WINDOW = pd.Timedelta(minutes=60)
HORIZONS = [5, 15, 30]  # minutes
TEST_BLOCKS = [(2006, 2007), (2008, 2009), (2010, 2011), (2012, 2013)]
MAIN_CLASSES = ['class_7', 'class_3']  # only classes with enough events after 2005
STATUS_CHANNELS = [f'channel_{i}' for i in (53, 54, 55, 56, 67, 68, 69)]  # <= 16 distinct values


def load_events(data_dir=DATA_DIR):
    """One row per event. Onset = earliest start over all affected channels."""
    labels = pd.read_csv(f'{data_dir}/labels.csv', parse_dates=['StartTime', 'EndTime'])
    types = pd.read_csv(f'{data_dir}/anomaly_types.csv').set_index('ID')
    events = labels.groupby('ID').agg(onset=('StartTime', 'min'), end=('EndTime', 'max')).join(types)
    events[['onset', 'end']] = events[['onset', 'end']].apply(lambda c: c.dt.tz_localize(None))
    return events.sort_values('onset')


def window_touches_event(grid, events):
    """True where [t - WINDOW, t] overlaps any event of any category, i.e. onset <= t <= end + WINDOW."""
    diff = np.zeros(len(grid) + 1, dtype=np.int32)
    np.add.at(diff, grid.searchsorted(events.onset, 'left'), 1)
    np.add.at(diff, grid.searchsorted(events.end + WINDOW, 'right'), -1)
    return np.cumsum(diff[:-1]) > 0


def window_has_nan(data_dir=OUT_DIR):
    """True where any channel is missing anywhere in [t - WINDOW, t] (or the window starts before the grid)."""
    nan = np.concatenate([pd.read_parquet(f'{data_dir}/{y}.parquet').isna().any(axis=1).to_numpy()
                          for y in range(pd.Timestamp(GRID_START).year, pd.Timestamp(GRID_END).year)])
    k = int(WINDOW / pd.Timedelta(FREQ)) + 1
    count = np.cumsum(np.concatenate([[0], nan]))
    out = np.ones(len(nan), dtype=bool)
    out[k - 1:] = (count[k:] - count[:-k]) > 0
    return out


def build_labels(events):
    grid = pd.date_range(GRID_START, GRID_END, freq=FREQ, inclusive='left')
    anomalies = events[events.Category == 'Anomaly']
    j = anomalies.onset.searchsorted(grid, 'right')  # next anomaly starting strictly after t
    has_next = j < len(anomalies)
    j = np.minimum(j, len(anomalies) - 1)
    minutes = np.where(has_next, (anomalies.onset.to_numpy()[j] - grid.to_numpy()) / np.timedelta64(1, 'm'), np.inf)

    df = pd.DataFrame({
        'valid': ~window_touches_event(grid, events) & ~window_has_nan(),
        'minutes_to_onset': minutes.astype('float32'),
        'next_id': pd.Categorical(np.where(has_next, anomalies.index.to_numpy()[j], None)),
    }, index=pd.Index(grid, name='time'))
    for h in HORIZONS:
        df[f'y_{h}'] = df.minutes_to_onset <= h
    return df


def folds(train_start=GRID_START):
    """Expanding-window folds: train, then one calibration year for picking alarm thresholds, then test.
    Each period stops WINDOW + max horizon before the next one so no window or label crosses a boundary."""
    gap = WINDOW + pd.Timedelta(minutes=max(HORIZONS))
    for first, last in TEST_BLOCKS:
        calib_start = pd.Timestamp(f'{first - 1}-01-01')
        test_start, test_end = pd.Timestamp(f'{first}-01-01'), pd.Timestamp(f'{last + 1}-01-01')
        yield {'name': f'test_{first}_{last}',
               'train': (pd.Timestamp(train_start), calib_start - gap),
               'calib': (calib_start, test_start - gap),
               'test': (test_start, test_end)}


def load_labels(data_dir=OUT_DIR):
    return pd.read_parquet(f'{data_dir}/labels.parquet')


def main():
    events = load_events()
    df = build_labels(events)
    df.to_parquet(f'{OUT_DIR}/labels.parquet')

    pos = df[df.valid & df.y_30]
    counts = pd.DataFrame({h: pos[pos.minutes_to_onset <= h].next_id.value_counts() for h in HORIZONS})
    events = events.join(counts.add_prefix('valid_pos_'))
    events['test_block'] = None
    for f in folds():
        in_test = events.onset.between(*f['test'], inclusive='left')
        events.loc[in_test, 'test_block'] = f['name']
    events.to_csv(f'{OUT_DIR}/events.csv')

    a = events[events.Category == 'Anomaly']
    print(f'valid times {df.valid.mean():.1%}  positives (5/15/30 min) '
          f'{[int((df.valid & df[f"y_{h}"]).sum()) for h in HORIZONS]}')
    print(f'anomalies with >=1 valid positive at 30 min: {int((a.valid_pos_30 > 0).sum())} of {len(a)}')
    print(f'done -> {OUT_DIR}/labels.parquet, events.csv')


if __name__ == '__main__':
    main()
