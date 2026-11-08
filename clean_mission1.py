"""Align ESA Mission 1 onto one 30s grid. Last-value hold only. No scaling or clipping."""

import gc
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, 'ESA-Mission1')
OUT_DIR = os.path.join(ROOT, 'cleaned', 'mission1_30s')
FREQ = '30s'  # ESA Mission 1 target rate (0.033 Hz)
MIN_HOLD = pd.Timedelta(minutes=10)
HOLD_STEPS = 3  # hold up to 3x a channel's typical step
GRID_START, GRID_END = '2000-01-01', '2014-01-01'


def align_channel(s, grid):
    s = s.sort_index()
    s = s[~s.index.duplicated(keep='last')]
    if s.index.tz is not None:
        s.index = s.index.tz_localize(None)
    step = s.index.to_series().diff().median()
    hold = max(MIN_HOLD, HOLD_STEPS * step)
    aligned = s.reindex(grid, method='ffill', tolerance=hold).astype('float32')
    return aligned, hold


def load_cleaned(start, end, channels=None, data_dir=OUT_DIR):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    frames = [pd.read_parquet(f'{data_dir}/{y}.parquet', columns=channels)
              for y in range(start.year, end.year + 1)]
    return pd.concat(frames).loc[start:end]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    channels = pd.read_csv(f'{DATA_DIR}/channels.csv').Channel.tolist()
    grid = pd.date_range(GRID_START, GRID_END, freq=FREQ, inclusive='left')

    mm_path = f'{OUT_DIR}/_values.f32'
    mm = np.memmap(mm_path, dtype='float32', mode='w+', shape=(len(grid), len(channels)))
    report = {}

    for i, ch in enumerate(channels):
        s = pd.read_pickle(f'{DATA_DIR}/channels/{ch}.zip')[ch]
        aligned, hold = align_channel(s, grid)
        mm[:, i] = aligned.to_numpy(np.float32)
        report[ch] = {'hold_seconds': hold.total_seconds(),
                      'nan_frac': float(np.isnan(mm[:, i]).mean())}
        print(f'{i + 1:2d}/{len(channels)}  {ch}  hold {hold}  NaN {report[ch]["nan_frac"]:.3%}',
              flush=True)

    mm.flush()
    del mm
    gc.collect()
    mm = np.memmap(mm_path, dtype='float32', mode='r', shape=(len(grid), len(channels)))

    pos = 0
    for year in range(2000, 2014):
        n = (366 if pd.Timestamp(f'{year}-01-01').is_leap_year else 365) * 24 * 120
        df = pd.DataFrame(np.array(mm[pos:pos + n]), index=grid[pos:pos + n], columns=channels)
        df.index.name = 'time'
        df.to_parquet(f'{OUT_DIR}/{year}.parquet')
        print(f'wrote {year} {df.shape}', flush=True)
        pos += n

    del mm
    gc.collect()
    os.remove(mm_path)

    meta = {
        'source': DATA_DIR,
        'freq': FREQ,
        'grid_start': GRID_START,
        'grid_end': GRID_END,
        'n_timestamps': int(len(grid)),
        'n_channels': len(channels),
        'did': [
            'drop duplicate timestamps (keep last)',
            'reindex to a 30s grid with last-value hold',
            f'hold at most max({MIN_HOLD}, {HOLD_STEPS}x typical step) per channel',
        ],
        'did_not': [
            'interpolate',
            'clip outliers or smooth',
            'standardize (fit later on training data only)',
            'remove anomalies or rare events',
            'include telecommands',
            'aggregate within a bin (last value only; spikes shorter than 30s can be lost)',
            'keep sampling density (raw rates change over time and rise before anomalies)',
        ],
        'channels': report,
    }
    with open(f'{OUT_DIR}/meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'done -> {OUT_DIR}', flush=True)


if __name__ == '__main__':
    main()
