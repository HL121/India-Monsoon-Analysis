from pathlib import Path

import numpy as np
import pandas as pd


MISSING_SENTINELS = {999: np.nan, 999.0: np.nan, 1e36: np.nan, 1e36 + 0.0: np.nan}


def read_miso(path):
    df = pd.read_csv(path, sep=r"\s+", header=0, engine="python")
    df = df.rename(
        columns={
            "Date": "date",
            "Year": "year",
            "DayIndex": "dayindex",
            "MISO1": "miso1",
            "MISO2": "miso2",
            "Amplitude": "amp",
            "Phase": "phase",
        }
    )

    for c in ["year", "dayindex", "miso1", "miso2", "amp", "phase"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date").sort_index()
    df = df.replace(MISSING_SENTINELS)
    return df[["miso1", "miso2", "amp"]].astype("float32")


def read_mjo(path, prithvi_start="1980-01-01", prithvi_end="2021-12-31"):
    daily_index = pd.date_range(prithvi_start, prithvi_end, freq="D")
    df = pd.read_csv(
        path,
        sep=r"\s+",
        skiprows=2,
        header=None,
        usecols=[0, 1, 2, 3, 4, 6],
        names=["year", "month", "day", "rmm1", "rmm2", "amp"],
        engine="python",
    )
    for c in ["year", "month", "day", "rmm1", "rmm2", "amp"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = pd.to_datetime(
        dict(year=df.year, month=df.month, day=df.day), errors="coerce"
    )
    df = df.dropna(subset=["date"]).set_index("date").sort_index()
    df = df.replace(MISSING_SENTINELS)
    df = df[["rmm1", "rmm2", "amp"]].astype("float32")
    daily = df.reindex(daily_index).interpolate(
        method="time", limit=3, limit_direction="both"
    )
    daily.index.name = "time"
    return daily.astype("float32")


def read_iod(path):
    df = pd.read_csv(path)
    df = df.rename(columns={"time": "date", "anom": "iod_anom"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["iod_anom"] = pd.to_numeric(df["iod_anom"], errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date").sort_index()
    return df[["iod_anom"]].astype("float32")


def read_target(config):
    target_type = config["target"]["type"].lower()
    path = config["target"]["path"]
    if target_type == "miso":
        return read_miso(path)
    if target_type == "mjo":
        return read_mjo(
            path,
            prithvi_start=config["target"].get("prithvi_start", "1980-01-01"),
            prithvi_end=config["target"].get("prithvi_end", "2021-12-31"),
        )
    if target_type == "iod":
        return read_iod(path)
    raise ValueError(f"Unsupported target type: {target_type}")


def load_latent_files(latent_dir, prefix, months):
    files = sorted(Path(latent_dir).glob(f"{prefix}_*.npy"))
    vectors = [np.load(f) for f in files]
    times = pd.to_datetime([f.stem.replace(f"{prefix}_", "") for f in files])
    mask = times.month.isin(months)
    vectors = [v for v, keep in zip(vectors, mask) if keep]
    times = times[mask]
    return np.stack(vectors, axis=0), times


def load_latents(latent_base, months):
    all_e, all_d, all_times, year_labels = [], [], [], []
    year_dirs = sorted(
        [d for d in Path(latent_base).iterdir() if d.is_dir() and d.name.isdigit()]
    )
    for year_dir in year_dirs:
        arr_e, times_e = load_latent_files(year_dir, "encoder", months)
        arr_d, times_d = load_latent_files(year_dir, "decoder", months)
        assert times_e.equals(times_d), (
            f"Encoder and decoder timestamps don't match in {year_dir}!"
        )
        year = int(year_dir.name)
        all_e.append(arr_e)
        all_d.append(arr_d)
        all_times.append(times_e)
        year_labels.extend([year] * len(times_e))

    return {
        "encoder": np.vstack(all_e),
        "decoder": np.vstack(all_d),
        "times": pd.DatetimeIndex(np.concatenate([t.values for t in all_times])),
        "years": np.array(year_labels),
    }


def align_target_to_latents(target_df, latent_times, columns):
    latent_dates = latent_times.normalize()
    return target_df.reindex(latent_dates)[columns]


def make_sequences(X, y, years, seq_len):
    X_seq, y_seq, seq_years = [], [], []
    for yr in np.unique(years):
        idx = np.where(years == yr)[0]
        X_year = X[idx]
        y_year = y[idx]
        for i in range(len(X_year) - seq_len):
            X_seq.append(X_year[i : i + seq_len])
            y_seq.append(y_year[i + seq_len])
            seq_years.append(yr)
    return np.array(X_seq), np.array(y_seq), np.array(seq_years)


def standardize_sequences(X_train, X_test):
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    return (X_train - mean) / std, (X_test - mean) / std


def prepare_sequences(config, seq_len):
    target = read_target(config)
    latent = load_latents(config["paths"]["latent_base"], config["data"]["months"])
    target_columns = config["target"]["columns"]
    aligned = align_target_to_latents(target, latent["times"], target_columns)
    y_all = aligned.values

    years = latent["years"]
    year_use = np.array(config["data"]["years"])
    mask = np.isin(years, year_use) & np.isfinite(y_all).all(axis=1)

    y = y_all[mask]
    years_use = years[mask]
    X_e = latent["encoder"][mask]
    X_d = latent["decoder"][mask]

    X_e_seq, y_seq, seq_years = make_sequences(X_e, y, years_use, seq_len)
    X_d_seq, _, _ = make_sequences(X_d, y, years_use, seq_len)
    return {
        "encoder": X_e_seq,
        "decoder": X_d_seq,
        "target": y_seq,
        "years": seq_years,
        "target_columns": target_columns,
    }

