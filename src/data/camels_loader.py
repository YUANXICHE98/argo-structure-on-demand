"""Minimal CAMELS loader used by the ARGO release scripts.

Expected files under data/camels_us:
- camels_attrs_v2_streamflow_v1p2.nc
- camels_attributes_v2.0.feather
- basin_mean_forcing/<forcing>/<huc>/<basin>_lump_<forcing>_forcing*.txt
"""

from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import Dataset


def load_camels_us(data_dir, forcing="nldas", use_531=True):
    data_dir = Path(data_dir)
    t0 = time.time()
    required = [data_dir / "camels_attrs_v2_streamflow_v1p2.nc", data_dir / "camels_attributes_v2.0.feather"]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing CAMELS files: " + ", ".join(missing))

    ds = xr.open_dataset(required[0])
    all_basin_ids = [str(x).zfill(8) for x in ds.station_id.values]
    discharge = ds["discharge"].values.astype(np.float32)
    times = pd.DatetimeIndex(ds.time.values)

    attrs = pd.read_feather(required[1])
    attrs.index = attrs.index.astype(str).str.zfill(8)
    forcing_dir = data_dir / "basin_mean_forcing" / forcing
    if not forcing_dir.exists():
        raise FileNotFoundError(f"Missing forcing directory: {forcing_dir}")

    available = {p.name[:8] for p in forcing_dir.glob("*/*.txt")}
    basins = sorted(set(all_basin_ids) & set(attrs.index) & available)
    if use_531:
        basins = [b for b in basins if np.isnan(discharge[:, all_basin_ids.index(b)]).mean() < 0.05]
    n, t = len(basins), len(times)
    features = ["Dayl", "PRCP", "SRAD", "SWE", "Tmax", "Tmin", "Vp"]
    meteo = np.full((t, n, len(features)), np.nan, dtype=np.float32)
    flow = np.full((t, n), np.nan, dtype=np.float32)

    for idx, bid in enumerate(basins):
        huc = str(attrs.loc[bid, "huc_02"]).zfill(2)
        candidates = [
            forcing_dir / huc / f"{bid}_lump_{forcing}_forcing_leap.txt",
            forcing_dir / huc / f"{bid}_lump_{forcing}_forcing.txt",
        ]
        fpath = next((p for p in candidates if p.exists()), None)
        if fpath is None:
            continue
        df = pd.read_csv(
            fpath,
            skiprows=4,
            sep="\t",
            names=["Year", "Mnth", "Day", "Hr", *features],
        )
        dates = pd.to_datetime(df[["Year", "Mnth", "Day"]].rename(columns={"Year": "year", "Mnth": "month", "Day": "day"}))
        df.index = dates
        meteo[:, idx, :] = df.reindex(times)[features].values.astype(np.float32)
        flow[:, idx] = discharge[:, all_basin_ids.index(bid)]

    numeric_cols = attrs.select_dtypes(include="number").columns.tolist()
    leakage_cols = {"q_mean", "runoff_ratio", "slope_fdc", "baseflow_index", "stream_elas", "q5", "q95", "high_q_freq", "high_q_dur", "low_q_freq", "low_q_dur", "zero_q_freq", "hfd_mean"}
    static_cols = [c for c in numeric_cols if c not in leakage_cols]
    static = np.nan_to_num(attrs.loc[basins, static_cols].values.astype(np.float32), nan=0.0)

    weeks = t // 7
    meteo_w = np.zeros((weeks, n, len(features)), dtype=np.float32)
    flow_w = np.full((weeks, n), np.nan, dtype=np.float32)
    for w in range(weeks):
        s, e = w * 7, (w + 1) * 7
        meteo_w[w, :, 1] = np.nansum(meteo[s:e, :, 1], axis=0)
        for j in [0, 2, 3, 4, 5, 6]:
            meteo_w[w, :, j] = np.nanmean(meteo[s:e, :, j], axis=0)
        flow_w[w] = np.nanmean(flow[s:e], axis=0)

    for arr in [meteo_w, flow_w]:
        mean = np.nanmean(arr, axis=0, keepdims=True)
        inds = np.where(np.isnan(arr))
        arr[inds] = np.take_along_axis(mean, inds[1][None, ...] if arr.ndim == 3 else inds[1][None, :], axis=1).reshape(-1)[: len(inds[0])]

    meteo_w = (meteo_w - meteo_w[: int(weeks * 0.7)].mean((0, 1), keepdims=True)) / (meteo_w[: int(weeks * 0.7)].std((0, 1), keepdims=True) + 1e-8)
    flow_w = np.log(np.maximum(flow_w, 0) + 0.001)
    flow_w = (flow_w - flow_w[: int(weeks * 0.7)].mean()) / (flow_w[: int(weeks * 0.7)].std() + 1e-8)
    static = (static - static.mean(0, keepdims=True)) / (static.std(0, keepdims=True) + 1e-8)

    print(f"Loaded CAMELS: {n} basins, {weeks} weeks, {time.time() - t0:.1f}s")
    return {"meteo": meteo_w, "flow": flow_w, "static": static, "basin_ids": basins, "N": n, "F": len(features), "D_s": static.shape[1]}


class CAMELSSpatialDataset(Dataset):
    def __init__(self, data, lookback=26, split="train", max_basins=None):
        self.meteo = data["meteo"]
        self.flow = data["flow"]
        self.static = data["static"]
        self.lookback = lookback
        self.n = min(max_basins or data["N"], data["N"])
        total = len(self.meteo) - lookback - 1
        a, b = int(total * 0.7), int(total * 0.85)
        ranges = {"train": range(0, a), "val": range(a, b), "test": range(b, total)}
        self.indices = list(ranges[split])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx]
        x = self.meteo[t : t + self.lookback, : self.n, :].transpose(1, 0, 2)
        y = self.flow[t + self.lookback, : self.n]
        static = self.static[: self.n]
        mask = np.isfinite(y).astype(np.float32)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(static, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)
