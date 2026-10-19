"""Geospatial foundation models: TerraMind (run locally) and TESSERA (pre-computed embeddings).

TerraMind 1.0 (IBM, ESA, Forschungszentrum Jülich; arXiv:2504.11171) is run on 224 x 224 chips of
Sentinel-2 L2A summer composites at 10 m. Inputs are DN without the +1000 baseline offset
(reflectance x 10000), which is what the model expects when no `offset` is passed.
"""
from __future__ import annotations

import numpy as np
import odc.stac
import xarray as xr

from . import pipeline as pl

BANDS12 = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
CHIP = 224
PATCH = 16
# Class order of TerraMind's LULC output = ESRI / Impact Observatory annual LULC v02 classes without
# their gaps (codes 1,2,4,5,7,8,9,10,11 -> 1..9). Verified against the real io-lulc map in the notebook.
TM_LULC = {0: "no data", 1: "water", 2: "trees", 3: "flooded vegetation", 4: "crops", 5: "built area",
           6: "bare ground", 7: "snow/ice", 8: "clouds", 9: "rangeland"}
IO_TO_TM = {0: 0, 1: 1, 2: 2, 4: 3, 5: 4, 7: 5, 8: 6, 9: 7, 10: 8, 11: 9}


def composite_s2_12(bbox, year: int, season=("07-01", "09-30"), resolution: int = 10) -> xr.DataArray:
    """12-band summer median composite (band, y, x), DN without offset, SCL-masked."""
    items = pl.search_scenes(bbox, f"{year}-{season[0]}", f"{year}-{season[1]}")
    ds = odc.stac.load(items, bands=BANDS12 + ["SCL"], bbox=bbox, crs=pl.CRS, resolution=resolution,
                       groupby="solar_day", chunks={"x": 1024, "y": 1024})
    pb = {it.datetime.date(): it.properties.get("s2:processing_baseline", "0") for it in items}
    off = xr.DataArray([pl.OFFSET_DN if pb.get(t.date(), "0") >= pl.OFFSET_BASELINE else 0.0
                        for t in ds.time.to_index()], dims="time", coords={"time": ds.time})
    valid = ds.SCL.isin(pl.SCL_VALID)
    arr = xr.concat([(ds[b].astype("float32") - off).where(valid & (ds[b] > 0)) for b in BANDS12],
                    dim=xr.DataArray(BANDS12, dims="band", name="band"))
    return arr.median("time").compute()


def _chips(arr: np.ndarray):
    """Split (C, H, W) into 224 x 224 chips (edge chips shifted inward); yields (i0, j0, chip)."""
    _, H, W = arr.shape
    rows = list(range(0, max(H - CHIP, 0) + 1, CHIP)) + ([H - CHIP] if H % CHIP else [])
    cols = list(range(0, max(W - CHIP, 0) + 1, CHIP)) + ([W - CHIP] if W % CHIP else [])
    for i0 in rows:
        for j0 in cols:
            yield i0, j0, arr[:, i0:i0 + CHIP, j0:j0 + CHIP]


def _fill(a: np.ndarray) -> np.ndarray:
    """Replace NaN (masked pixels) by the per-band median of the chip."""
    med = np.nanmedian(a.reshape(a.shape[0], -1), axis=1)
    return np.where(np.isnan(a), med[:, None, None], a)


def load_terramind(kind: str):
    """'generate' -> LULC generator; 'backbone' -> encoder returning patch embeddings."""
    from terratorch import BACKBONE_REGISTRY, FULL_MODEL_REGISTRY

    if kind == "generate":
        m = FULL_MODEL_REGISTRY.build("terramind_v1_base_generate", pretrained=True, modalities=["S2L2A"],
                                      output_modalities=["LULC"], timesteps=10, standardize=True)
    else:
        m = BACKBONE_REGISTRY.build("terramind_v1_base", pretrained=True, modalities=["S2L2A"])
    return m.eval()


def terramind_lulc(model, comp: xr.DataArray, seed: int = 0) -> xr.DataArray:
    """Generated land-cover class per 10 m pixel (TM_LULC codes), on the grid of `comp`."""
    import torch

    torch.manual_seed(seed)  # generation is stochastic (diffusion steps); fixed seed for reproducibility
    a = comp.values
    out = np.zeros(a.shape[1:], dtype="uint8")
    with torch.no_grad():
        for i0, j0, c in _chips(a):
            x = torch.from_numpy(_fill(c)).float().unsqueeze(0)
            lulc = model({"S2L2A": x})["LULC"][0]
            out[i0:i0 + CHIP, j0:j0 + CHIP] = lulc.argmax(0).numpy().astype("uint8")
    return xr.DataArray(out, coords={"y": comp.y, "x": comp.x}, dims=("y", "x"), name="terramind_lulc")


def terramind_embeddings(model, comp: xr.DataArray) -> xr.DataArray:
    """768-d patch embeddings (last encoder layer), one per 16 x 16 pixels (160 m at 10 m)."""
    import torch

    a = comp.values
    H, W = a.shape[1] // PATCH, a.shape[2] // PATCH
    out = np.full((H, W, 768), np.nan, dtype="float32")
    with torch.no_grad():
        for i0, j0, c in _chips(a):
            x = torch.from_numpy(_fill(c)).float().unsqueeze(0)
            # standardise with the pre-training statistics (the backbone does not do it itself)
            x = (x - _MEAN) / _STD
            e = model({"S2L2A": x})
            e = e[-1] if isinstance(e, (list, tuple)) else e        # last layer: (1, 196, 768)
            e = e[0].reshape(CHIP // PATCH, CHIP // PATCH, -1).numpy()
            pi, pj = i0 // PATCH, j0 // PATCH
            out[pi:pi + CHIP // PATCH, pj:pj + CHIP // PATCH] = e[:H - pi, :W - pj]
    ys = comp.y.values[PATCH // 2::PATCH][:H]
    xs = comp.x.values[PATCH // 2::PATCH][:W]
    return xr.DataArray(out, coords={"y": ys, "x": xs, "dim": np.arange(768)}, dims=("y", "x", "dim"))


def _stats():
    import torch
    from terratorch.models.backbones.terramind.model.terramind_register import (v1_pretraining_mean,
                                                                                v1_pretraining_std)
    k = "untok_sen2l2a@224"
    return (torch.tensor(v1_pretraining_mean[k]).view(1, -1, 1, 1),
            torch.tensor(v1_pretraining_std[k]).view(1, -1, 1, 1))


try:
    _MEAN, _STD = _stats()
except Exception:  # terratorch not installed: the GFM section of the notebook cannot run
    _MEAN = _STD = None


def io_lulc(bbox, year: int, like: xr.DataArray) -> xr.DataArray:
    """ESRI / Impact Observatory annual LULC (10 m) on the grid of `like`, recoded to TM_LULC codes."""
    item = next(pl.catalog().search(collections=["io-lulc-annual-v02"], bbox=bbox,
                                    datetime=f"{year}-01-01/{year}-12-31").items())
    m = odc.stac.load([item], bands=["data"], bbox=bbox, crs=pl.CRS, resolution=10,
                      resampling="nearest")["data"].isel(time=0)
    m = m.interp(x=like.x, y=like.y, method="nearest")
    lut = np.zeros(256, dtype="uint8")
    for k, v in IO_TO_TM.items():
        lut[k] = v
    return xr.DataArray(lut[m.values.astype("uint8")], coords={"y": like.y, "x": like.x}, dims=("y", "x"),
                        name="io_lulc")


def cosine_change(a: xr.DataArray, b: xr.DataArray) -> xr.DataArray:
    """1 - cosine similarity between two embedding grids (..., dim): 0 = unchanged, up to 2 = opposite."""
    num = (a * b).sum("dim")
    den = np.sqrt((a * a).sum("dim")) * np.sqrt((b * b).sum("dim"))
    return (1 - num / den).rename("embedding_change")


def cells_canopy_loss(cells, flag: xr.DataArray, lulc_before: xr.DataArray, lulc_after: xr.DataArray,
                      min_px: int = 25):
    """Per cell: share of flagged pixels mapped as trees before and no longer as trees after.

    `flag` (20 m) is brought to the 10 m LULC grid by nearest neighbour. Cells with fewer than
    `min_px` flagged tree pixels get NaN (too few to say). Returns a pandas Series aligned to `cells`.
    """
    import pandas as pd

    f = flag.astype("uint8").interp(x=lulc_before.x, y=lulc_before.y, method="nearest").values == 1
    trees_before = lulc_before.values == 2
    not_trees_after = lulc_after.values != 2
    xs, ys = lulc_before.x.values, lulc_before.y.values
    out = []
    for geom in cells.geometry:
        x0, y0, x1, y1 = geom.bounds
        j = (xs >= x0) & (xs < x1)
        i = (ys >= y0) & (ys < y1)
        base = (f & trees_before)[np.ix_(i, j)]
        n = int(base.sum())
        out.append(float(not_trees_after[np.ix_(i, j)][base].mean()) if n >= min_px else np.nan)
    return pd.Series(out, index=cells.index, name="canopy_no_longer_trees")


def tessera_change_and_clusters(bbox, like: xr.DataArray, years=(2023, 2025), k: int = 6,
                                cache_dir: str = "data/tessera", sample_per_tile: int = 20000, seed: int = 0):
    """TESSERA (geotessera, dataset v1) annual embeddings over `bbox`.

    Returns, on the grid of `like` (e.g. the 20 m alert grid):
      change   - 1 - cosine(embedding years[0], embedding years[1]), averaged from 10 m
      clusters - k-means cluster of the years[0] embedding (unsupervised land / forest types), mode from 10 m
    Embedding version v1 is used because v1.1 does not cover Sardinia; versions must not be mixed.
    """
    import os

    import rioxarray  # noqa: F401
    from geotessera import GeoTessera
    from rasterio.enums import Resampling
    from sklearn.cluster import MiniBatchKMeans

    os.makedirs(cache_dir, exist_ok=True)
    gt = GeoTessera(dataset_version="v1", cache_dir=f"{cache_dir}/cache", embeddings_dir=f"{cache_dir}/emb")
    emb = {}
    for y in years:
        tiles = gt.registry.load_blocks_for_region(bounds=bbox, year=y)
        for yy, lon, lat, e, crs, tr in gt.fetch_embeddings(tiles):
            emb[(y, round(lon, 2), round(lat, 2))] = (e, crs, tr)
    keys = sorted({(lon, lat) for (_, lon, lat) in emb if all((y, lon, lat) in emb for y in years)})

    def to_grid(arr, crs, tr, resampling, nodata=np.nan):
        H, W = arr.shape
        da = xr.DataArray(arr, dims=("y", "x"),
                          coords={"x": tr.c + tr.a * (np.arange(W) + 0.5), "y": tr.f + tr.e * (np.arange(H) + 0.5)})
        da = da.rio.write_crs(crs).rio.write_transform(tr).rio.write_nodata(nodata)
        return da.rio.reproject_match(like, resampling=resampling)

    rng = np.random.default_rng(seed)
    changes, samples = [], []
    for lon, lat in keys:
        a, crs, tr = emb[(years[0], lon, lat)]
        b = emb[(years[1], lon, lat)][0]
        den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
        chg = (1 - (a * b).sum(-1) / np.where(den == 0, np.nan, den)).astype("float32")
        changes.append(to_grid(chg, crs, tr, Resampling.average))
        flat = a.reshape(-1, a.shape[-1])
        samples.append(flat[rng.choice(len(flat), size=min(sample_per_tile, len(flat)), replace=False)])
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=5).fit(np.concatenate(samples))
    labels = []
    for lon, lat in keys:
        a, crs, tr = emb[(years[0], lon, lat)]
        lab = km.predict(a.reshape(-1, a.shape[-1])).reshape(a.shape[:2]).astype("float32")
        labels.append(to_grid(lab, crs, tr, Resampling.mode))
    # tiles do not overlap: outside its tile each reprojected layer is NaN, so max() mosaics them
    change = xr.concat(changes, "t").max("t").assign_coords(x=like.x, y=like.y).rename("tessera_change")
    clusters = xr.concat(labels, "t").max("t").assign_coords(x=like.x, y=like.y).rename("tessera_cluster")
    return change.drop_vars("spatial_ref", errors="ignore"), clusters.drop_vars("spatial_ref", errors="ignore")
