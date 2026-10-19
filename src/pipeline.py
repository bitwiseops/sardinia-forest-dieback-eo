"""Sentinel-2 L2A drought / dieback pipeline: search, preprocessing, indices, composites.

Every function is parametric (AOI, dates, thresholds) so the same code serves the
notebook and the MCP server.
"""
from __future__ import annotations

import numpy as np
import odc.stac
import pandas as pd
import planetary_computer
import pystac
import pystac_client
import xarray as xr

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
CRS = "EPSG:32632"  # UTM 32N, native CRS of Sardinian S2 tiles
BANDS = ["B02", "B03", "B04", "B08", "B8A", "B11", "B12"]
# Scene Classification Layer: keep 4 (vegetation) and 5 (not vegetated); drop
# 0 nodata, 1 saturated, 2 dark, 3 cloud shadow, 6 water, 7 unclassified,
# 8-9 cloud, 10 cirrus, 11 snow.
SCL_VALID = (4, 5)
# Processing baseline 04.00 (from 25 Jan 2022) stores DN = 10000 * reflectance + 1000.
OFFSET_BASELINE = "04.00"
OFFSET_DN = 1000.0


def catalog() -> pystac_client.Client:
    """Planetary Computer STAC client; asset URLs are signed automatically (no account)."""
    return pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)


def dedup_by_date(items) -> list[pystac.Item]:
    """One item per acquisition date: keep the most recent processing baseline.

    Reprocessed scenes appear twice in the catalogue with different baselines.
    """
    best: dict = {}
    for it in items:
        d = it.datetime.date()
        pb = it.properties.get("s2:processing_baseline", "0")
        if d not in best or pb > best[d].properties.get("s2:processing_baseline", "0"):
            best[d] = it
    return sorted(best.values(), key=lambda it: it.datetime)


def search_scenes(bbox, start: str, end: str, max_cloud: float = 20) -> list[pystac.Item]:
    """Sentinel-2 L2A scenes over `bbox` (lon/lat) between `start` and `end`, deduplicated."""
    items = catalog().search(collections=["sentinel-2-l2a"], bbox=bbox, datetime=f"{start}/{end}",
                             query={"eo:cloud_cover": {"lt": max_cloud}}).item_collection()
    return dedup_by_date(items)


def scenes_table(items) -> pd.DataFrame:
    return pd.DataFrame([{
        "date": it.datetime.date(),
        "tile": it.properties.get("s2:mgrs_tile"),
        "cloud_cover_%": round(it.properties.get("eo:cloud_cover", np.nan), 1),
        "baseline": it.properties.get("s2:processing_baseline"),
        "id": it.id,
    } for it in items])


def load_reflectance(items, bbox, resolution: int = 20, mask: bool = True) -> xr.Dataset:
    """Load bands as surface reflectance (0-1), removing the baseline offset and masking with SCL."""
    ds = odc.stac.load(items, bands=BANDS + ["SCL"], bbox=bbox, crs=CRS, resolution=resolution,
                       groupby="solar_day", chunks={})
    pb = {it.datetime.date(): it.properties.get("s2:processing_baseline", "0") for it in items}
    offset = xr.DataArray(
        [OFFSET_DN if pb.get(t.date(), "0") >= OFFSET_BASELINE else 0.0 for t in ds.time.to_index()],
        dims="time", coords={"time": ds.time})
    valid = ds.SCL.isin(SCL_VALID) if mask else xr.ones_like(ds.SCL, dtype=bool)
    out = xr.Dataset({b: ((ds[b].astype("float32") - offset) / 10000.0).where(valid & (ds[b] > 0))
                      for b in BANDS})
    out["valid"] = valid
    return out


def compute_indices(r: xr.Dataset) -> xr.Dataset:
    """NDVI (greenness), NDMI (canopy water), NBR (burn screening), CRswir (SWIR water absorption)."""
    return xr.Dataset({
        "ndvi": (r.B08 - r.B04) / (r.B08 + r.B04),
        "ndmi": (r.B8A - r.B11) / (r.B8A + r.B11),
        "nbr": (r.B8A - r.B12) / (r.B8A + r.B12),
        # continuum removal: B11 over the B8A-B12 line interpolated at 1610 nm (higher = drier)
        "crswir": r.B11 / (r.B8A + (1610 - 865) * (r.B12 - r.B8A) / (2190 - 865)),
    })


def forest_mask(bbox, like: xr.Dataset, resolution: int = 20) -> xr.DataArray:
    """ESA WorldCover 2021 tree cover (class 10) on the same grid as `like`."""
    item = next(catalog().search(collections=["esa-worldcover"], bbox=bbox,
                                 query={"esa_worldcover:product_version": {"eq": "2.0.0"}}).items())
    wc = odc.stac.load([item], bands=["map"], bbox=bbox, crs=CRS, resolution=resolution,
                       resampling="mode")["map"].isel(time=0)
    return (wc == 10).assign_coords(x=like.x, y=like.y).rename("forest")


def to_rgb(r: xr.Dataset, gain: float = 1 / 0.15) -> np.ndarray:
    """True-colour array (y, x, 3) clipped to 0-1 for plotting."""
    return np.clip(r[["B04", "B03", "B02"]].to_array().transpose("y", "x", "variable").values * gain, 0, 1)
