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


# ---------------------------------------------------------------------------
# Seasonal composites, disturbance screening, anomaly / recovery, outputs
# ---------------------------------------------------------------------------

def summer_composites(bbox, years, season=("07-01", "09-30"), resolution: int = 20,
                      max_cloud: float = 20, min_valid: float = 0.5):
    """Per-year median composites of the indices, the forest mask, and a per-acquisition table.

    The table holds, for every acquisition, the median of each index over forest pixels; dates
    where less than `min_valid` of the forest is cloud-free are kept but marked unreliable.
    """
    comps, rows, forest = [], [], None
    for y in years:
        items = search_scenes(bbox, f"{y}-{season[0]}", f"{y}-{season[1]}", max_cloud)
        r = load_reflectance(items, bbox, resolution)
        idx = compute_indices(r).compute()
        if forest is None:
            forest = forest_mask(bbox, idx, resolution)
        valid_frac = r.valid.where(forest).mean(["x", "y"]).compute()
        per_date = idx.where(forest).median(["x", "y"])
        for t in idx.time.values:
            row = {"date": pd.Timestamp(t), "year": y,
                   "forest_valid_frac": round(float(valid_frac.sel(time=t)), 3)}
            row.update({v: float(per_date[v].sel(time=t)) for v in idx.data_vars})
            rows.append(row)
        comps.append(idx.median("time").expand_dims(year=[y]))
    table = pd.DataFrame(rows)
    table["reliable"] = table.forest_valid_frac >= min_valid
    return xr.concat(comps, "year"), forest, table


def abrupt_disturbance(bbox, years, like: xr.Dataset, drop: float = 0.25, persist: float = 0.15,
                       season=("06-01", "10-31"), resolution: int = 20, max_cloud: float = 20) -> xr.DataArray:
    """Pixels with a sudden NBR drop that does not rebound within the season (fire, clear-cut).

    Drought lowers NBR gradually; fire and logging cause a step. A pixel is flagged when NBR falls
    by more than `drop` between two consecutive valid observations and the maximum of all later
    observations in the season is still more than `persist` below the pre-drop value.
    """
    out = xr.zeros_like(like[list(like.data_vars)[0]], dtype=bool)
    for y in years:
        items = search_scenes(bbox, f"{y}-{season[0]}", f"{y}-{season[1]}", max_cloud)
        nbr = compute_indices(load_reflectance(items, bbox, resolution)).nbr.compute()
        prev = nbr.ffill("time").shift(time=1)
        later_max = nbr.copy(data=np.fmax.accumulate(nbr.values[::-1], axis=0)[::-1])
        hit = (((prev - nbr) > drop) & ((prev - later_max) > persist)).any("time")
        out = out | hit.assign_coords(x=out.x, y=out.y)
    return out.rename("disturbed")


def anomaly_recovery(comp: xr.Dataset, var: str, valid: xr.DataArray, base_years, event_year: int,
                     recovery_year: int, sign: int = 1, landscape: bool = True):
    """Standardised anomaly in the event year and fraction of the drop recovered the year after.

    sign=-1 for indices where higher means drier (CRswir). With landscape=True the forest-wide
    median of each year is subtracted first, removing the climate signal shared by the whole AOI.
    z = (x_event - mu) / sigma over the baseline years;
    R = (x_recovery - x_event) / (mu - x_event): 1 = back to baseline, 0 = no recovery, <0 = worse.
    """
    x = comp[var].where(valid) * sign
    if landscape:
        x = x - x.median(["x", "y"])
    base = x.sel(year=list(base_years))
    mu, sd = base.median("year"), base.std("year")
    xe, xr_ = x.sel(year=event_year, drop=True), x.sel(year=recovery_year, drop=True)
    z = ((xe - mu) / sd).rename("z")
    rec = ((xr_ - xe) / (mu - xe)).rename("recovery")
    return z, rec


def cell_stats(flag: xr.DataArray, valid: xr.DataArray, z: xr.DataArray, rec: xr.DataArray,
               cell: int = 500, resolution: int = 20, min_valid: float = 0.5):
    """Aggregate pixel flags on a square grid; returns a GeoDataFrame ranked by flagged hectares."""
    import geopandas as gpd
    from shapely.geometry import box

    k = cell // resolution
    px_ha = resolution * resolution / 1e4
    F, V, Z, R = flag.values, valid.values, z.values, rec.values
    xs, ys = flag.x.values, flag.y.values
    rows = []
    for i in range(0, F.shape[0] - k + 1, k):
        for j in range(0, F.shape[1] - k + 1, k):
            v = V[i:i + k, j:j + k]
            if v.sum() < min_valid * k * k:
                continue
            f = F[i:i + k, j:j + k] & v
            xx, yy = xs[j:j + k], ys[i:i + k]
            rows.append({
                "geometry": box(xx.min() - resolution / 2, yy.min() - resolution / 2,
                                xx.max() + resolution / 2, yy.max() + resolution / 2),
                "forest_ha": round(float(v.sum()) * px_ha, 1),
                "flagged_ha": round(float(f.sum()) * px_ha, 1),
                "flagged_frac": round(float(f.sum() / v.sum()), 3),
                "z2024_median": round(float(np.nanmedian(np.where(v, Z[i:i + k, j:j + k], np.nan))), 2),
                "recovery_median_flagged": (round(float(np.nanmedian(R[i:i + k, j:j + k][f])), 2)
                                            if f.any() else None),
            })
    gdf = gpd.GeoDataFrame(rows, crs=CRS).sort_values("flagged_ha", ascending=False).reset_index(drop=True)
    gdf.insert(0, "rank", np.arange(1, len(gdf) + 1))
    c = gdf.to_crs(4326).geometry.centroid
    gdf["lat"], gdf["lon"] = c.y.round(5), c.x.round(5)
    return gdf


def flag_polygons(flag: xr.DataArray, min_area_ha: float = 1.0, resolution: int = 20):
    """Vectorise flagged pixels into patches; patches smaller than `min_area_ha` are sieved out."""
    import geopandas as gpd
    import rioxarray  # noqa: F401  (registers .rio)
    from rasterio import features
    from shapely.geometry import shape

    arr = flag.fillna(0).astype("uint8").values
    sieved = features.sieve(arr, size=int(min_area_ha * 1e4 / resolution ** 2), connectivity=8)
    transform = flag.rio.write_crs(CRS).rio.transform()
    geoms = [shape(g) for g, v in features.shapes(sieved, mask=sieved == 1, transform=transform) if v == 1]
    gdf = gpd.GeoDataFrame({"area_ha": [round(g.area / 1e4, 2) for g in geoms]}, geometry=geoms, crs=CRS)
    return gdf.sort_values("area_ha", ascending=False).reset_index(drop=True)


def export_cog(da: xr.DataArray, path: str, nodata=np.nan, dtype="float32") -> str:
    """Write a 2-D DataArray as a Cloud-Optimised GeoTIFF in the pipeline CRS."""
    import rioxarray  # noqa: F401

    out = da.astype(dtype)
    out = out.drop_vars([c for c in out.coords if c not in ("x", "y")])
    out.rio.write_crs(CRS, inplace=True).rio.write_nodata(nodata, inplace=True)
    out.rio.to_raster(path, driver="COG")
    return path
