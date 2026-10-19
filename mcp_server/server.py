"""MCP server exposing the Sentinel-2 forest-drought pipeline as three tools.

Run:  python -m mcp_server.server        (stdio transport; any MCP client can launch it)

Every tool returns a JSON-serialisable dict with the numbers the model should quote, plus the paths
of the files it wrote. Results are cached on disk by arguments, so asking the same question twice
does not re-download imagery; the result says whether it came from the cache.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import geopandas as gpd  # noqa: E402
import rioxarray  # noqa: E402,F401  (registers .rio)
import numpy as np  # noqa: E402
from mcp.server.fastmcp import FastMCP  # noqa: E402
from pyproj import Transformer  # noqa: E402
from shapely.geometry import box, shape  # noqa: E402

from src import pipeline as pl  # noqa: E402

OUT = ROOT / "outputs" / "mcp"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / "cache"
CACHE.mkdir(exist_ok=True)
MAX_AREA_KM2 = 400
YEARS_OK = range(2017, 2026)
RES = 20

mcp = FastMCP("sentinel2-forest-drought")


# ---------------------------------------------------------------------------------------------
def _cached(name: str, args: dict, fn):
    key = hashlib.sha1(json.dumps([name, args], sort_keys=True, default=str).encode()).hexdigest()[:16]
    f = CACHE / f"{name}_{key}.json"
    if f.exists():
        res = json.loads(f.read_text())
        res["cached"] = True
        return res
    t0 = time.time()
    res = fn()
    res["runtime_s"] = round(time.time() - t0, 1)
    if "error" not in res:
        f.write_text(json.dumps(res, default=str))
    res["cached"] = False
    return res


def _aoi(bbox=None, polygon=None):
    """Return (bbox, polygon in lon/lat) from either input, or raise ValueError with a readable message."""
    if polygon is not None:
        g = json.loads(polygon) if isinstance(polygon, str) else polygon
        if g.get("type") == "Feature":          # accept a GeoJSON Feature as well as a bare geometry
            g = g["geometry"]
        geom = shape(g)
    elif bbox is not None:
        if len(bbox) != 4:
            raise ValueError("bbox must be [lon_min, lat_min, lon_max, lat_max]")
        geom = box(*bbox)
    else:
        raise ValueError("give either bbox or polygon")
    lon0, lat0, lon1, lat1 = geom.bounds
    if not (-180 <= lon0 < lon1 <= 180 and -90 <= lat0 < lat1 <= 90):
        raise ValueError("coordinates must be lon/lat in degrees, with min < max")
    area = gpd.GeoSeries([geom], crs=4326).to_crs(pl.CRS).area.iloc[0] / 1e6
    if area > MAX_AREA_KM2:
        raise ValueError(f"area is {area:.0f} km2; the limit is {MAX_AREA_KM2} km2 to keep runtime to minutes")
    return tuple(round(v, 6) for v in geom.bounds), geom, area


def _period(p: str):
    try:
        a, b = p.split("/")
        np.datetime64(a), np.datetime64(b)
    except Exception:
        raise ValueError(f"period '{p}' must be 'YYYY-MM-DD/YYYY-MM-DD'")
    if int(a[:4]) not in YEARS_OK or int(b[:4]) not in YEARS_OK:
        raise ValueError("periods must fall within 2017-2025 (Sentinel-2 L2A archive used here)")
    return a, b


def _latlon(x, y):
    lon, lat = Transformer.from_crs(pl.CRS, 4326, always_xy=True).transform(x, y)
    return round(lat, 5), round(lon, 5)


# ---------------------------------------------------------------------------------------------
@mcp.tool()
def list_scenes(bbox: list[float], start: str, end: str, max_cloud: float = 20) -> dict:
    """List the Sentinel-2 L2A acquisitions available over an area and date window.

    Use this to check how much clear imagery exists before an analysis.
    bbox: [lon_min, lat_min, lon_max, lat_max] in degrees (WGS84). start, end: 'YYYY-MM-DD'.
    max_cloud: maximum tile-level cloud cover in percent.
    Returns the number of acquisitions and, for each, date, tile and cloud cover.
    """
    try:
        bb, _, _ = _aoi(bbox=bbox)
        _period(f"{start}/{end}")
    except ValueError as e:
        return {"error": str(e)}
    items = pl.search_scenes(bb, start, end, max_cloud)
    t = pl.scenes_table(items)
    return {"n_acquisitions": len(t), "collection": "sentinel-2-l2a (Planetary Computer)",
            "acquisitions": [{"date": str(r.date), "tile": r.tile, "cloud_cover_pct": r["cloud_cover_%"]}
                             for _, r in t.iterrows()]}


@mcp.tool()
def compare_vegetation(period_a: str, period_b: str, polygon: dict | str | None = None, bbox: list[float] | None = None,
                       index: str = "ndvi", drop_threshold: float = 0.10) -> dict:
    """Compare a vegetation index over the FOREST of an area between two periods, and locate where it dropped.

    Use for questions like "compare vegetation in this polygon between summer 2022 and 2024".
    period_a, period_b: 'YYYY-MM-DD/YYYY-MM-DD' (e.g. '2022-07-01/2022-09-30').
    polygon: GeoJSON geometry (lon/lat), as an object or a string; or bbox: [lon_min, lat_min, lon_max, lat_max].
    index: 'ndvi' (greenness) or 'ndmi' (canopy water content).
    drop_threshold: index decrease (absolute units) above which a pixel counts as 'dropped'.
    Returns median index in each period, absolute and % change, hectares of forest that dropped, the
    largest dropped patches with their centre (lat, lon) and area, and a GeoJSON file of the patches.
    Sentinel-2 L2A, clouds masked, per-pixel median per period, 20 m, forest = ESA WorldCover 2021.
    """
    try:
        bb, geom, area = _aoi(bbox, polygon)
        _period(period_a), _period(period_b)
        if index not in ("ndvi", "ndmi"):
            raise ValueError("index must be 'ndvi' or 'ndmi'")
    except ValueError as e:
        return {"error": str(e)}
    args = dict(bbox=bb, wkt=geom.wkt, a=period_a, b=period_b, index=index, thr=drop_threshold)

    def run():
        from rasterio import features

        ca, na = pl.period_composite(bb, *period_a.split("/"), RES)
        cb, nb = pl.period_composite(bb, *period_b.split("/"), RES)
        if ca is None or cb is None:
            return {"error": "no Sentinel-2 acquisitions with < 20 % cloud in one of the periods"}
        forest = pl.forest_mask(bb, ca, RES)
        g_utm = gpd.GeoSeries([geom], crs=4326).to_crs(pl.CRS).iloc[0]
        inside = features.geometry_mask([g_utm], out_shape=forest.shape,
                                        transform=ca.rio.write_crs(pl.CRS).rio.transform(), invert=True)
        m = forest.values & inside
        xa, xb = ca[index].where(m), cb[index].where(m)
        d = xb - xa
        drop = (d < -drop_threshold) & m
        px_ha = RES * RES / 1e4
        patches = pl.flag_polygons(drop, min_area_ha=1.0, resolution=RES)
        top = []
        for _, p in patches.head(5).iterrows():
            c = p.geometry.centroid
            lat, lon = _latlon(c.x, c.y)
            top.append({"lat": lat, "lon": lon, "area_ha": p.area_ha})
        f = OUT / f"compare_{index}_{period_a[:4]}_{period_b[:4]}_{hashlib.sha1(geom.wkt.encode()).hexdigest()[:6]}.geojson"
        if len(patches):
            patches.to_crs(4326).to_file(f, driver="GeoJSON")
        ma, mb = float(np.nanmedian(xa)), float(np.nanmedian(xb))
        return {
            "index": index, "period_a": period_a, "period_b": period_b,
            "acquisitions_a": na, "acquisitions_b": nb, "area_km2": round(area, 2),
            "forest_ha": round(float(m.sum()) * px_ha, 1),
            "median_a": round(ma, 3), "median_b": round(mb, 3),
            "change_abs": round(mb - ma, 3), "change_pct": round(100 * (mb - ma) / ma, 1),
            "dropped_ha": round(float(drop.sum()) * px_ha, 1),
            "dropped_pct_of_forest": round(100 * float(drop.sum()) / max(int(m.sum()), 1), 1),
            "drop_threshold": drop_threshold,
            "largest_dropped_patches": top,
            "patches_geojson": str(f.relative_to(ROOT)) if len(patches) else None,
        }

    return _cached("compare_vegetation", args, run)


@mcp.tool()
def dieback_alerts(bbox: list[float], event_year: int = 2024, recovery_year: int = 2025,
                   baseline_years: list[int] | None = None, z_threshold: float = -2.0,
                   recovery_threshold: float = 0.5, top_n: int = 5) -> dict:
    """Find forest stands that dried out in a drought year AND did not recover the year after.

    Use for questions about post-drought forest dieback or where a forest service should inspect first.
    bbox: [lon_min, lat_min, lon_max, lat_max]. event_year: the drought summer; recovery_year: the
    following summer; baseline_years: normal summers (default: the five before event_year).
    A forest pixel is flagged when its summer canopy-water index (NDMI), relative to the whole forest
    of the area, is at least |z_threshold| standard deviations below its own baseline in event_year
    AND it regained less than recovery_threshold (fraction) of that drop by recovery_year.
    Fires and clear-cuts (abrupt NBR drops) are excluded. Flags are aggregated on 500 m cells (25 ha).
    Returns forest and flagged hectares, the top_n priority cells (lat, lon, flagged ha), and file paths.
    """
    base = baseline_years or list(range(event_year - 5, event_year))
    try:
        bb, _, area = _aoi(bbox=bbox)
        years = sorted(set(base) | {event_year, recovery_year})
        if any(y not in YEARS_OK for y in years) or recovery_year <= event_year:
            raise ValueError("years must be within 2017-2025 and recovery_year > event_year")
    except ValueError as e:
        return {"error": str(e)}
    args = dict(bbox=bb, e=event_year, r=recovery_year, base=base, z=z_threshold, rt=recovery_threshold, n=top_n)

    def run():
        comp, forest, _ = pl.summer_composites(bb, years, ("07-01", "09-30"), RES)
        disturbed = pl.abrupt_disturbance(bb, [event_year, recovery_year], comp.isel(year=0), resolution=RES)
        valid = forest & ~disturbed
        z, rec = pl.anomaly_recovery(comp, "ndmi", valid, base, event_year, recovery_year, landscape=True)
        flag = (z <= z_threshold) & (rec < recovery_threshold) & valid
        cells = pl.cell_stats(flag, valid, z, rec, cell=500, resolution=RES)
        prio = cells[cells.flagged_frac >= 0.5]
        tag = hashlib.sha1(json.dumps(args, sort_keys=True).encode()).hexdigest()[:6]
        f_csv, f_geo = OUT / f"dieback_cells_{tag}.csv", OUT / f"dieback_cells_{tag}.geojson"
        cols = ["rank", "lat", "lon", "forest_ha", "flagged_ha", "flagged_frac", "recovery_median_flagged"]
        prio[cols].to_csv(f_csv, index=False)
        if len(prio):
            prio.to_crs(4326).to_file(f_geo, driver="GeoJSON")
        px_ha = RES * RES / 1e4
        return {
            "area_km2": round(area, 1), "event_year": event_year, "recovery_year": recovery_year,
            "baseline_years": base, "forest_ha": round(float(forest.sum()) * px_ha),
            "excluded_fire_or_logging_ha": round(float((disturbed & forest).sum()) * px_ha),
            "flagged_ha": round(float(flag.sum()) * px_ha),
            "flagged_pct_of_forest": round(100 * float(flag.sum() / valid.sum()), 1),
            "priority_cells_25ha": int(len(prio)),
            "top_cells": [{"rank": int(r["rank"]), "lat": r.lat, "lon": r.lon, "flagged_ha": r.flagged_ha,
                           "forest_ha": r.forest_ha} for _, r in prio.head(top_n).iterrows()],
            "cells_csv": str(f_csv.relative_to(ROOT)),
            "cells_geojson": str(f_geo.relative_to(ROOT)) if len(prio) else None,
        }

    return _cached("dieback_alerts", args, run)


if __name__ == "__main__":
    os.chdir(ROOT)
    mcp.run()  # stdio
