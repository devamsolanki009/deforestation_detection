"""Earth Engine helpers shared by the data scripts."""

from __future__ import annotations

import ee

from src.config import BBox, Config

S1_COLLECTION = "COPERNICUS/S1_GRD"
S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
S2_BANDS = ("B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12")
S1_BANDS = ("VV", "VH")

# Scene classification values kept as clear: vegetation, bare soil, water.
S2_CLEAR_SCL = (4, 5, 6)

# Every exported band is Int16 to keep files small. S1 dB values are stored multiplied by this.
S1_DB_SCALE = 100
NODATA = -32768


def initialize(config: Config) -> None:
    ee.Initialize(project=config.earth_engine.project)


def bbox_geometry(bbox: BBox) -> ee.Geometry:
    return ee.Geometry.Rectangle([bbox.west, bbox.south, bbox.east, bbox.north])


def quarters(bbox: BBox) -> dict[str, BBox]:
    """Split a box into four equal quarters, keyed by compass corner."""
    mid_lon = (bbox.west + bbox.east) / 2
    mid_lat = (bbox.south + bbox.north) / 2
    return {
        "NW": BBox(bbox.west, mid_lat, mid_lon, bbox.north),
        "NE": BBox(mid_lon, mid_lat, bbox.east, bbox.north),
        "SW": BBox(bbox.west, bbox.south, mid_lon, mid_lat),
        "SE": BBox(mid_lon, bbox.south, bbox.east, mid_lat),
    }


def season_dates(config: Config, year: int) -> tuple[ee.Date, ee.Date]:
    """Return the composite window for a year, with an exclusive end date."""
    start = ee.Date.fromYMD(year, config.season.start_month, 1)
    end = ee.Date.fromYMD(year, config.season.end_month, 1).advance(1, "month")
    return start, end


def _mask_s2_clouds(image: ee.Image) -> ee.Image:
    scl = image.select("SCL")
    clear = scl.eq(S2_CLEAR_SCL[0])
    for value in S2_CLEAR_SCL[1:]:
        clear = clear.Or(scl.eq(value))
    return image.select(list(S2_BANDS)).updateMask(clear)


def s2_collection(
    geometry: ee.Geometry, start: ee.Date, end: ee.Date, max_cloud_percent: float
) -> ee.ImageCollection:
    """Cloud-masked Sentinel-2 surface reflectance."""
    return (
        ee.ImageCollection(S2_COLLECTION)
        .filterBounds(geometry)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud_percent))
        .map(_mask_s2_clouds)
    )


def build_stack(config: Config, geometry: ee.Geometry) -> ee.Image:
    """Stack both years' composites with the raw Hansen bands as one Int16 image.

    Hansen stays raw so label rules can change locally without a new export. Band
    order is the order returned by stack_band_names.
    """
    images = []
    for year in (config.years.before, config.years.after):
        start, end = season_dates(config, year)
        s2 = s2_collection(geometry, start, end, config.composite.s2_max_cloud_percent)
        s1 = s1_collection(geometry, start, end).select(list(S1_BANDS))

        # Earth Engine's S1 collection is already in dB. Median needs no linear conversion
        # because it depends only on rank order.
        images.append(s2.median().rename([f"{band}_{year}" for band in S2_BANDS]))
        images.append(s1.median().multiply(S1_DB_SCALE).round().rename([f"{band}_{year}" for band in S1_BANDS]))
        images.append(s2.select(S2_BANDS[0]).count().rename(f"s2_clear_count_{year}"))

    # Hansen masks lossyear where there was no loss; 0 keeps that distinct from NODATA.
    hansen = ee.Image(config.earth_engine.hansen_asset)
    images.append(hansen.select(["lossyear", "treecover2000"]).unmask(0))
    return ee.Image.cat(images).unmask(NODATA).toInt16()


def stack_band_names(config: Config) -> list[str]:
    names = []
    for year in (config.years.before, config.years.after):
        names += [f"{band}_{year}" for band in (*S2_BANDS, *S1_BANDS)] + [f"s2_clear_count_{year}"]
    return names + ["lossyear", "treecover2000"]


def s1_collection(geometry: ee.Geometry, start: ee.Date, end: ee.Date) -> ee.ImageCollection:
    """Sentinel-1 ground-range scenes in the standard land mode with both polarisations."""
    return (
        ee.ImageCollection(S1_COLLECTION)
        .filterBounds(geometry)
        .filterDate(start, end)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
    )
