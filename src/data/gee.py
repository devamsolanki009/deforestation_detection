"""Earth Engine helpers shared by the data scripts."""

from __future__ import annotations

import ee

from src.config import BBox, Config

S1_COLLECTION = "COPERNICUS/S1_GRD"


def initialize(config: Config) -> None:
    ee.Initialize(project=config.earth_engine.project)


def bbox_geometry(bbox: BBox) -> ee.Geometry:
    return ee.Geometry.Rectangle([bbox.west, bbox.south, bbox.east, bbox.north])


def season_dates(config: Config, year: int) -> tuple[ee.Date, ee.Date]:
    """Return the composite window for a year, with an exclusive end date."""
    start = ee.Date.fromYMD(year, config.season.start_month, 1)
    end = ee.Date.fromYMD(year, config.season.end_month, 1).advance(1, "month")
    return start, end


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
