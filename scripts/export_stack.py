"""Export the composite stack from Earth Engine.

Areas:
    probe   a small square in the subset, to measure real EECU usage and validate bands
    subset  the test area from the config, which is also the NE quarter
    nw, sw, se  the other quarters of the region
    full    the whole region in one task

Run from the repo root:
    python -m scripts.export_stack --area probe --local   small direct download for checking
    python -m scripts.export_stack --area probe           batch task to Google Drive
    python -m scripts.export_stack --area subset --confirm
    python -m scripts.export_stack --area full --confirm

Subset and full runs consume real quota, so they need --confirm. A task with the same
name that is active or already completed blocks a new run unless --force is given.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

import ee

from src.config import DEFAULT_CONFIG, Config, load_config
from src.data import gee

PROBE_RADIUS_M = 2560  # gives a 5.12 km square
LOCAL_RADIUS_M = 1280  # direct downloads are limited in size
POLL_SECONDS = 20
BLOCKING_STATES = {"READY", "RUNNING", "COMPLETED"}
QUARTER_AREAS = ("nw", "sw", "se")  # the subset already covers the NE quarter


def area_geometry(config: Config, area: str, radius_m: int) -> ee.Geometry:
    if area == "full":
        return gee.bbox_geometry(config.region.bbox)
    if area == "subset":
        return gee.bbox_geometry(config.subset.bbox)
    if area in QUARTER_AREAS:
        return gee.bbox_geometry(gee.quarters(config.region.bbox)[area.upper()])
    box = config.subset.bbox
    center = ee.Geometry.Point([(box.west + box.east) / 2, (box.south + box.north) / 2])
    return center.buffer(radius_m).bounds()


def download_local(config: Config, image: ee.Image, geometry: ee.Geometry, path: Path) -> None:
    url = image.getDownloadURL({
        "format": "GEO_TIFF",
        "region": geometry,
        "crs": config.export.crs,
        "scale": config.export.scale_m,
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, path)
    print(f"Saved {path} ({path.stat().st_size / 1024**2:.1f} MiB)")


def find_existing_task(description: str) -> dict | None:
    for task in ee.batch.Task.list():
        status = task.status()
        if status.get("description") == description and status.get("state") in BLOCKING_STATES:
            return status
    return None


def start_drive_export(config: Config, image: ee.Image, geometry: ee.Geometry, description: str) -> ee.batch.Task:
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=description,
        folder=config.export.drive_folder,
        fileNamePrefix=description,
        region=geometry,
        crs=config.export.crs,
        scale=config.export.scale_m,
        maxPixels=int(1e10),
        fileDimensions=config.export.file_dimensions,
        fileFormat="GeoTIFF",
        formatOptions={"noData": gee.NODATA},
    )
    task.start()
    return task


def wait_for(task: ee.batch.Task) -> dict:
    while True:
        status = task.status()
        print(f"  {time.strftime('%H:%M:%S')} {status.get('state')}", flush=True)
        if status.get("state") in {"COMPLETED", "FAILED", "CANCELLED"}:
            return status
        time.sleep(POLL_SECONDS)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--area", choices=["probe", "subset", *QUARTER_AREAS, "full"], required=True)
    parser.add_argument("--local", action="store_true", help="direct download; probe area only")
    parser.add_argument("--confirm", action="store_true", help="required for subset and full")
    parser.add_argument("--force", action="store_true", help="run even if a matching task exists")
    parser.add_argument("--wait", action="store_true", help="poll until the task finishes")
    args = parser.parse_args()

    if args.local and args.area != "probe":
        parser.error("--local is only for the probe area")
    if args.area != "probe" and not args.confirm:
        parser.error(f"--area {args.area} uses real quota; pass --confirm to run it")

    config = load_config(args.config)
    gee.initialize(config)

    radius = LOCAL_RADIUS_M if args.local else PROBE_RADIUS_M
    geometry = area_geometry(config, args.area, radius)
    image = gee.build_stack(config, geometry)
    years = f"{config.years.before}_{config.years.after}"
    description = f"stack_{config.region.name}_{args.area}_{years}"

    manifest = {
        "area": args.area,
        "crs": config.export.crs,
        "scale_m": config.export.scale_m,
        "nodata": gee.NODATA,
        "s1_db_scale": gee.S1_DB_SCALE,
        "bands": gee.stack_band_names(config),
    }
    manifest_path = config.paths.raw_dir / f"{description}.manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if args.local:
        download_local(config, image, geometry, config.paths.raw_dir / f"{description}.local.tif")
        return 0

    existing = find_existing_task(description)
    if existing and not args.force:
        print(f"Not starting: task '{description}' is already {existing['state']}.")
        print(json.dumps(existing, indent=2))
        return 1

    task = start_drive_export(config, image, geometry, description)
    print(f"Started {description} ({task.id}), Drive folder {config.export.drive_folder}")
    if not args.wait:
        return 0

    status = wait_for(task)
    print(json.dumps(status, indent=2))
    return 0 if status.get("state") == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
