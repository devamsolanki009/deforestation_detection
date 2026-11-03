"""Demo app: pick an area and two years, see detected forest loss and hectares.

Run from the repo root, then open the printed local URL:
    python -m app.app
    python -m app.app --app-config configs/app.yaml --port 7860
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gradio as gr

from src.config import ConfigError
from src.inference.area import comparison_image, false_color, loss_overlay, save_geotiff
from src.inference.detector import DEFAULT_APP_CONFIG, Detector, bbox_around, load_app_config

INTRO = """# Forest loss detection: Sentinel-2 + Sentinel-1
Pick a point and two years. The app builds dry-season (June to September) composites in Earth Engine,
runs a Siamese U-Net on them and reports forest lost between the two years, in hectares.

**Limits:** the model was trained on 2020 to 2022 in São Félix do Xingu, Pará, Brazil. Other places
and year gaps run, but their accuracy is unknown. The default point is in the held-out test stripe.
"""

LEGEND = ("**Model vs Hansen:** yellow = both, red = model only, cyan = Hansen only, "
          "purple = model on loss Hansen dates to the before or after year (may still be in the period).")


def hectare_table(result, request, fetch_info, model_summary) -> str:
    h = result.hectares
    rows = [
        ("**Detected forest loss**", f"**{h['predicted_loss_ha']:,.1f} ha**"),
        ("… on Hansen loss dated between the years", f"{h['predicted_on_hansen_loss_between_years_ha']:,.1f} ha"),
        (f"… on Hansen loss dated {request.before} or {request.after}",
         f"{h['predicted_on_hansen_loss_in_edge_years_ha']:,.1f} ha"),
        (f"… on Hansen loss dated after {request.after}", f"{h['predicted_on_hansen_loss_after_period_ha']:,.1f} ha"),
        ("… with no Hansen loss", f"{h['predicted_with_no_hansen_loss_ha']:,.1f} ha"),
        ("Hansen loss dated between the years (reference)", f"{h['hansen_loss_between_years_ha']:,.1f} ha"),
        (f"Forest at the start of {request.before}", f"{h['forest_at_start_ha']:,.1f} ha"),
        ("Area analysed", f"{h['area_ha']:,.1f} ha ({request.grid.width} × {request.grid.height} px at 10 m)"),
    ]
    q = result.quality
    lines = ["| | |", "|---|---|", *[f"| {a} | {b} |" for a, b in rows]]
    lines.append("")
    lines.append(f"Imagery: median {q[f'median_s2_clear_views_{request.before}']:.0f} / "
                 f"{q[f'median_s2_clear_views_{request.after}']:.0f} cloud-free Sentinel-2 views per pixel; "
                 f"{q['missing_image_fraction']:.1%} of pixels missing.")
    source = ("cache, no quota used" if fetch_info["tiles_requested"] == 0
              else f"Earth Engine, {fetch_info['tiles_requested']} tile request(s) in {fetch_info['request_seconds']} s")
    lines.append(f"Imagery from {source}. Model `{model_summary['run']}`, threshold {model_summary['threshold']}, "
                 f"test F1 {model_summary['test_f1']} in its own evaluation.")
    if result.warnings:
        lines.append("")
        lines += [f"> ⚠️ {w}" for w in result.warnings]
    return "\n".join(lines)


def map_preview(lon: float, lat: float, size_km: float) -> str:
    """OpenStreetMap view of the box, to check the location before using quota."""
    box = bbox_around(lon, lat, size_km)
    pad_lon, pad_lat = (box.east - box.west), (box.north - box.south)
    view = f"{box.west - pad_lon},{box.south - pad_lat},{box.east + pad_lon},{box.north + pad_lat}"
    return (f'<iframe width="100%" height="320" style="border:0" '
            f'src="https://www.openstreetmap.org/export/embed.html?bbox={view}&layer=mapnik"></iframe>'
            f"<p>Box: W {box.west}, S {box.south}, E {box.east}, N {box.north}</p>")


def build_ui(detector: Detector) -> gr.Blocks:
    app = detector.app
    default = app.default_bbox
    default_lon, default_lat = (default.west + default.east) / 2, (default.south + default.north) / 2
    years = list(range(app.first_year, app.last_year + 1))
    max_side_km = int(app.max_area_km2 ** 0.5)

    def detect(lon, lat, size_km, before, after, progress=gr.Progress()):
        bbox = bbox_around(float(lon), float(lat), float(size_km))
        try:
            request = detector.validate(bbox, int(before), int(after))
        except ConfigError as exc:
            raise gr.Error(str(exc))

        progress(0.05, desc="Requesting imagery from Earth Engine")
        try:
            result, stack, bands, fetch_info = detector.run(
                request, progress=lambda done, total: progress(0.05 + 0.75 * done / total,
                                                              desc=f"Fetched tile {done}/{total}"))
        except Exception as exc:  # Earth Engine errors arrive as many types; show them, don't crash
            raise gr.Error(f"Request failed: {exc}")

        progress(0.9, desc="Saving outputs")
        output_dir = app.outputs_dir / f"{request.before}_{request.after}_{request.grid.key()}"
        geotiff = output_dir / "loss.tif"
        save_geotiff(geotiff, result)
        summary = output_dir / "summary.json"
        summary.write_text(json.dumps({"bbox": vars(bbox), "before": request.before, "after": request.after,
                                       "model": detector.model_summary, "hectares": result.hectares,
                                       "quality": result.quality, "warnings": result.warnings}, indent=2),
                           encoding="utf-8")

        before_image = false_color(stack, bands, request.before)
        after_image = false_color(stack, bands, request.after)
        return (before_image, after_image, loss_overlay(after_image, result.loss),
                comparison_image(after_image, result),
                hectare_table(result, request, fetch_info, detector.model_summary), [str(geotiff), str(summary)])

    with gr.Blocks(title="Forest loss detection") as ui:
        gr.Markdown(INTRO)
        with gr.Row():
            with gr.Column(scale=1):
                lon = gr.Number(value=round(default_lon, 4), label="Longitude (centre)", precision=5)
                lat = gr.Number(value=round(default_lat, 4), label="Latitude (centre)", precision=5)
                size = gr.Slider(1, max_side_km, value=5, step=0.5, label="Box side (km)")
                with gr.Row():
                    before = gr.Dropdown(years, value=2020, label="Before year")
                    after = gr.Dropdown(years, value=2022, label="After year")
                run = gr.Button("Detect forest loss", variant="primary")
                gr.Markdown(f"Each new area uses Earth Engine quota; repeats come from the cache. "
                            f"Limit {app.max_area_km2:.0f} km².")
            with gr.Column(scale=1):
                preview = gr.HTML(map_preview(default_lon, default_lat, 5))
        for control in (lon, lat, size):
            control.change(map_preview, [lon, lat, size], preview)

        with gr.Row():
            before_out = gr.Image(label="Before (false color: forest green, bare land pink)", type="numpy")
            after_out = gr.Image(label="After", type="numpy")
        with gr.Row():
            overlay_out = gr.Image(label="Detected loss (red) on the after image", type="numpy")
            compare_out = gr.Image(label="Model vs Hansen Global Forest Change", type="numpy")
        gr.Markdown(LEGEND)
        table_out = gr.Markdown()
        files_out = gr.File(label="Downloads: loss GeoTIFF (band 1 mask, band 2 probability %) and summary",
                            file_count="multiple")

        run.click(detect, [lon, lat, size, before, after],
                  [before_out, after_out, overlay_out, compare_out, table_out, files_out])
    return ui


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--app-config", type=Path, default=DEFAULT_APP_CONFIG)
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    app_config = load_app_config(args.app_config)
    detector = Detector(app_config)
    print(f"Model loaded: {detector.model_summary}", flush=True)
    # Only this machine can reach the app; outputs are saved under outputs/app.
    build_ui(detector).launch(server_name="127.0.0.1", server_port=args.port,
                              allowed_paths=[str(app_config.outputs_dir)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
