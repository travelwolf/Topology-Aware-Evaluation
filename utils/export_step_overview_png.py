from __future__ import annotations

import argparse
import logging
import shutil
import tempfile
import uuid
from pathlib import Path

from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Display.OCCViewer import OffscreenRenderer
from PIL import Image, ImageDraw, ImageFont

VIEW_SPECS = [
    ("main", "Main View", "View_Front"),
    ("side", "Side View", "View_Right"),
    ("top", "Top View", "View_Top"),
    ("iso", "Isometric View", "View_Iso"),
]

logging.getLogger("PIL").setLevel(logging.WARNING)


def read_step_shape(step_path: Path):
    reader = STEPControl_Reader()
    status = reader.ReadFile(str(step_path))
    if status != IFSelect_RetDone:
        raise RuntimeError(f"Failed to read STEP file: {step_path}")

    transfer_count = reader.TransferRoots()
    if transfer_count == 0:
        raise RuntimeError(f"No transferable roots found in: {step_path}")

    shape = reader.OneShape()
    if shape.IsNull():
        raise RuntimeError(f"STEP file contains no readable shape: {step_path}")
    return shape


def render_view_image(shape, view_method_name: str, width: int, height: int, output_path: Path) -> None:
    renderer = OffscreenRenderer(screen_size=(width, height))
    renderer.SetSize(width, height)
    renderer.EnableAntiAliasing()
    renderer.GetView().SetBackgroundColor(Quantity_Color(1.0, 1.0, 1.0, Quantity_TOC_RGB))
    renderer.DisplayShape(shape, update=True)
    getattr(renderer, view_method_name)()
    renderer.FitAll()
    renderer.ExportToImage(str(output_path))


def compose_overview(temp_paths: list[tuple[str, Path]], output_path: Path, cell_width: int, cell_height: int) -> None:
    margin = 24
    title_band = 56
    canvas = Image.new("RGB", (cell_width * 2 + margin * 3, cell_height * 2 + margin * 3), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for index, (label, image_path) in enumerate(temp_paths):
        row = index // 2
        col = index % 2
        x0 = margin + col * (cell_width + margin)
        y0 = margin + row * (cell_height + margin)
        with Image.open(image_path) as tile_image:
            tile = tile_image.convert("RGB")
            tile = tile.resize((cell_width, cell_height - title_band))
            canvas.paste(tile, (x0, y0 + title_band))
        draw.rectangle((x0, y0, x0 + cell_width, y0 + cell_height), outline=(180, 180, 180), width=1)
        draw.text((x0 + 12, y0 + 14), label, fill=(20, 20, 20), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_output_path = Path(tempfile.gettempdir()) / f"overview_{uuid.uuid4().hex}.png"
    try:
        canvas.save(str(temp_output_path))
        shutil.copyfile(temp_output_path, output_path)
    finally:
        temp_output_path.unlink(missing_ok=True)


def export_overview_png(step_path: Path, output_path: Path, width: int, height: int) -> None:
    shape = read_step_shape(step_path)
    cell_width = max(width // 2, 200)
    cell_height = max(height // 2, 160)

    temp_paths: list[tuple[str, Path]] = []
    try:
        for key, label, view_method_name in VIEW_SPECS:
            temp_path = Path(tempfile.gettempdir()) / f"{step_path.stem}_{key}_{uuid.uuid4().hex}.png"
            render_view_image(shape, view_method_name, cell_width, cell_height, temp_path)
            temp_paths.append((label, temp_path))

        compose_overview(temp_paths, output_path, cell_width, cell_height)
    finally:
        for _, temp_path in temp_paths:
            temp_path.unlink(missing_ok=True)


def iter_step_files(input_dir: Path, recursive: bool) -> list[Path]:
    patterns = ("*.step", "*.stp", "*.STEP", "*.STP")
    files: list[Path] = []
    for pattern in patterns:
        iterator = input_dir.rglob(pattern) if recursive else input_dir.glob(pattern)
        files.extend(path for path in iterator if path.is_file())
    return sorted(set(path.resolve() for path in files))


def export_overview_png_batch(input_dir: Path, output_dir: Path, width: int, height: int, recursive: bool) -> None:
    step_files = iter_step_files(input_dir, recursive=recursive)
    if not step_files:
        raise FileNotFoundError(f"No STEP files found in: {input_dir}")

    success_count = 0
    failure_count = 0
    for step_path in step_files:
        relative_parent = step_path.parent.relative_to(input_dir) if recursive else Path()
        target_dir = output_dir / relative_parent
        output_path = target_dir / f"{step_path.stem}.png"
        try:
            export_overview_png(step_path, output_path, width, height)
            success_count += 1
            print(f"[ok] {step_path} -> {output_path}")
        except Exception as exc:
            failure_count += 1
            print(f"[failed] {step_path}: {exc}")

    print(
        f"[summary] total={len(step_files)} success={success_count} "
        f"failed={failure_count} output_dir={output_dir}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a 4-view overview PNG from STEP files with PythonOCC.")
    parser.add_argument("step_file", type=Path, nargs="?", help="Input STEP file.")
    parser.add_argument("--input-dir", type=Path, help="Input directory containing STEP files.")
    parser.add_argument("--output-dir", type=Path, help="Output directory for batch mode.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output PNG path for single-file mode. Defaults to '<step_stem>_overview.png' next to the STEP file.",
    )
    parser.add_argument("--recursive", action="store_true", help="Recursively search STEP files under --input-dir.")
    parser.add_argument("--width", type=int, default=1600, help="Output image width in pixels.")
    parser.add_argument("--height", type=int, default=1200, help="Output image height in pixels.")
    args = parser.parse_args()

    if args.width <= 0 or args.height <= 0:
        raise ValueError("--width and --height must be positive integers")
    if args.step_file is None and args.input_dir is None:
        raise ValueError("Provide either a STEP file or --input-dir")
    if args.step_file is not None and args.input_dir is not None:
        raise ValueError("step_file and --input-dir are mutually exclusive")

    if args.input_dir is not None:
        input_dir = args.input_dir.expanduser().resolve()
        if not input_dir.exists():
            raise FileNotFoundError(input_dir)
        if not input_dir.is_dir():
            raise NotADirectoryError(input_dir)
        if args.output_dir is None:
            raise ValueError("--output-dir is required when using --input-dir")
        output_dir = args.output_dir.expanduser().resolve()
        export_overview_png_batch(input_dir, output_dir, args.width, args.height, recursive=args.recursive)
        return

    step_file = args.step_file.expanduser().resolve()
    if not step_file.exists():
        raise FileNotFoundError(step_file)
    output_path = args.output.expanduser().resolve() if args.output else step_file.with_name(f"{step_file.stem}_overview.png")
    export_overview_png(step_file, output_path, args.width, args.height)
    print(f"[ok] {step_file} -> {output_path}")


if __name__ == "__main__":
    main()


# python utils\export_step_overview_png.py ^
#   "data\step\a.step" ^
#   --output "outputs\views\a.png"

# python utils\export_step_overview_png.py ^
#   --input-dir "data\step" ^
#   --output-dir "outputs\views"


# python utils\export_step_overview_png.py ^
#   --input-dir "data\step" ^
#   --output-dir "outputs\views" ^
#   --recursive