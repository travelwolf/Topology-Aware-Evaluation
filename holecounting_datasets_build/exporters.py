from __future__ import annotations

import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path

import numpy as np

from .occ_utils import export_step

UTILS_DIR = Path(__file__).resolve().parents[1] / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.append(str(UTILS_DIR))

from export_step_overview_png import export_overview_png  # type: ignore  # noqa: E402
from step_to_point_npy import mesh_to_point_cloud, parse_rgb, shape_to_trimesh  # type: ignore  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


DEFAULT_RGB = parse_rgb("0.5,0.5,0.5")
DEFAULT_NUM_POINTS = 8192


def _copy_with_retry(src: Path, dst: Path, retries: int = 3, delay_seconds: float = 0.5) -> None:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            return
        except OSError as exc:
            last_error = exc
            if attempt == retries - 1:
                break
            time.sleep(delay_seconds * (attempt + 1))
    assert last_error is not None
    raise last_error


def export_step_file(shape, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_step(shape, output_path)


def export_point_cloud_file(
    shape,
    output_path: Path,
    num_points: int = DEFAULT_NUM_POINTS,
    rgb: np.ndarray = DEFAULT_RGB,
    linear_deflection: float = 0.08,
    angular_deflection: float = 0.4,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh = shape_to_trimesh(shape, linear_deflection, angular_deflection)
    point_cloud = mesh_to_point_cloud(mesh, num_points=num_points, rgb=rgb, normalize=False)
    temp_output_path = Path(tempfile.gettempdir()) / f"point_cloud_{uuid.uuid4().hex}.npy"
    try:
        np.save(temp_output_path, point_cloud.astype(np.float32))
        _copy_with_retry(temp_output_path, output_path)
    finally:
        temp_output_path.unlink(missing_ok=True)


def export_overview_image(step_path: Path, output_path: Path, width: int = 1600, height: int = 1200) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_overview_png(step_path, output_path, width, height)


def export_graph_bin_file(shape, output_path: Path) -> None:
    from model.STEP2BIN_GRAPH import save_graph_bin_from_step_path  # type: ignore

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_step_path = Path(tempfile.gettempdir()) / f"graph_export_{uuid.uuid4().hex}.step"
    try:
        export_step(shape, temp_step_path)
        save_graph_bin_from_step_path(temp_step_path, output_path)
    finally:
        temp_step_path.unlink(missing_ok=True)
