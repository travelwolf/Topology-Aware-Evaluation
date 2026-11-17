#!/usr/bin/env python
"""
Convert STEP files (.step/.stp) to PointLLM-compatible point clouds.

Output format:
    - .npy
    - shape: (N, 6)
    - columns: x, y, z, r, g, b
    - rgb range: [0, 1]

This script is intended for Windows and has been written against a Python
environment where `OCC`, `trimesh`, and `numpy` are available.

Example:
    python step_to_point_npy.py ^
        --input "data\\part.step" ^
        --output-dir "outputs\\points"

Batch mode:
    python step_to_point_npy.py ^
        --input "data\\step" ^
        --output-dir "outputs\\points" ^
        --recursive
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Sequence

import numpy as np
import trimesh
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_REVERSED
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TopoDS import topods_Face
from tqdm import tqdm


# =============================================================================
# Configuration
# =============================================================================
# If you run this script directly without command-line arguments, these values
# will be used automatically. Command-line arguments, when provided, override
# the values below.
DEFAULT_INPUT_PATH = r"data\step"
DEFAULT_OUTPUT_DIR = r"outputs\points"
DEFAULT_NUM_POINTS = 8192
DEFAULT_RGB = "0.5,0.5,0.5"
DEFAULT_NORMALIZE = False
DEFAULT_LINEAR_DEFLECTION = 0.1
DEFAULT_ANGULAR_DEFLECTION = 0.5
DEFAULT_RECURSIVE = True
#DEFAULT_MAX_WORKERS = max(1, (os.cpu_count() or 1) - 1)
DEFAULT_MAX_WORKERS = 2
DEFAULT_SKIP_EXISTING = True
DEFAULT_MAX_IN_FLIGHT_MULTIPLIER = 2
DEFAULT_FILE_TIMEOUT_SECONDS = 60


def parse_rgb(rgb_text: str) -> np.ndarray:
    parts = [item.strip() for item in rgb_text.split(",")]
    if len(parts) != 3:
        raise ValueError("--rgb must contain exactly 3 comma-separated values, e.g. 0.5,0.5,0.5")

    rgb = np.array([float(item) for item in parts], dtype=np.float32)
    if np.any(rgb < 0.0) or np.any(rgb > 1.0):
        raise ValueError("--rgb values must be within [0, 1]")
    return rgb


def pc_normalize(points_xyz: np.ndarray) -> np.ndarray:
    centroid = np.mean(points_xyz, axis=0)
    points_xyz = points_xyz - centroid
    scale = np.max(np.sqrt(np.sum(points_xyz ** 2, axis=1)))
    if scale > 0:
        points_xyz = points_xyz / scale
    return points_xyz


def collect_step_files(input_path: Path, recursive: bool) -> List[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in {".step", ".stp"}:
            raise ValueError(f"Unsupported file type: {input_path}")
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    pattern = "**/*" if recursive else "*"
    files = [
        path for path in input_path.glob(pattern)
        if path.is_file() and path.suffix.lower() in {".step", ".stp"}
    ]
    if not files:
        raise FileNotFoundError(f"No STEP files found under: {input_path}")
    return sorted(files)


def read_step_shape(step_path: Path):
    reader = STEPControl_Reader()
    status = reader.ReadFile(str(step_path))
    if status != IFSelect_RetDone:
        raise RuntimeError(f"Failed to read STEP file: {step_path}")

    reader.TransferRoots()
    shape = reader.OneShape()
    if shape.IsNull():
        raise RuntimeError(f"STEP file contains no readable shape: {step_path}")
    return shape


def shape_to_trimesh(shape, linear_deflection: float, angular_deflection: float) -> trimesh.Trimesh:
    mesher = BRepMesh_IncrementalMesh(shape, linear_deflection, False, angular_deflection, True)
    mesher.Perform()

    vertices: List[Sequence[float]] = []
    faces: List[Sequence[int]] = []

    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = topods_Face(explorer.Current())
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation(face, location)

        if triangulation is None or triangulation.NbNodes() == 0 or triangulation.NbTriangles() == 0:
            explorer.Next()
            continue

        transform = location.Transformation()
        face_vertex_offset = len(vertices)

        for node_idx in range(1, triangulation.NbNodes() + 1):
            point = triangulation.Node(node_idx).Transformed(transform)
            vertices.append((point.X(), point.Y(), point.Z()))

        for tri_idx in range(1, triangulation.NbTriangles() + 1):
            triangle = triangulation.Triangles().Value(tri_idx)
            n1, n2, n3 = triangle.Get()

            if face.Orientation() == TopAbs_REVERSED:
                faces.append((
                    face_vertex_offset + n1 - 1,
                    face_vertex_offset + n3 - 1,
                    face_vertex_offset + n2 - 1,
                ))
            else:
                faces.append((
                    face_vertex_offset + n1 - 1,
                    face_vertex_offset + n2 - 1,
                    face_vertex_offset + n3 - 1,
                ))
        explorer.Next()

    if not vertices or not faces:
        raise RuntimeError("Triangulation failed; the STEP file did not produce any mesh faces")

    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )

    if mesh.is_empty or mesh.area <= 0:
        raise RuntimeError("Generated mesh is empty or has zero area")

    return mesh


def mesh_to_point_cloud(mesh: trimesh.Trimesh, num_points: int, rgb: np.ndarray, normalize: bool) -> np.ndarray:
    sampled_xyz, _ = trimesh.sample.sample_surface(mesh, num_points)
    sampled_xyz = np.asarray(sampled_xyz, dtype=np.float32)

    if normalize:
        sampled_xyz = pc_normalize(sampled_xyz)

    rgb_array = np.repeat(rgb.reshape(1, 3), num_points, axis=0).astype(np.float32)
    return np.concatenate([sampled_xyz, rgb_array], axis=1).astype(np.float32)


def convert_one_file(
    step_path: Path,
    output_dir: Path,
    num_points: int,
    rgb: np.ndarray,
    normalize: bool,
    linear_deflection: float,
    angular_deflection: float,
) -> Path:
    shape = read_step_shape(step_path)
    mesh = shape_to_trimesh(shape, linear_deflection, angular_deflection)
    point_cloud = mesh_to_point_cloud(mesh, num_points, rgb, normalize)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{step_path.stem}_{num_points}.npy"
    np.save(output_path, point_cloud)
    return output_path


def convert_one_file_from_task(task: tuple[str, str, int, str, bool, float, float, bool]) -> tuple[bool, str, str]:
    step_path_str, output_dir_str, num_points, rgb_text, normalize, linear_deflection, angular_deflection, skip_existing = task
    step_path = Path(step_path_str)
    output_dir = Path(output_dir_str)
    output_path = output_dir / f"{step_path.stem}_{num_points}.npy"

    try:
        if skip_existing and output_path.exists():
            return True, str(step_path), f"SKIP {output_path}"

        rgb = parse_rgb(rgb_text)
        saved_path = convert_one_file(
            step_path=step_path,
            output_dir=output_dir,
            num_points=num_points,
            rgb=rgb,
            normalize=normalize,
            linear_deflection=linear_deflection,
            angular_deflection=angular_deflection,
        )
        return True, str(step_path), str(saved_path)
    except Exception as exc:
        return False, str(step_path), str(exc)


def build_worker_command(
    step_file: Path,
    output_dir: Path,
    num_points: int,
    rgb_text: str,
    normalize: bool,
    linear_deflection: float,
    angular_deflection: float,
    skip_existing: bool,
) -> List[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-single-file",
        str(step_file),
        "--output-dir",
        str(output_dir),
        "--num-points",
        str(num_points),
        "--rgb",
        rgb_text,
        "--linear-deflection",
        str(linear_deflection),
        "--angular-deflection",
        str(angular_deflection),
    ]
    if normalize:
        command.append("--normalize")
    if skip_existing:
        command.append("--skip-existing")
    return command


def run_parallel(
    step_files: List[Path],
    output_dir: Path,
    num_points: int,
    rgb_text: str,
    normalize: bool,
    linear_deflection: float,
    angular_deflection: float,
    max_workers: int,
    skip_existing: bool,
    file_timeout_seconds: int,
) -> tuple[int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)

    succeeded = 0
    failed = 0
    skipped = 0
    failure_log_path = output_dir / "step_to_point_failures.log"

    worker_count = max(1, min(max_workers, len(step_files)))
    max_in_flight = max(worker_count * DEFAULT_MAX_IN_FLIGHT_MULTIPLIER, worker_count)
    print(f"[INFO] Launching {worker_count} worker(s) as isolated subprocesses")
    print(f"[INFO] Max in-flight tasks: {max_in_flight}")
    print(f"[INFO] Per-file timeout: {file_timeout_seconds}s")

    def task_iter():
        for step_file in step_files:
            yield (
                step_file,
                num_points,
                rgb_text,
                normalize,
                linear_deflection,
                angular_deflection,
                skip_existing,
            )

    task_iterator = iter(task_iter())
    pending: dict[subprocess.Popen, tuple[Path, float]] = {}

    def submit_next_task() -> bool:
        try:
            task = next(task_iterator)
        except StopIteration:
            return False

        step_file, task_num_points, task_rgb_text, task_normalize, task_linear_deflection, task_angular_deflection, task_skip_existing = task
        command = build_worker_command(
            step_file=step_file,
            output_dir=output_dir,
            num_points=task_num_points,
            rgb_text=task_rgb_text,
            normalize=task_normalize,
            linear_deflection=task_linear_deflection,
            angular_deflection=task_angular_deflection,
            skip_existing=task_skip_existing,
        )
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        pending[process] = (step_file, time.monotonic())
        return True

    for _ in range(min(max_in_flight, len(step_files))):
        if not submit_next_task():
            break

    with tqdm(total=len(step_files), desc="Converting STEP", unit="file") as progress:
        while pending:
            finished_any = False
            for process, (step_file, start_time) in list(pending.items()):
                return_code = process.poll()
                elapsed = time.monotonic() - start_time

                if return_code is None and elapsed > file_timeout_seconds:
                    process.kill()
                    stdout_text, stderr_text = process.communicate()
                    message = (
                        f"Timeout after {file_timeout_seconds}s"
                        + (f" | stderr: {stderr_text.strip()}" if stderr_text.strip() else "")
                    )
                    failed += 1
                    with failure_log_path.open("a", encoding="utf-8") as failure_log:
                        failure_log.write(f"{step_file}\t{message}\n")
                    pending.pop(process)
                    progress.update(1)
                    finished_any = True
                    continue

                if return_code is None:
                    continue

                stdout_text, stderr_text = process.communicate()
                pending.pop(process)

                if return_code == 0:
                    output_path = output_dir / f"{step_file.stem}_{num_points}.npy"
                    if skip_existing and output_path.exists() and "SKIP" in stdout_text:
                        skipped += 1
                    else:
                        succeeded += 1
                else:
                    failed += 1
                    message = stderr_text.strip() or stdout_text.strip() or f"Worker exited with code {return_code}"
                    with failure_log_path.open("a", encoding="utf-8") as failure_log:
                        failure_log.write(f"{step_file}\t{message}\n")

                progress.update(1)
                finished_any = True

            while len(pending) < max_in_flight:
                if not submit_next_task():
                    break

            progress.set_postfix_str(f"ok={succeeded} skip={skipped} fail={failed} running={len(pending)}")
            if not finished_any:
                time.sleep(0.2)

    print(f"[INFO] Summary: succeeded={succeeded}, skipped={skipped}, failed={failed}")
    if failed > 0:
        print(f"[INFO] Failure log: {failure_log_path}")
    return succeeded, failed


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert STEP files to PointLLM-compatible (N, 6) .npy point clouds")
    parser.add_argument(
        "--input",
        help="Input STEP file path or a directory containing .step/.stp files",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory to save generated .npy point clouds",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        help="Number of sampled points per file, default: 8192",
    )
    parser.add_argument(
        "--rgb",
        help="Uniform RGB color in [0,1], default: 0.5,0.5,0.5",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize xyz into a unit sphere after sampling",
    )
    parser.add_argument(
        "--linear-deflection",
        type=float,
        help="Linear deflection used for STEP tessellation, default: 0.1",
    )
    parser.add_argument(
        "--angular-deflection",
        type=float,
        help="Angular deflection used for STEP tessellation, default: 0.5",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search STEP files when --input is a directory",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        help="Number of worker processes used for parallel conversion",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip output files that already exist",
    )
    parser.add_argument(
        "--file-timeout",
        type=int,
        help="Timeout in seconds for each file, default: 180",
    )
    parser.add_argument(
        "--worker-single-file",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(normalize=None, recursive=None, skip_existing=None)
    return parser


def main() -> int:
    parser = build_argparser()
    args = parser.parse_args()

    input_value = args.input or DEFAULT_INPUT_PATH
    output_value = args.output_dir or DEFAULT_OUTPUT_DIR
    num_points = args.num_points if args.num_points is not None else DEFAULT_NUM_POINTS
    rgb_text = args.rgb if args.rgb is not None else DEFAULT_RGB
    normalize = args.normalize if args.normalize is not None else DEFAULT_NORMALIZE
    linear_deflection = args.linear_deflection if args.linear_deflection is not None else DEFAULT_LINEAR_DEFLECTION
    angular_deflection = args.angular_deflection if args.angular_deflection is not None else DEFAULT_ANGULAR_DEFLECTION
    recursive = args.recursive if args.recursive is not None else DEFAULT_RECURSIVE
    max_workers = args.max_workers if args.max_workers is not None else DEFAULT_MAX_WORKERS
    skip_existing = args.skip_existing if args.skip_existing is not None else DEFAULT_SKIP_EXISTING
    file_timeout_seconds = args.file_timeout if args.file_timeout is not None else DEFAULT_FILE_TIMEOUT_SECONDS

    if not input_value:
        raise ValueError("Please set DEFAULT_INPUT_PATH at the top of the script, or pass --input explicitly")

    input_path = Path(input_value).expanduser().resolve()
    output_dir = Path(output_value).expanduser().resolve()
    parse_rgb(rgb_text)

    if num_points <= 0:
        raise ValueError("--num-points must be a positive integer")
    if max_workers <= 0:
        raise ValueError("--max-workers must be a positive integer")
    if file_timeout_seconds <= 0:
        raise ValueError("--file-timeout must be a positive integer")

    if args.worker_single_file:
        ok, _, message = convert_one_file_from_task(
            (
                args.worker_single_file,
                str(output_dir),
                num_points,
                rgb_text,
                normalize,
                linear_deflection,
                angular_deflection,
                skip_existing,
            )
        )
        if ok:
            print(message)
            return 0
        print(message, file=sys.stderr)
        return 1

    step_files = collect_step_files(input_path, recursive)
    print(f"[INFO] Found {len(step_files)} STEP file(s)")
    print(f"[INFO] Output directory: {output_dir}")
    print(
        f"[INFO] num_points={num_points}, rgb={parse_rgb(rgb_text).tolist()}, "
        f"normalize={normalize}, recursive={recursive}, max_workers={max_workers}, "
        f"skip_existing={skip_existing}, file_timeout={file_timeout_seconds}s"
    )

    _, failed = run_parallel(
        step_files=step_files,
        output_dir=output_dir,
        num_points=num_points,
        rgb_text=rgb_text,
        normalize=normalize,
        linear_deflection=linear_deflection,
        angular_deflection=angular_deflection,
        max_workers=max_workers,
        skip_existing=skip_existing,
        file_timeout_seconds=file_timeout_seconds,
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
