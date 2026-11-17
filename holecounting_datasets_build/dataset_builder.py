from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import tempfile
import uuid
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from holecounting_datasets_build.category_generators import CATEGORY_BUILDERS, instantiate_category
from holecounting_datasets_build.exporters import (
    DEFAULT_NUM_POINTS,
    export_graph_bin_file,
    export_overview_image,
    export_point_cloud_file,
    export_step_file,
)
from holecounting_datasets_build.hole_driller import apply_holes
from holecounting_datasets_build.hole_sampler import choose_condition, sample_condition_hole_count, sample_hole_request
from holecounting_datasets_build.occ_utils import ensure_valid, get_bbox

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


QUIET_MODE = False
MODEL_ID_PATTERN = re.compile(r"^model_(\d{6})$")


def progress_write(message: str) -> None:
    if QUIET_MODE:
        return
    if tqdm is not None:
        tqdm.write(message)
    else:
        print(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build multi-modal hole-counting dataset samples.")
    parser.add_argument("--output-root", type=Path, default=Path("holecounting_datasets_build") / "test_outputs")
    parser.add_argument("--num-models", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument("--zero-hole-ratio", type=float, default=0.08)
    parser.add_argument(
        "--categories",
        nargs="*",
        default=sorted(CATEGORY_BUILDERS.keys()),
        help="Optional subset of categories to generate.",
    )
    parser.add_argument("--linear-deflection", type=float, default=0.08)
    parser.add_argument("--angular-deflection", type=float, default=0.4)
    parser.add_argument("--hole-clearance", type=float, default=0.8)
    parser.add_argument("--export-step", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--export-point", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--export-image", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--export-label", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--export-bin", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resume", action="store_true", help="Resume from existing manifest.csv and skip completed model_ids.")
    parser.add_argument("--quiet", action="store_true", help="Keep progress bar but suppress extra terminal logs.")
    return parser.parse_args()


def prepare_dirs(output_root: Path, export_flags: dict[str, bool]) -> dict[str, Path]:
    dirs = {
        "step": output_root / "step",
        "point": output_root / "point",
        "image": output_root / "image",
        "label": output_root / "label",
        "bin": output_root / "bin",
        "preview": output_root / "preview",
    }
    for key, path in dirs.items():
        if key == "preview" or export_flags.get(key, False):
            path.mkdir(parents=True, exist_ok=True)
    return dirs


def _write_label_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_manifest_row(manifest_path: Path, row: dict) -> None:
    file_exists = manifest_path.exists()
    with manifest_path.open("a", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "model_id",
                "category",
                "condition_level",
                "has_internal_structure",
                "hole_count",
                "step_path",
                "point_path",
                "image_path",
                "bin_path",
                "label_path",
            ],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _append_failure_log(failure_log_path: Path, row: dict) -> None:
    failure_log_path.parent.mkdir(parents=True, exist_ok=True)
    with failure_log_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_completed_model_ids(manifest_path: Path, max_index: int) -> set[str]:
    if not manifest_path.exists():
        return set()
    completed_ids: set[str] = set()
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            model_id = (row.get("model_id") or "").strip()
            match = MODEL_ID_PATTERN.match(model_id)
            if not match:
                continue
            if int(match.group(1)) <= max_index:
                completed_ids.add(model_id)
    return completed_ids


def build_one_model(
    model_id: str,
    category: str,
    rng: random.Random,
    dirs: dict[str, Path],
    linear_deflection: float,
    angular_deflection: float,
    zero_hole_ratio: float,
    hole_clearance: float,
    output_root: Path,
    export_flags: dict[str, bool],
    failure_log_path: Path | None = None,
) -> dict | None:
    instance = instantiate_category(category, rng)
    condition_level = choose_condition(rng, instance.allowed_conditions)
    target_hole_count = sample_condition_hole_count(rng, condition_level, zero_hole_ratio)

    shape = instance.shape
    hole_records: list[dict] = []
    accepted_requests = []
    attempts = 0
    max_attempts = max(10, target_hole_count * 10)

    while len(hole_records) < target_hole_count and attempts < max_attempts:
        request = sample_hole_request(
            rng=rng,
            condition_level=condition_level,
            preferred_axes=instance.preferred_axes,
            bbox=get_bbox(shape),
            clearance=hole_clearance,
            existing_requests=accepted_requests,
        )
        if request is None:
            break
        candidate_shape, new_records = apply_holes(shape, [request])
        if new_records:
            shape = candidate_shape
            hole_records.extend(new_records)
            accepted_requests.append(request)
        attempts += 1

    for hole_index, hole_record in enumerate(hole_records, start=1):
        hole_record["hole_id"] = hole_index

    failure_log_path = failure_log_path or (output_root / "failure_log.jsonl")
    if len(hole_records) != target_hole_count:
        _append_failure_log(
            failure_log_path,
            {
                "model_id": model_id,
                "category": instance.category,
                "condition_level": condition_level,
                "failure_reason": "underfilled_hole_count",
                "target_hole_count": target_hole_count,
                "accepted_hole_count": len(hole_records),
                "attempt_count": attempts,
            },
        )
        return None

    ensure_valid(shape, model_id)

    step_path = dirs["step"] / f"{model_id}.step"
    point_path = dirs["point"] / f"{model_id}.npy"
    image_path = dirs["image"] / f"{model_id}_overview.png"
    bin_path = dirs["bin"] / f"{model_id}.bin"
    label_path = dirs["label"] / f"{model_id}.json"

    if export_flags["step"]:
        export_step_file(shape, step_path)
    if export_flags["point"]:
        export_point_cloud_file(
            shape,
            point_path,
            num_points=DEFAULT_NUM_POINTS,
            linear_deflection=linear_deflection,
            angular_deflection=angular_deflection,
        )
    if export_flags["image"]:
        if export_flags["step"]:
            export_overview_image(step_path, image_path)
        else:
            temp_step_path = Path(tempfile.gettempdir()) / f"{model_id}_{uuid.uuid4().hex}.step"
            try:
                export_step_file(shape, temp_step_path)
                export_overview_image(temp_step_path, image_path)
            finally:
                temp_step_path.unlink(missing_ok=True)
    if export_flags["bin"]:
        export_graph_bin_file(shape, bin_path)

    payload = {
        "model_id": model_id,
        "category": instance.category,
        "condition_level": condition_level,
        "has_internal_structure": instance.has_internal_structure,
        "base_params": instance.base_params,
        "hole_count": len(hole_records),
        "target_hole_count": target_hole_count,
        "holes": hole_records,
        "exports": {},
    }
    if export_flags["step"]:
        payload["exports"]["step"] = str(step_path.relative_to(output_root))
    if export_flags["point"]:
        payload["exports"]["point"] = str(point_path.relative_to(output_root))
    if export_flags["image"]:
        payload["exports"]["image"] = str(image_path.relative_to(output_root))
    if export_flags["bin"]:
        payload["exports"]["bin"] = str(bin_path.relative_to(output_root))
    if export_flags["label"]:
        _write_label_json(label_path, payload)
    _append_manifest_row(
        output_root / "manifest.csv",
        {
            "model_id": model_id,
            "category": instance.category,
            "condition_level": condition_level,
            "has_internal_structure": instance.has_internal_structure,
            "hole_count": len(hole_records),
            "step_path": str(step_path.relative_to(output_root)) if export_flags["step"] else "",
            "point_path": str(point_path.relative_to(output_root)) if export_flags["point"] else "",
            "image_path": str(image_path.relative_to(output_root)) if export_flags["image"] else "",
            "bin_path": str(bin_path.relative_to(output_root)) if export_flags["bin"] else "",
            "label_path": str(label_path.relative_to(output_root)) if export_flags["label"] else "",
        },
    )
    return payload


def main() -> int:
    args = parse_args()
    global QUIET_MODE
    QUIET_MODE = args.quiet
    if args.num_models <= 0:
        raise ValueError("--num-models must be positive")
    if not 0.0 <= args.zero_hole_ratio < 1.0:
        raise ValueError("--zero-hole-ratio must be in [0, 1)")
    for category in args.categories:
        if category not in CATEGORY_BUILDERS:
            raise ValueError(f"Unknown category: {category}")

    export_flags = {
        "step": args.export_step,
        "point": args.export_point,
        "image": args.export_image,
        "label": args.export_label,
        "bin": args.export_bin,
    }
    if not any(export_flags.values()):
        raise ValueError("At least one export target must be enabled.")

    output_root = args.output_root.expanduser().resolve()
    dirs = prepare_dirs(output_root, export_flags)
    rng = random.Random(args.seed)
    manifest_path = output_root / "manifest.csv"
    failure_log_path = output_root / "failure_log.jsonl"
    completed_model_ids = _load_completed_model_ids(manifest_path, args.num_models) if args.resume else set()
    completed_count = len(completed_model_ids)
    pending_indices = [
        index
        for index in range(1, args.num_models + 1)
        if f"model_{index:06d}" not in completed_model_ids
    ]

    progress_write(f"[info] output_root={output_root}")
    progress_write(f"[info] num_models={args.num_models}, seed={args.seed}")
    progress_write(f"[info] categories={args.categories}")
    progress_write(f"[info] export_flags={export_flags}")
    if args.resume:
        progress_write(
            f"[info] resume=True completed={completed_count} pending={len(pending_indices)} "
            f"manifest={manifest_path}"
        )

    if tqdm is not None:
        progress = tqdm(total=args.num_models, initial=completed_count, desc="build dataset", dynamic_ncols=True)
    else:
        progress = None

    success_count = completed_count
    pending_cursor = 0
    next_index = args.num_models + 1 if args.resume else 1

    while success_count < args.num_models:
        if args.resume and pending_cursor < len(pending_indices):
            index = pending_indices[pending_cursor]
            pending_cursor += 1
        else:
            index = next_index
            next_index += 1
        category = rng.choice(args.categories)
        model_id = f"model_{index:06d}"
        try:
            payload = build_one_model(
                model_id=model_id,
                category=category,
                rng=rng,
                dirs=dirs,
                linear_deflection=args.linear_deflection,
                angular_deflection=args.angular_deflection,
                zero_hole_ratio=args.zero_hole_ratio,
                hole_clearance=args.hole_clearance,
                output_root=output_root,
                export_flags=export_flags,
                failure_log_path=failure_log_path,
            )
        except Exception as exc:
            _append_failure_log(
                failure_log_path,
                {
                    "model_id": model_id,
                    "category": category,
                    "failure_reason": "build_exception",
                    "error": repr(exc),
                },
            )
            payload = None

        if payload is None:
            if progress is not None:
                progress.set_postfix(model_id=model_id, status="retry")
            continue

        success_count += 1
        if progress is not None:
            progress.update(1)
            progress.set_postfix(
                model_id=model_id,
                category=payload["category"],
                condition=payload["condition_level"],
                holes=payload["hole_count"],
            )

    if progress is not None:
        progress.close()

    progress_write(f"[ok] manifest -> {output_root / 'manifest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
