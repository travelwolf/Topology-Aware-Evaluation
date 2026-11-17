from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

from holecounting_datasets_build.category_generators import instantiate_category
from holecounting_datasets_build.final_brep_recount import read_step_shape
from holecounting_datasets_build.hole_sampler import HoleRequest, sample_hole_request
from holecounting_datasets_build import occ_utils
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomAbs import GeomAbs_Cylinder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Produce reviewer-facing dataset composition statistics and a stratified "
            "dense audit of the hole-candidate predicate."
        )
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--lofo-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--samples-per-cell", type=int, default=5)
    parser.add_argument("--dense-samples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20261117)
    parser.add_argument("--families", nargs="*", default=None)
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--accepted-audit", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def normalize_relpath(value: str) -> Path:
    return Path(value.replace("\\", "/"))


def dataset_statistics(
    dataset_root: Path, lofo_summary_path: Path, output_dir: Path
) -> dict[str, object]:
    manifest = read_csv(dataset_root / "manifest.csv")
    categories = sorted({row["category"] for row in manifest})
    counts = sorted({int(row["hole_count"]) for row in manifest})
    matrix = {category: Counter() for category in categories}
    for row in manifest:
        matrix[row["category"]][int(row["hole_count"])] += 1

    matrix_rows: list[dict[str, object]] = []
    for category in categories:
        record: dict[str, object] = {"family": category}
        for count in counts:
            record[f"count_{count}"] = matrix[category][count]
        record["total"] = sum(matrix[category].values())
        matrix_rows.append(record)
    total_row: dict[str, object] = {"family": "TOTAL"}
    for count in counts:
        total_row[f"count_{count}"] = sum(matrix[category][count] for category in categories)
    total_row["total"] = len(manifest)
    matrix_rows.append(total_row)
    write_csv(output_dir / "family_by_count_matrix.csv", matrix_rows)

    split_rows: list[dict[str, object]] = []
    for path in sorted((dataset_root / "splits").glob("*manifest.csv")):
        split_rows.append({"protocol": "random", "split": path.stem, "samples": len(read_csv(path))})

    lofo = json.loads(lofo_summary_path.read_text(encoding="utf-8"))
    lofo_rows: list[dict[str, object]] = []
    for family, fold in lofo["folds"].items():
        lofo_rows.append(
            {
                "family": family,
                "fold_index": fold["index"],
                "train": fold["train"],
                "validation": fold["val"],
                "test": fold["test"],
            }
        )
    write_csv(output_dir / "lofo_split_sizes.csv", sorted(lofo_rows, key=lambda item: int(item["fold_index"])))
    write_csv(output_dir / "random_split_sizes.csv", split_rows)

    return {
        "num_samples": len(manifest),
        "num_families": len(categories),
        "hole_count_min": min(counts),
        "hole_count_max": max(counts),
        "nonempty_family_count_cells": sum(
            matrix[category][count] > 0 for category in categories for count in counts
        ),
        "random_splits": split_rows,
        "lofo_train_range": [min(row["train"] for row in lofo_rows), max(row["train"] for row in lofo_rows)],
        "lofo_validation_range": [
            min(row["validation"] for row in lofo_rows),
            max(row["validation"] for row in lofo_rows),
        ],
        "lofo_test_range": [min(row["test"] for row in lofo_rows), max(row["test"] for row in lofo_rows)],
    }


def stratified_rows(
    rows: list[dict[str, str]], samples_per_cell: int, seed: int, *, positive_only: bool
) -> list[dict[str, str]]:
    grouped: defaultdict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        count = int(row["hole_count"])
        if positive_only and count == 0:
            continue
        grouped[(row["category"], count)].append(row)
    rng = random.Random(seed)
    selected: list[dict[str, str]] = []
    for key in sorted(grouped):
        candidates = grouped[key]
        selected.extend(rng.sample(candidates, min(samples_per_cell, len(candidates))))
    return selected


def request_from_record(record: dict[str, object]) -> HoleRequest:
    return HoleRequest(
        diameter=float(record["diameter"]),
        size_level=str(record["size_level"]),
        axis_name=str(record["axis_name"]),
        axis_origin=tuple(float(value) for value in record["axis_origin"]),
        axis_direction=tuple(float(value) for value in record["axis_direction"]),
        tool_length=float(record["tool_length"]),
    )


def dense_material_intervals(shape, request: HoleRequest, sample_count: int):
    return occ_utils._sample_material_intervals(
        shape,
        tuple(request.axis_origin),
        occ_utils._normalize(tuple(request.axis_direction)),
        float(request.tool_length),
        float(request.diameter) / 2.0,
        sample_count=sample_count,
        tol=1e-7,
    )


def strict_cylindrical_component_count(shape, request: HoleRequest) -> int:
    """Count coaxial wall components with tighter, independent face tolerances."""
    origin = tuple(float(value) for value in request.axis_origin)
    direction = occ_utils._normalize(tuple(float(value) for value in request.axis_direction))
    radius = float(request.diameter) / 2.0
    radius_tol = max(radius * 0.02, 0.01)
    axis_tol = max(radius * 0.03, 0.03)
    alignment_cosine = 0.999

    matching: dict[int, set[int]] = {}
    edge_to_faces: defaultdict[int, set[int]] = defaultdict(set)
    for face in occ_utils._iter_faces(shape):
        surface = BRepAdaptor_Surface(face)
        if surface.GetType() != GeomAbs_Cylinder:
            continue
        cylinder = surface.Cylinder()
        face_direction = occ_utils._normalize(
            (
                float(cylinder.Axis().Direction().X()),
                float(cylinder.Axis().Direction().Y()),
                float(cylinder.Axis().Direction().Z()),
            )
        )
        if abs(occ_utils._dot(face_direction, direction)) < alignment_cosine:
            continue
        if abs(float(cylinder.Radius()) - radius) > radius_tol:
            continue
        location = cylinder.Axis().Location()
        delta = (
            float(location.X()) - origin[0],
            float(location.Y()) - origin[1],
            float(location.Z()) - origin[2],
        )
        axial = occ_utils._dot(delta, direction)
        radial = (
            delta[0] - axial * direction[0],
            delta[1] - axial * direction[1],
            delta[2] - axial * direction[2],
        )
        if math.sqrt(occ_utils._dot(radial, radial)) > axis_tol:
            continue
        face_hash = occ_utils._shape_hash(face)
        edges = {occ_utils._shape_hash(edge) for edge in occ_utils._iter_edges(face)}
        matching[face_hash] = edges
        for edge_hash in edges:
            edge_to_faces[edge_hash].add(face_hash)

    remaining = set(matching)
    component_count = 0
    while remaining:
        component_count += 1
        stack = [remaining.pop()]
        while stack:
            current = stack.pop()
            for edge_hash in matching[current]:
                for neighbor in edge_to_faces[edge_hash]:
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        stack.append(neighbor)
    return component_count


def dense_validate(before_shape, after_shape, request: HoleRequest, volume_delta: float, sample_count: int):
    if volume_delta <= 0.0:
        return {"accepted": False, "reject_reason": "non_positive_volume_delta"}
    before_intervals = dense_material_intervals(before_shape, request, sample_count)
    if not before_intervals:
        return {"accepted": False, "reject_reason": "no_material_along_axis"}
    if len(before_intervals) > 1:
        return {"accepted": False, "reject_reason": "discontinuous_material_along_axis"}
    after_intervals = dense_material_intervals(after_shape, request, sample_count)
    if after_intervals:
        return {"accepted": False, "reject_reason": "residual_material_after_cut"}

    before_components = strict_cylindrical_component_count(before_shape, request)
    after_components = strict_cylindrical_component_count(after_shape, request)
    channel_delta = max(0, after_components - before_components)
    accepted = channel_delta == 1
    return {
        "accepted": accepted,
        "reject_reason": None if accepted else "topology_validation_failed",
        "channel_component_count": channel_delta,
        "before_interval_count": len(before_intervals),
        "after_interval_count": len(after_intervals),
    }


def audit_existing_acceptances(
    dataset_root: Path,
    output_dir: Path,
    samples_per_cell: int,
    dense_samples: int,
    seed: int,
    families: set[str] | None,
) -> dict[str, object]:
    manifest = read_csv(dataset_root / "manifest.csv")
    if families:
        manifest = [row for row in manifest if row["category"] in families]
    selected = stratified_rows(manifest, samples_per_cell, seed, positive_only=True)
    rows: list[dict[str, object]] = []
    sample_failures = 0
    checked_holes = 0
    failed_holes = 0
    for index, manifest_row in enumerate(selected, start=1):
        label_path = dataset_root / normalize_relpath(manifest_row["label_path"])
        step_path = dataset_root / normalize_relpath(manifest_row["step_path"])
        label = json.loads(label_path.read_text(encoding="utf-8"))
        shape = read_step_shape(step_path)
        model_failed = False
        for hole in label.get("holes", []):
            request = request_from_record(hole)
            residual = dense_material_intervals(shape, request, dense_samples)
            topology = occ_utils.analyze_drilled_hole_topology(shape, request)
            strict_components = strict_cylindrical_component_count(shape, request)
            # Later non-merging cuts can split an opening wire without changing the
            # underlying channel.  The final-solid audit therefore checks the two
            # invariant properties: no residual material on the recorded axis and
            # at least one connected cylindrical wall component on that axis.
            accepted = not residual and strict_components >= 1
            checked_holes += 1
            failed_holes += int(not accepted)
            model_failed = model_failed or not accepted
            rows.append(
                {
                    "model_id": manifest_row["model_id"],
                    "family": manifest_row["category"],
                    "label_count": int(manifest_row["hole_count"]),
                    "hole_id": int(hole["hole_id"]),
                    "dense_residual_interval_count": len(residual),
                    "opening_pair_count": int(topology["opening_pair_count"]),
                    "channel_component_count": int(topology["channel_component_count"]),
                    "strict_channel_component_count": strict_components,
                    "audit_passed": accepted,
                }
            )
        sample_failures += int(model_failed)
        if index % 25 == 0 or index == len(selected):
            print(f"accepted-audit {index}/{len(selected)}")
    write_csv(output_dir / "accepted_label_dense_audit.csv", rows)
    return {
        "stratification": "family x positive hole count",
        "samples_per_nonempty_cell": samples_per_cell,
        "sampled_models": len(selected),
        "checked_holes": checked_holes,
        "false_acceptance_models": sample_failures,
        "false_acceptance_holes": failed_holes,
        "model_false_acceptance_rate": sample_failures / len(selected) if selected else 0.0,
        "hole_false_acceptance_rate": failed_holes / checked_holes if checked_holes else 0.0,
        "dense_axis_samples": dense_samples,
        "classifier_tolerance": 1e-7,
        "strict_wall_radius_tolerance": "max(0.02r, 0.01)",
        "strict_wall_axis_distance_tolerance": "max(0.03r, 0.03)",
        "strict_wall_alignment_cosine": 0.999,
    }


def replay_candidate_audit(
    dataset_root: Path,
    output_dir: Path,
    samples_per_cell: int,
    dense_samples: int,
    seed: int,
    families: set[str] | None,
) -> dict[str, object]:
    manifest = read_csv(dataset_root / "manifest.csv")
    if families:
        manifest = [row for row in manifest if row["category"] in families]
    blueprints = stratified_rows(manifest, samples_per_cell, seed + 1, positive_only=True)
    rng = random.Random(seed + 2)
    event_rows: list[dict[str, object]] = []
    model_rows: list[dict[str, object]] = []
    by_family: defaultdict[str, Counter[str]] = defaultdict(Counter)
    total = Counter()

    for model_index, blueprint in enumerate(blueprints, start=1):
        family = blueprint["category"]
        condition = blueprint["condition_level"]
        target = int(blueprint["hole_count"])
        instance = instantiate_category(family, rng)
        shape = instance.shape
        accepted_requests: list[HoleRequest] = []
        accepted_count = 0
        attempts = 0
        max_attempts = max(10, target * 10)
        model_events = Counter()

        while accepted_count < target and attempts < max_attempts:
            request = sample_hole_request(
                rng=rng,
                condition_level=condition,
                preferred_axes=instance.preferred_axes,
                bbox=occ_utils.get_bbox(shape),
                clearance=0.8,
                existing_requests=accepted_requests,
            )
            if request is None:
                model_events["placement_sampling_exhausted"] += 1
                total["placement_sampling_exhausted"] += 1
                by_family[family]["placement_sampling_exhausted"] += 1
                break

            attempts += 1
            total["geometry_candidates"] += 1
            by_family[family]["geometry_candidates"] += 1
            model_events["geometry_candidates"] += 1
            tool = occ_utils.make_cylinder(
                request.axis_origin,
                request.axis_direction,
                request.diameter / 2.0,
                request.tool_length,
            )
            before_volume = occ_utils.shape_volume(shape)
            candidate = occ_utils.cut(shape, tool)
            valid_shape = True
            try:
                occ_utils.ensure_valid(candidate, f"audit_{model_index}_{attempts}")
            except RuntimeError:
                valid_shape = False
            after_volume = occ_utils.shape_volume(candidate) if valid_shape else before_volume
            volume_delta = before_volume - after_volume
            if valid_shape:
                original = occ_utils.validate_hole_candidate(shape, candidate, request, volume_delta)
                dense = dense_validate(shape, candidate, request, volume_delta, dense_samples)
            else:
                original = {"accepted": False, "reject_reason": "invalid_shape"}
                dense = {"accepted": False, "reject_reason": "invalid_shape"}

            original_accepted = bool(original.get("accepted", False))
            dense_accepted = bool(dense.get("accepted", False))
            if original_accepted:
                total["original_accepted"] += 1
                by_family[family]["original_accepted"] += 1
                model_events["original_accepted"] += 1
            else:
                total["original_rejected"] += 1
                by_family[family]["original_rejected"] += 1
                model_events["original_rejected"] += 1
            if original_accepted and not dense_accepted:
                total["false_acceptances"] += 1
                by_family[family]["false_acceptances"] += 1
                model_events["false_acceptances"] += 1
            if not original_accepted and dense_accepted:
                total["false_rejections"] += 1
                by_family[family]["false_rejections"] += 1
                model_events["false_rejections"] += 1

            event_rows.append(
                {
                    "audit_model": model_index,
                    "family": family,
                    "condition_level": condition,
                    "target_hole_count": target,
                    "candidate_index": attempts,
                    "diameter": request.diameter,
                    "axis_name": request.axis_name,
                    "original_accepted": original_accepted,
                    "dense_accepted": dense_accepted,
                    "original_reject_reason": original.get("reject_reason") or "",
                    "dense_reject_reason": dense.get("reject_reason") or "",
                    "false_acceptance": original_accepted and not dense_accepted,
                    "false_rejection": (not original_accepted) and dense_accepted,
                }
            )

            if original_accepted:
                shape = candidate
                accepted_requests.append(request)
                accepted_count += 1

        model_rows.append(
            {
                "audit_model": model_index,
                "family": family,
                "condition_level": condition,
                "target_hole_count": target,
                "accepted_hole_count": accepted_count,
                "geometry_candidates": model_events["geometry_candidates"],
                "rejected_candidates": model_events["original_rejected"],
                "false_acceptances": model_events["false_acceptances"],
                "false_rejections": model_events["false_rejections"],
                "underfilled": accepted_count != target,
            }
        )
        total["underfilled_models"] += int(accepted_count != target)
        by_family[family]["models"] += 1
        by_family[family]["underfilled_models"] += int(accepted_count != target)
        if model_index % 10 == 0 or model_index == len(blueprints):
            print(f"candidate-replay {model_index}/{len(blueprints)}")

    write_csv(output_dir / "candidate_events.csv", event_rows)
    write_csv(output_dir / "candidate_models.csv", model_rows)
    family_rows: list[dict[str, object]] = []
    for family in sorted(by_family):
        values = by_family[family]
        candidates = values["geometry_candidates"]
        rejected = values["original_rejected"]
        family_rows.append(
            {
                "family": family,
                "models": values["models"],
                "geometry_candidates": candidates,
                "accepted_candidates": values["original_accepted"],
                "rejected_candidates": rejected,
                "candidate_rejection_rate": rejected / candidates if candidates else 0.0,
                "false_acceptances": values["false_acceptances"],
                "false_rejections": values["false_rejections"],
                "underfilled_models": values["underfilled_models"],
            }
        )
    write_csv(output_dir / "candidate_rejection_by_family.csv", family_rows)

    candidates = total["geometry_candidates"]
    rejected = total["original_rejected"]
    return {
        "stratification": "family x positive hole count",
        "samples_per_nonempty_cell": samples_per_cell,
        "audit_models": len(blueprints),
        "geometry_candidates": candidates,
        "accepted_candidates": total["original_accepted"],
        "rejected_candidates": rejected,
        "candidate_rejection_rate": rejected / candidates if candidates else 0.0,
        "false_acceptances": total["false_acceptances"],
        "false_rejections": total["false_rejections"],
        "false_acceptance_rate": total["false_acceptances"] / total["original_accepted"]
        if total["original_accepted"]
        else 0.0,
        "false_rejection_rate": total["false_rejections"] / rejected if rejected else 0.0,
        "underfilled_models": total["underfilled_models"],
        "placement_sampling_exhausted": total["placement_sampling_exhausted"],
        "dense_axis_samples": dense_samples,
        "classifier_tolerance": 1e-7,
        "strict_wall_radius_tolerance": "max(0.02r, 0.01)",
        "strict_wall_axis_distance_tolerance": "max(0.03r, 0.03)",
        "strict_wall_alignment_cosine": 0.999,
    }


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {
        "seed": args.seed,
        "families": args.families or "all",
        "dataset_statistics": dataset_statistics(dataset_root, args.lofo_summary, output_dir),
    }
    if args.accepted_audit:
        summary["accepted_label_audit"] = audit_existing_acceptances(
            dataset_root,
            output_dir,
            args.samples_per_cell,
            args.dense_samples,
            args.seed,
            set(args.families) if args.families else None,
        )
    if args.replay:
        summary["candidate_replay_audit"] = replay_candidate_audit(
            dataset_root,
            output_dir,
            args.samples_per_cell,
            args.dense_samples,
            args.seed,
            set(args.families) if args.families else None,
        )
    (output_dir / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
