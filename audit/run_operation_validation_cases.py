from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from holecounting_datasets_build.hole_driller import validate_hole_candidate
from holecounting_datasets_build.hole_sampler import HoleRequest, _safe_band, holes_would_merge
from holecounting_datasets_build.occ_utils import (
    cut,
    ensure_valid,
    export_step,
    fuse,
    make_box,
    make_cylinder,
    shape_volume,
)


def _request(
    *,
    diameter: float = 10.0,
    origin: tuple[float, float, float] = (30.0, 25.0, -2.0),
    length: float = 24.0,
) -> HoleRequest:
    return HoleRequest(
        diameter=diameter,
        size_level="large",
        axis_name="z",
        axis_origin=origin,
        axis_direction=(0.0, 0.0, 1.0),
        tool_length=length,
    )


def _validate_geometry_case(
    *,
    name: str,
    before,
    after,
    request: HoleRequest,
    expected_acceptance: bool,
    expected_reason: str | None,
    output_dir: Path,
) -> dict[str, object]:
    ensure_valid(before, f"{name}_before")
    ensure_valid(after, f"{name}_after")
    export_step(before, output_dir / f"{name}_before.step")
    export_step(after, output_dir / f"{name}_after.step")

    volume_delta = shape_volume(before) - shape_volume(after)
    result = validate_hole_candidate(
        before,
        after,
        request,
        volume_delta=volume_delta,
    )
    passed = result.accepted == expected_acceptance and result.reject_reason == expected_reason
    return {
        "case": name,
        "stage": "operation_validation",
        "expected_acceptance": expected_acceptance,
        "actual_acceptance": result.accepted,
        "expected_reject_reason": expected_reason,
        "actual_reject_reason": result.reject_reason,
        "removed_volume": volume_delta,
        "opening_pair_count": result.opening_pair_count,
        "channel_component_count": result.channel_component_count,
        "passed": passed,
        "validation": asdict(result),
    }


def _valid_through_case(output_dir: Path) -> dict[str, object]:
    before = make_box((0.0, 0.0, 0.0), (60.0, 50.0, 20.0))
    request = _request()
    tool = make_cylinder(
        request.axis_origin,
        request.axis_direction,
        request.diameter / 2.0,
        request.tool_length,
    )
    return _validate_geometry_case(
        name="valid_through_hole",
        before=before,
        after=cut(before, tool),
        request=request,
        expected_acceptance=True,
        expected_reason=None,
        output_dir=output_dir,
    )


def _blind_case(output_dir: Path) -> dict[str, object]:
    before = make_box((0.0, 0.0, 0.0), (60.0, 50.0, 20.0))
    request = _request()
    blind_tool = make_cylinder(
        request.axis_origin,
        request.axis_direction,
        request.diameter / 2.0,
        12.0,
    )
    return _validate_geometry_case(
        name="blind_hole",
        before=before,
        after=cut(before, blind_tool),
        request=request,
        expected_acceptance=False,
        expected_reason="residual_material_after_cut",
        output_dir=output_dir,
    )


def _tangential_case(output_dir: Path) -> dict[str, object]:
    before = make_box((0.0, 0.0, 0.0), (60.0, 50.0, 20.0))
    request = _request(origin=(0.0, 25.0, -2.0))
    tool = make_cylinder(
        request.axis_origin,
        request.axis_direction,
        request.diameter / 2.0,
        request.tool_length,
    )
    after = cut(before, tool)
    ensure_valid(before, "tangential_cut_before")
    ensure_valid(after, "tangential_cut_after")
    export_step(before, output_dir / "tangential_cut_before.step")
    export_step(after, output_dir / "tangential_cut_after.step")

    x_band = _safe_band(0.0, 60.0)
    y_band = _safe_band(0.0, 50.0)
    x, y, _ = request.axis_origin
    inside_sampling_band = x_band[0] <= x <= x_band[1] and y_band[0] <= y <= y_band[1]
    return {
        "case": "tangential_cut",
        "stage": "candidate_sampling",
        "expected_acceptance": False,
        "actual_acceptance": inside_sampling_band,
        "expected_reject_reason": "outside_transverse_sampling_band",
        "actual_reject_reason": None if inside_sampling_band else "outside_transverse_sampling_band",
        "axis_transverse_position": [x, y],
        "allowed_x_band": list(x_band),
        "allowed_y_band": list(y_band),
        "passed": not inside_sampling_band,
    }


def _merged_channel_case() -> dict[str, object]:
    first = _request(origin=(32.0, 25.0, -2.0))
    second = _request(origin=(38.0, 25.0, -2.0))
    blocked = holes_would_merge(second, first, clearance=0.8)
    return {
        "case": "merged_channels",
        "stage": "candidate_sampling",
        "expected_acceptance": False,
        "actual_acceptance": not blocked,
        "expected_reject_reason": "insufficient_centerline_clearance",
        "actual_reject_reason": "insufficient_centerline_clearance" if blocked else None,
        "centerline_distance": 6.0,
        "required_minimum_distance": 10.8,
        "passed": blocked,
    }


def _multi_layer_case(output_dir: Path) -> dict[str, object]:
    lower = make_box((0.0, 0.0, 0.0), (60.0, 50.0, 8.0))
    upper = make_box((0.0, 0.0, 14.0), (60.0, 50.0, 8.0))
    side_bridge = make_box((0.0, 0.0, 0.0), (6.0, 50.0, 22.0))
    before = fuse(fuse(lower, upper), side_bridge)
    request = _request(diameter=8.0, length=26.0)
    tool = make_cylinder(
        request.axis_origin,
        request.axis_direction,
        request.diameter / 2.0,
        request.tool_length,
    )
    return _validate_geometry_case(
        name="multi_layer_cut",
        before=before,
        after=cut(before, tool),
        request=request,
        expected_acceptance=False,
        expected_reason="discontinuous_material_along_axis",
        output_dir=output_dir,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run deterministic cases for the operation-level hole validator."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = [
        _valid_through_case(output_dir),
        _blind_case(output_dir),
        _tangential_case(output_dir),
        _merged_channel_case(),
        _multi_layer_case(output_dir),
    ]
    for record in records:
        print(
            f"{record['case']}: accepted={record['actual_acceptance']} "
            f"reason={record['actual_reject_reason']} passed={record['passed']}"
        )

    report = {
        "validator": "operation_level",
        "num_cases": len(records),
        "num_passed": sum(bool(record["passed"]) for record in records),
        "all_passed": all(bool(record["passed"]) for record in records),
        "cases": records,
    }
    (output_dir / "operation_validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
