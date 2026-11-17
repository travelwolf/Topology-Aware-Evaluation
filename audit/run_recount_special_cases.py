from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from holecounting_datasets_build.final_brep_recount import recount_through_holes
from holecounting_datasets_build.hole_driller import validate_hole_candidate
from holecounting_datasets_build.occ_utils import (
    cut,
    ensure_valid,
    export_step,
    fuse,
    make_box,
    make_cylinder,
    shape_volume,
)


def _through_hole():
    base = make_box((0.0, 0.0, 0.0), (60.0, 50.0, 20.0))
    tool = make_cylinder((30.0, 25.0, -2.0), (0.0, 0.0, 1.0), 5.0, 24.0)
    return cut(base, tool)


def _two_independent_holes():
    base = make_box((0.0, 0.0, 0.0), (70.0, 50.0, 20.0))
    first = make_cylinder((22.0, 25.0, -2.0), (0.0, 0.0, 1.0), 4.0, 24.0)
    second = make_cylinder((48.0, 25.0, -2.0), (0.0, 0.0, 1.0), 5.0, 24.0)
    return cut(cut(base, first), second)


def _blind_hole():
    base = make_box((0.0, 0.0, 0.0), (60.0, 50.0, 20.0))
    tool = make_cylinder((30.0, 25.0, 10.0), (0.0, 0.0, 1.0), 5.0, 12.0)
    return cut(base, tool)


def _tangential_cut():
    base = make_box((0.0, 0.0, 0.0), (60.0, 50.0, 20.0))
    tool = make_cylinder((0.0, 25.0, -2.0), (0.0, 0.0, 1.0), 5.0, 24.0)
    return cut(base, tool)


def _merged_channels():
    base = make_box((0.0, 0.0, 0.0), (70.0, 50.0, 20.0))
    first = make_cylinder((32.0, 25.0, -2.0), (0.0, 0.0, 1.0), 5.0, 24.0)
    second = make_cylinder((38.0, 25.0, -2.0), (0.0, 0.0, 1.0), 5.0, 24.0)
    return cut(cut(base, first), second)


def _multi_layer_operation():
    lower = make_box((0.0, 0.0, 0.0), (60.0, 50.0, 8.0))
    upper = make_box((0.0, 0.0, 14.0), (60.0, 50.0, 8.0))
    before = fuse(lower, upper)
    request = SimpleNamespace(
        diameter=8.0,
        axis_origin=(30.0, 25.0, -2.0),
        axis_direction=(0.0, 0.0, 1.0),
        tool_length=26.0,
    )
    tool = make_cylinder(
        request.axis_origin,
        request.axis_direction,
        request.diameter / 2.0,
        request.tool_length,
    )
    after = cut(before, tool)
    validation = validate_hole_candidate(
        before,
        after,
        request,
        volume_delta=shape_volume(before) - shape_volume(after),
    )
    return before, after, validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic B-rep recount cases.")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = [
        ("through_hole", _through_hole(), 1),
        ("two_independent_holes", _two_independent_holes(), 2),
        ("blind_hole", _blind_hole(), 0),
        ("tangential_cut", _tangential_cut(), 0),
        ("merged_channels", _merged_channels(), 0),
    ]
    records: list[dict[str, object]] = []
    for name, shape, expected in cases:
        ensure_valid(shape, name)
        export_step(shape, output_dir / f"{name}.step")
        result = recount_through_holes(shape)
        record = {
            "case": name,
            "expected_count": expected,
            "predicted_count": result.through_hole_count,
            "passed": result.through_hole_count == expected,
            "result": result.to_dict(),
        }
        records.append(record)
        print(
            f"{name}: expected={expected} predicted={result.through_hole_count} "
            f"passed={record['passed']}"
        )

    before, after, validation = _multi_layer_operation()
    ensure_valid(before, "multi_layer_before")
    ensure_valid(after, "multi_layer_after")
    export_step(before, output_dir / "multi_layer_before.step")
    export_step(after, output_dir / "multi_layer_after.step")
    multi_layer_recount = recount_through_holes(after)
    recount_passed = multi_layer_recount.through_hole_count == 1
    records.append(
        {
            "case": "multi_layer_final_brep",
            "expected_count": 1,
            "predicted_count": multi_layer_recount.through_hole_count,
            "passed": recount_passed,
            "result": multi_layer_recount.to_dict(),
        }
    )
    print(
        "multi_layer_final_brep: expected=1 "
        f"predicted={multi_layer_recount.through_hole_count} passed={recount_passed}"
    )
    multi_layer_passed = (
        not validation.accepted
        and validation.reject_reason == "discontinuous_material_along_axis"
    )
    records.append(
        {
            "case": "multi_layer_generation_guard",
            "expected_acceptance": False,
            "actual_acceptance": validation.accepted,
            "expected_reject_reason": "discontinuous_material_along_axis",
            "actual_reject_reason": validation.reject_reason,
            "passed": multi_layer_passed,
            "validation": asdict(validation),
        }
    )
    print(
        "multi_layer_operation: expected_acceptance=False "
        f"actual_acceptance={validation.accepted} reason={validation.reject_reason} "
        f"passed={multi_layer_passed}"
    )

    report = {
        "num_cases": len(records),
        "num_passed": sum(bool(record["passed"]) for record in records),
        "all_passed": all(bool(record["passed"]) for record in records),
        "cases": records,
    }
    (output_dir / "special_case_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

