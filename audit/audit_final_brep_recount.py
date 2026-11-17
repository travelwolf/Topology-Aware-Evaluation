from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from holecounting_datasets_build.final_brep_recount import recount_step_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit manifest labels using a history-free recount of final STEP B-reps."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def _resolve_step_path(dataset_root: Path, row: dict[str, str]) -> Path:
    value = row.get("step_path") or row.get("step") or ""
    if not value:
        raise ValueError("Manifest row has no step_path")
    # Dataset manifests were originally written on Windows and retain
    # backslashes when copied to Linux. Normalize separators before resolving
    # the path so the same manifest is portable across both systems.
    path = Path(value.replace("\\", "/"))
    return path if path.is_absolute() else dataset_root / path


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    manifest_path = (args.manifest or dataset_root / "manifest.csv").expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if args.limit is not None:
        rows = rows[: max(args.limit, 0)]

    audit_rows: list[dict[str, object]] = []
    family_totals: Counter[str] = Counter()
    family_matches: Counter[str] = Counter()
    mismatch_reasons: defaultdict[str, Counter[str]] = defaultdict(Counter)

    for index, row in enumerate(rows, start=1):
        model_id = row.get("model_id", f"row_{index:06d}")
        category = row.get("category", "unknown")
        expected = int(row.get("hole_count", "0"))
        step_path = _resolve_step_path(dataset_root, row)
        record: dict[str, object] = {
            "model_id": model_id,
            "category": category,
            "step_path": str(step_path),
            "expected_hole_count": expected,
        }
        try:
            result = recount_step_file(step_path)
            predicted = int(result.through_hole_count)
            matched = predicted == expected
            rejected = Counter(
                component.rejection_reason or "accepted"
                for component in result.components
            )
            record.update(
                {
                    "predicted_hole_count": predicted,
                    "matched": matched,
                    "valid_shape": result.valid_shape,
                    "cylindrical_face_count": result.cylindrical_face_count,
                    "wall_component_count": result.cylindrical_wall_component_count,
                    "component_outcomes": json.dumps(rejected, ensure_ascii=False, sort_keys=True),
                    "error": "",
                }
            )
            family_totals[category] += 1
            family_matches[category] += int(matched)
            if not matched:
                for reason, count in rejected.items():
                    mismatch_reasons[category][reason] += count
        except Exception as exc:  # keep the complete audit running and expose failures
            record.update(
                {
                    "predicted_hole_count": "",
                    "matched": False,
                    "valid_shape": False,
                    "cylindrical_face_count": "",
                    "wall_component_count": "",
                    "component_outcomes": "",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            family_totals[category] += 1
        audit_rows.append(record)
        print(
            f"[{index:05d}/{len(rows):05d}] {model_id} expected={expected} "
            f"predicted={record['predicted_hole_count']} matched={record['matched']}"
        )

    fieldnames = list(audit_rows[0]) if audit_rows else []
    with (output_dir / "recount_predictions.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(audit_rows)

    family_rows = []
    for category in sorted(family_totals):
        total = family_totals[category]
        matches = family_matches[category]
        family_rows.append(
            {
                "category": category,
                "total": total,
                "matches": matches,
                "agreement_rate": matches / total if total else 0.0,
                "mismatch_component_reasons": json.dumps(
                    mismatch_reasons[category], ensure_ascii=False, sort_keys=True
                ),
            }
        )
    with (output_dir / "family_agreement.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(family_rows[0]) if family_rows else [])
        if family_rows:
            writer.writeheader()
            writer.writerows(family_rows)

    total = len(audit_rows)
    matches = sum(bool(row["matched"]) for row in audit_rows)
    errors = sum(bool(row["error"]) for row in audit_rows)
    summary = {
        "manifest": str(manifest_path),
        "dataset_root": str(dataset_root),
        "num_samples": total,
        "num_matches": matches,
        "num_mismatches": total - matches,
        "num_errors": errors,
        "agreement_rate": matches / total if total else 0.0,
        "history_fields_used": [],
        "output_files": ["recount_predictions.csv", "family_agreement.csv"],
    }
    (output_dir / "recount_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if errors == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

