from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    family_dirs = sorted(path.parent for path in args.input_root.glob("*/audit_summary.json"))
    if not family_dirs:
        raise SystemExit("No completed family summaries found")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    accepted_keys = [
        "sampled_models",
        "checked_holes",
        "false_acceptance_models",
        "false_acceptance_holes",
    ]
    replay_keys = [
        "audit_models",
        "geometry_candidates",
        "accepted_candidates",
        "rejected_candidates",
        "false_acceptances",
        "false_rejections",
        "underfilled_models",
        "placement_sampling_exhausted",
    ]
    accepted = {key: 0 for key in accepted_keys}
    replay = {key: 0 for key in replay_keys}
    family_rows: list[dict[str, object]] = []
    accepted_rows: list[dict[str, str]] = []
    event_rows: list[dict[str, str]] = []
    model_rows: list[dict[str, str]] = []

    for family_dir in family_dirs:
        summary = json.loads((family_dir / "audit_summary.json").read_text(encoding="utf-8"))
        for key in accepted_keys:
            accepted[key] += int(summary["accepted_label_audit"][key])
        for key in replay_keys:
            replay[key] += int(summary["candidate_replay_audit"][key])
        family_rows.extend(read_csv(family_dir / "candidate_rejection_by_family.csv"))
        accepted_rows.extend(read_csv(family_dir / "accepted_label_dense_audit.csv"))
        event_rows.extend(read_csv(family_dir / "candidate_events.csv"))
        model_rows.extend(read_csv(family_dir / "candidate_models.csv"))

    accepted["model_false_acceptance_rate"] = (
        accepted["false_acceptance_models"] / accepted["sampled_models"]
        if accepted["sampled_models"]
        else 0.0
    )
    accepted["hole_false_acceptance_rate"] = (
        accepted["false_acceptance_holes"] / accepted["checked_holes"]
        if accepted["checked_holes"]
        else 0.0
    )
    replay["candidate_rejection_rate"] = (
        replay["rejected_candidates"] / replay["geometry_candidates"]
        if replay["geometry_candidates"]
        else 0.0
    )
    replay["false_acceptance_rate"] = (
        replay["false_acceptances"] / replay["accepted_candidates"]
        if replay["accepted_candidates"]
        else 0.0
    )
    replay["false_rejection_rate"] = (
        replay["false_rejections"] / replay["rejected_candidates"]
        if replay["rejected_candidates"]
        else 0.0
    )

    combined = {
        "completed_families": len(family_dirs),
        "accepted_label_audit": accepted,
        "candidate_replay_audit": replay,
        "dense_axis_samples": 2048,
        "classifier_tolerance": 1e-7,
    }
    (args.output_dir / "combined_audit_summary.json").write_text(
        json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(args.output_dir / "candidate_rejection_by_family.csv", family_rows)
    write_csv(args.output_dir / "accepted_label_dense_audit.csv", accepted_rows)
    write_csv(args.output_dir / "candidate_events.csv", event_rows)
    write_csv(args.output_dir / "candidate_models.csv", model_rows)

    first = family_dirs[0]
    for name in ("family_by_count_matrix.csv", "random_split_sizes.csv", "lofo_split_sizes.csv"):
        shutil.copy2(first / name, args.output_dir / name)

    rates = {row["family"]: 100.0 * float(row["candidate_rejection_rate"]) for row in family_rows}
    matrix = read_csv(args.output_dir / "family_by_count_matrix.csv")
    display_rows: list[dict[str, object]] = []
    for row in matrix:
        family = row["family"]
        display: dict[str, object] = dict(row)
        display["candidate_rejection_percent"] = "--" if family == "TOTAL" else f"{rates[family]:.2f}"
        display_rows.append(display)
    write_csv(args.output_dir / "family_count_and_rejection.csv", display_rows)

    print(json.dumps(combined, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
