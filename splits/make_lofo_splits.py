from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260610)
    args = parser.parse_args()

    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if not rows or "category" not in fieldnames or "hole_count" not in fieldnames:
        raise ValueError("Manifest must contain category and hole_count columns.")

    families = sorted({row["category"] for row in rows})
    summary: dict[str, object] = {
        "source_manifest": str(args.manifest.resolve()),
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "folds": {},
    }
    for fold_index, held_out in enumerate(families):
        test_rows = [row for row in rows if row["category"] == held_out]
        candidates = [row for row in rows if row["category"] != held_out]
        groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
        for row in candidates:
            groups[(row["category"], row["hole_count"])].append(row)

        val_ids: set[str] = set()
        for group_index, key in enumerate(sorted(groups)):
            group = list(groups[key])
            random.Random(args.seed + fold_index * 1000 + group_index).shuffle(group)
            count = max(1, round(len(group) * args.val_ratio)) if len(group) > 1 else 0
            val_ids.update(row["model_id"] for row in group[:count])

        train_rows = [row for row in candidates if row["model_id"] not in val_ids]
        val_rows = [row for row in candidates if row["model_id"] in val_ids]
        fold_dir = args.output_dir / f"fold_{fold_index:02d}_{held_out}"
        write_rows(fold_dir / "train.csv", fieldnames, train_rows)
        write_rows(fold_dir / "val.csv", fieldnames, val_rows)
        write_rows(fold_dir / "test.csv", fieldnames, test_rows)
        summary["folds"][held_out] = {
            "index": fold_index,
            "train": len(train_rows),
            "val": len(val_rows),
            "test": len(test_rows),
            "train_families": sorted({row["category"] for row in train_rows}),
            "test_hole_counts": dict(sorted(defaultdict(int, {
                str(h): sum(row["hole_count"] == str(h) for row in test_rows)
                for h in range(9)
            }).items())),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
