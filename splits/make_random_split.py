from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split hole-counting manifest into train/val manifests.")
    parser.add_argument("--manifest", type=Path, required=True, help="Input manifest.csv path.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for split manifests.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation ratio inside each category.")
    parser.add_argument("--seed", type=int, default=20260610, help="Random seed.")
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_manifest(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def split_by_category(
    rows: list[dict[str, str]],
    val_ratio: float,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    rng = random.Random(seed)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["category"]].append(row)

    train_rows: list[dict[str, str]] = []
    val_rows: list[dict[str, str]] = []
    for category, category_rows in sorted(grouped.items()):
        shuffled = list(category_rows)
        rng.shuffle(shuffled)
        raw_val_count = int(round(len(shuffled) * val_ratio))
        if len(shuffled) > 1:
            val_count = min(max(raw_val_count, 1), len(shuffled) - 1)
        else:
            val_count = 0
        val_rows.extend(shuffled[:val_count])
        train_rows.extend(shuffled[val_count:])
        print(
            f"[split] category={category} total={len(shuffled)} "
            f"train={len(shuffled) - val_count} val={val_count}"
        )
    return train_rows, val_rows


def main() -> int:
    args = parse_args()
    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("--val-ratio must be in (0, 1)")

    manifest_path = args.manifest.expanduser().resolve()
    rows = read_manifest(manifest_path)
    if not rows:
        raise ValueError(f"Manifest is empty: {manifest_path}")

    output_dir = args.output_dir.expanduser().resolve()
    fieldnames = list(rows[0].keys())
    train_rows, val_rows = split_by_category(rows, args.val_ratio, args.seed)

    combined_rows: list[dict[str, str]] = []
    for row in train_rows:
        combined_rows.append({**row, "split": "train"})
    for row in val_rows:
        combined_rows.append({**row, "split": "val"})

    write_manifest(output_dir / "train_manifest.csv", train_rows, fieldnames)
    write_manifest(output_dir / "val_manifest.csv", val_rows, fieldnames)
    write_manifest(output_dir / "split_manifest.csv", combined_rows, fieldnames + ["split"])

    print(f"[ok] train_manifest -> {output_dir / 'train_manifest.csv'}")
    print(f"[ok] val_manifest -> {output_dir / 'val_manifest.csv'}")
    print(f"[ok] split_manifest -> {output_dir / 'split_manifest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
