from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

import torch
from PIL import Image, ImageDraw
from torch import nn

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


FEATURE_NAMES = [
    "num_faces",
    "num_edges",
    "num_vertices",
    "num_wires",
    "num_cyl_faces",
    "num_cyl_components",
    "num_hole_like_cyl_components",
    "num_circular_edges",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate simple rule-based and linear hole-counting baselines.")
    parser.add_argument("--split-root", type=Path, required=True, help="Directory that contains train_manifest.csv and val_manifest.csv.")
    parser.add_argument("--data-root", type=Path, default=None, help="Dataset root that contains step/ point/ image/ label/ bin.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, default=None)
    parser.add_argument("--max-hole-count", type=int, default=8)
    parser.add_argument("--linear-epochs", type=int, default=100)
    parser.add_argument("--linear-lr", type=float, default=0.05)
    parser.add_argument("--linear-weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260630)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    return parser.parse_args()


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _normalize_relative_path(rel_path: str) -> Path:
    normalized = rel_path.replace("\\", "/").strip()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    normalized = normalized.lstrip("/")
    return Path(normalized)


def compute_holecount_metrics(predictions, labels, num_classes: int) -> dict[str, object]:
    prediction_tensor = torch.as_tensor(predictions, dtype=torch.long)
    label_tensor = torch.as_tensor(labels, dtype=torch.long)
    if label_tensor.numel() == 0:
        return {
            "acc": 0.0,
            "balanced_acc": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "weighted_precision": 0.0,
            "weighted_recall": 0.0,
            "weighted_f1": 0.0,
            "micro_precision": 0.0,
            "micro_recall": 0.0,
            "micro_f1": 0.0,
            "mae": 0.0,
            "rmse": 0.0,
            "within_1_acc": 0.0,
            "confusion_matrix": [[0 for _ in range(num_classes)] for _ in range(num_classes)],
            "per_class_metrics": [],
        }

    confusion = torch.bincount(label_tensor * num_classes + prediction_tensor, minlength=num_classes * num_classes)
    confusion = confusion.reshape(num_classes, num_classes).to(torch.float32)
    tp = confusion.diag()
    support = confusion.sum(dim=1)
    predicted = confusion.sum(dim=0)

    precision = torch.where(predicted > 0, tp / predicted, torch.zeros_like(tp))
    recall = torch.where(support > 0, tp / support, torch.zeros_like(tp))
    f1 = torch.where((precision + recall) > 0, 2 * precision * recall / (precision + recall), torch.zeros_like(tp))

    valid_class_mask = support > 0
    valid_class_count = valid_class_mask.sum().clamp_min(1)
    support_sum = support.sum().clamp_min(1.0)

    acc = tp.sum() / support_sum
    macro_precision = precision[valid_class_mask].sum() / valid_class_count
    macro_recall = recall[valid_class_mask].sum() / valid_class_count
    macro_f1 = f1[valid_class_mask].sum() / valid_class_count
    weighted_precision = (precision * support).sum() / support_sum
    weighted_recall = (recall * support).sum() / support_sum
    weighted_f1 = (f1 * support).sum() / support_sum

    abs_error = (prediction_tensor - label_tensor).abs().to(torch.float32)
    mae = abs_error.mean()
    rmse = torch.sqrt((abs_error ** 2).mean())
    within_1_acc = (abs_error <= 1).to(torch.float32).mean()

    per_class_metrics = []
    confusion_int = confusion.to(torch.int64)
    for class_id in range(num_classes):
        per_class_metrics.append(
            {
                "class_id": class_id,
                "support": int(support[class_id].item()),
                "precision": precision[class_id].item(),
                "recall": recall[class_id].item(),
                "f1": f1[class_id].item(),
            }
        )

    return {
        "acc": acc.item(),
        "balanced_acc": macro_recall.item(),
        "macro_precision": macro_precision.item(),
        "macro_recall": macro_recall.item(),
        "macro_f1": macro_f1.item(),
        "weighted_precision": weighted_precision.item(),
        "weighted_recall": weighted_recall.item(),
        "weighted_f1": weighted_f1.item(),
        "micro_precision": acc.item(),
        "micro_recall": acc.item(),
        "micro_f1": acc.item(),
        "mae": mae.item(),
        "rmse": rmse.item(),
        "within_1_acc": within_1_acc.item(),
        "confusion_matrix": confusion_int.tolist(),
        "per_class_metrics": per_class_metrics,
    }


def analyze_loss_stability(
    loss_history: list[float],
    window_size: int = 10,
    std_threshold: float = 5e-3,
    improvement_threshold: float = 5e-3,
) -> dict[str, float | bool | int]:
    if not loss_history:
        return {
            "is_stable": False,
            "window_size": 0,
            "final_loss": 0.0,
            "best_loss": 0.0,
            "last_window_mean": 0.0,
            "last_window_std": 0.0,
            "relative_improvement": 0.0,
        }

    effective_window = max(1, min(window_size, len(loss_history)))
    tail = torch.tensor(loss_history[-effective_window:], dtype=torch.float32)
    last_window_mean = float(tail.mean().item())
    last_window_std = float(tail.std(unbiased=False).item())
    relative_improvement = float(max(0.0, (float(tail[0].item()) - float(tail[-1].item())) / max(abs(float(tail[0].item())), 1e-8)))
    is_stable = bool(last_window_std <= std_threshold and relative_improvement <= improvement_threshold)
    return {
        "is_stable": is_stable,
        "window_size": effective_window,
        "final_loss": float(loss_history[-1]),
        "best_loss": float(min(loss_history)),
        "last_window_mean": last_window_mean,
        "last_window_std": last_window_std,
        "relative_improvement": relative_improvement,
    }


def save_loss_artifacts(output_dir: Path, loss_history: list[float], stability_summary: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "loss_history.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", "loss"])
        writer.writeheader()
        for epoch, loss_value in enumerate(loss_history, start=1):
            writer.writerow({"epoch": epoch, "loss": float(loss_value)})

    (output_dir / "loss_summary.json").write_text(
        json.dumps(stability_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    width = 960
    height = 540
    margin_left = 70
    margin_right = 30
    margin_top = 30
    margin_bottom = 60
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    plot_left = margin_left
    plot_top = margin_top
    plot_right = width - margin_right
    plot_bottom = height - margin_bottom
    draw.rectangle([plot_left, plot_top, plot_right, plot_bottom], outline="black", width=2)

    if loss_history:
        min_loss = min(loss_history)
        max_loss = max(loss_history)
        loss_span = max(max_loss - min_loss, 1e-8)
        x_span = max(len(loss_history) - 1, 1)
        points: list[tuple[float, float]] = []
        for index, loss_value in enumerate(loss_history):
            x = plot_left + (plot_right - plot_left) * (index / x_span)
            y = plot_bottom - (plot_bottom - plot_top) * ((float(loss_value) - min_loss) / loss_span)
            points.append((x, y))
        if len(points) >= 2:
            draw.line(points, fill="blue", width=3)
        else:
            x, y = points[0]
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill="blue", outline="blue")
        draw.text((plot_left, plot_bottom + 10), "epoch", fill="black")
        draw.text((plot_left, 5), f"loss curve | final={loss_history[-1]:.4f} | stable={stability_summary.get('is_stable', False)}", fill="black")
        draw.text((plot_right - 100, plot_bottom + 10), str(len(loss_history)), fill="black")
        draw.text((5, plot_top), f"{max_loss:.4f}", fill="black")
        draw.text((5, plot_bottom - 10), f"{min_loss:.4f}", fill="black")

    image.save(output_dir / "loss_curve.png")


def predict_cylindrical_rule(feature_row: dict[str, object], max_hole_count: int) -> int:
    prediction = int(feature_row.get("num_hole_like_cyl_components", feature_row.get("num_cyl_components", 0)) or 0)
    return max(0, min(max_hole_count, prediction))


def predict_ring_rule(feature_row: dict[str, object], max_hole_count: int) -> int:
    circular_edges = int(feature_row.get("num_circular_edges", 0) or 0)
    prediction = circular_edges // 2
    return max(0, min(max_hole_count, prediction))


class TopologyLinearClassifier:
    def __init__(
        self,
        max_hole_count: int,
        feature_names: list[str] | None = None,
        epochs: int = 300,
        lr: float = 0.05,
        weight_decay: float = 1e-4,
        seed: int = 20260630,
    ) -> None:
        self.max_hole_count = max_hole_count
        self.feature_names = feature_names or FEATURE_NAMES
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.seed = seed
        self.mean: torch.Tensor | None = None
        self.std: torch.Tensor | None = None
        self.model: nn.Linear | None = None
        self.loss_history: list[float] = []
        self.loss_stability_summary: dict[str, object] | None = None

    def _rows_to_tensor(self, rows: list[dict[str, object]]) -> torch.Tensor:
        matrix = []
        for row in rows:
            matrix.append([float(row.get(name, 0.0) or 0.0) for name in self.feature_names])
        return torch.tensor(matrix, dtype=torch.float32)

    def fit(self, rows: list[dict[str, object]], labels: list[int]) -> None:
        if not rows:
            raise ValueError("Training rows must not be empty")
        torch.manual_seed(self.seed)
        features = self._rows_to_tensor(rows)
        targets = torch.tensor(labels, dtype=torch.long)
        self.mean = features.mean(dim=0, keepdim=True)
        self.std = features.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
        normalized = (features - self.mean) / self.std

        self.model = nn.Linear(len(self.feature_names), self.max_hole_count + 1)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.CrossEntropyLoss()
        self.loss_history = []

        self.model.train(True)
        epoch_iterator = range(self.epochs)
        if tqdm is not None:
            epoch_iterator = tqdm(epoch_iterator, total=self.epochs, desc="train topology_linear", leave=False, dynamic_ncols=True)
        for epoch_index in epoch_iterator:
            optimizer.zero_grad(set_to_none=True)
            logits = self.model(normalized)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            self.loss_history.append(float(loss.item()))
            if tqdm is not None:
                predictions = logits.argmax(dim=-1)
                acc = (predictions == targets).to(torch.float32).mean().item()
                epoch_iterator.set_postfix(epoch=epoch_index + 1, loss=f"{loss.item():.4f}", acc=f"{acc:.4f}")
        self.loss_stability_summary = analyze_loss_stability(self.loss_history)

    def predict(self, rows: list[dict[str, object]]) -> list[int]:
        if self.model is None or self.mean is None or self.std is None:
            raise RuntimeError("TopologyLinearClassifier must be fitted before prediction")
        features = self._rows_to_tensor(rows)
        normalized = (features - self.mean) / self.std
        self.model.train(False)
        with torch.no_grad():
            logits = self.model(normalized)
        return logits.argmax(dim=-1).tolist()


def _load_occ_modules():
    try:
        from OCC.Core.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
        from OCC.Core.GeomAbs import GeomAbs_Circle, GeomAbs_Cylinder
        from OCC.Core.IFSelect import IFSelect_RetDone
        from OCC.Core.STEPControl import STEPControl_Reader
        from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_VERTEX, TopAbs_WIRE
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopoDS import topods_Edge, topods_Face
    except ModuleNotFoundError as exc:
        raise ImportError(
            "pythonOCC is required to extract topology-rule baseline features from STEP files. "
            "Install OCC in the current environment, or provide a prebuilt --feature-cache."
        ) from exc

    return {
        "BRepAdaptor_Curve": BRepAdaptor_Curve,
        "BRepAdaptor_Surface": BRepAdaptor_Surface,
        "GeomAbs_Circle": GeomAbs_Circle,
        "GeomAbs_Cylinder": GeomAbs_Cylinder,
        "IFSelect_RetDone": IFSelect_RetDone,
        "STEPControl_Reader": STEPControl_Reader,
        "TopAbs_EDGE": TopAbs_EDGE,
        "TopAbs_FACE": TopAbs_FACE,
        "TopAbs_VERTEX": TopAbs_VERTEX,
        "TopAbs_WIRE": TopAbs_WIRE,
        "TopExp_Explorer": TopExp_Explorer,
        "topods_Edge": topods_Edge,
        "topods_Face": topods_Face,
    }


def _iter_shapes(shape, shape_type):
    occ = _load_occ_modules()
    explorer = occ["TopExp_Explorer"](shape, shape_type)
    while explorer.More():
        yield explorer.Current()
        explorer.Next()


def _shape_hash(shape) -> int:
    return int(shape.HashCode(2147483647))


def read_step_shape(step_path: Path):
    occ = _load_occ_modules()
    reader = occ["STEPControl_Reader"]()
    status = reader.ReadFile(str(step_path))
    if status != occ["IFSelect_RetDone"]:
        raise RuntimeError(f"Failed to read STEP file: {step_path}")
    reader.TransferRoots()
    shape = reader.OneShape()
    if shape.IsNull():
        raise RuntimeError(f"STEP file contains no readable shape: {step_path}")
    return shape


def _build_cylindrical_components(shape) -> list[dict[str, int]]:
    occ = _load_occ_modules()
    edge_type = occ["TopAbs_EDGE"]
    face_type = occ["TopAbs_FACE"]
    cylindrical_face_hashes: set[int] = set()
    face_to_edges: dict[int, set[int]] = {}
    edge_to_faces: dict[int, set[int]] = {}
    edge_is_circular: dict[int, bool] = {}

    for face_shape in _iter_shapes(shape, face_type):
        face = occ["topods_Face"](face_shape)
        surface = occ["BRepAdaptor_Surface"](face)
        if surface.GetType() != occ["GeomAbs_Cylinder"]:
            continue
        face_hash = _shape_hash(face)
        cylindrical_face_hashes.add(face_hash)
        face_edges: set[int] = set()
        for edge_shape in _iter_shapes(face, edge_type):
            edge = occ["topods_Edge"](edge_shape)
            edge_hash = _shape_hash(edge)
            face_edges.add(edge_hash)
            edge_to_faces.setdefault(edge_hash, set()).add(face_hash)
            if edge_hash not in edge_is_circular:
                curve = occ["BRepAdaptor_Curve"](edge)
                edge_is_circular[edge_hash] = curve.GetType() == occ["GeomAbs_Circle"]
        face_to_edges[face_hash] = face_edges

    components: list[dict[str, int]] = []
    remaining = set(cylindrical_face_hashes)
    while remaining:
        seed = remaining.pop()
        stack = [seed]
        component_faces = {seed}
        while stack:
            current = stack.pop()
            for edge_hash in face_to_edges.get(current, set()):
                for neighbor in edge_to_faces.get(edge_hash, set()):
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        component_faces.add(neighbor)
                        stack.append(neighbor)

        boundary_circular_edges = 0
        for face_hash in component_faces:
            for edge_hash in face_to_edges.get(face_hash, set()):
                shared_face_count = len(edge_to_faces.get(edge_hash, set()) & component_faces)
                if shared_face_count == 1 and edge_is_circular.get(edge_hash, False):
                    boundary_circular_edges += 1
        components.append(
            {
                "num_faces": len(component_faces),
                "num_boundary_circular_edges": boundary_circular_edges,
            }
        )
    return components


def extract_topology_features(step_path: Path) -> dict[str, int | str]:
    occ = _load_occ_modules()
    shape = read_step_shape(step_path)

    num_faces = sum(1 for _ in _iter_shapes(shape, occ["TopAbs_FACE"]))
    num_edges = sum(1 for _ in _iter_shapes(shape, occ["TopAbs_EDGE"]))
    num_vertices = sum(1 for _ in _iter_shapes(shape, occ["TopAbs_VERTEX"]))
    num_wires = sum(1 for _ in _iter_shapes(shape, occ["TopAbs_WIRE"]))

    num_circular_edges = 0
    for edge_shape in _iter_shapes(shape, occ["TopAbs_EDGE"]):
        edge = occ["topods_Edge"](edge_shape)
        curve = occ["BRepAdaptor_Curve"](edge)
        if curve.GetType() == occ["GeomAbs_Circle"]:
            num_circular_edges += 1

    num_cyl_faces = 0
    for face_shape in _iter_shapes(shape, occ["TopAbs_FACE"]):
        face = occ["topods_Face"](face_shape)
        surface = occ["BRepAdaptor_Surface"](face)
        if surface.GetType() == occ["GeomAbs_Cylinder"]:
            num_cyl_faces += 1

    cyl_components = _build_cylindrical_components(shape)
    num_hole_like_cyl_components = sum(
        1 for component in cyl_components if component["num_boundary_circular_edges"] >= 2
    )
    return {
        "step_path": str(step_path),
        "num_faces": num_faces,
        "num_edges": num_edges,
        "num_vertices": num_vertices,
        "num_wires": num_wires,
        "num_cyl_faces": num_cyl_faces,
        "num_cyl_components": len(cyl_components),
        "num_hole_like_cyl_components": num_hole_like_cyl_components,
        "num_circular_edges": num_circular_edges,
    }


def load_feature_cache(cache_path: Path) -> dict[str, dict[str, object]]:
    if not cache_path.exists():
        return {}
    with cache_path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    cache: dict[str, dict[str, object]] = {}
    for row in rows:
        cache[row["model_id"]] = {
            "model_id": row["model_id"],
            "step_rel_path": row["step_rel_path"],
            "step_path": row["step_path"],
            **{name: int(row[name]) for name in FEATURE_NAMES},
        }
    return cache


def save_feature_cache(cache_path: Path, rows: list[dict[str, object]]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model_id", "step_rel_path", "step_path", *FEATURE_NAMES]
    with cache_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def build_feature_rows(
    manifest_rows: list[dict[str, str]],
    data_root: Path,
    cache_path: Path,
    limit: int | None = None,
    progress_desc: str | None = None,
) -> list[dict[str, object]]:
    cache = load_feature_cache(cache_path)
    merged_rows: list[dict[str, object]] = []
    updated_cache_rows: dict[str, dict[str, object]] = dict(cache)
    selected_rows = manifest_rows[:limit] if limit is not None else manifest_rows

    iterator = selected_rows
    if tqdm is not None:
        iterator = tqdm(selected_rows, total=len(selected_rows), desc=progress_desc or "extract topology", leave=False, dynamic_ncols=True)

    for row in iterator:
        model_id = row["model_id"]
        if model_id in cache:
            feature_row = dict(cache[model_id])
        else:
            step_rel_path = row["step_path"]
            step_path = data_root / _normalize_relative_path(step_rel_path)
            feature_row = {
                "model_id": model_id,
                "step_rel_path": step_rel_path,
                **extract_topology_features(step_path),
            }
            updated_cache_rows[model_id] = feature_row

        merged_rows.append(
            {
                "model_id": model_id,
                "category": row["category"],
                "label": int(row["hole_count"]),
                **{name: int(feature_row[name]) for name in FEATURE_NAMES},
            }
        )

    save_feature_cache(cache_path, list(updated_cache_rows.values()))
    return merged_rows


def save_baseline_artifacts(
    output_dir: Path,
    baseline_name: str,
    predictions: list[int],
    rows: list[dict[str, object]],
    max_hole_count: int,
    extra_summary: dict[str, object] | None = None,
) -> dict[str, object]:
    baseline_dir = output_dir / baseline_name
    baseline_dir.mkdir(parents=True, exist_ok=True)
    labels = [int(row["label"]) for row in rows]
    metrics = compute_holecount_metrics(predictions, labels, num_classes=max_hole_count + 1)
    summary = {
        key: value
        for key, value in metrics.items()
        if key not in {"confusion_matrix", "per_class_metrics"}
    }
    summary["baseline_name"] = baseline_name
    summary["num_samples"] = len(rows)
    if extra_summary:
        summary.update(extra_summary)

    (baseline_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (baseline_dir / "confusion_matrix.json").write_text(
        json.dumps(metrics["confusion_matrix"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (baseline_dir / "per_class_metrics.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["class_id", "support", "precision", "recall", "f1"])
        writer.writeheader()
        writer.writerows(metrics["per_class_metrics"])
    with (baseline_dir / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["model_id", "category", "label", "prediction", "abs_error", *FEATURE_NAMES],
        )
        writer.writeheader()
        for row, prediction in zip(rows, predictions):
            payload = {
                "model_id": row["model_id"],
                "category": row["category"],
                "label": row["label"],
                "prediction": prediction,
                "abs_error": abs(int(prediction) - int(row["label"])),
            }
            for name in FEATURE_NAMES:
                payload[name] = row[name]
            writer.writerow(payload)
    return summary


def main() -> int:
    args = parse_args()
    split_root = args.split_root.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve() if args.data_root is not None else split_root.parent
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.feature_cache.expanduser().resolve() if args.feature_cache is not None else output_dir / "topology_feature_cache.csv"

    train_manifest = split_root / "train_manifest.csv"
    val_manifest = split_root / "val_manifest.csv"
    train_manifest_rows = read_manifest(train_manifest)
    val_manifest_rows = read_manifest(val_manifest)

    train_rows = build_feature_rows(
        train_manifest_rows,
        data_root=data_root,
        cache_path=cache_path,
        limit=args.train_limit,
        progress_desc="extract train topology",
    )
    val_rows = build_feature_rows(
        val_manifest_rows,
        data_root=data_root,
        cache_path=cache_path,
        limit=args.val_limit,
        progress_desc="extract val topology",
    )

    linear_classifier = TopologyLinearClassifier(
        max_hole_count=args.max_hole_count,
        epochs=args.linear_epochs,
        lr=args.linear_lr,
        weight_decay=args.linear_weight_decay,
        seed=args.seed,
    )
    linear_classifier.fit(train_rows, [int(row["label"]) for row in train_rows])
    topology_linear_predictions = linear_classifier.predict(val_rows)
    cylindrical_rule_predictions = [predict_cylindrical_rule(row, max_hole_count=args.max_hole_count) for row in val_rows]
    ring_rule_predictions = [predict_ring_rule(row, max_hole_count=args.max_hole_count) for row in val_rows]

    summaries = [
        save_baseline_artifacts(
            output_dir,
            "topology_linear",
            topology_linear_predictions,
            val_rows,
            args.max_hole_count,
            extra_summary=linear_classifier.loss_stability_summary,
        ),
        save_baseline_artifacts(output_dir, "cylindrical_rule", cylindrical_rule_predictions, val_rows, args.max_hole_count),
        save_baseline_artifacts(output_dir, "ring_rule", ring_rule_predictions, val_rows, args.max_hole_count),
    ]
    save_loss_artifacts(
        output_dir / "topology_linear",
        loss_history=linear_classifier.loss_history,
        stability_summary=linear_classifier.loss_stability_summary or analyze_loss_stability([]),
    )
    (output_dir / "baseline_comparison.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
