from __future__ import annotations

from dataclasses import dataclass
from random import Random


SIZE_RANGES = {
    "micro": (0.10, 0.30),
    "medium": (0.80, 2.00),
    "large": (3.00, 8.00),
}


@dataclass
class HoleRequest:
    diameter: float
    size_level: str
    axis_name: str
    axis_origin: tuple[float, float, float]
    axis_direction: tuple[float, float, float]
    tool_length: float


AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def _uniform(rng: Random, lo: float, hi: float) -> float:
    return round(rng.uniform(lo, hi), 4)


def _sample_diameter(rng: Random, size_level: str) -> float:
    lo, hi = SIZE_RANGES[size_level]
    return _uniform(rng, lo, hi)


def sample_condition_hole_count(rng: Random, condition_level: str, zero_hole_ratio: float) -> int:
    if rng.random() < zero_hole_ratio:
        return 0
    if condition_level == "C0":
        return rng.randint(1, 6)
    if condition_level == "C1":
        return rng.randint(1, 5)
    if condition_level == "C2":
        return rng.randint(1, 6)
    return rng.randint(2, 8)


def choose_condition(rng: Random, allowed_conditions: list[str]) -> str:
    return rng.choice(allowed_conditions)


def _safe_band(min_value: float, max_value: float, margin_ratio: float = 0.18) -> tuple[float, float]:
    span = max(max_value - min_value, 1.0)
    return min_value + span * margin_ratio, max_value - span * margin_ratio


def _sample_axis_position(
    rng: Random,
    axis_name: str,
    bbox: tuple[float, float, float, float, float, float],
    tool_length: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    xmin, ymin, zmin, xmax, ymax, zmax = bbox
    x_band = _safe_band(xmin, xmax)
    y_band = _safe_band(ymin, ymax)
    z_band = _safe_band(zmin, zmax)
    margin = tool_length * 0.08

    if axis_name == "x":
        return (xmin - margin, _uniform(rng, *y_band), _uniform(rng, *z_band)), (1.0, 0.0, 0.0)
    if axis_name == "y":
        return (_uniform(rng, *x_band), ymin - margin, _uniform(rng, *z_band)), (0.0, 1.0, 0.0)
    return (_uniform(rng, *x_band), _uniform(rng, *y_band), zmin - margin), (0.0, 0.0, 1.0)


def _sample_size_level_for_condition(rng: Random, condition_level: str) -> str:
    if condition_level == "C0":
        return rng.choices(["medium", "large"], weights=[0.45, 0.55], k=1)[0]
    if condition_level == "C1":
        return rng.choices(["micro", "medium"], weights=[0.75, 0.25], k=1)[0]
    if condition_level == "C2":
        return rng.choices(["medium", "large"], weights=[0.55, 0.45], k=1)[0]
    return rng.choices(["micro", "medium", "large"], weights=[0.55, 0.30, 0.15], k=1)[0]


def _position_signature(request: HoleRequest) -> tuple[float, float]:
    x, y, z = request.axis_origin
    if request.axis_name == "x":
        return y, z
    if request.axis_name == "y":
        return x, z
    return x, y


def _same_axis_center_distance(request_a: HoleRequest, request_b: HoleRequest) -> float:
    a0, a1 = _position_signature(request_a)
    b0, b1 = _position_signature(request_b)
    return ((a0 - b0) ** 2 + (a1 - b1) ** 2) ** 0.5


def _cross_axis_separation(request_a: HoleRequest, request_b: HoleRequest) -> float:
    ax = AXIS_INDEX[request_a.axis_name]
    bx = AXIS_INDEX[request_b.axis_name]
    remaining_index = ({0, 1, 2} - {ax, bx}).pop()
    return abs(request_a.axis_origin[remaining_index] - request_b.axis_origin[remaining_index])


def holes_would_merge(
    request_a: HoleRequest,
    request_b: HoleRequest,
    clearance: float,
) -> bool:
    merge_threshold = request_a.diameter / 2.0 + request_b.diameter / 2.0 + clearance
    if request_a.axis_name == request_b.axis_name:
        return _same_axis_center_distance(request_a, request_b) <= merge_threshold
    return _cross_axis_separation(request_a, request_b) <= merge_threshold


def sample_hole_request(
    rng: Random,
    condition_level: str,
    preferred_axes: list[str],
    bbox: tuple[float, float, float, float, float, float],
    clearance: float,
    existing_requests: list[HoleRequest] | None = None,
    max_attempts: int = 32,
) -> HoleRequest | None:
    xmin, ymin, zmin, xmax, ymax, zmax = bbox
    dx = xmax - xmin
    dy = ymax - ymin
    dz = zmax - zmin
    tool_length = max(dx, dy, dz) * 2.6 + 20.0

    existing_requests = existing_requests or []
    for _ in range(max_attempts):
        axis_name = rng.choice(preferred_axes)
        size_level = _sample_size_level_for_condition(rng, condition_level)
        axis_origin, axis_direction = _sample_axis_position(rng, axis_name, bbox, tool_length)
        request = HoleRequest(
            diameter=_sample_diameter(rng, size_level),
            size_level=size_level,
            axis_name=axis_name,
            axis_origin=axis_origin,
            axis_direction=axis_direction,
            tool_length=tool_length,
        )
        if any(holes_would_merge(request, existing, clearance) for existing in existing_requests):
            continue
        return request
    return None
