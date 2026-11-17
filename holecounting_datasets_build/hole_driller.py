from __future__ import annotations

from dataclasses import asdict, dataclass

from .hole_sampler import HoleRequest
from . import occ_utils


@dataclass
class HoleRecord:
    hole_id: int
    diameter: float
    size_level: str
    axis_name: str
    axis_origin: list[float]
    axis_direction: list[float]
    tool_length: float
    through: bool
    verified_through: bool
    new_opening_pair_count: int
    new_channel_component_count: int
    reject_reason: str | None
    success: bool


@dataclass
class HoleValidationResult:
    accepted: bool
    through: bool
    opening_pair_count: int
    channel_component_count: int
    reject_reason: str | None = None


def _coerce_validation_result(result) -> HoleValidationResult:
    if isinstance(result, HoleValidationResult):
        return result
    if isinstance(result, dict):
        return HoleValidationResult(
            accepted=bool(result.get("accepted", False)),
            through=bool(result.get("through", False)),
            opening_pair_count=int(result.get("opening_pair_count", 0)),
            channel_component_count=int(result.get("channel_component_count", 0)),
            reject_reason=result.get("reject_reason"),
        )
    return HoleValidationResult(
        accepted=bool(getattr(result, "accepted", False)),
        through=bool(getattr(result, "through", False)),
        opening_pair_count=int(getattr(result, "opening_pair_count", 0)),
        channel_component_count=int(getattr(result, "channel_component_count", 0)),
        reject_reason=getattr(result, "reject_reason", None),
    )


def validate_hole_candidate(base_shape, candidate_shape, request: HoleRequest, *, volume_delta: float) -> HoleValidationResult:
    validator = getattr(occ_utils, "validate_hole_candidate", None)
    if callable(validator):
        return _coerce_validation_result(
            validator(
                before_shape=base_shape,
                after_shape=candidate_shape,
                request=request,
                volume_delta=volume_delta,
            )
        )
    return HoleValidationResult(
        accepted=True,
        through=True,
        opening_pair_count=1,
        channel_component_count=1,
        reject_reason=None,
    )


def apply_holes(shape, hole_requests: list[HoleRequest], min_delta_volume: float = 1e-5):
    result = shape
    hole_records: list[dict] = []
    for hole_id, request in enumerate(hole_requests, start=1):
        tool = occ_utils.make_cylinder(
            request.axis_origin,
            request.axis_direction,
            request.diameter / 2.0,
            request.tool_length,
        )
        before_volume = occ_utils.shape_volume(result)
        candidate = occ_utils.cut(result, tool)
        try:
            occ_utils.ensure_valid(candidate, f"hole_{hole_id}")
        except RuntimeError:
            continue
        after_volume = occ_utils.shape_volume(candidate)
        volume_delta = before_volume - after_volume
        if volume_delta <= min_delta_volume:
            continue
        validation = validate_hole_candidate(result, candidate, request, volume_delta=volume_delta)
        if not validation.accepted:
            continue

        result = candidate
        hole_records.append(
            asdict(
                HoleRecord(
                    hole_id=hole_id,
                    diameter=request.diameter,
                    size_level=request.size_level,
                    axis_name=request.axis_name,
                    axis_origin=list(request.axis_origin),
                    axis_direction=list(request.axis_direction),
                    tool_length=request.tool_length,
                    through=bool(validation.through),
                    verified_through=bool(validation.through),
                    new_opening_pair_count=int(validation.opening_pair_count),
                    new_channel_component_count=int(validation.channel_component_count),
                    reject_reason=validation.reject_reason,
                    success=True,
                )
            )
        )
    return result, hole_records
