from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from math import acos, pi, sqrt
from pathlib import Path
from typing import Iterable

from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.BRepClass3d import BRepClass3d_SolidClassifier
from OCC.Core.GeomAbs import GeomAbs_Circle, GeomAbs_Cylinder
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_IN, TopAbs_ON, TopAbs_VERTEX
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import topods_Edge, topods_Face
from OCC.Core.gp import gp_Pnt


Vector3 = tuple[float, float, float]


@dataclass(frozen=True)
class RecountConfig:
    """Numerical tolerances used by the history-free B-rep recount."""

    classifier_tolerance: float = 1e-6
    axis_alignment_cosine: float = 0.999
    circle_normal_alignment_cosine: float = 0.999
    relative_radius_tolerance: float = 0.02
    absolute_radius_tolerance: float = 0.02
    relative_axis_tolerance: float = 0.03
    absolute_axis_tolerance: float = 0.03
    relative_center_tolerance: float = 0.03
    absolute_center_tolerance: float = 0.03
    full_circle_angular_tolerance: float = 0.08
    relative_probe_offset: float = 0.08
    absolute_probe_offset: float = 0.02


@dataclass(frozen=True)
class CircleArc:
    edge_hash: int
    center: Vector3
    normal: Vector3
    radius: float
    span: float


@dataclass
class BoundaryLoop:
    center: Vector3
    normal: Vector3
    radius: float
    angular_span: float
    edge_hashes: list[int] = field(default_factory=list)
    is_closed: bool = False
    axial_projection_min: float = 0.0
    axial_projection_max: float = 0.0
    negative_probe_state: str = "unknown"
    positive_probe_state: str = "unknown"
    is_open_boundary: bool = False


@dataclass
class WallComponentResult:
    component_id: int
    face_count: int
    boundary_edge_count: int
    circular_arc_count: int
    complete_loop_count: int
    open_boundary_count: int
    radius: float | None
    axis_direction: Vector3 | None
    axes_are_coaxial: bool
    classified_as_through_hole: bool
    rejection_reason: str | None
    loops: list[BoundaryLoop] = field(default_factory=list)


@dataclass
class RecountResult:
    through_hole_count: int
    cylindrical_face_count: int
    cylindrical_wall_component_count: int
    valid_shape: bool
    components: list[WallComponentResult]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


@dataclass(frozen=True)
class _CylinderFace:
    face_hash: int
    axis_location: Vector3
    axis_direction: Vector3
    radius: float
    # Keep repeated occurrences. A periodic seam normally occurs twice in one
    # cylindrical face and must not be mistaken for a physical boundary edge.
    edge_hashes: tuple[int, ...]


def _dot(a: Vector3, b: Vector3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _sub(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _scale(a: Vector3, scalar: float) -> Vector3:
    return (a[0] * scalar, a[1] * scalar, a[2] * scalar)


def _norm(a: Vector3) -> float:
    return sqrt(max(_dot(a, a), 0.0))


def _normalize(a: Vector3) -> Vector3:
    length = _norm(a)
    if length <= 1e-12:
        raise ValueError("Cannot normalize a zero-length vector")
    return (a[0] / length, a[1] / length, a[2] / length)


def _distance(a: Vector3, b: Vector3) -> float:
    return _norm(_sub(a, b))


def _axis_distance(point: Vector3, axis_origin: Vector3, axis_direction: Vector3) -> float:
    delta = _sub(point, axis_origin)
    projection = _scale(axis_direction, _dot(delta, axis_direction))
    return _norm(_sub(delta, projection))


def _direction_angle_degrees(a: Vector3, b: Vector3) -> float:
    cosine = min(1.0, max(-1.0, abs(_dot(_normalize(a), _normalize(b)))))
    return acos(cosine) * 180.0 / pi


def _shape_hash(shape) -> int:
    return int(shape.HashCode(2147483647))


def _iter_faces(shape) -> Iterable[object]:
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        yield topods_Face(explorer.Current())
        explorer.Next()


def _iter_edges(shape) -> Iterable[object]:
    explorer = TopExp_Explorer(shape, TopAbs_EDGE)
    while explorer.More():
        yield topods_Edge(explorer.Current())
        explorer.Next()


def read_step_shape(step_path: str | Path):
    step_path = Path(step_path).expanduser().resolve()
    reader = STEPControl_Reader()
    status = reader.ReadFile(str(step_path))
    if status != IFSelect_RetDone:
        raise RuntimeError(f"Failed to read STEP file: {step_path}")
    reader.TransferRoots()
    shape = reader.OneShape()
    if shape.IsNull():
        raise RuntimeError(f"STEP file contains no readable shape: {step_path}")
    return shape


def _extract_cylindrical_faces(shape) -> tuple[dict[int, _CylinderFace], dict[int, object]]:
    cylinders: dict[int, _CylinderFace] = {}
    edge_objects: dict[int, object] = {}
    for face in _iter_faces(shape):
        surface = BRepAdaptor_Surface(face)
        if surface.GetType() != GeomAbs_Cylinder:
            continue
        cylinder = surface.Cylinder()
        axis = cylinder.Axis()
        location = axis.Location()
        direction = axis.Direction()
        face_edges: list[int] = []
        for edge in _iter_edges(face):
            edge_hash = _shape_hash(edge)
            face_edges.append(edge_hash)
            edge_objects.setdefault(edge_hash, edge)
        face_hash = _shape_hash(face)
        cylinders[face_hash] = _CylinderFace(
            face_hash=face_hash,
            axis_location=(float(location.X()), float(location.Y()), float(location.Z())),
            axis_direction=_normalize(
                (float(direction.X()), float(direction.Y()), float(direction.Z()))
            ),
            radius=float(cylinder.Radius()),
            edge_hashes=tuple(face_edges),
        )
    return cylinders, edge_objects


def _faces_belong_to_same_wall(
    first: _CylinderFace, second: _CylinderFace, config: RecountConfig
) -> bool:
    reference_radius = max(first.radius, second.radius)
    return (
        abs(first.radius - second.radius) <= _radius_tolerance(reference_radius, config)
        and abs(_dot(first.axis_direction, second.axis_direction)) >= config.axis_alignment_cosine
        and _axis_distance(
            second.axis_location, first.axis_location, first.axis_direction
        )
        <= _axis_tolerance(reference_radius, config)
    )


def _coaxial_wall_groups(
    cylinders: dict[int, _CylinderFace], config: RecountConfig
) -> list[set[int]]:
    """Group all patches of one geometric channel wall.

    A shell or a partitioned solid can split one drilled channel into several
    topologically disconnected cylindrical patches.  Edge adjacency therefore
    cannot define an independent channel.  The final B-rep still identifies the
    channel geometrically: its patches share one axis and one radius.  This
    grouping uses only those final-surface descriptors.
    """
    remaining = set(cylinders)
    components: list[set[int]] = []
    while remaining:
        seed = remaining.pop()
        component = {seed}
        stack = [seed]
        while stack:
            current = stack.pop()
            for neighbor in list(remaining):
                if _faces_belong_to_same_wall(
                    cylinders[current], cylinders[neighbor], config
                ):
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)
        components.append(component)
    return components


def _component_boundary_edges(
    component: set[int], cylinders: dict[int, _CylinderFace]
) -> set[int]:
    edge_counts: dict[int, int] = {}
    for face_hash in component:
        for edge_hash in cylinders[face_hash].edge_hashes:
            edge_counts[edge_hash] = edge_counts.get(edge_hash, 0) + 1
    return {edge_hash for edge_hash, count in edge_counts.items() if count == 1}


def _edge_vertex_hashes(edge) -> list[int]:
    hashes: list[int] = []
    explorer = TopExp_Explorer(edge, TopAbs_VERTEX)
    while explorer.More():
        hashes.append(_shape_hash(explorer.Current()))
        explorer.Next()
    return hashes


def _connected_boundary_edge_groups(
    boundary_edges: set[int], edge_objects: dict[int, object]
) -> list[set[int]]:
    edge_vertices = {
        edge_hash: _edge_vertex_hashes(edge_objects[edge_hash])
        for edge_hash in boundary_edges
    }
    vertex_to_edges: dict[int, set[int]] = {}
    for edge_hash, vertex_hashes in edge_vertices.items():
        for vertex_hash in vertex_hashes:
            vertex_to_edges.setdefault(vertex_hash, set()).add(edge_hash)

    remaining = set(boundary_edges)
    groups: list[set[int]] = []
    while remaining:
        seed = remaining.pop()
        group = {seed}
        stack = [seed]
        while stack:
            current = stack.pop()
            for vertex_hash in edge_vertices[current]:
                for neighbor in vertex_to_edges.get(vertex_hash, set()):
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        group.add(neighbor)
                        stack.append(neighbor)
        groups.append(group)
    return groups


def _boundary_group_is_closed(
    edge_group: set[int], edge_objects: dict[int, object]
) -> bool:
    degrees: dict[int, int] = {}
    for edge_hash in edge_group:
        edge = edge_objects[edge_hash]
        vertex_hashes = _edge_vertex_hashes(edge)
        if len(vertex_hashes) == 1 and bool(edge.Closed()):
            degrees[vertex_hashes[0]] = degrees.get(vertex_hashes[0], 0) + 2
        else:
            for vertex_hash in vertex_hashes:
                degrees[vertex_hash] = degrees.get(vertex_hash, 0) + 1
    if not degrees:
        return all(bool(edge_objects[edge_hash].Closed()) for edge_hash in edge_group)
    return all(degree == 2 for degree in degrees.values())


def _sample_edge_points(edge, samples: int = 9) -> list[Vector3]:
    curve = BRepAdaptor_Curve(edge)
    first = float(curve.FirstParameter())
    last = float(curve.LastParameter())
    if samples <= 1 or abs(last - first) <= 1e-12:
        parameters = [(first + last) * 0.5]
    else:
        parameters = [first + (last - first) * index / (samples - 1) for index in range(samples)]
    points: list[Vector3] = []
    for parameter in parameters:
        point = curve.Value(parameter)
        points.append((float(point.X()), float(point.Y()), float(point.Z())))
    return points


def _build_boundary_loops(
    boundary_edges: set[int],
    edge_objects: dict[int, object],
    axis_origin: Vector3,
    axis_direction: Vector3,
    radius: float,
) -> list[BoundaryLoop]:
    loops: list[BoundaryLoop] = []
    for edge_group in _connected_boundary_edge_groups(boundary_edges, edge_objects):
        points: list[Vector3] = []
        circular_span = 0.0
        for edge_hash in sorted(edge_group):
            edge = edge_objects[edge_hash]
            try:
                points.extend(_sample_edge_points(edge))
            except RuntimeError:
                continue
            arc = _extract_circle_arc(edge_hash, edge)
            if arc is not None:
                circular_span += arc.span
        if not points:
            continue
        projections = [_dot(_sub(point, axis_origin), axis_direction) for point in points]
        station = sum(projections) / len(projections)
        probe_point = _add(axis_origin, _scale(axis_direction, station))
        loops.append(
            BoundaryLoop(
                center=probe_point,
                normal=axis_direction,
                radius=radius,
                angular_span=min(circular_span, 2.0 * pi),
                edge_hashes=sorted(edge_group),
                is_closed=_boundary_group_is_closed(edge_group, edge_objects),
                axial_projection_min=min(projections),
                axial_projection_max=max(projections),
            )
        )
    return loops


def _extract_circle_arc(edge_hash: int, edge) -> CircleArc | None:
    try:
        curve = BRepAdaptor_Curve(edge)
        if curve.GetType() != GeomAbs_Circle:
            return None
        circle = curve.Circle()
        location = circle.Location()
        direction = circle.Axis().Direction()
        first = float(curve.FirstParameter())
        last = float(curve.LastParameter())
        span = abs(last - first)
        if span <= 1e-8:
            return None
        return CircleArc(
            edge_hash=edge_hash,
            center=(float(location.X()), float(location.Y()), float(location.Z())),
            normal=_normalize(
                (float(direction.X()), float(direction.Y()), float(direction.Z()))
            ),
            radius=float(circle.Radius()),
            span=min(span, 2.0 * pi),
        )
    except (RuntimeError, ValueError):
        return None


def _radius_tolerance(radius: float, config: RecountConfig) -> float:
    return max(config.absolute_radius_tolerance, abs(radius) * config.relative_radius_tolerance)


def _center_tolerance(radius: float, config: RecountConfig) -> float:
    return max(config.absolute_center_tolerance, abs(radius) * config.relative_center_tolerance)


def _axis_tolerance(radius: float, config: RecountConfig) -> float:
    return max(config.absolute_axis_tolerance, abs(radius) * config.relative_axis_tolerance)


def _same_circle(a: CircleArc, b: CircleArc, config: RecountConfig) -> bool:
    reference_radius = max(a.radius, b.radius)
    return (
        abs(a.radius - b.radius) <= _radius_tolerance(reference_radius, config)
        and _distance(a.center, b.center) <= _center_tolerance(reference_radius, config)
        and abs(_dot(a.normal, b.normal)) >= config.circle_normal_alignment_cosine
    )


def _cluster_circle_arcs(arcs: list[CircleArc], config: RecountConfig) -> list[BoundaryLoop]:
    remaining = set(range(len(arcs)))
    loops: list[BoundaryLoop] = []
    while remaining:
        seed_index = remaining.pop()
        member_indices = {seed_index}
        stack = [seed_index]
        while stack:
            current = stack.pop()
            for candidate in list(remaining):
                if _same_circle(arcs[current], arcs[candidate], config):
                    remaining.remove(candidate)
                    member_indices.add(candidate)
                    stack.append(candidate)
        members = [arcs[index] for index in sorted(member_indices)]
        total_span = min(sum(member.span for member in members), 2.0 * pi)
        loops.append(
            BoundaryLoop(
                center=members[0].center,
                normal=members[0].normal,
                radius=sum(member.radius for member in members) / len(members),
                angular_span=total_span,
                edge_hashes=[member.edge_hash for member in members],
            )
        )
    return loops


def _classify_probe(shape, point: Vector3, config: RecountConfig) -> str:
    classifier = BRepClass3d_SolidClassifier(shape)
    classifier.Perform(gp_Pnt(*point), config.classifier_tolerance)
    state = classifier.State()
    if state == TopAbs_IN:
        return "inside"
    if state == TopAbs_ON:
        return "boundary"
    return "outside"


def _probe_side_state(
    shape,
    center: Vector3,
    direction: Vector3,
    base_offset: float,
    config: RecountConfig,
) -> str:
    for multiplier in (1.0, 2.0, 4.0):
        point = _add(center, _scale(direction, base_offset * multiplier))
        state = _classify_probe(shape, point, config)
        if state != "boundary":
            return state
    return "boundary"


def _annotate_open_boundary(shape, loop: BoundaryLoop, config: RecountConfig) -> None:
    offset = max(config.absolute_probe_offset, loop.radius * config.relative_probe_offset)
    normal = _normalize(loop.normal)
    negative = _probe_side_state(shape, loop.center, _scale(normal, -1.0), offset, config)
    positive = _probe_side_state(shape, loop.center, normal, offset, config)
    loop.negative_probe_state = negative
    loop.positive_probe_state = positive
    loop.is_open_boundary = negative == "outside" and positive == "outside"


def _component_axes_are_coaxial(
    faces: list[_CylinderFace], config: RecountConfig
) -> tuple[bool, float | None, Vector3 | None, str | None]:
    if not faces:
        return False, None, None, "empty_component"
    reference = faces[0]
    radius_tolerance = _radius_tolerance(reference.radius, config)
    axis_tolerance = _axis_tolerance(reference.radius, config)
    for face in faces[1:]:
        if abs(face.radius - reference.radius) > radius_tolerance:
            return False, reference.radius, reference.axis_direction, "mixed_radii"
        if abs(_dot(face.axis_direction, reference.axis_direction)) < config.axis_alignment_cosine:
            return False, reference.radius, reference.axis_direction, "mixed_axis_directions"
        if _axis_distance(face.axis_location, reference.axis_location, reference.axis_direction) > axis_tolerance:
            return False, reference.radius, reference.axis_direction, "offset_axes"
    return True, reference.radius, reference.axis_direction, None


def _classify_component(
    shape,
    component_id: int,
    component: set[int],
    cylinders: dict[int, _CylinderFace],
    edge_objects: dict[int, object],
    config: RecountConfig,
) -> WallComponentResult:
    faces = [cylinders[face_hash] for face_hash in sorted(component)]
    axes_are_coaxial, radius, axis_direction, axis_reason = _component_axes_are_coaxial(
        faces, config
    )
    boundary_edges = _component_boundary_edges(component, cylinders)
    arcs = [
        arc
        for edge_hash in sorted(boundary_edges)
        if (arc := _extract_circle_arc(edge_hash, edge_objects[edge_hash])) is not None
    ]
    loops = (
        _build_boundary_loops(
            boundary_edges,
            edge_objects,
            axis_origin=faces[0].axis_location,
            axis_direction=faces[0].axis_direction,
            radius=faces[0].radius,
        )
        if faces
        else []
    )
    complete_loops = [loop for loop in loops if loop.is_closed]
    for loop in complete_loops:
        _annotate_open_boundary(shape, loop, config)
    open_loops = [loop for loop in complete_loops if loop.is_open_boundary]

    rejection_reason: str | None = None
    if not axes_are_coaxial:
        rejection_reason = axis_reason
    elif len(complete_loops) < 2:
        rejection_reason = "fewer_than_two_closed_boundary_loops"
    elif len(open_loops) != len(complete_loops):
        rejection_reason = "one_or_more_boundaries_are_not_open"
    elif axis_direction is None or radius is None:
        rejection_reason = "missing_axis_descriptor"
    else:
        if any(
            _axis_distance(loop.center, faces[0].axis_location, axis_direction)
            > _axis_tolerance(radius, config)
            for loop in open_loops
        ):
            rejection_reason = "one_or_more_openings_are_off_axis"
        else:
            axial_separation = max(
                loop.axial_projection_max for loop in open_loops
            ) - min(loop.axial_projection_min for loop in open_loops)
            if axial_separation <= _center_tolerance(radius, config):
                rejection_reason = "openings_not_axially_separated"

    classified = rejection_reason is None
    return WallComponentResult(
        component_id=component_id,
        face_count=len(component),
        boundary_edge_count=len(boundary_edges),
        circular_arc_count=len(arcs),
        complete_loop_count=len(complete_loops),
        open_boundary_count=len(open_loops),
        radius=radius,
        axis_direction=axis_direction,
        axes_are_coaxial=axes_are_coaxial,
        classified_as_through_hole=classified,
        rejection_reason=rejection_reason,
        loops=complete_loops,
    )


def recount_through_holes(shape, config: RecountConfig | None = None) -> RecountResult:
    """Count independent through channels from the final B-rep only.

    The function does not accept or inspect operation records, cutter poses,
    sampled targets, or construction order. A counted channel must be represented
    by one coaxial, equal-radius cylindrical wall group.  Two boundary loops are
    sufficient for a single material layer; additional open loops are retained
    when the same channel crosses a shell or internal partition.  Every retained
    loop must open to void on both axial sides.
    """

    config = config or RecountConfig()
    valid_shape = bool(BRepCheck_Analyzer(shape).IsValid())
    if not valid_shape:
        return RecountResult(
            through_hole_count=0,
            cylindrical_face_count=0,
            cylindrical_wall_component_count=0,
            valid_shape=False,
            components=[],
        )

    cylinders, edge_objects = _extract_cylindrical_faces(shape)
    components = _coaxial_wall_groups(cylinders, config)
    results = [
        _classify_component(
            shape,
            component_id=index,
            component=component,
            cylinders=cylinders,
            edge_objects=edge_objects,
            config=config,
        )
        for index, component in enumerate(components, start=1)
    ]
    return RecountResult(
        through_hole_count=sum(result.classified_as_through_hole for result in results),
        cylindrical_face_count=len(cylinders),
        cylindrical_wall_component_count=len(components),
        valid_shape=True,
        components=results,
    )


def recount_step_file(
    step_path: str | Path, config: RecountConfig | None = None
) -> RecountResult:
    return recount_through_holes(read_step_shape(step_path), config=config)


__all__ = [
    "BoundaryLoop",
    "RecountConfig",
    "RecountResult",
    "WallComponentResult",
    "read_step_shape",
    "recount_step_file",
    "recount_through_holes",
]

