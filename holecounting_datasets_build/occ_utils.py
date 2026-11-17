from __future__ import annotations

from math import sqrt
from pathlib import Path

from OCC.Core.BRep import BRep_Tool
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.BRepClass3d import BRepClass3d_SolidClassifier
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.BRepGProp import brepgprop_VolumeProperties
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCone, BRepPrimAPI_MakeCylinder
from OCC.Core.GeomAbs import GeomAbs_Cylinder
from OCC.Core.GProp import GProp_GProps
from OCC.Core.STEPControl import STEPControl_AsIs, STEPControl_Writer
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_IN, TopAbs_ON, TopAbs_VERTEX
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import topods_Edge, topods_Face, topods_Vertex
from OCC.Core.gp import gp_Ax2, gp_Dir, gp_Pnt, gp_Trsf, gp_Vec


def ensure_valid(shape, name: str) -> None:
    analyzer = BRepCheck_Analyzer(shape)
    if not analyzer.IsValid():
        raise RuntimeError(f"{name} is not a valid OCC shape")


def export_step(shape, output_path: Path) -> None:
    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs)
    status = writer.Write(str(output_path))
    if status != 1:
        raise RuntimeError(f"Failed to write STEP file: {output_path}")


def make_box(origin: tuple[float, float, float], size: tuple[float, float, float]):
    return BRepPrimAPI_MakeBox(gp_Pnt(*origin), *size).Shape()


def make_cylinder(
    origin: tuple[float, float, float],
    direction: tuple[float, float, float],
    radius: float,
    height: float,
):
    axis = gp_Ax2(gp_Pnt(*origin), gp_Dir(*direction))
    return BRepPrimAPI_MakeCylinder(axis, radius, height).Shape()


def make_cone(
    origin: tuple[float, float, float],
    direction: tuple[float, float, float],
    radius_1: float,
    radius_2: float,
    height: float,
):
    axis = gp_Ax2(gp_Pnt(*origin), gp_Dir(*direction))
    return BRepPrimAPI_MakeCone(axis, radius_1, radius_2, height).Shape()


def fuse(shape_a, shape_b):
    return BRepAlgoAPI_Fuse(shape_a, shape_b).Shape()


def cut(shape_a, shape_b):
    return BRepAlgoAPI_Cut(shape_a, shape_b).Shape()


def translate_shape(shape, dx: float, dy: float, dz: float):
    trsf = gp_Trsf()
    trsf.SetTranslation(gp_Vec(dx, dy, dz))
    return BRepBuilderAPI_Transform(shape, trsf, True).Shape()


def shape_volume(shape) -> float:
    props = GProp_GProps()
    brepgprop_VolumeProperties(shape, props)
    return float(props.Mass())


def get_bbox(shape) -> tuple[float, float, float, float, float, float]:
    bbox = Bnd_Box()
    brepbndlib.Add(shape, bbox)
    return bbox.Get()


def _dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _normalize(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    norm = sqrt(_dot(vector, vector))
    if norm <= 1e-12:
        raise ValueError("Axis direction must be non-zero")
    return (vector[0] / norm, vector[1] / norm, vector[2] / norm)


def _point_on_axis(
    origin: tuple[float, float, float], direction: tuple[float, float, float], distance: float
) -> tuple[float, float, float]:
    return (
        origin[0] + direction[0] * distance,
        origin[1] + direction[1] * distance,
        origin[2] + direction[2] * distance,
    )


def _project_to_axis(
    point: tuple[float, float, float], origin: tuple[float, float, float], direction: tuple[float, float, float]
) -> float:
    delta = (point[0] - origin[0], point[1] - origin[1], point[2] - origin[2])
    return _dot(delta, direction)


def _iter_faces(shape):
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        yield topods_Face(explorer.Current())
        explorer.Next()


def _iter_edges(shape):
    explorer = TopExp_Explorer(shape, TopAbs_EDGE)
    while explorer.More():
        yield topods_Edge(explorer.Current())
        explorer.Next()


def _iter_vertices(shape):
    explorer = TopExp_Explorer(shape, TopAbs_VERTEX)
    while explorer.More():
        yield topods_Vertex(explorer.Current())
        explorer.Next()


def _shape_hash(shape) -> int:
    return int(shape.HashCode(2147483647))


def _sample_material_intervals(
    shape,
    origin: tuple[float, float, float],
    direction: tuple[float, float, float],
    length: float,
    radius: float,
    sample_count: int | None = None,
    tol: float = 1e-6,
) -> list[tuple[float, float]]:
    direction = _normalize(direction)
    if sample_count is None:
        sample_count = max(96, min(512, int(max(length / max(radius, 1e-3), 1.0) * 8)))

    classifier = BRepClass3d_SolidClassifier(shape)
    positions = [length * index / max(sample_count - 1, 1) for index in range(sample_count)]
    inside_flags: list[bool] = []
    for position in positions:
        point = _point_on_axis(origin, direction, position)
        classifier.Perform(gp_Pnt(*point), tol)
        inside_flags.append(classifier.State() in (TopAbs_IN, TopAbs_ON))

    intervals: list[tuple[float, float]] = []
    start_index: int | None = None
    for index, is_inside in enumerate(inside_flags):
        if is_inside and start_index is None:
            start_index = index
        if not is_inside and start_index is not None:
            intervals.append((positions[start_index], positions[index - 1]))
            start_index = None
    if start_index is not None:
        intervals.append((positions[start_index], positions[-1]))
    return intervals


def _face_matches_drill(
    face,
    axis_direction: tuple[float, float, float],
    radius: float,
    radius_tol: float,
    alignment_tol: float,
) -> bool:
    surface = BRepAdaptor_Surface(face)
    if surface.GetType() != GeomAbs_Cylinder:
        return False
    cylinder = surface.Cylinder()
    face_axis = cylinder.Axis().Direction()
    face_direction = _normalize((face_axis.X(), face_axis.Y(), face_axis.Z()))
    if abs(_dot(face_direction, axis_direction)) < alignment_tol:
        return False
    return abs(float(cylinder.Radius()) - radius) <= radius_tol


def _collect_axis_aligned_cylindrical_components(
    shape,
    origin: tuple[float, float, float],
    axis_direction: tuple[float, float, float],
    radius: float,
) -> list[dict[str, object]]:
    matching_faces: dict[int, object] = {}
    face_to_edges: dict[int, set[int]] = {}
    edge_to_faces: dict[int, set[int]] = {}
    edge_objects: dict[int, object] = {}
    radius_tol = max(radius * 0.12, 0.05)
    alignment_tol = 0.985

    for face in _iter_faces(shape):
        if not _face_matches_drill(face, axis_direction, radius, radius_tol, alignment_tol):
            continue
        face_hash = _shape_hash(face)
        matching_faces[face_hash] = face
        face_edges: set[int] = set()
        for edge in _iter_edges(face):
            edge_hash = _shape_hash(edge)
            face_edges.add(edge_hash)
            edge_to_faces.setdefault(edge_hash, set()).add(face_hash)
            edge_objects[edge_hash] = edge
        face_to_edges[face_hash] = face_edges

    components: list[dict[str, object]] = []
    remaining = set(matching_faces.keys())
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

        boundary_edges: list[object] = []
        for face_hash in component_faces:
            for edge_hash in face_to_edges.get(face_hash, set()):
                if len(edge_to_faces.get(edge_hash, set()) & component_faces) == 1:
                    boundary_edges.append(edge_objects[edge_hash])

        projections: list[float] = []
        for edge in boundary_edges:
            for vertex in _iter_vertices(edge):
                point = BRep_Tool.Pnt(vertex)
                projections.append(
                    _project_to_axis((float(point.X()), float(point.Y()), float(point.Z())), origin, axis_direction)
                )

        components.append(
            {
                "face_hashes": component_faces,
                "boundary_projections": projections,
            }
        )
    return components


def _count_opening_pairs(components: list[dict[str, object]], radius: float) -> int:
    opening_pairs = 0
    end_tol = max(radius * 0.8, 0.1)
    for component in components:
        projections = [float(value) for value in component["boundary_projections"]]
        if not projections:
            continue
        start = min(projections)
        end = max(projections)
        has_start = any(abs(value - start) <= end_tol for value in projections)
        has_end = any(abs(value - end) <= end_tol for value in projections)
        if has_start and has_end and end - start > end_tol:
            opening_pairs += 1
    return opening_pairs


def analyze_drilled_hole_topology(shape, request) -> dict[str, int]:
    axis_direction = _normalize(tuple(request.axis_direction))
    radius = float(request.diameter) / 2.0
    components = _collect_axis_aligned_cylindrical_components(
        shape,
        tuple(request.axis_origin),
        axis_direction,
        radius,
    )
    return {
        "opening_pair_count": _count_opening_pairs(components, radius),
        "channel_component_count": len(components),
    }


def validate_hole_candidate(before_shape, after_shape, request, volume_delta: float) -> dict[str, object]:
    axis_origin = tuple(request.axis_origin)
    axis_direction = _normalize(tuple(request.axis_direction))
    tool_length = float(request.tool_length)
    radius = float(request.diameter) / 2.0

    if volume_delta <= 0.0:
        return {
            "accepted": False,
            "through": False,
            "opening_pair_count": 0,
            "channel_component_count": 0,
            "reject_reason": "non_positive_volume_delta",
        }

    intervals_before = _sample_material_intervals(before_shape, axis_origin, axis_direction, tool_length, radius)
    if not intervals_before:
        return {
            "accepted": False,
            "through": False,
            "opening_pair_count": 0,
            "channel_component_count": 0,
            "reject_reason": "no_material_along_axis",
        }
    if len(intervals_before) > 1:
        return {
            "accepted": False,
            "through": False,
            "opening_pair_count": 0,
            "channel_component_count": 0,
            "reject_reason": "discontinuous_material_along_axis",
        }

    intervals_after = _sample_material_intervals(after_shape, axis_origin, axis_direction, tool_length, radius)
    if intervals_after:
        return {
            "accepted": False,
            "through": False,
            "opening_pair_count": 0,
            "channel_component_count": 0,
            "reject_reason": "residual_material_after_cut",
        }

    before_topology = analyze_drilled_hole_topology(before_shape, request)
    after_topology = analyze_drilled_hole_topology(after_shape, request)
    new_opening_pair_count = max(0, after_topology["opening_pair_count"] - before_topology["opening_pair_count"])
    new_channel_component_count = max(
        0,
        after_topology["channel_component_count"] - before_topology["channel_component_count"],
    )
    accepted = new_opening_pair_count == 1 and new_channel_component_count == 1
    reject_reason = None if accepted else "topology_validation_failed"
    return {
        "accepted": accepted,
        "through": accepted,
        "opening_pair_count": new_opening_pair_count,
        "channel_component_count": new_channel_component_count,
        "reject_reason": reject_reason,
    }
