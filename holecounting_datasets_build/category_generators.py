from __future__ import annotations

from dataclasses import dataclass
from random import Random

from .internal_structures import (
    add_top_boss,
    carve_dual_cavity_with_center_rib,
    carve_labyrinth_channels,
    carve_partition_cavities,
    carve_sealed_box_cavity,
    carve_sealed_nozzle_chambers,
)
from .occ_utils import ensure_valid, fuse, make_box, make_cone, make_cylinder


@dataclass
class CategoryInstance:
    category: str
    shape: object
    base_params: dict
    has_internal_structure: bool
    preferred_axes: list[str]
    allowed_conditions: list[str]


def _uniform(rng: Random, lo: float, hi: float) -> float:
    return round(rng.uniform(lo, hi), 4)


def build_solid_block(rng: Random) -> CategoryInstance:
    lx = _uniform(rng, 70.0, 130.0)
    ly = _uniform(rng, 45.0, 95.0)
    lz = _uniform(rng, 25.0, 75.0)
    shape = make_box((0.0, 0.0, 0.0), (lx, ly, lz))
    if rng.random() < 0.65:
        bx = lx * _uniform(rng, 0.28, 0.55)
        by = ly * _uniform(rng, 0.28, 0.55)
        bz = lz * _uniform(rng, 0.15, 0.35)
        shape = add_top_boss(shape, (lx, ly, lz), (bx, by, bz))
    ensure_valid(shape, "solid_block")
    return CategoryInstance("solid_block", shape, {"size": [lx, ly, lz]}, False, ["x", "y", "z"], ["C0", "C1"])


def build_l_bracket(rng: Random) -> CategoryInstance:
    leg_a = _uniform(rng, 70.0, 130.0)
    leg_b = _uniform(rng, 60.0, 120.0)
    thickness = _uniform(rng, 12.0, 28.0)
    depth = _uniform(rng, 50.0, 100.0)
    vertical = make_box((0.0, 0.0, 0.0), (thickness, depth, leg_b))
    horizontal = make_box((0.0, 0.0, 0.0), (leg_a, depth, thickness))
    shape = fuse(vertical, horizontal)
    ensure_valid(shape, "l_bracket")
    return CategoryInstance(
        "l_bracket",
        shape,
        {"leg_a": leg_a, "leg_b": leg_b, "thickness": thickness, "depth": depth},
        False,
        ["x", "z", "y"],
        ["C0", "C1"],
    )


def build_thin_plate(rng: Random) -> CategoryInstance:
    lx = _uniform(rng, 100.0, 180.0)
    ly = _uniform(rng, 70.0, 140.0)
    lz = _uniform(rng, 6.0, 18.0)
    shape = make_box((0.0, 0.0, 0.0), (lx, ly, lz))
    if rng.random() < 0.6:
        rim_height = lz * _uniform(rng, 0.35, 0.9)
        rim = make_box((0.0, 0.0, lz), (lx, ly * _uniform(rng, 0.12, 0.22), rim_height))
        shape = fuse(shape, rim)
    ensure_valid(shape, "thin_plate")
    return CategoryInstance("thin_plate", shape, {"size": [lx, ly, lz]}, False, ["z", "x", "y"], ["C0", "C1"])


def build_solid_disc(rng: Random) -> CategoryInstance:
    radius = _uniform(rng, 35.0, 70.0)
    height = _uniform(rng, 10.0, 28.0)
    shape = make_cylinder((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), radius, height)
    if rng.random() < 0.6:
        boss_radius = radius * _uniform(rng, 0.18, 0.45)
        boss_height = height * _uniform(rng, 0.4, 1.1)
        boss = make_cylinder((0.0, 0.0, height), (0.0, 0.0, 1.0), boss_radius, boss_height)
        shape = fuse(shape, boss)
    ensure_valid(shape, "solid_disc")
    return CategoryInstance("solid_disc", shape, {"radius": radius, "height": height}, False, ["z", "x", "y"], ["C0", "C1"])


def build_stepped_prism(rng: Random) -> CategoryInstance:
    base = (_uniform(rng, 80.0, 150.0), _uniform(rng, 50.0, 100.0), _uniform(rng, 24.0, 42.0))
    shape = make_box((0.0, 0.0, 0.0), base)
    stage_count = rng.randint(1, 3)
    current_height = base[2]
    params = {"base": list(base), "stage_count": stage_count}
    for stage_idx in range(stage_count):
        sx = base[0] * _uniform(rng, 0.35, 0.72)
        sy = base[1] * _uniform(rng, 0.35, 0.72)
        sz = _uniform(rng, 10.0, 26.0)
        offset_x = _uniform(rng, 0.0, max(base[0] - sx, 1.0))
        offset_y = _uniform(rng, 0.0, max(base[1] - sy, 1.0))
        stage = make_box((offset_x, offset_y, current_height), (sx, sy, sz))
        shape = fuse(shape, stage)
        current_height += sz
        params[f"stage_{stage_idx}"] = [sx, sy, sz, offset_x, offset_y]
    ensure_valid(shape, "stepped_prism")
    return CategoryInstance("stepped_prism", shape, params, False, ["x", "z", "y"], ["C0", "C1"])


def build_sealed_box(rng: Random) -> CategoryInstance:
    outer_size = (_uniform(rng, 90.0, 140.0), _uniform(rng, 70.0, 120.0), _uniform(rng, 50.0, 90.0))
    wall = _uniform(rng, 6.0, 12.0)
    floor = _uniform(rng, 7.0, 14.0)
    roof = _uniform(rng, 7.0, 14.0)
    shape = make_box((0.0, 0.0, 0.0), outer_size)
    shape = carve_sealed_box_cavity(shape, outer_size, wall, floor, roof)
    ensure_valid(shape, "sealed_box")
    return CategoryInstance(
        "sealed_box",
        shape,
        {"outer_size": list(outer_size), "wall": wall, "floor": floor, "roof": roof},
        True,
        ["x", "y", "z"],
        ["C2", "C1+C2"],
    )


def build_double_shell_block(rng: Random) -> CategoryInstance:
    outer_size = (_uniform(rng, 100.0, 150.0), _uniform(rng, 70.0, 110.0), _uniform(rng, 55.0, 95.0))
    wall = _uniform(rng, 8.0, 13.0)
    floor = _uniform(rng, 8.0, 14.0)
    roof = _uniform(rng, 8.0, 14.0)
    rib_width = _uniform(rng, 8.0, 16.0)
    shape = make_box((0.0, 0.0, 0.0), outer_size)
    shape = carve_dual_cavity_with_center_rib(shape, outer_size, wall, floor, roof, rib_width)
    ensure_valid(shape, "double_shell_block")
    return CategoryInstance(
        "double_shell_block",
        shape,
        {"outer_size": list(outer_size), "wall": wall, "floor": floor, "roof": roof, "rib_width": rib_width},
        True,
        ["x", "y", "z"],
        ["C2", "C1+C2"],
    )


def build_ribbed_housing(rng: Random) -> CategoryInstance:
    outer_size = (_uniform(rng, 100.0, 160.0), _uniform(rng, 80.0, 130.0), _uniform(rng, 55.0, 95.0))
    wall = _uniform(rng, 8.0, 12.0)
    floor = _uniform(rng, 8.0, 14.0)
    roof = _uniform(rng, 8.0, 14.0)
    rib_width = _uniform(rng, 10.0, 18.0)
    shape = make_box((0.0, 0.0, 0.0), outer_size)
    shape = carve_dual_cavity_with_center_rib(shape, outer_size, wall, floor, roof, rib_width)
    if rng.random() < 0.7:
        top_boss = make_box(
            (outer_size[0] * 0.25, outer_size[1] * 0.25, outer_size[2]),
            (outer_size[0] * 0.30, outer_size[1] * 0.22, outer_size[2] * 0.18),
        )
        shape = fuse(shape, top_boss)
    ensure_valid(shape, "ribbed_housing")
    return CategoryInstance(
        "ribbed_housing",
        shape,
        {"outer_size": list(outer_size), "wall": wall, "floor": floor, "roof": roof, "rib_width": rib_width},
        True,
        ["x", "y", "z"],
        ["C2", "C1+C2"],
    )


def build_partition_housing(rng: Random) -> CategoryInstance:
    outer_size = (_uniform(rng, 110.0, 160.0), _uniform(rng, 75.0, 120.0), _uniform(rng, 50.0, 88.0))
    wall = _uniform(rng, 8.0, 13.0)
    floor = _uniform(rng, 8.0, 14.0)
    roof = _uniform(rng, 8.0, 14.0)
    partition = _uniform(rng, 8.0, 16.0)
    shape = make_box((0.0, 0.0, 0.0), outer_size)
    shape = carve_partition_cavities(shape, outer_size, wall, floor, roof, partition)
    ensure_valid(shape, "partition_housing")
    return CategoryInstance(
        "partition_housing",
        shape,
        {"outer_size": list(outer_size), "wall": wall, "floor": floor, "roof": roof, "partition": partition},
        True,
        ["x", "y", "z"],
        ["C2", "C1+C2"],
    )


def build_sealed_nozzle(rng: Random) -> CategoryInstance:
    body_radius = _uniform(rng, 14.0, 26.0)
    body_length = _uniform(rng, 45.0, 85.0)
    tip_length = _uniform(rng, 14.0, 28.0)
    tip_radius = body_radius * _uniform(rng, 0.35, 0.65)
    body = make_cylinder((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), body_radius, body_length)
    tip = make_cone((0.0, 0.0, body_length), (0.0, 0.0, 1.0), body_radius, tip_radius, tip_length)
    shape = fuse(body, tip)
    chamber_radius_a = body_radius * _uniform(rng, 0.28, 0.46)
    chamber_radius_b = body_radius * _uniform(rng, 0.16, 0.30)
    shape = carve_sealed_nozzle_chambers(shape, body_radius, body_length, chamber_radius_a, chamber_radius_b)
    ensure_valid(shape, "sealed_nozzle")
    return CategoryInstance(
        "sealed_nozzle",
        shape,
        {
            "body_radius": body_radius,
            "body_length": body_length,
            "tip_length": tip_length,
            "tip_radius": tip_radius,
            "chamber_radius_a": chamber_radius_a,
            "chamber_radius_b": chamber_radius_b,
        },
        True,
        ["x", "y", "z"],
        ["C2", "C1+C2"],
    )


def build_labyrinth_block(rng: Random) -> CategoryInstance:
    outer_size = (_uniform(rng, 110.0, 160.0), _uniform(rng, 80.0, 120.0), _uniform(rng, 50.0, 80.0))
    wall = _uniform(rng, 8.0, 12.0)
    floor = _uniform(rng, 8.0, 12.0)
    roof = _uniform(rng, 8.0, 12.0)
    shape = make_box((0.0, 0.0, 0.0), outer_size)
    shape = carve_labyrinth_channels(shape, outer_size, wall, floor, roof)
    ensure_valid(shape, "labyrinth_block")
    return CategoryInstance(
        "labyrinth_block",
        shape,
        {"outer_size": list(outer_size), "wall": wall, "floor": floor, "roof": roof},
        True,
        ["x", "y", "z"],
        ["C2", "C1+C2"],
    )


def build_dome_cover(rng: Random) -> CategoryInstance:
    base_radius = _uniform(rng, 35.0, 65.0)
    skirt_height = _uniform(rng, 14.0, 28.0)
    dome_height = _uniform(rng, 16.0, 36.0)
    top_radius = base_radius * _uniform(rng, 0.25, 0.55)
    skirt = make_cylinder((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), base_radius, skirt_height)
    dome = make_cone((0.0, 0.0, skirt_height), (0.0, 0.0, 1.0), base_radius, top_radius, dome_height)
    shape = fuse(skirt, dome)
    cavity = make_cone(
        (0.0, 0.0, skirt_height * 0.15),
        (0.0, 0.0, 1.0),
        base_radius * 0.62,
        top_radius * 0.62,
        dome_height * 0.82,
    )
    shape = fuse(shape, make_cylinder((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), base_radius * 0.75, skirt_height * 0.22))
    shape = cut_cover(shape, cavity)
    ensure_valid(shape, "dome_cover")
    return CategoryInstance(
        "dome_cover",
        shape,
        {
            "base_radius": base_radius,
            "skirt_height": skirt_height,
            "dome_height": dome_height,
            "top_radius": top_radius,
        },
        True,
        ["x", "y", "z"],
        ["C2", "C1+C2"],
    )


def cut_cover(shape, cavity):
    from .occ_utils import cut

    return cut(shape, cavity)


CATEGORY_BUILDERS = {
    "solid_block": build_solid_block,
    "l_bracket": build_l_bracket,
    "thin_plate": build_thin_plate,
    "solid_disc": build_solid_disc,
    "stepped_prism": build_stepped_prism,
    "sealed_box": build_sealed_box,
    "double_shell_block": build_double_shell_block,
    "ribbed_housing": build_ribbed_housing,
    "partition_housing": build_partition_housing,
    "sealed_nozzle": build_sealed_nozzle,
    "labyrinth_block": build_labyrinth_block,
    "dome_cover": build_dome_cover,
}


def instantiate_category(category: str, rng: Random) -> CategoryInstance:
    return CATEGORY_BUILDERS[category](rng)
