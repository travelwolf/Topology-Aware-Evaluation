from __future__ import annotations

from .occ_utils import cut, fuse, make_box, make_cylinder


def carve_sealed_box_cavity(
    outer_shape,
    outer_size: tuple[float, float, float],
    wall: float,
    floor: float,
    roof: float,
):
    lx, ly, lz = outer_size
    cavity = make_box((wall, wall, floor), (lx - 2 * wall, ly - 2 * wall, lz - floor - roof))
    return cut(outer_shape, cavity)


def carve_dual_cavity_with_center_rib(
    outer_shape,
    outer_size: tuple[float, float, float],
    wall: float,
    floor: float,
    roof: float,
    rib_width: float,
):
    lx, ly, lz = outer_size
    usable_x = lx - 2 * wall
    cavity_width = max(usable_x - rib_width, rib_width) / 2.0
    left = make_box((wall, wall, floor), (cavity_width, ly - 2 * wall, lz - floor - roof))
    right = make_box(
        (wall + cavity_width + rib_width, wall, floor),
        (cavity_width, ly - 2 * wall, lz - floor - roof),
    )
    return cut(cut(outer_shape, left), right)


def carve_partition_cavities(
    outer_shape,
    outer_size: tuple[float, float, float],
    wall: float,
    floor: float,
    roof: float,
    partition_width: float,
):
    lx, ly, lz = outer_size
    inner_y = ly - 2 * wall
    cavity_height = lz - floor - roof
    left = make_box((wall, wall, floor), (lx - 2 * wall, inner_y * 0.35, cavity_height))
    right = make_box(
        (wall, wall + inner_y * 0.35 + partition_width, floor),
        (lx - 2 * wall, inner_y * 0.35, cavity_height),
    )
    return cut(cut(outer_shape, left), right)


def carve_labyrinth_channels(
    outer_shape,
    outer_size: tuple[float, float, float],
    wall: float,
    floor: float,
    roof: float,
):
    lx, ly, lz = outer_size
    cavity_height = lz - floor - roof
    channel_a = make_box((wall, wall, floor), (lx * 0.55, ly * 0.22, cavity_height))
    channel_b = make_box((lx * 0.28, ly * 0.38, floor), (lx * 0.45, ly * 0.22, cavity_height))
    channel_c = make_box((wall, ly * 0.62, floor), (lx * 0.58, ly * 0.18, cavity_height))
    return cut(cut(cut(outer_shape, channel_a), channel_b), channel_c)


def add_top_boss(base_shape, footprint: tuple[float, float, float], boss_size: tuple[float, float, float]):
    lx, ly, lz = footprint
    bx, by, bz = boss_size
    boss = make_box(((lx - bx) / 2.0, (ly - by) / 2.0, lz), (bx, by, bz))
    return fuse(base_shape, boss)


def carve_sealed_nozzle_chambers(
    outer_shape,
    body_radius: float,
    body_length: float,
    chamber_radius_a: float,
    chamber_radius_b: float,
):
    chamber_a = make_cylinder((0.0, 0.0, body_length * 0.16), (0.0, 0.0, 1.0), chamber_radius_a, body_length * 0.34)
    chamber_b = make_cylinder((0.0, 0.0, body_length * 0.60), (0.0, 0.0, 1.0), chamber_radius_b, body_length * 0.18)
    if chamber_radius_a >= body_radius or chamber_radius_b >= body_radius:
        raise RuntimeError("Nozzle chamber radius exceeds outer body")
    return cut(cut(outer_shape, chamber_a), chamber_b)
