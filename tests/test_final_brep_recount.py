from __future__ import annotations

import unittest

try:
    from holecounting_datasets_build.final_brep_recount import recount_through_holes
    from holecounting_datasets_build.occ_utils import cut, fuse, make_box, make_cylinder
except ModuleNotFoundError as exc:  # local environments may not contain pythonOCC
    recount_through_holes = None
    OCC_IMPORT_ERROR = exc
else:
    OCC_IMPORT_ERROR = None


@unittest.skipIf(recount_through_holes is None, f"pythonOCC unavailable: {OCC_IMPORT_ERROR}")
class FinalBRepRecountTests(unittest.TestCase):
    def test_counts_one_through_hole(self) -> None:
        base = make_box((0.0, 0.0, 0.0), (60.0, 50.0, 20.0))
        tool = make_cylinder((30.0, 25.0, -2.0), (0.0, 0.0, 1.0), 5.0, 24.0)
        result = recount_through_holes(cut(base, tool))
        self.assertEqual(result.through_hole_count, 1)

    def test_rejects_blind_hole(self) -> None:
        base = make_box((0.0, 0.0, 0.0), (60.0, 50.0, 20.0))
        tool = make_cylinder((30.0, 25.0, 10.0), (0.0, 0.0, 1.0), 5.0, 12.0)
        result = recount_through_holes(cut(base, tool))
        self.assertEqual(result.through_hole_count, 0)

    def test_rejects_tangential_cut(self) -> None:
        base = make_box((0.0, 0.0, 0.0), (60.0, 50.0, 20.0))
        tool = make_cylinder((0.0, 25.0, -2.0), (0.0, 0.0, 1.0), 5.0, 24.0)
        result = recount_through_holes(cut(base, tool))
        self.assertEqual(result.through_hole_count, 0)

    def test_rejects_merged_channels_as_two_independent_holes(self) -> None:
        base = make_box((0.0, 0.0, 0.0), (70.0, 50.0, 20.0))
        first = make_cylinder((32.0, 25.0, -2.0), (0.0, 0.0, 1.0), 5.0, 24.0)
        second = make_cylinder((38.0, 25.0, -2.0), (0.0, 0.0, 1.0), 5.0, 24.0)
        result = recount_through_holes(cut(cut(base, first), second))
        self.assertEqual(result.through_hole_count, 0)

    def test_counts_one_channel_across_separated_material_layers(self) -> None:
        lower = make_box((0.0, 0.0, 0.0), (60.0, 50.0, 8.0))
        upper = make_box((0.0, 0.0, 14.0), (60.0, 50.0, 8.0))
        tool = make_cylinder((30.0, 25.0, -2.0), (0.0, 0.0, 1.0), 4.0, 26.0)
        result = recount_through_holes(cut(fuse(lower, upper), tool))
        self.assertEqual(result.through_hole_count, 1)


if __name__ == "__main__":
    unittest.main()

