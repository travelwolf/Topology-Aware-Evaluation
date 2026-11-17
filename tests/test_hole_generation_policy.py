from __future__ import annotations

import importlib
import json
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch


def _load_hole_driller_with_stub_occ_utils():
    fake_occ_utils = types.ModuleType("holecounting_datasets_build.occ_utils")
    fake_occ_utils.cut = lambda shape_a, shape_b: "candidate-shape"
    fake_occ_utils.ensure_valid = lambda shape, name: None
    fake_occ_utils.make_cylinder = lambda *args, **kwargs: "tool-shape"
    fake_occ_utils.shape_volume = lambda shape: 10.0 if shape == "base-shape" else 8.0

    package = importlib.import_module("holecounting_datasets_build")
    # The package can already expose the real OCC module when geometry tests run
    # first. Patch both sys.modules and the package attribute so test order does
    # not change which implementation hole_driller imports.
    with patch.dict(
        sys.modules, {"holecounting_datasets_build.occ_utils": fake_occ_utils}
    ), patch.object(package, "occ_utils", fake_occ_utils, create=True):
        sys.modules.pop("holecounting_datasets_build.hole_driller", None)
        module = importlib.import_module("holecounting_datasets_build.hole_driller")
        return importlib.reload(module)


def _load_dataset_builder_with_stubs():
    fake_category_generators = types.ModuleType("holecounting_datasets_build.category_generators")
    fake_category_generators.CATEGORY_BUILDERS = {"demo": object()}
    fake_category_generators.instantiate_category = lambda category, rng: None

    fake_exporters = types.ModuleType("holecounting_datasets_build.exporters")
    fake_exporters.DEFAULT_NUM_POINTS = 8192
    fake_exporters.export_graph_bin_file = lambda *args, **kwargs: None
    fake_exporters.export_overview_image = lambda *args, **kwargs: None
    fake_exporters.export_point_cloud_file = lambda *args, **kwargs: None
    fake_exporters.export_step_file = lambda *args, **kwargs: None

    fake_hole_driller = types.ModuleType("holecounting_datasets_build.hole_driller")
    fake_hole_driller.apply_holes = lambda *args, **kwargs: None

    fake_hole_sampler = types.ModuleType("holecounting_datasets_build.hole_sampler")
    fake_hole_sampler.choose_condition = lambda rng, allowed_conditions: "C0"
    fake_hole_sampler.sample_condition_hole_count = lambda rng, condition_level, zero_hole_ratio: 0
    fake_hole_sampler.sample_hole_request = lambda **kwargs: None

    fake_occ_utils = types.ModuleType("holecounting_datasets_build.occ_utils")
    fake_occ_utils.ensure_valid = lambda shape, name: None
    fake_occ_utils.get_bbox = lambda shape: (0.0, 0.0, 0.0, 10.0, 10.0, 10.0)

    patched_modules = {
        "holecounting_datasets_build.category_generators": fake_category_generators,
        "holecounting_datasets_build.exporters": fake_exporters,
        "holecounting_datasets_build.hole_driller": fake_hole_driller,
        "holecounting_datasets_build.hole_sampler": fake_hole_sampler,
        "holecounting_datasets_build.occ_utils": fake_occ_utils,
    }

    with patch.dict(sys.modules, patched_modules):
        sys.modules.pop("holecounting_datasets_build.dataset_builder", None)
        module = importlib.import_module("holecounting_datasets_build.dataset_builder")
        return importlib.reload(module)


class HoleDrillerPolicyTests(unittest.TestCase):
    def test_apply_holes_skips_candidate_when_topology_validator_rejects(self) -> None:
        hole_driller = _load_hole_driller_with_stub_occ_utils()
        request = hole_driller.HoleRequest(
            diameter=2.0,
            size_level="medium",
            axis_name="z",
            axis_origin=(0.0, 0.0, 0.0),
            axis_direction=(0.0, 0.0, 1.0),
            tool_length=10.0,
        )
        validator_result = SimpleNamespace(
            accepted=False,
            through=False,
            opening_pair_count=0,
            channel_component_count=0,
            reject_reason="topology_validation_failed",
        )

        with patch.object(hole_driller, "validate_hole_candidate", return_value=validator_result, create=True):
            result_shape, hole_records = hole_driller.apply_holes("base-shape", [request])

        self.assertEqual(result_shape, "base-shape")
        self.assertEqual(hole_records, [])

    def test_apply_holes_records_verified_metadata_from_validator(self) -> None:
        hole_driller = _load_hole_driller_with_stub_occ_utils()
        request = hole_driller.HoleRequest(
            diameter=3.0,
            size_level="large",
            axis_name="x",
            axis_origin=(0.0, 0.0, 0.0),
            axis_direction=(1.0, 0.0, 0.0),
            tool_length=12.0,
        )
        validator_result = SimpleNamespace(
            accepted=True,
            through=True,
            opening_pair_count=1,
            channel_component_count=1,
            reject_reason=None,
        )

        with patch.object(hole_driller, "validate_hole_candidate", return_value=validator_result, create=True):
            _, hole_records = hole_driller.apply_holes("base-shape", [request])

        self.assertEqual(len(hole_records), 1)
        self.assertIs(hole_records[0]["verified_through"], True)
        self.assertEqual(hole_records[0]["new_opening_pair_count"], 1)
        self.assertEqual(hole_records[0]["new_channel_component_count"], 1)
        self.assertIsNone(hole_records[0]["reject_reason"])


class DatasetBuilderPolicyTests(unittest.TestCase):
    def test_build_one_model_discards_underfilled_samples(self) -> None:
        dataset_builder = _load_dataset_builder_with_stubs()
        instance = SimpleNamespace(
            shape="base-shape",
            category="demo",
            has_internal_structure=False,
            base_params={"demo": True},
            preferred_axes=["z"],
            allowed_conditions=["C0"],
        )
        request = SimpleNamespace(
            diameter=2.0,
            size_level="medium",
            axis_name="z",
            axis_origin=(0.0, 0.0, 0.0),
            axis_direction=(0.0, 0.0, 1.0),
            tool_length=10.0,
        )

        with TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir)
            dirs = {
                "step": output_root / "step",
                "point": output_root / "point",
                "image": output_root / "image",
                "label": output_root / "label",
                "bin": output_root / "bin",
                "preview": output_root / "preview",
            }
            for path in dirs.values():
                path.mkdir(parents=True, exist_ok=True)

            with patch.object(dataset_builder, "instantiate_category", return_value=instance), patch.object(
                dataset_builder, "sample_condition_hole_count", return_value=2
            ), patch.object(dataset_builder, "sample_hole_request", side_effect=[request, None]), patch.object(
                dataset_builder,
                "apply_holes",
                return_value=(
                    "candidate-shape",
                    [
                        {
                            "hole_id": 1,
                            "diameter": 2.0,
                            "size_level": "medium",
                            "axis_name": "z",
                            "axis_origin": [0.0, 0.0, 0.0],
                            "axis_direction": [0.0, 0.0, 1.0],
                            "tool_length": 10.0,
                            "through": True,
                            "success": True,
                        }
                    ],
                ),
            ), patch.object(dataset_builder, "_append_manifest_row") as append_manifest, patch.object(
                dataset_builder, "_write_label_json"
            ) as write_label:
                payload = dataset_builder.build_one_model(
                    model_id="model_000001",
                    category="demo",
                    rng=SimpleNamespace(choice=lambda items: items[0]),
                    dirs=dirs,
                    linear_deflection=0.08,
                    angular_deflection=0.4,
                    zero_hole_ratio=0.0,
                    hole_clearance=0.8,
                    output_root=output_root,
                    export_flags={"step": False, "point": False, "image": False, "label": True, "bin": False},
                )

        self.assertIsNone(payload)
        append_manifest.assert_not_called()
        write_label.assert_not_called()

    def test_build_one_model_logs_failure_reason_for_underfilled_samples(self) -> None:
        dataset_builder = _load_dataset_builder_with_stubs()
        instance = SimpleNamespace(
            shape="base-shape",
            category="demo",
            has_internal_structure=False,
            base_params={"demo": True},
            preferred_axes=["z"],
            allowed_conditions=["C0"],
        )

        with TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir)
            dirs = {
                "step": output_root / "step",
                "point": output_root / "point",
                "image": output_root / "image",
                "label": output_root / "label",
                "bin": output_root / "bin",
                "preview": output_root / "preview",
            }
            for path in dirs.values():
                path.mkdir(parents=True, exist_ok=True)

            with patch.object(dataset_builder, "instantiate_category", return_value=instance), patch.object(
                dataset_builder, "sample_condition_hole_count", return_value=1
            ), patch.object(dataset_builder, "sample_hole_request", return_value=None):
                payload = dataset_builder.build_one_model(
                    model_id="model_000001",
                    category="demo",
                    rng=SimpleNamespace(choice=lambda items: items[0]),
                    dirs=dirs,
                    linear_deflection=0.08,
                    angular_deflection=0.4,
                    zero_hole_ratio=0.0,
                    hole_clearance=0.8,
                    output_root=output_root,
                    export_flags={"step": False, "point": False, "image": False, "label": False, "bin": False},
                )

            self.assertIsNone(payload)
            failure_log_path = output_root / "failure_log.jsonl"
            self.assertTrue(failure_log_path.exists())
            failure_record = json.loads(failure_log_path.read_text(encoding="utf-8").strip())
            self.assertEqual(failure_record["model_id"], "model_000001")
            self.assertEqual(failure_record["failure_reason"], "underfilled_hole_count")
            self.assertEqual(failure_record["target_hole_count"], 1)
            self.assertEqual(failure_record["accepted_hole_count"], 0)

    def test_main_retries_until_target_success_count(self) -> None:
        dataset_builder = _load_dataset_builder_with_stubs()

        with TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir)
            args = SimpleNamespace(
                output_root=output_root,
                num_models=2,
                seed=123,
                zero_hole_ratio=0.0,
                categories=["demo"],
                linear_deflection=0.08,
                angular_deflection=0.4,
                hole_clearance=0.8,
                export_step=False,
                export_point=False,
                export_image=False,
                export_label=True,
                export_bin=False,
                resume=False,
                quiet=True,
            )
            build_results = [
                None,
                {"category": "demo", "condition_level": "C0", "hole_count": 1},
                {"category": "demo", "condition_level": "C1", "hole_count": 2},
            ]

            with patch.object(dataset_builder, "parse_args", return_value=args), patch.object(
                dataset_builder,
                "prepare_dirs",
                return_value={
                    "step": output_root / "step",
                    "point": output_root / "point",
                    "image": output_root / "image",
                    "label": output_root / "label",
                    "bin": output_root / "bin",
                    "preview": output_root / "preview",
                },
            ), patch.object(dataset_builder, "build_one_model", side_effect=build_results) as build_one_model:
                result = dataset_builder.main()

        self.assertEqual(result, 0)
        self.assertEqual(build_one_model.call_count, 3)
        self.assertEqual(
            [call.kwargs["model_id"] for call in build_one_model.call_args_list],
            ["model_000001", "model_000002", "model_000003"],
        )


if __name__ == "__main__":
    unittest.main()
