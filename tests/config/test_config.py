"""Tests for configuration parsing, validation, and path handling."""

import tempfile
import unittest
from pathlib import Path

from align.config import (
    ArchitectureConfig,
    CanonicalizeConfig,
    MatchConfig,
    ObjectiveConfig,
    PathConfig,
    RunConfig,
    RuntimeConfig,
    SelectionConfig,
    SolverStep,
    load_align_config,
    validate_architecture,
    validate_paths,
)


class TestCanonicalizeConfig(unittest.TestCase):
    """Strict validation and typing for the canonicalize stage config."""

    def test_epsilon_string_converted_to_float(self) -> None:
        config = CanonicalizeConfig(epsilon="1e-8")
        self.assertIsInstance(config.epsilon, float)
        self.assertAlmostEqual(config.epsilon, 1e-8)

    def test_epsilon_int_converted_to_float(self) -> None:
        config = CanonicalizeConfig(epsilon=1)
        self.assertIsInstance(config.epsilon, float)
        self.assertEqual(config.epsilon, 1.0)

    def test_include_bias_in_norm_string_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "include_bias_in_norm.*bool"):
            CanonicalizeConfig(include_bias_in_norm="false")

    def test_default_types(self) -> None:
        config = CanonicalizeConfig()
        self.assertIsNone(config.strategy)
        self.assertIsInstance(config.epsilon, float)
        self.assertIsInstance(config.degenerate_channels, str)
        self.assertIsInstance(config.include_bias_in_norm, bool)
        self.assertIsInstance(config.activation, str)

    def test_invalid_strategy_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown canonicalize.strategy"):
            CanonicalizeConfig(strategy="minimum")

    def test_invalid_degenerate_channels_rejected(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "Unknown canonicalize.degenerate_channels"
        ):
            CanonicalizeConfig(degenerate_channels="drop")

    def test_invalid_activation_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown canonicalize.activation"):
            CanonicalizeConfig(activation="swish")

    def test_method_kwargs_typed(self) -> None:
        config = CanonicalizeConfig(
            strategy="unit_norm",
            epsilon="1e-8",
            degenerate_channels="preserve",
            include_bias_in_norm=True,
            activation="relu",
        )
        kwargs = config.canonicalizer_kwargs
        self.assertIsInstance(kwargs["epsilon"], float)
        self.assertEqual(kwargs["strategy"], "unit_norm")
        self.assertIsInstance(kwargs["include_bias_in_norm"], bool)
        self.assertEqual(kwargs["activation"], "relu")

    def test_from_mapping_rejects_unknown_option(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown canonicalize config option"):
            CanonicalizeConfig.from_mapping({"unexpected": 1})

    def test_from_mapping_rejects_quoted_boolean(self) -> None:
        with self.assertRaisesRegex(ValueError, "include_bias_in_norm.*bool"):
            CanonicalizeConfig.from_mapping({"include_bias_in_norm": "false"})


class TestObjectiveConfig(unittest.TestCase):
    def test_string_shorthand(self) -> None:
        config = ObjectiveConfig.from_mapping("diagonal_fisher")
        self.assertEqual(config.type, "diagonal_fisher")

    def test_mapping_with_weights_path(self) -> None:
        config = ObjectiveConfig.from_mapping(
            {"type": "diagonal_fisher", "weights_path": "w.npz"}
        )
        self.assertEqual(config.kwargs, {"weights_path": "w.npz"})

    def test_relative_fisher_metrics_path(self) -> None:
        config = ObjectiveConfig.from_mapping(
            {"type": "relative_fisher", "metrics_path": "grams.npz"}
        )
        self.assertEqual(config.kwargs, {"metrics_path": "grams.npz"})

    def test_unknown_type_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown match.objective.type"):
            ObjectiveConfig(type="cosine")


class TestMatchConfig(unittest.TestCase):
    def test_rejects_unknown_option(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown match config option"):
            MatchConfig.from_mapping({"unexpected": 1})

    def test_barycenter_passes_parsed_and_validated(self) -> None:
        config = MatchConfig.from_mapping({"barycenter_passes": 2})
        self.assertEqual(config.barycenter_passes, 2)
        with self.assertRaisesRegex(ValueError, "barycenter_passes"):
            MatchConfig.from_mapping({"barycenter_passes": 0})
        with self.assertRaisesRegex(ValueError, "barycenter_passes"):
            MatchConfig.from_mapping({"barycenter_passes": True})

    def test_default_objective_and_solvers(self) -> None:
        config = MatchConfig.from_mapping({})
        self.assertEqual(config.objective.type, "euclidean")
        self.assertEqual(len(config.solvers), 1)
        self.assertEqual(config.solvers[0].solver, "lap")

    def test_solvers_round_trip(self) -> None:
        config = MatchConfig.from_mapping(
            {
                "objective": {"type": "diagonal_fisher"},
                "solvers": [{"solver": "lap", "max_sweeps": 5}],
            }
        )
        self.assertEqual(config.objective.type, "diagonal_fisher")
        self.assertEqual(config.solvers_payload()[0]["max_sweeps"], 5)

    def test_empty_solvers_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty list"):
            MatchConfig.from_mapping({"solvers": []})


class TestSolverStep(unittest.TestCase):
    def test_rejects_unknown_option(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown match solver config option"):
            SolverStep.from_mapping({"solver": "lap", "max_iters": 3})

    def test_rejects_quoted_booleans(self) -> None:
        with self.assertRaisesRegex(ValueError, "record_loss_history.*bool"):
            SolverStep.from_mapping(
                {"solver": "sinkhorn", "record_loss_history": "false"}
            )
        with self.assertRaisesRegex(ValueError, "harden.*bool"):
            SolverStep.from_mapping({"solver": "sinkhorn", "harden": "true"})

    def test_groups_round_trip(self) -> None:
        step = SolverStep.from_mapping(
            {"solver": "lap", "groups": ["core/Conv_1"], "max_sweeps": 5}
        )
        self.assertEqual(step.groups, ("core/Conv_1",))
        self.assertEqual(
            step.to_dict(),
            {
                "solver": "lap",
                "tolerance": 0.0,
                "max_sweeps": 5,
                "groups": ["core/Conv_1"],
            },
        )

    def test_components_round_trip(self) -> None:
        step = SolverStep.from_mapping(
            {"solver": "lap", "components": ["*/ffn"], "max_sweeps": 5}
        )
        self.assertEqual(step.components, ("*/ffn",))
        self.assertEqual(
            step.to_dict(),
            {
                "solver": "lap",
                "tolerance": 0.0,
                "max_sweeps": 5,
                "components": ["*/ffn"],
            },
        )

    def test_rejects_groups_and_components_together(self) -> None:
        with self.assertRaisesRegex(ValueError, "not both"):
            SolverStep.from_mapping(
                {"solver": "lap", "groups": ["g"], "components": ["b"]}
            )


class TestArchitectureConfig(unittest.TestCase):
    def test_defaults_to_mlp(self) -> None:
        config = ArchitectureConfig()
        self.assertEqual(config.family, "mlp")
        self.assertEqual(config.recipe_kwargs, {})

    def test_unknown_family_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown architecture family"):
            ArchitectureConfig(family="unknown_family")

    def test_options_validated_per_family(self) -> None:
        config = ArchitectureConfig.from_mapping(
            {"family": "residual_convnet", "parameter_root": "core"}
        )
        self.assertEqual(config.recipe_kwargs["parameter_root"], "core")

    def test_unknown_option_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown architecture"):
            ArchitectureConfig.from_mapping({"family": "mlp", "unexpected": 1})

    def test_as_dict_round_trip(self) -> None:
        config = ArchitectureConfig(
            family="convnet", options={"parameter_root": "core"}
        )
        self.assertEqual(
            config.as_dict(), {"family": "convnet", "parameter_root": "core"}
        )


class TestRunConfig(unittest.TestCase):
    def _minimal(self, **extra) -> dict:
        payload = {"pipeline": ["canonicalize", "match"]}
        payload.update(extra)
        return payload

    def test_requires_non_empty_pipeline(self) -> None:
        with self.assertRaisesRegex(ValueError, "pipeline"):
            RunConfig.from_mapping({"pipeline": []})

    def test_rejects_unknown_stage(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown pipeline stage"):
            RunConfig.from_mapping({"pipeline": ["matching"]})

    def test_rejects_unknown_top_level_option(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown top-level config option"):
            RunConfig.from_mapping(self._minimal(unexpected=1))

    def test_section_present_but_not_in_pipeline_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "not in the pipeline"):
            RunConfig.from_mapping({"pipeline": ["canonicalize"], "match": {}})

    def test_active_stages_returns_pipeline(self) -> None:
        config = RunConfig.from_mapping(
            self._minimal(pipeline=["match", "canonicalize"])
        )
        self.assertEqual(config.active_stages(), ["match", "canonicalize"])

    def test_only_canonicalize_leaves_match_none(self) -> None:
        config = RunConfig.from_mapping({"pipeline": ["canonicalize"]})
        self.assertIsNotNone(config.canonicalize)
        self.assertIsNone(config.match)

    def test_center_softmax_head_stage_parses(self) -> None:
        config = RunConfig.from_mapping(
            {
                "pipeline": ["canonicalize", "center_softmax_head", "match"],
                "center_softmax_head": {"head": "core.Dense_0"},
            }
        )
        self.assertIsNotNone(config.center_softmax_head)
        self.assertEqual(config.center_softmax_head.head_path, ("core", "Dense_0"))
        self.assertEqual(
            config.as_dict()["center_softmax_head"], {"head": "core.Dense_0"}
        )

    def test_center_softmax_head_defaults_to_detection(self) -> None:
        config = RunConfig.from_mapping({"pipeline": ["center_softmax_head"]})
        self.assertIsNotNone(config.center_softmax_head)
        self.assertIsNone(config.center_softmax_head.head)
        self.assertIsNone(config.center_softmax_head.head_path)

    def test_center_softmax_head_section_requires_stage(self) -> None:
        with self.assertRaisesRegex(ValueError, "not in the pipeline"):
            RunConfig.from_mapping({"pipeline": ["match"], "center_softmax_head": {}})

    def test_center_softmax_head_rejects_unknown_option(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown"):
            RunConfig.from_mapping(
                {
                    "pipeline": ["center_softmax_head"],
                    "center_softmax_head": {"kernel": "core.Dense_0"},
                }
            )

    def test_as_dict_round_trip(self) -> None:
        config = RunConfig.from_mapping(
            self._minimal(architecture={"family": "convnet"})
        )
        payload = config.as_dict()
        self.assertEqual(payload["pipeline"], ["canonicalize", "match"])
        self.assertEqual(payload["architecture"], {"family": "convnet"})


class TestLoaderAndValidation(unittest.TestCase):
    def test_load_config_merges_yaml_and_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_file = root / "config.yaml"
            config_file.write_text(
                f"""
paths:
  experiment_root: {root}
pipeline: [canonicalize, match]
selection:
  samples_per_chain: 2
  reference_chain: 0
runtime:
  resume: false
                """
            )
            overrides = {
                "selection": {"reference_chain": 3, "max_total": 4},
                "pipeline": ["canonicalize"],
                "runtime": {"dry_run": True},
            }
            config = load_align_config(config_file, overrides)
            self.assertEqual(config.selection.samples_per_chain, 2)
            self.assertEqual(config.selection.reference_chain, 3)
            self.assertEqual(config.selection.max_total, 4)
            self.assertEqual(config.active_stages(), ["canonicalize"])
            self.assertIsNone(config.match)
            self.assertTrue(config.runtime.dry_run)
            self.assertEqual(config.paths.sample_format, "pytree_npz")

    def test_path_config_validates_sample_format(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_file = root / "config.yaml"
            config_file.write_text(
                f"""
pipeline: [canonicalize]
paths:
  experiment_root: {root}
  sample_format: missing
                """
            )
            with self.assertRaises(ValueError):
                load_align_config(config_file)

    def test_residual_convnet_accepted_by_validator(self) -> None:
        config = RunConfig(architecture=ArchitectureConfig(family="residual_convnet"))
        validate_architecture(config)

    def test_validate_paths_checks_samples_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            samples_dir = root / "samples"
            samples_dir.mkdir()
            config = RunConfig(paths=PathConfig(experiment_root=root))
            validate_paths(config)
            samples_dir.rmdir()
            with self.assertRaises(FileNotFoundError):
                validate_paths(config)

    def test_section_configs_reject_unknown_options(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown paths config option"):
            PathConfig.from_mapping({"experiment_root": ".", "unexpected": 1})
        with self.assertRaisesRegex(ValueError, "Unknown selection config option"):
            SelectionConfig.from_mapping({"reference_chain": 0, "unexpected": 1})
        with self.assertRaisesRegex(ValueError, "Unknown runtime config option"):
            RuntimeConfig.from_mapping({"resume": False, "unexpected": 1})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
