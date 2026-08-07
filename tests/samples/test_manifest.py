"""Tests for sample manifest discovery, serialization, and filtering."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pytest

from align.sample_manifest import SampleManifest


def _write_sample(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, weight=np.array([1.0]))


class ManifestTests(unittest.TestCase):
    def test_manifest_build_and_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            samples_dir = root / "samples"
            _write_sample(samples_dir / "0" / "sample_0.npz")
            _write_sample(samples_dir / "0" / "sample_1.npz")
            _write_sample(samples_dir / "1" / "sample_0.npz")
            tree_path = root / "tree"
            tree_path.write_bytes(b"dummy")

            manifest = SampleManifest.build(
                samples_dir,
                tree_path,
                chain_indices=[0],
                samples_per_chain=1,
                reference_chain=0,
                reference_sample=0,
            )
            self.assertEqual(manifest.total, 1)
            self.assertEqual(manifest.reference_index, 0)
            self.assertEqual(manifest.filters["chain_indices"], [0])

            manifest_path = root / "manifest.json"
            manifest.save(manifest_path)
            manifest_payload = json.loads(manifest_path.read_text())
            self.assertEqual(manifest_payload["reference_chain"], 0)
            self.assertEqual(manifest_payload["reference_sample"], 0)
            shared_tree_path = manifest.ensure_shared_tree(manifest_path)
            manifest.ensure_shared_records(manifest_path)
            records_sidecar = manifest_path.with_name(
                f"{manifest_path.stem}.records.npy"
            )
            paths_sidecar = manifest_path.with_name(f"{manifest_path.stem}.paths.bin")
            tree_override = manifest_path.with_name(
                f"{manifest_path.stem}.tree_path.txt"
            )
            tree_copy = manifest_path.with_name(f"{manifest_path.stem}.treedef.pkl")

            self.assertTrue(records_sidecar.exists())
            self.assertTrue(paths_sidecar.exists())
            self.assertTrue(tree_override.exists())
            self.assertTrue(tree_copy.exists())
            self.assertEqual(shared_tree_path, tree_copy.resolve())

            loaded = SampleManifest.load(manifest_path)
            self.assertEqual(loaded.records[0].label, "chain0_sample0")
            self.assertEqual(loaded.digest(), manifest.digest())
            self.assertEqual(Path(loaded.tree_path), tree_copy.resolve())


def _sample_tree(tmp_path: Path, counts: dict[int, int]) -> tuple[Path, Path]:
    samples_dir = tmp_path / "samples"
    for chain, count in counts.items():
        for sample in range(count):
            _write_sample(samples_dir / str(chain) / f"sample_{sample}.npz")
    tree_path = tmp_path / "tree"
    tree_path.write_bytes(b"dummy")
    return samples_dir, tree_path


def test_requested_chains_must_all_exist(tmp_path):
    samples_dir, tree_path = _sample_tree(tmp_path, {0: 2})
    with pytest.raises(ValueError, match=r"Requested chains.*\[1\]"):
        SampleManifest.build(
            samples_dir,
            tree_path,
            chain_indices=[0, 1],
            reference_chain=0,
        )


def test_samples_per_chain_is_an_exact_request(tmp_path):
    samples_dir, tree_path = _sample_tree(tmp_path, {0: 3, 1: 2})
    with pytest.raises(ValueError, match=r"chain 1: 2/3"):
        SampleManifest.build(
            samples_dir,
            tree_path,
            samples_per_chain=3,
            reference_chain=0,
        )


def test_step_is_accounted_for_in_exact_per_chain_count(tmp_path):
    samples_dir, tree_path = _sample_tree(tmp_path, {0: 4, 1: 3})
    with pytest.raises(ValueError, match=r"chain 1: 2/3"):
        SampleManifest.build(
            samples_dir,
            tree_path,
            samples_per_chain=3,
            sample_step=2,
            reference_chain=0,
        )


def test_per_chain_and_global_limits_are_mutually_exclusive(tmp_path):
    samples_dir, tree_path = _sample_tree(tmp_path, {0: 2})
    with pytest.raises(ValueError, match="mutually exclusive"):
        SampleManifest.build(
            samples_dir,
            tree_path,
            samples_per_chain=1,
            max_total=1,
            reference_chain=0,
        )


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("sample_step", 0, "sample_step must be positive"),
        ("samples_per_chain", 0, "samples_per_chain must be positive"),
        ("max_total", 0, "max_total must be positive"),
    ],
)
def test_selection_limits_must_be_positive(tmp_path, keyword, value, message):
    samples_dir, tree_path = _sample_tree(tmp_path, {0: 2})
    with pytest.raises(ValueError, match=message):
        SampleManifest.build(
            samples_dir,
            tree_path,
            reference_chain=0,
            **{keyword: value},
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
