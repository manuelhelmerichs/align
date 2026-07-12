"""Tests for sample manifest discovery, serialization, and filtering."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
