"""Tests for shared runtime stage-pipeline bookkeeping."""

import jax.numpy as jnp

from align.runtime.pipeline import StagePipeline
from align.runtime.stages import StageResult
from align.sample_manifest import SampleRecord
from align.samples import WeightSample


class _CollectSink:
    def __init__(self) -> None:
        self.outputs = []

    def write(self, output) -> None:
        self.outputs.append(output)


class _Loader:
    def __init__(self, samples):
        self.samples = samples

    def load(self, record):
        return self.samples[record.index]


class _CanonicalizeStage:
    supports_batching = False
    prefers_gpu = False

    def process_single(self, record, sample):
        value = sample.params["x"] + 1
        return StageResult(
            sample=sample.with_params({"x": value}),
            scales={"g": jnp.asarray([record.index + 1])},
            diagnostics={"plan": "dense_chain"},
        )


class _MatchStage:
    supports_batching = True
    prefers_gpu = False

    def __init__(self) -> None:
        self.batch_calls = 0

    def process_single(self, record, sample):
        return self._result(record, sample)

    def process_batch(self, records, samples):
        self.batch_calls += 1
        return [
            self._result(record, sample)
            for record, sample in zip(records, samples, strict=True)
        ]

    def _result(self, record, sample):
        value = sample.params["x"] + 10
        return StageResult(
            sample=sample.with_params({"x": value}),
            transforms={"g": jnp.eye(1, dtype=jnp.uint8)},
            diagnostics={
                "reference": record.index == 0,
                "transform_families": {"g": "permutation"},
            },
        )


def _record(index: int) -> SampleRecord:
    return SampleRecord(
        index=index,
        chain_id=0,
        sample_id=index,
        relative_path=f"0/sample_{index}.npz",
    )


def test_stage_pipeline_collects_artifacts_diagnostics_and_stage_outputs() -> None:
    record = _record(0)
    sink = _CollectSink()
    pipeline = StagePipeline(
        [("canonicalize", _CanonicalizeStage()), ("match", _MatchStage())],
        save_intermediate=True,
    )

    pipeline.run(record, WeightSample({"x": jnp.asarray(0)}), sink)

    assert len(sink.outputs) == 1
    output = sink.outputs[0]
    assert output.record is record
    assert int(output.sample.params["x"]) == 11
    assert int(output.stage_outputs["canonicalize"].params["x"]) == 1
    assert output.diagnostics == {
        "canonicalize": {"plan": "dense_chain"},
        "match": {
            "reference": True,
            "transform_families": {"g": "permutation"},
        },
    }
    assert set(output.scales) == {"g"}
    assert set(output.transforms) == {"g"}


def test_stage_pipeline_batches_only_final_match_stage() -> None:
    records = [_record(0), _record(1)]
    samples = {
        record.index: WeightSample({"x": jnp.asarray(record.index)})
        for record in records
    }
    sink = _CollectSink()
    match_stage = _MatchStage()
    started: list[int] = []
    pipeline = StagePipeline(
        [("canonicalize", _CanonicalizeStage()), ("match", match_stage)],
        save_intermediate=True,
    )

    pipeline.run_records(
        records,
        _Loader(samples),
        sink,
        batch_size=2,
        on_record_start=lambda record: started.append(record.index),
    )

    assert match_stage.batch_calls == 1
    assert started == [0, 1]
    assert [int(output.sample.params["x"]) for output in sink.outputs] == [11, 12]
    assert [
        int(output.stage_outputs["canonicalize"].params["x"]) for output in sink.outputs
    ] == [1, 2]
    assert [output.diagnostics["match"]["reference"] for output in sink.outputs] == [
        True,
        False,
    ]
