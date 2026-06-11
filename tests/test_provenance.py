"""Provenance bundles: layout, content, and the no-bare-expression rule."""

import json

from evals.eval1_eh_trace import build_npr, target_eom
from noether.npr.latex import render
from noether.provenance.bundle import ResultBundle, write_bundle
from noether.verify.checks import CheckResult
from noether.verify.ladder import LadderReport


def make_bundle():
    report = LadderReport(
        results=[
            CheckResult(
                name="well-formed",
                rung="V0",
                passed=True,
                detail="ok",
                computed_by="structural",
            ),
        ]
    )
    return ResultBundle(
        session_id="s1",
        result_id="r1",
        result_tex=render(target_eom()) + " = 0",
        result_expr=target_eom().model_dump(),
        npr_snapshot=build_npr(resolved=True),
        plan=[{"capability": "vary", "description": "test"}],
        computed=[],
        ladder=report,
        narrative="# test narrative\n",
    )


class TestBundle:
    def test_layout(self, tmp_path):
        base = write_bundle(tmp_path, make_bundle())
        assert (base / "result.json").exists()
        assert (base / "assumptions.json").exists()
        assert (base / "plan.json").exists()
        assert (base / "checks.json").exists()
        assert (base / "narrative.md").exists()
        assert (base / "scripts").is_dir()
        assert (base / "raw").is_dir()

    def test_assumptions_carry_conventions(self, tmp_path):
        base = write_bundle(tmp_path, make_bundle())
        assumptions = json.loads((base / "assumptions.json").read_text())
        assert assumptions["conventions"]["id"] == "noether-default-v1"
        assert assumptions["geometry"]["connection"]["type"] == "levi-civita"

    def test_verified_flag_tracks_ladder(self, tmp_path):
        bundle = make_bundle()
        bundle.ladder.results[0].passed = False
        base = write_bundle(tmp_path, bundle)
        result = json.loads((base / "result.json").read_text())
        assert result["verified"] is False
