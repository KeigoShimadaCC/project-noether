"""Write and load result bundles.

Layout (docs/02_TECH_SPEC.md section 7):
  results/<session>/<result-id>/
    result.json  assumptions.json  plan.json  checks.json
    scripts/     raw/              narrative.md
"""

import json
from pathlib import Path

from pydantic import BaseModel, Field

from noether.kernels.base import ComputedResult
from noether.npr.schema import NPR
from noether.verify.ladder import LadderReport


class ResultBundle(BaseModel):
    session_id: str
    result_id: str
    result_tex: str
    result_expr: dict | None = None
    npr_snapshot: NPR
    plan: list[dict] = Field(default_factory=list)
    computed: list[ComputedResult] = Field(default_factory=list)
    ladder: LadderReport
    narrative: str = ""


def write_bundle(root: Path, bundle: ResultBundle) -> Path:
    base = root / bundle.session_id / bundle.result_id
    (base / "scripts").mkdir(parents=True, exist_ok=True)
    (base / "raw").mkdir(parents=True, exist_ok=True)

    _dump(
        base / "result.json",
        {
            "result_tex": bundle.result_tex,
            "result_expr": bundle.result_expr,
            "verified": bundle.ladder.all_passed,
        },
    )
    _dump(base / "assumptions.json", bundle.npr_snapshot.model_dump(mode="json"))
    _dump(base / "plan.json", bundle.plan)
    _dump(base / "checks.json", bundle.ladder.model_dump(mode="json"))

    for i, computed in enumerate(bundle.computed):
        ext = {"cadabra": "cdb", "python-sympy": "py", "wolfram": "wl"}.get(
            computed.script.language, "txt"
        )
        (base / "scripts" / f"{i:02d}_{computed.kernel_name}.{ext}").write_text(
            computed.script.source
        )
        _dump(
            base / "raw" / f"{i:02d}_{computed.kernel_name}.json",
            {
                "kernel_version": computed.kernel_version,
                "stdout": computed.raw.stdout,
                "stderr": computed.raw.stderr,
                "returncode": computed.raw.returncode,
                "duration_s": computed.raw.duration_s,
                "notes": computed.notes,
            },
        )

    (base / "narrative.md").write_text(bundle.narrative)
    return base


def _dump(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str))
