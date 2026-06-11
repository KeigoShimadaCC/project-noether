"""Minimal CLI: prove the loop end to end from a terminal.

H1 surface: `noether kernels`, `noether eval1 [--results DIR]`.
The conversational front grows here later; physics state stays server-side.
"""

import argparse
import sys
import uuid
from pathlib import Path

from noether.kernels.base import Capability, KernelTask, KernelUnavailable
from noether.kernels.cadabra import CadabraAdapter
from noether.kernels.sympy_kernel import SympyKernelAdapter
from noether.npr.latex import render
from noether.orchestrator.session import Session
from noether.provenance.bundle import ResultBundle, write_bundle
from noether.verify.checks import (
    DivergenceFreeCheck,
    EqualOnBackgroundCheck,
    SymmetricCheck,
    WellFormedCheck,
)
from noether.verify.ladder import run_ladder


def _adapters() -> dict:
    return {"cadabra": CadabraAdapter(), "sympy": SympyKernelAdapter()}


def cmd_kernels(_args) -> int:
    for name, adapter in _adapters().items():
        status = adapter.version() if adapter.available() else "NOT INSTALLED"
        caps = ", ".join(sorted(c.value for c in adapter.capabilities()))
        print(f"{name:10s} {status:30s} [{caps}]")
    return 0


def cmd_eval1(args) -> int:
    from evals import eval1_eh_trace as ev1

    adapters = _adapters()
    session = Session(session_id=f"eval1-{uuid.uuid4().hex[:8]}")

    # INGEST with open ambiguities, ELICIT with the documented answers.
    session.ingest(ev1.build_npr(resolved=False))
    print("Elicitation:")
    for amb in session.npr.unresolved_ambiguities():
        answer = ev1.ELICITATION_ANSWERS[amb.id]
        print(f"  Q: {amb.question}")
        print(f"  A: {answer}")
        session.resolve(amb.id, answer)

    plan = session.plan()
    print(f"\nPlan ({plan.task_type}):")
    for i, step in enumerate(plan.steps, 1):
        print(f"  {i}. [{step.capability.value}] {step.description}")

    # COMPUTE: cadabra derivation if available.
    computed = []
    derivation_note = ""
    cadabra = adapters["cadabra"]
    if cadabra.available():
        task = KernelTask(
            capability=Capability.VARY,
            description="EH trace-form metric variation",
            payload={"template": "eval1_eh_trace"},
        )
        result = cadabra.run(task)
        computed.append(result)
        residue_zero = result.value["checks"].get("residue_zero")
        derivation_note = f"cadabra derivation residue check: residue_zero={residue_zero}"
        print(f"\n{derivation_note}")
        if residue_zero != "True":
            print("DERIVATION CHECK FAILED; refusing to present result as verified.")
    else:
        derivation_note = (
            "cadabra2 not installed: symbolic derivation step SKIPPED; "
            "presenting the documented target with component verification only"
        )
        print(f"\n{derivation_note}")

    # VERIFY: the ladder on the target EOM.
    target = ev1.target_eom()
    checks = [
        WellFormedCheck(expected_free=[ev1.MU, ev1.NU]),
        SymmetricCheck(),
        DivergenceFreeCheck(),
        EqualOnBackgroundCheck(rhs=ev1.target_eom_expanded()),
    ]
    report = run_ladder(target, checks, adapters)
    for line in report.summary().splitlines():
        print(f"  {line}")
    for check in report.results:
        computed.extend(check.artifacts)

    # PRESENT + provenance bundle.
    eom_tex = render(target) + " = 0"
    verified = report.all_passed and (
        not cadabra.available() or computed[0].value["checks"].get("residue_zero") == "True"
    )
    print(f"\nResult: {eom_tex}")
    print(f"Verified: {verified}")

    bundle = ResultBundle(
        session_id=session.session_id,
        result_id="eom-metric",
        result_tex=eom_tex,
        result_expr=target.model_dump(),
        npr_snapshot=session.npr,
        plan=[s.model_dump() for s in plan.steps],
        computed=computed,
        ladder=report,
        narrative=(
            "# Eval 1: Einstein-Hilbert in trace form\n\n"
            "Assumptions: see assumptions.json (noether-default-v1).\n\n"
            f"Derivation: {derivation_note}\n\n"
            f"Checks:\n{report.summary()}\n\n"
            f"Result: $ {eom_tex} $\n"
        ),
    )
    path = write_bundle(Path(args.results), bundle)
    session.record_result("eom-metric")
    print(f"Provenance bundle: {path}")
    return 0 if verified else 1


def main() -> int:
    parser = argparse.ArgumentParser(prog="noether")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("kernels", help="list kernel adapters and availability")
    p1 = sub.add_parser("eval1", help="run eval 1 end to end")
    p1.add_argument("--results", default="results", help="provenance bundle root")
    args = parser.parse_args()
    try:
        if args.command == "kernels":
            return cmd_kernels(args)
        if args.command == "eval1":
            return cmd_eval1(args)
    except KernelUnavailable as exc:
        print(f"kernel unavailable: {exc}", file=sys.stderr)
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
