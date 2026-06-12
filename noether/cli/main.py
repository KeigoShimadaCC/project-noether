"""Minimal CLI: prove the loop end to end from a terminal.

H1 surface: `noether kernels`, `noether ingest "<lagrangian>"`,
`noether elicit "<lagrangian>"`, `noether eval{1..5} [--results DIR]`
(eval5 gates Horizon 2).
The conversational front grows here later; physics state stays server-side.
"""

import argparse
import sys
import uuid
from pathlib import Path

from noether.kernels.base import Capability, KernelTask, KernelUnavailable
from noether.kernels.cadabra import CadabraAdapter
from noether.kernels.sympy_kernel import SympyKernelAdapter
from noether.llm import CliLLMAdapter, LLMError
from noether.npr.latex import render
from noether.npr.parse import ParseError
from noether.orchestrator.elicit import apply_resolutions, propose_resolutions
from noether.orchestrator.ingest import ingest_action
from noether.orchestrator.planner import AmbiguityBlocked, build_plan
from noether.orchestrator.session import Session
from noether.provenance.bundle import ResultBundle, write_bundle
from noether.verify.ladder import run_ladder

EVAL_KEYS = ("eval1", "eval1s", "eval2", "eval3", "eval3s", "eval4", "eval5")


def _adapters() -> dict:
    return {"cadabra": CadabraAdapter(), "sympy": SympyKernelAdapter()}


def cmd_kernels(_args) -> int:
    for name, adapter in _adapters().items():
        status = adapter.version() if adapter.available() else "NOT INSTALLED"
        caps = ", ".join(sorted(c.value for c in adapter.capabilities()))
        print(f"{name:10s} {status:30s} [{caps}]")
    return 0


def cmd_ingest(args) -> int:
    """Parse an action and show the draft NPR plus the questions that block it.

    This is the INGEST beat in isolation: it never answers a physics question,
    it only surfaces them, so the printed draft is deliberately un-plannable.
    """
    try:
        result = ingest_action(args.measure, args.lagrangian)
    except ParseError as exc:
        print(f"parse error: {exc}", file=sys.stderr)
        return 2

    print(f"Action:  \\int {args.measure} ( {args.lagrangian} )\n")
    print(f"Parsed Lagrangian (NPR -> LaTeX): {render(result.lagrangian)}\n")

    print("Objects (syntactic classification; roles provisional):")
    for obj in result.npr.objects:
        print(f"  {obj.name:6s} {obj.kind:13s} rank={obj.rank} role={obj.role}*")

    print("\nOpen questions (must be resolved by a human before planning):")
    for amb in result.npr.ambiguities:
        opts = ", ".join(amb.options)
        print(f"  [{amb.id}] ({amb.kind}) {amb.question}")
        print(f"      options: {opts}")

    print(
        f"\nWell-posed: {result.npr.is_well_posed()} "
        "(planning is structurally blocked until every question is answered)."
    )
    return 0


def cmd_elicit(args) -> int:
    """Ingest an action, then let an auto-detected agent CLI PROPOSE answers.

    Proposals are unconfirmed by default: nothing is resolved and planning
    stays blocked. With --accept-llm the human explicitly delegates
    confirmation to the model, applies its on-menu proposals, and (if every
    question is answered) prints the plan.
    """
    try:
        result = ingest_action(args.measure, args.lagrangian)
    except ParseError as exc:
        print(f"parse error: {exc}", file=sys.stderr)
        return 2
    npr = result.npr

    print(f"Action:  \\int {args.measure} ( {args.lagrangian} )\n")
    print("Open questions:")
    for amb in npr.ambiguities:
        print(f"  [{amb.id}] {amb.question}")
        print(f"      options: {', '.join(amb.options)}")

    llm = CliLLMAdapter()
    if not llm.available():
        print(
            "\nNo agent CLI detected (looked for codex, claude, gemini, droid). "
            "Answer the questions manually, then plan."
        )
        return 0

    print(f"\nDetected LLM: {llm.name} ({llm.version()})")
    try:
        proposal = propose_resolutions(npr, llm)
    except LLMError as exc:
        print(f"elicitation failed: {exc}", file=sys.stderr)
        return 2

    print("Proposed resolutions (UNCONFIRMED; a human must accept each):")
    for p in proposal.proposals:
        shown = p.choice if p.choice is not None else "(no valid proposal)"
        print(f"  [{p.ambiguity_id}] -> {shown}")
        if p.rationale:
            print(f"      rationale: {p.rationale}")

    if not args.accept_llm:
        print(
            "\nWell-posed: False. Re-run with --accept-llm to delegate "
            "confirmation to the model, or resolve manually."
        )
        return 0

    confirmations = {p.ambiguity_id: p.choice for p in proposal.proposals if p.choice is not None}
    confirmed = apply_resolutions(npr, confirmations)
    print(f"\n--accept-llm: applied {len(confirmations)} model proposal(s) as confirmed.")
    if not confirmed.is_well_posed():
        missing = [a.id for a in confirmed.unresolved_ambiguities()]
        print(f"Still blocked: no valid proposal for {', '.join(missing)}.")
        return 1
    try:
        plan = build_plan(confirmed)
    except AmbiguityBlocked as exc:
        print(f"still blocked: {exc}", file=sys.stderr)
        return 1
    print(f"\nPlan ({plan.task_type}):")
    for i, step in enumerate(plan.steps, 1):
        print(f"  {i}. [{step.capability.value}] {step.description}")
    return 0


def cmd_serve(args) -> int:
    """Run the HTTP session API. FastAPI/uvicorn live behind the [server]
    extra so the core package stays lean."""
    try:
        import uvicorn

        from noether.orchestrator.store import SessionStore
        from noether.server import create_app
    except ImportError:
        print(
            'server dependencies missing: pip install -e ".[server]"',
            file=sys.stderr,
        )
        return 2
    store = SessionStore(Path(args.store)) if args.store else None
    app = create_app(store=store)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def cmd_mcp(args) -> int:
    try:
        from noether.mcp import create_mcp_server
        from noether.orchestrator.store import SessionStore
    except ImportError:
        print('mcp dependencies missing: pip install -e ".[mcp]"', file=sys.stderr)
        return 2
    store = SessionStore(Path(args.store)) if args.store else None
    create_mcp_server(store=store).run()
    return 0


def run_eval(key: str, results_root: str) -> int:
    from evals.registry import component_task, get_spec

    spec = get_spec(key)
    adapters = _adapters()
    session = Session(session_id=f"{spec.key}-{uuid.uuid4().hex[:8]}")

    print(f"== {spec.key}: {spec.title} ==\n")

    # INGEST with open ambiguities, ELICIT with the documented answers.
    session.ingest(spec.build_npr(False))
    print("Elicitation:")
    for amb in session.npr.unresolved_ambiguities():
        answer = spec.answers[amb.id]
        print(f"  Q: {amb.question}")
        print(f"  A: {answer}")
        session.resolve(amb.id, answer)

    plan = session.plan()
    print(f"\nPlan ({plan.task_type}):")
    for i, step in enumerate(plan.steps, 1):
        print(f"  {i}. [{step.capability.value}] {step.description}")

    # COMPUTE: audited cadabra derivations if the kernel is present.
    cadabra = adapters["cadabra"]
    cadabra_results = []
    derivation_ok = True
    derivation_notes = []
    if not spec.cadabra_runs:
        pass
    elif cadabra.available():
        print("\nKernel derivations (cadabra):")
        for run in spec.cadabra_runs:
            result = cadabra.run(
                KernelTask(
                    capability=Capability.VARY,
                    description=run.description,
                    payload={"template": run.template},
                )
            )
            cadabra_results.append(result)
            checks = result.value["checks"]
            oks = {k: checks.get(k) for k in run.required_true}
            ok = all(v == "True" for v in oks.values())
            derivation_ok = derivation_ok and ok
            line = f"{run.description}: " + ", ".join(f"{k}={v}" for k, v in oks.items())
            derivation_notes.append(line)
            print(f"  {'PASS' if ok else 'FAIL'} {line}")
            if not ok:
                print("  DERIVATION CHECK FAILED; result will not be presented as verified.")
    else:
        note = (
            "cadabra2 not installed: symbolic derivation SKIPPED; "
            "presenting documented targets with component verification only"
        )
        derivation_notes.append(note)
        print(f"\n{note}")

    # VERIFY + PRESENT + provenance bundle, one per presented result.
    exit_code = 0
    for presented in spec.results:
        target = presented.expr()
        report = run_ladder(target, presented.ladder(), adapters)
        computed = list(cadabra_results)
        for check in report.results:
            computed.extend(check.artifacts)

        extra_ok = True
        extra_lines = []
        for description, payload in presented.component_tasks:
            result = adapters["sympy"].run(component_task(description, payload))
            computed.append(result)
            passed = bool(result.value["passed"])
            extra_ok = extra_ok and passed
            extra_lines.append(
                f"[{'PASS' if passed else 'FAIL'}] (sympy) {description}: {result.value['detail']}"
            )

        eom_tex = render(target) + presented.tex_suffix
        verified = report.all_passed and extra_ok and derivation_ok
        print(f"\nResult [{presented.result_id}]: {eom_tex}")
        for line in report.summary().splitlines():
            print(f"  {line}")
        for line in extra_lines:
            print(f"  {line}")
        print(f"  Verified: {verified}")

        bundle = ResultBundle(
            session_id=session.session_id,
            result_id=presented.result_id,
            result_tex=eom_tex,
            result_expr=target.model_dump(),
            npr_snapshot=session.npr,
            plan=[s.model_dump() for s in plan.steps],
            computed=computed,
            ladder=report,
            narrative=(
                f"# {spec.title} [{presented.result_id}]\n\n"
                "Assumptions: see assumptions.json (noether-default-v1).\n\n"
                "Derivation:\n"
                + "".join(f"- {n}\n" for n in derivation_notes)
                + ("".join(f"- note: {n}\n" for n in spec.notes))
                + f"\nChecks:\n{report.summary()}\n"
                + "".join(f"{line}\n" for line in extra_lines)
                + f"\nResult: $ {eom_tex} $\n"
            ),
        )
        path = write_bundle(Path(results_root), bundle)
        session.record_result(presented.result_id)
        print(f"  Provenance bundle: {path}")
        if not verified:
            exit_code = 1
    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(prog="noether")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("kernels", help="list kernel adapters and availability")
    ing = sub.add_parser("ingest", help="parse a LaTeX action into a draft NPR + questions")
    ing.add_argument(
        "lagrangian",
        help="the scalar Lagrangian density, e.g. '-\\tfrac14 F_{\\mu\\nu} F^{\\mu\\nu}'",
    )
    ing.add_argument(
        "--measure", default=r"d^4x \sqrt{-g}", help="action measure (default: d^4x \\sqrt{-g})"
    )
    el = sub.add_parser("elicit", help="ingest, then let an agent CLI propose answers")
    el.add_argument("lagrangian", help="the scalar Lagrangian density")
    el.add_argument("--measure", default=r"d^4x \sqrt{-g}", help="action measure")
    el.add_argument(
        "--accept-llm",
        action="store_true",
        help="delegate confirmation to the model: apply on-menu proposals and plan",
    )
    srv = sub.add_parser("serve", help="run the HTTP session API (requires [server] extra)")
    srv.add_argument("--host", default="127.0.0.1", help="bind host (default: 127.0.0.1)")
    srv.add_argument("--port", type=int, default=8754, help="bind port (default: 8754)")
    srv.add_argument("--store", default=None, help="session store directory")
    mcp_p = sub.add_parser("mcp", help="run the MCP stdio server (requires [mcp] extra)")
    mcp_p.add_argument("--store", default=None, help="session store directory")
    for key in EVAL_KEYS:
        p = sub.add_parser(key, help=f"run {key} end to end")
        p.add_argument("--results", default="results", help="provenance bundle root")
    args = parser.parse_args()
    try:
        if args.command == "kernels":
            return cmd_kernels(args)
        if args.command == "ingest":
            return cmd_ingest(args)
        if args.command == "elicit":
            return cmd_elicit(args)
        if args.command == "serve":
            return cmd_serve(args)
        if args.command == "mcp":
            return cmd_mcp(args)
        if args.command in EVAL_KEYS:
            return run_eval(args.command, args.results)
    except KernelUnavailable as exc:
        print(f"kernel unavailable: {exc}", file=sys.stderr)
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
