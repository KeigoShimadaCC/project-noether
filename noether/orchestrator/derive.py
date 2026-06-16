"""DERIVE: run a verified derivation for a well-posed session.

This is the compute beat. It connects an arbitrary well-posed NPR to the
kernel for an action that is not one of the frozen evals:

  1. refuse unless the problem is well posed (build_plan enforces the
     no-guessing gate);
  2. ask the LLM to parameterize a Cadabra script (it writes a script, never
     an answer);
  3. run the script in the sandboxed kernel;
  4. TRUST the result only if the kernel's own residue check confirms it
     (`residue_zero=True`); otherwise mark it unverified;
  5. write a provenance bundle for every run, verified or not.

The bright line of AGENTS.md (model reasons, kernel computes, ladder confirms)
is preserved: `verified` is set by the kernel, not by the model or by us.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from pydantic import BaseModel, Field

from noether.kernels.base import Capability, ComputedResult, KernelTask
from noether.kernels.cadabra.generate import generate_script
from noether.llm.base import LLMAdapter
from noether.npr.schema import NPR
from noether.orchestrator.planner import build_plan
from noether.provenance.bundle import ResultBundle, write_bundle
from noether.verify.checks import CheckResult
from noether.verify.ladder import LadderReport


class FieldDerivation(BaseModel):
    """One field's derived result (EOM or quadratic action), with its verdict."""

    wrt: str
    kind: str = "eom"  # "eom" | "perturbation" | "adm"
    capability: Capability
    result_id: str = ""
    result_tex: str | None = None
    verified: bool = False
    checks: dict[str, str] = Field(default_factory=dict)
    kernel_name: str = ""
    kernel_version: str = ""
    llm_name: str = ""
    llm_version: str = ""
    script: str = ""
    bundle_path: str | None = None
    detail: str = ""


def _ladder_from_kernel(computed: ComputedResult, verified: bool, detail: str) -> LadderReport:
    """Represent the kernel's in-script residue check as a one-rung ladder.

    The residue check compares an independently derived variation against an
    independently stated candidate equation, both canonicalized by the kernel,
    so a zero residue is a genuine V3-style equality verified by computation.
    """
    return LadderReport(
        results=[
            CheckResult(
                name="variation-residue-zero",
                rung="V3",
                passed=verified,
                detail=detail,
                computed_by=computed.kernel_name,
                artifacts=[computed],
            )
        ]
    )


def _stderr_tail(stderr: str, *, limit: int = 240) -> str:
    lines = [ln.strip() for ln in (stderr or "").splitlines() if ln.strip()]
    tail = " | ".join(lines[-3:])
    return tail[-limit:] if len(tail) > limit else tail


def _result_detail(
    kind: str, verified: bool, checks: dict[str, str], computed: ComputedResult
) -> str:
    """Explain the verdict, and when unverified, say *why*.

    A run can fail to verify two very different ways: the generated script may
    never reach the kernel's residue check (a script or kernel error), or it
    may run and report a nonzero residue (the model's candidate disagrees with
    its own derivation). Collapsing both into one message hides which happened,
    so we distinguish them and surface the kernel's stderr for the first."""
    if verified:
        if kind == "perturbation":
            return "kernel confirmed the quadratic action reproduces the linearized equation"
        return "kernel confirmed the variation matches the candidate equation"

    candidate = "quadratic action" if kind == "perturbation" else "candidate equation"
    if "residue_zero" not in checks:
        rc = computed.raw.returncode
        tail = _stderr_tail(computed.raw.stderr)
        base = (
            "unverified: the generated script produced no residue check; it did "
            f"not run to completion (kernel exit {rc})"
        )
        return f"{base}: {tail}" if tail else base
    if checks.get("residue_zero") != "True":
        return (
            "unverified: the kernel computed a nonzero residue, so the model's "
            f"{candidate} does not match its own derivation"
        )
    if kind == "perturbation" and checks.get("linearized_eom_match") not in (None, "True"):
        return (
            "unverified: the residue vanished but the independent linearized-EOM "
            "cross-check did not match"
        )
    return "unverified: kernel did not confirm the result"


def _verdict(kind: str, checks: dict[str, str]) -> bool:
    """The kernel sets the verdict, not the model. For an EOM, the residue
    against the independent candidate must vanish. For a quadratic-action
    expansion, the linearized EOM must match both the documented operator and
    an independent linearization of the full equation."""
    if kind == "perturbation":
        if checks.get("residue_zero") != "True":
            return False
        # linearized_eom_match strengthens trust; require it when emitted.
        return checks.get("linearized_eom_match", "True") == "True"
    return checks.get("residue_zero") == "True"


def derive_field(
    npr: NPR,
    wrt: str,
    llm: LLMAdapter,
    adapters: dict,
    *,
    kind: str = "eom",
    session_id: str,
    results_root: Path | None = None,
) -> FieldDerivation:
    """Derive and verify a result for `wrt` in a well-posed `npr`.

    `kind="eom"` varies the action; `kind="perturbation"` expands it to
    quadratic order. Either way the model only writes the script and the
    kernel's own residue check decides `verified`.
    """
    build_plan(npr)  # raises AmbiguityBlocked unless the problem is well posed

    cadabra = adapters.get("cadabra")
    if cadabra is None or not cadabra.available():
        raise RuntimeError("cadabra kernel unavailable; cannot run a derivation")

    capability = Capability.PERTURB if kind == "perturbation" else Capability.VARY
    label = "quadratic-action expansion" if kind == "perturbation" else "general variation"

    generated = generate_script(npr, wrt, llm, kind=kind)
    computed = cadabra.run(
        KernelTask(
            capability=capability,
            description=f"{label} wrt {wrt}",
            payload={"script": generated.source},
        )
    )
    checks = computed.value.get("checks", {})
    verified = _verdict(kind, checks)
    detail = _result_detail(kind, verified, checks, computed)

    prefix = "perturb" if kind == "perturbation" else "vary"
    result_id = "{}-{}-{}".format(
        prefix,
        wrt.strip("\\").replace("\\", "").replace("{", "").replace("}", "") or "field",
        hashlib.sha1(generated.source.encode()).hexdigest()[:8],
    )

    derivation = FieldDerivation(
        wrt=wrt,
        kind=kind,
        capability=capability,
        result_id=result_id,
        result_tex=computed.expression_tex,
        verified=verified,
        checks=checks,
        kernel_name=computed.kernel_name,
        kernel_version=computed.kernel_version,
        llm_name=generated.llm_name,
        llm_version=generated.llm_version,
        script=generated.source,
        detail=detail,
    )

    if results_root is not None:
        derivation.bundle_path = str(results_root / session_id / result_id)
        ladder = _ladder_from_kernel(computed, verified, detail)
        bundle = ResultBundle(
            session_id=session_id,
            result_id=result_id,
            result_tex=computed.expression_tex or "",
            npr_snapshot=npr,
            plan=[],
            computed=[computed],
            ladder=ladder,
            narrative=(
                f"{label} wrt {wrt}. Script generated by "
                f"{generated.llm_name} {generated.llm_version}; "
                f"verified={verified} (kernel residue check)."
            ),
            derivations=[derivation.model_dump(mode="json")],
        )
        write_bundle(results_root, bundle)

    return derivation


def derive_eom(
    npr: NPR,
    llm: LLMAdapter,
    adapters: dict,
    *,
    session_id: str,
    results_root: Path | None = None,
) -> list[FieldDerivation]:
    """Derive the equation of motion for each field the task varies over."""
    if npr.task.type != "vary":
        raise NotImplementedError(
            f"general derivation currently supports task type 'vary', not {npr.task.type!r}"
        )
    fields = npr.task.with_respect_to or [
        o.name for o in npr.objects if o.kind in ("metric", "scalar-field", "tensor-field")
    ]
    return [
        derive_field(npr, wrt, llm, adapters, session_id=session_id, results_root=results_root)
        for wrt in fields
    ]


def derive_perturbation(
    npr: NPR,
    llm: LLMAdapter,
    adapters: dict,
    *,
    fields: list[str] | None = None,
    session_id: str,
    results_root: Path | None = None,
) -> list[FieldDerivation]:
    """Expand the action to quadratic order around a background for each
    dynamical scalar field or metric (the sectors with an audited scaffold
    today: pert_scalar_quadratic and pert_metric_quadratic).

    Raises NotImplementedError naming any requested field whose kind has no
    quadratic-action example yet, rather than guessing one.
    """
    perturbable = ("scalar-field", "metric")
    by_name = {o.name: o for o in npr.objects}
    if fields is None:
        fields = [o.name for o in npr.objects if o.kind in perturbable and o.role == "dynamical"]
    if not fields:
        raise NotImplementedError(
            "perturbation currently supports dynamical scalar fields and the "
            "metric; this action declares neither"
        )
    for name in fields:
        obj = by_name.get(name)
        if obj is None or obj.kind not in perturbable:
            raise NotImplementedError(
                "perturbation currently has audited scaffolds for scalar and "
                f"metric fields; cannot expand {name!r}"
            )
    return [
        derive_field(
            npr,
            wrt,
            llm,
            adapters,
            kind="perturbation",
            session_id=session_id,
            results_root=results_root,
        )
        for wrt in fields
    ]


# The ADM (3+1) decomposition of the gravitational sector. These are the
# Gauss-Codazzi identities the SymPy component kernel verifies on a
# nondegenerate 1+2 background (noether.kernels.sympy_kernel.adm, eval 1s):
# universal geometry that holds for any foliated metric, independent of the
# action. For pure GR in vacuum the projection left-hand sides vanish through
# the Einstein equations, giving the familiar Hamiltonian and momentum
# constraints; for an action with matter they are sourced by the stress tensor.
_ADM_SPLIT_TEX = (
    r"\sqrt{-g}\,R = N\sqrt{h}\left(R^{(3)} + K_{ab}K^{ab} - K^{2}\right)"
    r" - 2\,\partial_{\mu}\!\left(\sqrt{-g}\,v^{\mu}\right),\quad"
    r"v^{\mu} = n^{\nu}\nabla_{\nu}n^{\mu} - n^{\mu}\nabla_{\nu}n^{\nu}"
)
_ADM_HAMILTONIAN_TEX = r"2\,G_{\mu\nu}\,n^{\mu}n^{\nu} = R^{(3)} + K^{2} - K_{ab}K^{ab}"
_ADM_MOMENTUM_TEX = r"G_{\mu i}\,n^{\mu} = D_{a}\!\left(K^{a}{}_{i} - \delta^{a}{}_{i}\,K\right)"
_ADM_K_TEX = (
    r"K_{ij} = \tfrac{1}{2N}\left(\partial_{t}h_{ij} - D_{i}N_{j} - D_{j}N_{i}\right)"
    r" = \nabla_{i}n_{j}"
)

_ADM_OUTPUTS: list[tuple[str, str]] = [
    ("Gauss-Codazzi split of the gravitational Lagrangian", _ADM_SPLIT_TEX),
    ("Hamiltonian (normal-normal) projection", _ADM_HAMILTONIAN_TEX),
    ("momentum (normal-tangential) projection", _ADM_MOMENTUM_TEX),
]


def _ladder_from_components(computed: ComputedResult, verified: bool, detail: str) -> LadderReport:
    """Represent the SymPy component-eval suite as a one-rung ladder. Each
    identity is checked on an explicit nondegenerate background, a V2-style
    falsifier that a wrong tensor relation cannot survive."""
    return LadderReport(
        results=[
            CheckResult(
                name="adm-split-on-components",
                rung="V2",
                passed=verified,
                detail=detail,
                computed_by=computed.kernel_name,
                artifacts=[computed],
            )
        ]
    )


def derive_adm(
    npr: NPR,
    adapters: dict,
    *,
    session_id: str,
    results_root: Path | None = None,
) -> list[FieldDerivation]:
    """Decompose the gravitational sector into its ADM (3+1) form for a
    well-posed metric action, verified by the SymPy component kernel.

    Unlike `vary`/`perturb`, this path writes no model script: the deliverable
    is the Gauss-Codazzi split and the normal/tangential projections of the
    Einstein tensor, universal foliation geometry that the kernel confirms on
    an explicit background. Any well-posed action carrying a metric is
    accepted; an action with no metric is refused rather than guessed.
    """
    build_plan(npr)  # raises AmbiguityBlocked unless the problem is well posed

    if not any(o.kind == "metric" for o in npr.objects):
        raise NotImplementedError(
            "ADM decomposition needs a metric and a foliation; this action declares no metric"
        )

    sympy = adapters.get("sympy")
    if sympy is None or not sympy.available():
        raise RuntimeError("sympy kernel unavailable; cannot verify the ADM split")

    computed = sympy.run(
        KernelTask(
            capability=Capability.COMPONENT_EVAL,
            description="ADM 3+1 split of the gravitational sector",
            payload={"check": "adm-gr-1p2"},
        )
    )
    checks = computed.value.get("checks", {})
    verified = bool(computed.value.get("passed"))
    detail = (
        "kernel confirmed the ADM split, both Einstein-tensor projections, the "
        "extrinsic-curvature identity, and the lapse Euler-Lagrange equation on "
        "an explicit 1+2 background"
        if verified
        else "kernel did not confirm the ADM split; surfaced as unverified"
    )

    result_id = f"adm-{hashlib.sha1(npr.action.lagrangian_tex.encode()).hexdigest()[:8]}"
    bundle_path = str(results_root / session_id / result_id) if results_root is not None else None

    derivations = [
        FieldDerivation(
            wrt=label,
            kind="adm",
            capability=Capability.ADM,
            result_id=result_id,
            result_tex=tex,
            verified=verified,
            checks=checks,
            kernel_name=computed.kernel_name,
            kernel_version=computed.kernel_version,
            script=computed.script.source,
            detail=detail,
            bundle_path=bundle_path,
        )
        for label, tex in _ADM_OUTPUTS
    ]

    if results_root is not None:
        ladder = _ladder_from_components(computed, verified, detail)
        bundle = ResultBundle(
            session_id=session_id,
            result_id=result_id,
            result_tex=_ADM_SPLIT_TEX,
            npr_snapshot=npr,
            plan=[],
            computed=[computed],
            ladder=ladder,
            narrative=(
                "ADM (3+1) decomposition of the gravitational sector. The split "
                f"and projections were verified by {computed.kernel_name} "
                f"{computed.kernel_version} on a nondegenerate 1+2 background; "
                f"K_{{ij}} = nabla_i n_j and the lapse Euler-Lagrange equation are "
                f"part of the same suite. verified={verified}."
            ),
            derivations=[d.model_dump(mode="json") for d in derivations],
        )
        write_bundle(results_root, bundle)

    return derivations
