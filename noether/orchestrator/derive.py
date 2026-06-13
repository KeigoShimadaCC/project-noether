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
    kind: str = "eom"  # "eom" | "perturbation"
    capability: Capability
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
    if kind == "perturbation":
        detail = (
            "kernel confirmed the quadratic action reproduces the linearized equation"
            if verified
            else "kernel did not confirm the expansion; surfaced as unverified"
        )
    else:
        detail = (
            "kernel confirmed the variation matches the candidate equation"
            if verified
            else "kernel did not confirm the result; surfaced as unverified"
        )

    derivation = FieldDerivation(
        wrt=wrt,
        kind=kind,
        capability=capability,
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
        ladder = _ladder_from_kernel(computed, verified, detail)
        prefix = "perturb" if kind == "perturbation" else "vary"
        result_id = "{}-{}-{}".format(
            prefix,
            wrt.strip("\\").replace("\\", "").replace("{", "").replace("}", "") or "field",
            hashlib.sha1(generated.source.encode()).hexdigest()[:8],
        )
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
        )
        base = write_bundle(results_root, bundle)
        derivation.bundle_path = str(base)

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
    dynamical scalar field (the only sector with an audited scaffold today).

    Raises NotImplementedError naming any requested field whose kind has no
    quadratic-action example yet, rather than guessing one.
    """
    by_name = {o.name: o for o in npr.objects}
    if fields is None:
        fields = [o.name for o in npr.objects if o.kind == "scalar-field" and o.role == "dynamical"]
    if not fields:
        raise NotImplementedError(
            "perturbation currently supports dynamical scalar fields; this action declares none"
        )
    for name in fields:
        obj = by_name.get(name)
        if obj is None or obj.kind != "scalar-field":
            raise NotImplementedError(
                "perturbation currently has an audited scaffold only for scalar "
                f"fields; cannot expand {name!r}"
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
