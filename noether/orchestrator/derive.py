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
from noether.kernels.cadabra.blocks import (
    Decomposition,
    assemble_metric_eom_script,
    assemble_scalar_eom_script,
    block_summary,
    compose_display_tex,
    compose_metric_display_tex,
    decompose_metric,
    decompose_scalar,
    has_g4g5_terms,
)
from noether.kernels.cadabra.generate import generate_script
from noether.kernels.cadabra.horndeski_g4g5 import (
    SORTCOVDS_BLOCKER,
    assemble_g4_metric_eom_script,
    assemble_g4_scalar_eom_script,
)
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
    teaching: str = ""  # teaching narration (reasoned, not verified; distinct from detail)
    # The active convention block at derivation time. Explicit and named;
    # never silently assumed. Populated for ADM derivations and threaded
    # through the results payload so the consumer can see which conventions
    # produced the result.
    conventions: dict[str, str] = Field(default_factory=dict)


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


def _compositional_detail(
    verified: bool, dec: Decomposition, checks: dict[str, str], computed: ComputedResult
) -> str:
    """Explain a compositional verdict, naming the blocks that composed."""
    blocks = ", ".join(block_summary(dec.matches))
    if verified:
        return (
            "kernel verified the assembled action against the candidate built "
            f"from its blocks ({blocks})"
        )
    if "residue_zero" not in checks:
        rc = computed.raw.returncode
        tail = _stderr_tail(computed.raw.stderr)
        base = (
            f"unverified: the assembled script ({blocks}) produced no residue "
            f"check; it did not run to completion (kernel exit {rc})"
        )
        return f"{base}: {tail}" if tail else base
    return f"unverified: the kernel computed a nonzero residue for the assembled action ({blocks})"


def _is_palatini_eh_connection_variation(npr: NPR, wrt: str) -> bool:
    """True when the derivation is a connection variation of the pure
    Palatini Einstein-Hilbert action (g^{mu nu} R_{mu nu}(Gamma) with no
    other matter fields). In this case, the verified projective-family
    result from the eval2_palatini_connection template is available, so
    we route directly to it rather than the unverified general LLM path.

    The detection checks:
      1. wrt names a connection object;
      2. the connection type is independent;
      3. there are no dynamical scalar, vector, or tensor matter fields
         (only the metric g, the connection Gamma, and curvature shorthands);
      4. the Lagrangian involves curvature (R).

    This is intentionally narrow: Palatini scalar-tensor F(phi)R(Gamma),
    Einstein-Cartan with spin sources, and other metric-affine theories
    have different physics and must still route through the general path
    (or their own templates) rather than being silently routed to the
    pure-EH result.
    """
    by_name = {o.name: o for o in npr.objects}
    wrt_obj = by_name.get(wrt)
    if wrt_obj is None or wrt_obj.kind != "connection":
        return False
    if npr.geometry.connection.type != "independent":
        return False
    # No dynamical matter fields other than g and Gamma.
    _matter_kinds = {"scalar-field", "tensor-field"}
    matter_objects = [
        o for o in npr.objects if o.kind in _matter_kinds and o.role == "dynamical"
    ]
    if matter_objects:
        return False
    # No coupling functions that would make this non-pure-EH.
    coupling_objects = [o for o in npr.objects if o.kind == "function"]
    if coupling_objects:
        return False
    # The Lagrangian involves curvature (R or similar shorthand).
    has_curvature = any(o.name in ("R", "G", "C", "W") for o in npr.objects)
    if not has_curvature:
        return False
    return True


_PROJECTIVE_FAMILY_TEX = (
    r"\Gamma^{\lambda}_{\mu\nu}"
    r" = \{^{\lambda}_{\mu\nu}\}_g"
    r" + \delta^{\lambda}_{\nu} A_{\mu}"
)

_PROJECTIVE_FREEDOM_DETAIL = (
    "The connection is determined only up to the projective mode: "
    r"\Gamma = \mathrm{LC}(g) + \delta^{\lambda}_{\nu} A_{\mu}"
    r" with A_{\mu} arbitrary. The projective family annihilates the "
    "connection equation (solution_zero=True) and the Ricci shift "
    r"is exactly dA (ricci_shift_is_dA=True), so the metric equation "
    r"reduces to G_{\mu\nu}(g)=0. The connection is never uniquely fixed."
)


def _compositional_decomposition(npr: NPR, wrt: str, kind: str) -> Decomposition | None:
    """Decompose the Lagrangian into building blocks, when the compositional
    path applies: an EOM for a dynamical scalar field rendered as phi, or the
    metric EOM of an action whose scalar (if any) is rendered as phi. Returns
    None when the path does not apply; a partial Decomposition (``full`` False)
    when some term matches no registered block.

    Connection fields are NOT handled compositionally (there are no registered
    building blocks for the connection equation yet); they route to the
    model-written script path with the connection-variation worked example
    (generate._variation_key -> vary-connection).

    The assembled scripts render the scalar as `phi`, so a scalar field under
    any other name routes to the model path rather than being mislabeled."""
    if kind != "eom":
        return None
    obj = next((o for o in npr.objects if o.name == wrt), None)
    if obj is None:
        return None
    # Connection variation has no compositional blocks; it always routes to
    # the model-written script path (with the vary-connection worked example).
    if obj.kind == "connection":
        return None
    if obj.kind == "scalar-field" and wrt == "phi":
        return decompose_scalar(npr.action.lagrangian, wrt)
    if obj.kind == "metric":
        scalars = [o.name for o in npr.objects if o.kind == "scalar-field"]
        if any(s != "phi" for s in scalars):
            return None
        return decompose_metric(npr.action.lagrangian, "phi")
    return None


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


def attempt_g4g5_eom(
    cadabra_adapter,
    npr: NPR,
    *,
    session_id: str = "",
    results_root: Path | None = None,
) -> list[FieldDerivation]:
    """Attempt the held G4(phi,X)R / G5 Horndeski EOM as best-effort,
    producing ``FieldDerivation`` objects for the scalar and metric equations.

    This runs the hand-audited Cadabra scripts for the scalar and metric EOM
    variations and constructs ``FieldDerivation`` objects from the results.
    The result is always honest: either fully verified (residue 0 on both
    EOMs, ``verified=True``) or gated (``verified=False`` with a non-empty
    ``detail`` naming the blocker). It is never ``verified=True`` with a gate
    unmet, satisfying VAL-EOM-013.

    The scalar EOM script checks that no third derivatives of phi survive the
    IBP (``scalar_eom_second_order``). The metric EOM script checks that
    third derivatives are present after expanding wrapped terms
    (``metric_eom_has_third_derivs``), confirming the SortCovDs blocker is
    real. Neither script produces a ``residue_zero`` check because there is
    no target equation to compare against; the full closure requires both the
    scalar and metric EOMs to residue-check, and the metric EOM cannot close
    without normal-ordering.

    When ``results_root`` is not None, a provenance bundle is written for each
    derivation (one per result_id), mirroring the general path's persistence
    block so the gated result reloads via GET /sessions/{id}/results, MCP
    noether_results, and derivations.json (VAL-CROSS-021).

    Returns:
        List of two ``FieldDerivation`` objects: one for the scalar EOM
        (wrt="phi") and one for the metric EOM (wrt="g").

    The result satisfies the XOR condition from VAL-EOM-013:
        (verified and residue_zero=="True")
        XOR
        (not verified and detail != "")
    """
    # Run the scalar EOM script.
    scalar_script = assemble_g4_scalar_eom_script()
    scalar_computed = cadabra_adapter.run(
        KernelTask(
            capability=Capability.SUBSTITUTE,
            description="G4 scalar EOM best-effort",
            payload={"script": scalar_script},
        )
    )
    scalar_checks = scalar_computed.value.get("checks", {})

    # Run the metric EOM script.
    metric_script = assemble_g4_metric_eom_script()
    metric_computed = cadabra_adapter.run(
        KernelTask(
            capability=Capability.SUBSTITUTE,
            description="G4 metric EOM best-effort",
            payload={"script": metric_script},
        )
    )
    metric_checks = metric_computed.value.get("checks", {})

    # Both EOMs must residue-check for the closure to be verified.
    scalar_residue_zero = scalar_checks.get("residue_zero") == "True"
    metric_residue_zero = metric_checks.get("residue_zero") == "True"
    both_verified = scalar_residue_zero and metric_residue_zero

    # Construct the detail: always non-empty so a gated result is
    # distinguishable from a verified one by both verified and detail
    # (VAL-GUIDE-012/013).
    detail = (
        "kernel verified the G4 scalar and metric EOMs (both residue checks passed)"
        if both_verified
        else SORTCOVDS_BLOCKER
    )

    # Build FieldDerivation objects. Even the gated G4/G5 result carries its
    # named convention block (conventions are always explicit; every other
    # derivation path already populates this via _convention_block).
    conv_block = _convention_block(npr)
    scalar_derivation = FieldDerivation(
        wrt="phi",
        kind="eom",
        capability=Capability.VARY,
        result_id=(
            f"g4g5-scalar-{hashlib.sha1(scalar_script.encode()).hexdigest()[:8]}"
        ),
        result_tex=scalar_computed.expression_tex,
        verified=both_verified,
        checks=scalar_checks,
        kernel_name=scalar_computed.kernel_name,
        kernel_version=scalar_computed.kernel_version,
        script=scalar_script,
        detail=detail,
        conventions=conv_block,
    )

    metric_derivation = FieldDerivation(
        wrt="g",
        kind="eom",
        capability=Capability.VARY,
        result_id=(
            f"g4g5-metric-{hashlib.sha1(metric_script.encode()).hexdigest()[:8]}"
        ),
        result_tex=metric_computed.expression_tex,
        verified=both_verified,
        checks=metric_checks,
        kernel_name=metric_computed.kernel_name,
        kernel_version=metric_computed.kernel_version,
        script=metric_script,
        detail=detail,
        conventions=conv_block,
    )

    # Persist each derivation's provenance bundle, mirroring the general
    # path's shared persistence block.  This is the fix for VAL-CROSS-021:
    # gated G4/G5 best-effort EOM derivations must be persisted so they
    # reload via GET /sessions/{id}/results, MCP noether_results, and the
    # provenance bundle derivations.json.  Without this, the g4g5 branch
    # in derive_field early-returns before the shared write_bundle section,
    # and _record (server/app.py, mcp/server.py) already records the id
    # but read_results skips results whose bundle is missing.
    if results_root is not None:
        for derivation, computed in [
            (scalar_derivation, scalar_computed),
            (metric_derivation, metric_computed),
        ]:
            derivation.bundle_path = str(
                results_root / session_id / derivation.result_id
            )
            ladder = _ladder_from_kernel(computed, both_verified, detail)
            narrative = (
                f"best-effort G4/G5 EOM wrt {derivation.wrt}. "
                f"Script from hand-audited horndeski_g4g5; "
                f"verified={both_verified} "
                f"(kernel diagnostic checks: "
                f"{', '.join(f'{k}={v}' for k, v in derivation.checks.items())}). "
                f"{detail}"
            )
            bundle = ResultBundle(
                session_id=session_id,
                result_id=derivation.result_id,
                result_tex=derivation.result_tex or "",
                npr_snapshot=npr,
                plan=[],
                computed=[computed],
                ladder=ladder,
                narrative=narrative,
                derivations=[derivation.model_dump(mode="json")],
            )
            write_bundle(results_root, bundle)

    return [scalar_derivation, metric_derivation]


def _convention_block(npr: NPR) -> dict[str, str]:
    """Extract the load-bearing convention entries from the NPR for inclusion
    in a FieldDerivation's convention block. Every entry is named and explicit;
    nothing is silently assumed."""
    c = npr.conventions
    block: dict[str, str] = {
        "signature": c.signature,
        "torsion_sign": c.torsion_sign,
        "nonmetricity_definition": c.nonmetricity_definition,
        "ricci_contraction": c.ricci_contraction,
        "contortion_sign": c.contortion_sign,
        "disformation_sign": c.disformation_sign,
        "convention_id": c.id,
    }
    # Foliation/normal convention: the sign of the extrinsic curvature
    # and the normal direction. These are convention fields on the NPR,
    # not hardcoded for any signature.
    if c.foliation_normal == "future-directed":
        if c.signature == "mostly-plus":
            block["foliation_normal"] = "n_mu=(-N,0,...,0) timelike"
        else:
            block["foliation_normal"] = "n_mu=(+N,0,...,0) timelike"
    else:  # past-directed
        if c.signature == "mostly-plus":
            block["foliation_normal"] = "n_mu=(+N,0,...,0) timelike"
        else:
            block["foliation_normal"] = "n_mu=(-N,0,...,0) timelike"
    if c.K_sign == "+1":
        block["K_sign"] = "+1 (K_{ij}=+nabla_i n_j expansion-positive)"
    else:
        block["K_sign"] = "-1 (K_{ij}=-nabla_i n_j expansion-negative)"
    # For metric-affine NPRs, also surface the field-strength definition
    # since it affects the connection-sector constraints.
    if npr.geometry.connection.type == "independent":
        block["field_strength_definition"] = c.field_strength_definition
    return block


def _geometry_teaching(npr: NPR, wrt: str, kind: str) -> str:
    """Generate teaching narration for a derivation on a metric-affine NPR.

    The teaching contrasts the physical consequences of the geometric
    choices the user made (torsion -> spin coupling, non-metricity ->
    length non-conservation, projective freedom), not a bare restatement
    of the menu. It remains on the teaching channel and resolves nothing.
    It mutates no NPR and sets no result expression.

    For Levi-Civita NPRs (the default), the teaching is empty: there are
    no geometric tradeoffs to narrate. For metric-affine NPRs with an
    independent connection, the teaching explains what the geometry means
    for the derivation at hand.
    """
    conn = npr.geometry.connection
    if conn.type != "independent":
        return ""

    parts: list[str] = []
    has_torsion = conn.torsion
    has_nonmetricity = conn.nonmetricity

    if kind == "eom":
        if wrt and any(
            o.name == wrt and o.kind == "connection" for o in npr.objects
        ):
            # Connection variation teaching
            parts.append(
                "The independent connection equation is algebraic in "
                "the distortion tensors: it constrains the connection "
                "without time derivatives, so the connection carries no "
                "independent propagating degrees of freedom."
            )
            if has_torsion:
                parts.append(
                    "With torsion allowed, the contortion K(T) couples "
                    "to the spin current of matter fields. A nonzero "
                    "spin density sources torsion, so the geometry "
                    "responds to intrinsic angular momentum rather than "
                    "just energy-momentum."
                )
            if has_nonmetricity:
                parts.append(
                    "With non-metricity allowed, the disformation L(Q) "
                    "means the covariant derivative of the metric is no "
                    "longer zero: parallel transport does not preserve "
                    "vector length. This couples to the dilation and "
                    "shear currents of matter (the dilation trace of the "
                    "hypermomentum)."
                )
            parts.append(
                "The projective freedom (Gamma -> Gamma + delta^lambda_nu "
                "A_mu for arbitrary A_mu) is a gauge redundancy of the "
                "connection equation for pure Palatini Einstein-Hilbert "
                "gravity: the connection is determined only up to this "
                "family, never uniquely fixed."
            )
        else:
            # Metric variation teaching on a metric-affine background
            parts.append(
                "The metric equation varies the action with respect to "
                "g while treating the connection as independent: curvature "
                "is not varied with the metric, and the resulting field "
                "equation involves the symmetric part of the Ricci tensor."
            )
            if has_torsion:
                parts.append(
                    "Torsion introduces a spin-current coupling: the "
                    "antisymmetric part of the affine connection allows "
                    "matter with intrinsic spin to source torsion "
                    "nonlinearly, modifying the effective stress-energy "
                    "felt by the metric."
                )
            if has_nonmetricity:
                parts.append(
                    "Non-metricity means length is not conserved under "
                    "parallel transport: the covariant derivative of the "
                    "metric is Q_{lambda mu nu} rather than zero. This "
                    "introduces dilation and shear currents that modify "
                    "the metric equation beyond the standard Einstein form."
                )
    elif kind == "perturbation":
        if has_torsion or has_nonmetricity:
            torsion_text = (
                "torsion fluctuations contribute to the connection "
                "perturbation dG"
                if has_torsion
                else ""
            )
            nonmetricity_text = (
                "non-metricity fluctuations contribute to dG "
                "alongside the metric fluctuation h"
                if has_nonmetricity
                else ""
            )
            fragments = [t for t in [torsion_text, nonmetricity_text] if t]
            parts.append(
                "On a metric-affine background the quadratic action "
                "retains the connection fluctuation dG alongside the "
                "metric fluctuation h. "
                + " and ".join(fragments)
                + ". These cross-terms between h and dG are "
                "characteristic of the metric-affine perturbation "
                "structure and have no Levi-Civita analogue."
            )
    elif kind == "adm":
        if has_torsion:
            parts.append(
                "In the ADM decomposition, torsion projects into "
                "spatial (T^i_{jk}), normal-upper (T^n_{jk}), and "
                "mixed (T^i_{nk}) pieces. The contortion K(T) enters "
                "the connection constraint structure, and a nonzero "
                "spin current sources primary torsion constraints."
            )
        if has_nonmetricity:
            parts.append(
                "Non-metricity in the ADM split produces spatial "
                "(Q_{ijk}), normal-first (Q_{nij}), and mixed "
                "(Q_{inj}) pieces. The disformation L(Q) introduces "
                "additional structure that makes the Dirac constraint "
                "chain harder to close, requiring action-specific "
                "analysis."
            )

    return " ".join(parts) if parts else ""


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

    # When varying an independent connection, the capability is
    # INDEPENDENT_CONNECTION (matching the planner step), not the generic VARY.
    # A connection field must never be silently routed to the metric worked
    # example (VAL-EOM-006).
    by_name = {o.name: o for o in npr.objects}
    wrt_obj = by_name.get(wrt)
    if kind == "perturbation":
        capability = Capability.PERTURB
    elif wrt_obj is not None and wrt_obj.kind == "connection":
        capability = Capability.INDEPENDENT_CONNECTION
    else:
        capability = Capability.VARY
    label = "quadratic-action expansion" if kind == "perturbation" else "general variation"

    # Compositional path: when the scalar Lagrangian fully decomposes into
    # registered building blocks, assemble one Cadabra script for the user's
    # actual action and let the kernel residue-check it. No model is involved,
    # and the result renders in collapsed shorthand. Otherwise fall back to the
    # model-written script path. A partial decomposition is left to the model
    # rather than guessing at the unmatched term.
    dec = _compositional_decomposition(npr, wrt, kind)
    if dec is not None and dec.full:
        if dec.field == "g":
            script = assemble_metric_eom_script(dec.matches)
            display_tex = compose_metric_display_tex(dec.matches)
        else:
            script = assemble_scalar_eom_script(dec.matches, wrt)
            display_tex = compose_display_tex(dec.matches, wrt)
        computed = cadabra.run(
            KernelTask(
                capability=capability,
                description=f"compositional variation wrt {wrt}",
                payload={"script": script},
            )
        )
        checks = computed.value.get("checks", {})
        verified = checks.get("residue_zero") == "True"
        detail = _compositional_detail(verified, dec, checks, computed)
        result_tex = display_tex if verified else computed.expression_tex
        source = script
        llm_name, llm_version = "compositional", "blocks-v1"
    elif (
        kind == "eom"
        and _is_palatini_eh_connection_variation(npr, wrt)
    ):
        # Palatini Einstein-Hilbert connection variation: route to the
        # verified projective-family result from the eval2_palatini_connection
        # template (VAL-EOM-004). The general LLM-written script path
        # produces verified=false with a nonzero residue; the template
        # carries the genuine solution_zero and ricci_shift_is_dA checks.
        # Do NOT fabricate verification: the surfaced result must carry the
        # genuine checks from the template run.
        from noether.kernels.cadabra import templates as _templates

        template_name = "eval2_palatini_connection"
        source = _templates.get(template_name)
        computed = cadabra.run(
            KernelTask(
                capability=Capability.INDEPENDENT_CONNECTION,
                description="Palatini EH connection variation (template)",
                payload={"template": template_name},
            )
        )
        checks = computed.value.get("checks", {})
        solution_zero = checks.get("solution_zero") == "True"
        ricci_shift = checks.get("ricci_shift_is_dA") == "True"
        verified = solution_zero and ricci_shift
        if verified:
            detail = _PROJECTIVE_FREEDOM_DETAIL
            result_tex = _PROJECTIVE_FAMILY_TEX
        else:
            # The template checks failed (should not happen on a correct
            # kernel, but be honest about it).
            detail = (
                "unverified: the Palatini connection template checks did "
                f"not both pass (solution_zero={checks.get('solution_zero')}, "
                f"ricci_shift_is_dA={checks.get('ricci_shift_is_dA')})"
            )
            result_tex = computed.expression_tex
        llm_name, llm_version = "template", template_name
    elif (
        kind == "eom"
        and wrt in ("g", "phi")
        and has_g4g5_terms(npr.action.lagrangian, "phi")
    ):
        # Best-effort G4(phi,X)R / G5 Horndeski path (VAL-EOM-013).
        # The compositional decomposition cannot match these terms (they
        # need normal-ordering to verify), so the model-written path would
        # also fail to produce a verified result. Instead, use the
        # hand-audited G4/G5 scripts from horndeski_g4g5.py, which run the
        # diagnostic checks and return an honest gated result.
        derivations = attempt_g4g5_eom(
            cadabra,
            npr=npr,
            session_id=session_id,
            results_root=results_root,
        )
        by_wrt = {d.wrt: d for d in derivations}
        if wrt in by_wrt:
            return by_wrt[wrt]
        # wrt is not in the G4/G5 results (e.g. a connection field);
        # fall through to the model-written script path.
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
        result_tex = computed.expression_tex
        source = generated.source
        llm_name, llm_version = generated.llm_name, generated.llm_version
    else:
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
        result_tex = computed.expression_tex
        source = generated.source
        llm_name, llm_version = generated.llm_name, generated.llm_version

    prefix = "perturb" if kind == "perturbation" else "vary"
    result_id = "{}-{}-{}".format(
        prefix,
        wrt.strip("\\").replace("\\", "").replace("{", "").replace("}", "") or "field",
        hashlib.sha1(source.encode()).hexdigest()[:8],
    )

    derivation = FieldDerivation(
        wrt=wrt,
        kind=kind,
        capability=capability,
        result_id=result_id,
        result_tex=result_tex,
        verified=verified,
        checks=checks,
        kernel_name=computed.kernel_name,
        kernel_version=computed.kernel_version,
        llm_name=llm_name,
        llm_version=llm_version,
        script=source,
        detail=detail,
        teaching=_geometry_teaching(npr, wrt, kind),
        conventions=_convention_block(npr),
    )

    if results_root is not None:
        derivation.bundle_path = str(results_root / session_id / result_id)
        ladder = _ladder_from_kernel(computed, verified, detail)
        if llm_name == "compositional":
            blocks = ", ".join(block_summary(dec.matches)) if dec is not None else ""
            narrative = (
                f"{label} wrt {wrt}. Assembled compositionally from building "
                f"blocks ({blocks}) and residue-checked by the kernel; "
                f"verified={verified}."
            )
        elif llm_name == "template":
            narrative = (
                f"{label} wrt {wrt}. Run from frozen template "
                f"{llm_version}; verified={verified} "
                f"(kernel checks: {', '.join(f'{k}={v}' for k, v in checks.items())}). "
                f"{detail}"
            )
        else:
            narrative = (
                f"{label} wrt {wrt}. Script generated by {llm_name} "
                f"{llm_version}; verified={verified} (kernel residue check)."
            )
        bundle = ResultBundle(
            session_id=session_id,
            result_id=result_id,
            result_tex=result_tex or "",
            npr_snapshot=npr,
            plan=[],
            computed=[computed],
            ladder=ladder,
            narrative=narrative,
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
    # When the connection is independent, include connection objects in the
    # default field list so derive_eom covers the connection equation too.
    default_kinds: set[str] = {"metric", "scalar-field", "tensor-field"}
    if npr.geometry.connection.type == "independent":
        default_kinds.add("connection")
    fields = npr.task.with_respect_to or [
        o.name for o in npr.objects if o.kind in default_kinds
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
    dynamical scalar field, metric, or gauge potential (the sectors with an
    audited scaffold today: pert_scalar_quadratic, pert_kessence_quadratic for
    an X-dependent scalar, pert_metric_quadratic, and the gauge scaffolds
    pert_gauge_quadratic (Maxwell) / pert_yang_mills_quadratic), plus the
    metric-affine scaffold pert_metric_affine_quadratic when the connection
    is independent.

    Raises NotImplementedError naming any requested field whose kind has no
    quadratic-action example yet, rather than guessing one.
    """
    by_name = {o.name: o for o in npr.objects}
    has_independent_connection = (
        getattr(npr.geometry.connection, "type", None) == "independent"
    )

    def _supported(o) -> bool:
        # a gauge potential is rank 1; the field strength (rank 2) is not the
        # perturbed degree of freedom, so it falls through to the refusal.
        if o.kind in ("scalar-field", "metric"):
            return True
        # On a metric-affine background the metric perturbation scaffold
        # (pert_metric_affine_quadratic) includes the connection fluctuation
        # dG automatically alongside the metric fluctuation h. The connection
        # object itself is NOT perturbed independently; there is no separate
        # connection perturbation scaffold.
        if o.kind == "tensor-field" and o.rank == 1:
            return True
        return False

    if fields is None:
        fields = [o.name for o in npr.objects if _supported(o) and o.role == "dynamical"]
        # On a metric-affine background, always include the metric
        # perturbation which carries the connection fluctuation.
        if has_independent_connection and "g" not in fields:
            g_obj = next((o for o in npr.objects if o.kind == "metric"), None)
            if g_obj and g_obj.name not in fields:
                fields.append(g_obj.name)
    if not fields:
        raise NotImplementedError(
            "perturbation currently supports dynamical scalar fields, the metric, "
            "and rank-1 gauge potentials; this action declares none"
        )
    for name in fields:
        obj = by_name.get(name)
        if obj is None or not _supported(obj):
            raise NotImplementedError(
                "perturbation currently has audited scaffolds for scalar fields, "
                "the metric, and rank-1 gauge potentials (Maxwell / Yang-Mills); "
                f"cannot expand {name!r}"
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

# Metric-affine ADM: connection foliation decomposition and constraints.
_ADM_CONNECTION_FOLIATION_TEX = (
    r"\Gamma^{\lambda}_{\mu\nu}"
    r" = \{^{\lambda}_{\mu\nu}\}_g"
    r" + K^{\lambda}_{\mu\nu}(T)"
    r" + L^{\lambda}_{\mu\nu}(Q)"
)
_ADM_TORSION_FOLIATION_TEX = (
    r"T^{\lambda}_{\mu\nu}:\;"
    r"T^{i}_{jk}\;\text{(spatial)},\;"
    r"T^{n}_{jk}\;\text{(normal-upper)},\;"
    r"T^{i}_{nk}\;\text{(mixed)}"
)
_ADM_NONMETRICITY_FOLIATION_TEX = (
    r"Q_{\lambda\mu\nu}:\;"
    r"Q_{ijk}\;\text{(spatial)},\;"
    r"Q_{nij}\;\text{(normal-first)},\;"
    r"Q_{inj}\;\text{(mixed)}"
)
_ADM_K_SIGN_TEX = (
    r"K_{ij} = +\nabla_i n_j\;\text{(expansion-positive, }n_\mu = (-N,0,\ldots,0)\text{)}"
)
_ADM_CONNECTION_CONSTRAINTS_TEX = (
    r"\text{Primary: }\delta S/\delta\Gamma\;\text{algebraic}"
    r"\;\Rightarrow\;\Gamma\;\text{non-dynamical}"
    r";\;\text{Secondary: Dirac chain gated}"
)
_ADM_CONNECTION_CONSTRAINTS_Q_TEX = (
    r"\text{Primary: }\delta S/\delta\Gamma\;\text{involves }K(T),L(Q)"
    r";\;\text{Dirac chain closure requires action-specific analysis}"
)

_ADM_AFFINE_OUTPUTS: list[tuple[str, str, str]] = [
    ("connection foliation decomposition", _ADM_CONNECTION_FOLIATION_TEX, ""),
    ("torsion foliation pieces", _ADM_TORSION_FOLIATION_TEX, ""),
    ("non-metricity foliation pieces", _ADM_NONMETRICITY_FOLIATION_TEX, ""),
    ("extrinsic curvature convention", _ADM_K_SIGN_TEX, ""),
    (
        "connection-sector constraints",
        _ADM_CONNECTION_CONSTRAINTS_TEX,
        (
            "The independent connection Gamma decomposes as LC(g) + K(T) + L(Q) "
            "along the foliation. When the connection EOM is algebraic in the "
            "contortion K (as on a metric-compatible torsionful background), the "
            "connection components carry no time derivatives and generate primary "
            "constraints in the Dirac sense. For pure Palatini EH on a "
            "metric-compatible background the projective gauge freedom generates "
            "first-class constraints, and the Dirac chain closes. When non-metricity "
            "is present, the disformation L(Q) introduces additional structure "
            "requiring action-specific analysis, and the Dirac chain closure is "
            "gated as unverified."
        ),
    ),
]

_ADM_MATTER_HYPERMOMENTUM_TEX = (
    r"\Delta^{\lambda}_{\mu\nu} = \tau^{\lambda}_{\mu\nu}"
    r" + \tfrac{1}{n}\delta^{\lambda}_{\mu}\Delta_{\nu}"
    r" + \sigma^{\lambda}_{\mu\nu}"
    r";\;\text{spin/dilation/shear enter constraint structure}"
)

_ADM_MATTER_HYPERMOMENTUM_NARRATIVE = (
    "When matter couples to the independent connection, the hypermomentum "
    "Delta^lambda_{mu nu} enters the connection-sector constraints as a "
    "source. The spin part tau (antisymmetric, traceless) sources the "
    "torsion primary constraint, the dilation trace Delta_nu sources "
    "the projective constraint, and the shear part sigma (symmetric, "
    "traceless) sources the non-metricity constraint. On a metric-compatible "
    "background (Q=0) the Dirac chain closes; with non-metricity it is gated."
)


def _action_has_hypermomentum(npr: NPR) -> bool:
    """Determine whether the action has matter that couples to the
    independent connection (has nonzero hypermomentum).

    The hypermomentum Delta^lambda_{mu nu} is nonzero when any matter
    field in the action depends on the connection Gamma. Detection:
    - A vector/gauge field with F = covariant curl has hypermomentum
      (Delta = -2 A_lambda F^{mu nu}, nonzero).
    - A gauge field with F = dA has zero hypermomentum.
    - A scalar in F(phi) R(Gamma) Palatini has connection coupling
      through the non-constant F(phi) term (dF sources the connection
      equation), contributing to the disformation/non-metricity source.
    - Pure gravity (only metric and connection) has zero hypermomentum.

    Convention: noether-default-v1 + metric-affine-v1.
    """
    # Check for vector/gauge fields with covariant-curl field-strength.
    # In the NPR schema, a gauge potential is a rank-1 tensor-field.
    field_strength = getattr(
        npr.conventions, "field_strength_definition", "exterior-derivative"
    )
    has_gauge_covcurl = field_strength == "covariant-curl" and any(
        o.kind == "tensor-field" and o.rank == 1 for o in npr.objects
    )

    # Check for scalar fields that could couple through F(phi)R(Gamma)
    has_scalar_coupling = any(
        o.kind == "scalar-field" for o in npr.objects
    )

    # If the connection is independent and there is matter (beyond
    # just the metric and connection), the action may have hypermomentum
    has_only_geometry = all(
        o.kind in ("metric", "connection") for o in npr.objects
    )

    return (
        has_gauge_covcurl
        or (has_scalar_coupling and not has_only_geometry)
    )


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

    For a metric-affine NPR (independent connection with torsion and/or
    non-metricity), the result additionally exposes the connection's own
    degrees of freedom decomposed along the foliation (the Gamma = LC + K(T)
    + L(Q) split projected into normal/tangential parts), surfaces torsion
    and non-metricity pieces explicitly, distinguishes constraint pieces from
    evolution pieces, and identifies connection-sector primary/secondary
    constraints. If the Dirac chain cannot be closed, the constraint piece
    carries verified=false with a detail naming the blocker.
    """
    build_plan(npr)  # raises AmbiguityBlocked unless the problem is well posed

    if not any(o.kind == "metric" for o in npr.objects):
        metric_name = npr.geometry.metric_name
        raise NotImplementedError(
            f"ADM decomposition needs a metric and a foliation; "
            f"this action declares no metric object '{metric_name}'"
        )

    sympy = adapters.get("sympy")
    if sympy is None or not sympy.available():
        raise RuntimeError("sympy kernel unavailable; cannot verify the ADM split")

    has_independent_connection = (
        getattr(npr.geometry.connection, "type", None) == "independent"
    )

    # --- Metric-sector checks (GR ADM, always run) ---
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

    # --- Connection-sector checks (metric-affine ADM) ---
    if has_independent_connection:
        affine_computed = sympy.run(
            KernelTask(
                capability=Capability.COMPONENT_EVAL,
                description="ADM 3+1 metric-affine connection decomposition",
                payload={"check": "adm-affine-1p2"},
            )
        )
        affine_checks = affine_computed.value.get("checks", {})
        affine_verified = bool(affine_computed.value.get("passed"))
        affine_detail = (
            "kernel confirmed the post-Riemannian decomposition on the "
            "foliated background, the torsion and non-metricity foliation "
            "projections, and the distortion spatial projections on an "
            "explicit 1+2 metric-affine background"
            if affine_verified
            else "kernel did not confirm the metric-affine ADM decomposition; "
            "surfaced as unverified"
        )
    else:
        affine_computed = None
        affine_checks = {}
        affine_verified = False
        affine_detail = ""

    result_id = f"adm-{hashlib.sha1(npr.action.lagrangian_tex.encode()).hexdigest()[:8]}"
    bundle_path = str(results_root / session_id / result_id) if results_root is not None else None
    conv_block = _convention_block(npr)

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
            conventions=conv_block,
        )
        for label, tex in _ADM_OUTPUTS
    ]

    # Add metric-affine pieces when the connection is independent.
    if has_independent_connection:
        # Determine whether the Dirac chain can close based on the
        # algebraic nature of the connection EOM.
        has_nonmetricity = npr.geometry.connection.nonmetricity
        # When Q != 0, the Dirac chain cannot be closed in general
        # (the disformation L(Q) introduces additional structure that
        # requires action-specific analysis). Gate the constraint piece.
        dirac_closeable = not has_nonmetricity

        for label, tex, default_teaching in _ADM_AFFINE_OUTPUTS:
            piece_verified = affine_verified
            piece_detail = affine_detail
            piece_checks = affine_checks
            piece_teaching = default_teaching

            # The connection-sector constraints piece carries a more
            # specific verdict depending on the Dirac chain closure.
            if label == "connection-sector constraints":
                if not dirac_closeable:
                    piece_verified = False
                    piece_detail = (
                        "Dirac chain cannot be closed for the general "
                        "metric-affine case with non-metricity (Q != 0): "
                        "the disformation L(Q) introduces additional "
                        "structure that requires action-specific analysis. "
                        "Primary constraints from the algebraic connection "
                        "EOM are identified, but secondary constraints "
                        "and their consistency require further treatment. "
                        "Gated with a stated reason."
                    )
                else:
                    piece_detail = (
                        "On a metric-compatible (Q=0) torsionful background, "
                        "the connection EOM is algebraic in K (no "
                        "derivative-of-K terms). Primary constraints: the "
                        "algebraic EOM constrains all Gamma components "
                        "without time derivatives. Secondary constraints: "
                        "for pure Palatini EH the projective gauge freedom "
                        "generates first-class constraints. "
                        + affine_detail
                    )

            derivations.append(
                FieldDerivation(
                    wrt=label,
                    kind="adm",
                    capability=Capability.ADM,
                    result_id=result_id,
                    result_tex=tex,
                    verified=piece_verified,
                    checks=piece_checks,
                    kernel_name=(
                        affine_computed.kernel_name
                        if affine_computed
                        else computed.kernel_name
                    ),
                    kernel_version=(
                        affine_computed.kernel_version
                        if affine_computed
                        else computed.kernel_version
                    ),
                    script=(
                        affine_computed.script.source
                        if affine_computed
                        else computed.script.source
                    ),
                    detail=piece_detail,
                    bundle_path=bundle_path,
                    teaching=piece_teaching,
                    conventions=conv_block,
                )
            )

        # --- Matter hypermomentum contribution (VAL-ADM-015) ---
        # When the action has matter that couples to the independent
        # connection (has hypermomentum), a constraint piece names the
        # matter contribution. Pure-gravity sessions carry no such piece.
        has_matter_hypermomentum = _action_has_hypermomentum(npr)
        if has_matter_hypermomentum:
            matter_verified = affine_verified
            matter_detail = affine_detail
            matter_checks = affine_checks
            matter_teaching = _ADM_MATTER_HYPERMOMENTUM_NARRATIVE
            if not dirac_closeable:
                matter_verified = False
                matter_detail = (
                    "Matter hypermomentum enters the connection-sector "
                    "constraints as a source (spin/dilation/shear), but "
                    "the Dirac chain cannot be closed for the general "
                    "metric-affine case with non-metricity (Q != 0): "
                    "the disformation L(Q) introduces additional structure "
                    "that requires action-specific analysis. "
                    "Gated with a stated reason."
                )
            else:
                matter_detail = (
                    "Matter hypermomentum enters the connection-sector "
                    "constraints as a source: spin (antisymmetric, "
                    "traceless) sources the torsion primary constraint, "
                    "dilation (trace vector) sources the projective "
                    "constraint, shear (symmetric, traceless) sources "
                    "the non-metricity constraint. "
                    + affine_detail
                )
            derivations.append(
                FieldDerivation(
                    wrt="matter hypermomentum contribution",
                    kind="adm",
                    capability=Capability.ADM,
                    result_id=result_id,
                    result_tex=_ADM_MATTER_HYPERMOMENTUM_TEX,
                    verified=matter_verified,
                    checks=matter_checks,
                    kernel_name=(
                        affine_computed.kernel_name
                        if affine_computed
                        else computed.kernel_name
                    ),
                    kernel_version=(
                        affine_computed.kernel_version
                        if affine_computed
                        else computed.kernel_version
                    ),
                    script=(
                        affine_computed.script.source
                        if affine_computed
                        else computed.script.source
                    ),
                    detail=matter_detail,
                    bundle_path=bundle_path,
                    teaching=matter_teaching,
                    conventions=conv_block,
                )
            )

    if results_root is not None:
        ladder = _ladder_from_components(computed, verified, detail)
        all_computed = [computed]
        if affine_computed is not None:
            all_computed.append(affine_computed)
        narrative = (
            "ADM (3+1) decomposition of the gravitational sector. The split "
            f"and projections were verified by {computed.kernel_name} "
            f"{computed.kernel_version} on a nondegenerate 1+2 background; "
            f"K_{{ij}} = nabla_i n_j and the lapse Euler-Lagrange equation are "
            f"part of the same suite. verified={verified}."
        )
        if has_independent_connection:
            narrative += (
                f" Connection-sector decomposition verified={affine_verified} "
                f"on a metric-affine background with torsion and non-metricity."
            )
        bundle = ResultBundle(
            session_id=session_id,
            result_id=result_id,
            result_tex=_ADM_SPLIT_TEX,
            npr_snapshot=npr,
            plan=[],
            computed=all_computed,
            ladder=ladder,
            narrative=narrative,
            derivations=[d.model_dump(mode="json") for d in derivations],
        )
        write_bundle(results_root, bundle)

    return derivations
