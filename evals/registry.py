"""Eval registry: one declarative spec per eval, consumed by the CLI runner.

Each spec describes the full four-beat loop for its eval: the NPR with its
documented elicitation answers, the audited cadabra templates with the kernel
checks that must come back True, and the presented results with their
verification ladders and extra component-evaluation tasks.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from noether.kernels.base import Capability, KernelTask
from noether.npr.ast import Expr
from noether.npr.schema import NPR


@dataclass(frozen=True)
class CadabraRun:
    description: str
    template: str
    required_true: tuple[str, ...]


@dataclass(frozen=True)
class PresentedResult:
    result_id: str
    expr: Callable[[], Expr]
    tex_suffix: str
    ladder: Callable[[], list[Any]]
    component_tasks: tuple[tuple[str, dict], ...] = ()


@dataclass(frozen=True)
class EvalSpec:
    key: str
    title: str
    build_npr: Callable[[bool], NPR]
    answers: dict[str, str]
    cadabra_runs: tuple[CadabraRun, ...]
    results: tuple[PresentedResult, ...]
    notes: tuple[str, ...] = field(default=())


def _eval1() -> EvalSpec:
    from evals import eval1_eh_trace as m
    from noether.verify.checks import (
        DivergenceFreeCheck,
        EqualOnBackgroundCheck,
        SymmetricCheck,
        WellFormedCheck,
    )

    return EvalSpec(
        key="eval1",
        title="Einstein-Hilbert in trace form",
        build_npr=m.build_npr,
        answers=m.ELICITATION_ANSWERS,
        cadabra_runs=(
            CadabraRun("EH trace-form metric variation", "eval1_eh_trace", ("residue_zero",)),
        ),
        results=(
            PresentedResult(
                result_id="eom-metric",
                expr=m.target_eom,
                tex_suffix=" = 0",
                ladder=lambda: [
                    WellFormedCheck(expected_free=[m.MU, m.NU]),
                    SymmetricCheck(),
                    DivergenceFreeCheck(),
                    EqualOnBackgroundCheck(rhs=m.target_eom_expanded()),
                ],
            ),
        ),
    )


def _eval2() -> EvalSpec:
    from evals import eval2_palatini as m
    from noether.verify.checks import WellFormedCheck

    return EvalSpec(
        key="eval2",
        title="Palatini gravity (independent torsionful connection)",
        build_npr=m.build_npr,
        answers=m.ELICITATION_ANSWERS,
        cadabra_runs=(
            CadabraRun("Palatini metric variation", "eval2_palatini_metric", ("residue_zero",)),
            CadabraRun(
                "Palatini connection variation and projective solution",
                "eval2_palatini_connection",
                ("solution_zero", "ricci_shift_is_dA"),
            ),
        ),
        results=(
            PresentedResult(
                result_id="eom-metric",
                expr=m.target_metric_eom,
                tex_suffix=" = 0",
                ladder=lambda: [WellFormedCheck(expected_free=[m.MU, m.NU])],
                component_tasks=(
                    (
                        "projective family inert on background (seed 11/4)",
                        {
                            "check": "palatini-projective-inert",
                            "metric": {"kind": "random-diagonal", "seed": 11, "dim": 3},
                            "seed": 4,
                        },
                    ),
                    (
                        "projective family inert on background (seed 23/9)",
                        {
                            "check": "palatini-projective-inert",
                            "metric": {"kind": "random-diagonal", "seed": 23, "dim": 4},
                            "seed": 9,
                        },
                    ),
                ),
            ),
        ),
        notes=(
            "connection EOM solved by Gamma = LC(g) + delta^lam_nu A_mu; "
            "metric equation reduces to G_{mu nu}(g) = 0",
        ),
    )


def _eval3() -> EvalSpec:
    from evals import eval3_scalar_tensor as m
    from noether.verify.checks import WellFormedCheck

    bg = {"kind": "random-diagonal", "seed": 31, "dim": 4}
    phi = {"phi": {"kind": "random-scalar", "seed": 17}}
    return EvalSpec(
        key="eval3",
        title="Scalar-tensor gravity F(phi) R",
        build_npr=m.build_npr,
        answers=m.ELICITATION_ANSWERS,
        cadabra_runs=(
            CadabraRun(
                "scalar-tensor metric variation",
                "eval3_scalar_tensor_metric",
                ("residue_zero",),
            ),
            CadabraRun(
                "scalar-tensor scalar variation",
                "eval3_scalar_tensor_scalar",
                ("residue_zero",),
            ),
        ),
        results=(
            PresentedResult(
                result_id="eom-metric",
                expr=m.target_metric_eom,
                tex_suffix=" = 0",
                ladder=lambda: [WellFormedCheck(expected_free=[m.MU, m.NU])],
                component_tasks=(
                    (
                        "minimal-limit EOM symmetric on background",
                        {
                            "check": "symmetric",
                            "expr": m.minimal_limit_metric_eom().model_dump(),
                            "metric": bg,
                            "fields": phi,
                        },
                    ),
                    (
                        "generalized Bianchi link on background",
                        {
                            "check": "equal",
                            "lhs": m.bianchi_link_lhs().model_dump(),
                            "rhs": m.bianchi_link_rhs().model_dump(),
                            "metric": bg,
                            "fields": phi,
                        },
                    ),
                ),
            ),
            PresentedResult(
                result_id="eom-scalar",
                expr=m.target_scalar_eom,
                tex_suffix=" = 0",
                ladder=lambda: [WellFormedCheck(expected_free=[])],
            ),
        ),
    )


def _eval4() -> EvalSpec:
    from evals import eval4_maxwell as m
    from noether.verify.checks import WellFormedCheck

    return EvalSpec(
        key="eval4",
        title="Maxwell on a fixed curved background",
        build_npr=m.build_npr,
        answers=m.ELICITATION_ANSWERS,
        cadabra_runs=(
            CadabraRun(
                "Maxwell variation (only A varied; g background)",
                "eval4_maxwell",
                ("residue_zero",),
            ),
        ),
        results=(
            PresentedResult(
                result_id="eom-gauge",
                expr=m.target_eom,
                tex_suffix=" = 0",
                ladder=lambda: [WellFormedCheck(expected_free=[m.NU])],
                component_tasks=(
                    (
                        "Noether identity off shell (random antisymmetric F)",
                        {
                            "check": "zero",
                            "expr": m.noether_identity_expr().model_dump(),
                            "metric": {"kind": "random-diagonal", "seed": 19, "dim": 4},
                            "fields": {"F": {"kind": "random-antisymmetric", "seed": 12}},
                        },
                    ),
                ),
            ),
        ),
        notes=("role discipline: the metric is never varied",),
    )


def _eval5() -> EvalSpec:
    from evals import eval5_gauss_bonnet as m
    from noether.verify.checks import WellFormedCheck

    h = m.lanczos_shorthand().model_dump()
    return EvalSpec(
        key="eval5",
        title="Gauss-Bonnet (Lovelock p=2), dimension-dependent identities",
        build_npr=m.build_npr,
        answers=m.ELICITATION_ANSWERS,
        cadabra_runs=(
            CadabraRun(
                "Lovelock p=2 delta algebra in symbolic D (GB scalar + Lanczos form)",
                "eval5_gauss_bonnet",
                ("gb_scalar_zero", "lanczos_form_zero"),
            ),
        ),
        results=(
            PresentedResult(
                result_id="eom-metric",
                expr=m.target_eom,
                tex_suffix=" = 0",
                ladder=lambda: [WellFormedCheck(expected_free=[m.MU, m.NU])],
                component_tasks=(
                    (
                        "D=4: Lanczos tensor identically zero (GB scalar nonzero background)",
                        {"check": "zero", "expr": h, "metric": {"kind": "warped-product-4d"}},
                    ),
                    (
                        "D=5: Lanczos tensor symmetric",
                        {
                            "check": "symmetric",
                            "expr": h,
                            "metric": {
                                "kind": "sparse-diagonal",
                                "seed": 7,
                                "dim": 5,
                                "curved": 3,
                            },
                        },
                    ),
                    (
                        "D=5: Lanczos tensor divergence-free (Lovelock property)",
                        {
                            "check": "divergence-zero",
                            "expr": h,
                            "metric": {
                                "kind": "sparse-diagonal",
                                "seed": 7,
                                "dim": 5,
                                "curved": 3,
                            },
                        },
                    ),
                ),
            ),
        ),
        notes=(
            "in D=4 the equation is IDENTICALLY zero (Gauss-Bonnet is topological); "
            "dynamical only for D >= 5",
            "variational derivation with Bianchi reduction is Horizon 2 scope; "
            "kernel evidence here: Lovelock delta algebra (cadabra, symbolic D) "
            "+ component Lovelock properties (sympy); field-equation form cited "
            "from Lovelock 1971",
        ),
    )


_BUILDERS: dict[str, Callable[[], EvalSpec]] = {
    "eval1": _eval1,
    "eval2": _eval2,
    "eval3": _eval3,
    "eval4": _eval4,
    "eval5": _eval5,
}


def get_spec(key: str) -> EvalSpec:
    if key not in _BUILDERS:
        raise KeyError(f"no eval spec named {key!r}")
    return _BUILDERS[key]()


def component_task(description: str, payload: dict) -> KernelTask:
    return KernelTask(
        capability=Capability.COMPONENT_EVAL, description=description, payload=payload
    )
