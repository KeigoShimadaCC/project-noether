"""Task -> DAG of capability-tagged steps.

The planner refuses to plan while the ambiguity ledger is non-empty. That is
the structural form of "ambiguity is resolved by asking, not by guessing".
"""

from pydantic import BaseModel, Field

from noether.kernels.base import Capability
from noether.npr.schema import NPR


class AmbiguityBlocked(RuntimeError):
    def __init__(self, questions: list[str]):
        self.questions = questions
        super().__init__("cannot plan: unresolved ambiguities remain: " + "; ".join(questions))


class PlanStep(BaseModel):
    capability: Capability
    description: str
    payload: dict = Field(default_factory=dict)


class Plan(BaseModel):
    task_type: str
    steps: list[PlanStep]
    verification: list[str]  # names of checks the result class requires


def build_plan(npr: NPR) -> Plan:
    unresolved = npr.unresolved_ambiguities()
    if unresolved:
        raise AmbiguityBlocked([a.question for a in unresolved])

    if npr.task.type == "vary":
        steps = [
            PlanStep(
                capability=Capability.SUBSTITUTE,
                description="expand composite shorthands and declared definitions",
            ),
            PlanStep(
                capability=Capability.VARY,
                description=(
                    "vary the action with respect to " + ", ".join(npr.task.with_respect_to)
                ),
                payload={"with_respect_to": npr.task.with_respect_to},
            ),
            PlanStep(
                capability=Capability.IBP,
                description="integrate by parts; record surface terms",
            ),
            PlanStep(
                capability=Capability.CANONICALIZE,
                description="canonicalize indices and merge equivalent terms",
            ),
        ]
        if npr.geometry.connection.type == "independent":
            steps.insert(
                1,
                PlanStep(
                    capability=Capability.INDEPENDENT_CONNECTION,
                    description="set up independent connection (torsion="
                    f"{npr.geometry.connection.torsion}, nonmetricity="
                    f"{npr.geometry.connection.nonmetricity})",
                ),
            )
        verification = [
            "well-formed",
            "symmetric-rank2",
            "divergence-free",
            "equal-on-background",
        ]
        return Plan(task_type="vary", steps=steps, verification=verification)

    raise NotImplementedError(
        f"task type {npr.task.type!r} has no plan template yet (Horizon gate)"
    )
