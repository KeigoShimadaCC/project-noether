"""Run a list of checks against a result expression and summarize."""

from pydantic import BaseModel, Field

from noether.kernels.base import KernelAdapter
from noether.npr.ast import Expr
from noether.verify.checks import CheckResult


class LadderReport(BaseModel):
    results: list[CheckResult] = Field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)

    def summary(self) -> str:
        lines = []
        for r in self.results:
            status = "PASS" if r.passed else "FAIL"
            lines.append(f"[{r.rung}] {r.name}: {status} ({r.computed_by}) {r.detail}")
        return "\n".join(lines)


def run_ladder(expr: Expr, checks: list, adapters: dict[str, KernelAdapter]) -> LadderReport:
    report = LadderReport()
    for check in checks:
        report.results.append(check.run(expr, adapters))
    return report
