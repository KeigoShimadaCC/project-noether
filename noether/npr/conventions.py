"""Convention blocks (AGENTS.md section 5).

Every expression that crosses a kernel boundary carries one of these. No code
in this repository may assume a convention silently.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class Conventions(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    dimension: int | str  # int, or a symbol name like "D" for symbolic dimension
    signature: Literal["mostly-plus", "mostly-minus"]
    # riemann_sign "+1" means
    # R^rho_{sigma mu nu} = +(d Gamma^rho_{nu sigma}/d x^mu - ...)
    riemann_sign: Literal["+1", "-1"]
    # "first-third": R_{mu nu} = R^lambda_{mu lambda nu}
    ricci_contraction: Literal["first-third"]
    symmetrization_weight: Literal["1/n!", "1"]


NOETHER_DEFAULT_V1 = Conventions(
    id="noether-default-v1",
    dimension=4,
    signature="mostly-plus",
    riemann_sign="+1",
    ricci_contraction="first-third",
    symmetrization_weight="1/n!",
)
