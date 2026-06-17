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
    # torsion_sign "+1" means
    # T^lambda_{mu nu} = Gamma^lambda_{mu nu} - Gamma^lambda_{nu mu}
    torsion_sign: Literal["+1", "-1"]
    # "nabla-g": Q_{lambda mu nu} = nabla_lambda g_{mu nu}
    # "minus-nabla-g": Q_{lambda mu nu} = -nabla_lambda g_{mu nu}
    nonmetricity_definition: Literal["nabla-g", "minus-nabla-g"]
    # Contortion sign: "+1" means K^lambda_{mu nu} = +(1/2)(T^lambda_{mu nu}
    #   + g^{lambda sigma} g_{mu tau} T^tau_{sigma nu}
    #   + g^{lambda sigma} g_{nu tau} T^tau_{sigma mu}), per metric-affine-v1.
    # "-1" flips the leading factor to -(1/2).
    contortion_sign: Literal["+1", "-1"]
    # Disformation sign: "+1" means L^lambda_{mu nu} = +(1/2) g^{lambda rho}
    #   (-Q_{mu nu rho} - Q_{nu rho mu} + Q_{rho mu nu}), per metric-affine-v1.
    # "-1" flips the leading factor to -(1/2).
    disformation_sign: Literal["+1", "-1"]
    # "first-third": R_{mu nu} = R^lambda_{mu lambda nu}
    # "first-fourth": R_{mu nu} = R^lambda_{mu nu lambda}
    ricci_contraction: Literal["first-third", "first-fourth"]
    symmetrization_weight: Literal["1/n!", "1"]


NOETHER_DEFAULT_V1 = Conventions(
    id="noether-default-v1",
    dimension=4,
    signature="mostly-plus",
    riemann_sign="+1",
    torsion_sign="+1",
    nonmetricity_definition="nabla-g",
    contortion_sign="+1",
    disformation_sign="+1",
    ricci_contraction="first-third",
    symmetrization_weight="1/n!",
)
