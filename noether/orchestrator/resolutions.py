"""Map confirmed ledger answers onto the NPR fields they decide.

Recording an answer in the ambiguity ledger is not enough when the answer
carries task semantics: the planner reads `task.with_respect_to`, not the
ledger. Every resolve path (session, elicit confirmation) funnels through
`propagate_resolution` so a confirmed choice and the task can never disagree.

Only declared object names are accepted; connective words in eval-style
options ("g and phi", "g only") fall out naturally because they are not
declared objects. A free-form answer that names no declared field leaves the
task untouched rather than guessing.
"""

from __future__ import annotations

import re

from noether.npr.schema import NPR, Ambiguity

_RICCI_AMBIGUITY_ID = "amb-ricci-contraction"


def _set_connection_family(npr: NPR) -> None:
    connection = npr.geometry.connection
    if connection.type == "levi-civita":
        connection.family = "riemannian"
    elif connection.metric_compatible and connection.torsion and not connection.nonmetricity:
        connection.family = "riemann-cartan"
    else:
        connection.family = "metric-affine"


def _ensure_ricci_contraction_ambiguity(npr: NPR) -> None:
    if any(amb.id == _RICCI_AMBIGUITY_ID for amb in npr.ambiguities):
        return
    npr.ambiguities.append(
        Ambiguity(
            id=_RICCI_AMBIGUITY_ID,
            question=(
                "With an independent connection the Ricci tensor need not be symmetric. "
                "Which Ricci contraction convention should Noether use?"
            ),
            kind="conventional",
            options=["first-third", "first-fourth"],
        )
    )


def _remove_ricci_contraction_ambiguity(npr: NPR) -> None:
    npr.ambiguities = [amb for amb in npr.ambiguities if amb.id != _RICCI_AMBIGUITY_ID]


def _propagate_geometry_resolution(npr: NPR, ambiguity: Ambiguity) -> None:
    connection = npr.geometry.connection

    if ambiguity.id == "amb-connection":
        if ambiguity.resolution == "independent":
            connection.type = "independent"
            _ensure_ricci_contraction_ambiguity(npr)
        elif ambiguity.resolution == "levi-civita":
            connection.type = "levi-civita"
            connection.torsion = False
            connection.nonmetricity = False
            connection.metric_compatible = True
            _remove_ricci_contraction_ambiguity(npr)
        _set_connection_family(npr)
        return

    if ambiguity.id == "amb-torsion":
        connection.torsion = ambiguity.resolution in {"torsion-present", "torsion-allowed"}
        _set_connection_family(npr)
        return

    if ambiguity.id == "amb-nonmetricity":
        connection.nonmetricity = ambiguity.resolution in {
            "nonmetricity-present",
            "nonmetricity-allowed",
        }
        _set_connection_family(npr)
        return

    if ambiguity.id == "amb-metric-compatibility":
        connection.metric_compatible = ambiguity.resolution == "metric-compatible"
        _set_connection_family(npr)
        return

    if ambiguity.id == _RICCI_AMBIGUITY_ID and ambiguity.resolution:
        npr.conventions = npr.conventions.model_copy(
            update={"ricci_contraction": ambiguity.resolution}
        )


def propagate_resolution(npr: NPR, ambiguity: Ambiguity) -> None:
    if ambiguity.id == "amb-vary-wrt" and ambiguity.resolution:
        tokens = [t for t in re.split(r"[^A-Za-z0-9_\\]+", ambiguity.resolution) if t]
        declared = {obj.name for obj in npr.objects}
        fields = [t for t in tokens if t in declared]
        if fields and npr.task.type == "vary":
            npr.task.with_respect_to = fields
        return

    if ambiguity.id == "amb-kinetic-X" and ambiguity.resolution == "independent-field":
        # The human overrode the convention default: X is not the kinetic
        # shorthand of the scalar but a field in its own right.
        for obj in npr.objects:
            if obj.name == "X":
                obj.kind = "scalar-field"
                obj.role = "dynamical"
                obj.definition_tex = None
                break
        return

    if ambiguity.resolution:
        _propagate_geometry_resolution(npr, ambiguity)
