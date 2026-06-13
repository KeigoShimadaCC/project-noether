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


def propagate_resolution(npr: NPR, ambiguity: Ambiguity) -> None:
    if ambiguity.id == "amb-vary-wrt" and ambiguity.resolution:
        tokens = [t for t in re.split(r"[^A-Za-z0-9_\\]+", ambiguity.resolution) if t]
        declared = {obj.name for obj in npr.objects}
        fields = [t for t in tokens if t in declared]
        if fields and npr.task.type == "vary":
            npr.task.with_respect_to = fields
