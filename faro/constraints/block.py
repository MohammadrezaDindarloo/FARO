"""A uniform container for constraint expressions.

Every constraint block in `faro.constraints` returns one of these, so the NLP
builders in Milestones 3-5 can assemble Eqs. 14 / 15 / 17 without knowing anything
about what each block means.

Sign convention throughout FARO, matching the paper's own notation:

    eq   == 0      equalities
    ineq <= 0      inequalities

The paper writes its constraint groups the same way (`contact(q, c) <= 0`,
`collision(q) <= 0`, `limits(q) <= 0` in Eq. 14), so a block's rows drop straight
into an NLP with bounds `lbg = ubg = 0` for equalities and `lbg = -inf, ubg = 0`
for inequalities.

`labels` exists for debugging. When a 5000-row NLP comes back infeasible, the
single most useful thing is knowing which row it was; carrying a human-readable
name per row costs nothing at solve time and saves hours.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import casadi as ca


def _n_rows(expr) -> int:
    return 0 if expr is None else int(expr.shape[0])


@dataclass
class ConstraintBlock:
    """Equalities and inequalities from one constraint definition.

    Attributes
    ----------
    name : which block produced this, e.g. "contact[left_foot_sole->floor]".
    eq : column vector of expressions constrained to equal zero.
    ineq : column vector of expressions constrained to be <= 0.
    eq_labels, ineq_labels : one label per row, naming the paper sub-equation.
    """

    name: str
    eq: ca.SX | None = None
    ineq: ca.SX | None = None
    eq_labels: list[str] = field(default_factory=list)
    ineq_labels: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Labels must line up with rows or debugging output silently misattributes
        # violations to the wrong constraint, which is worse than having no labels.
        if self.eq is not None and len(self.eq_labels) != _n_rows(self.eq):
            raise ValueError(
                f"{self.name}: {len(self.eq_labels)} eq labels for {_n_rows(self.eq)} rows"
            )
        if self.ineq is not None and len(self.ineq_labels) != _n_rows(self.ineq):
            raise ValueError(
                f"{self.name}: {len(self.ineq_labels)} ineq labels for {_n_rows(self.ineq)} rows"
            )

    @property
    def n_eq(self) -> int:
        return _n_rows(self.eq)

    @property
    def n_ineq(self) -> int:
        return _n_rows(self.ineq)

    def __repr__(self) -> str:  # pragma: no cover - display only
        return f"ConstraintBlock({self.name!r}, n_eq={self.n_eq}, n_ineq={self.n_ineq})"


def merge(name: str, blocks: list[ConstraintBlock]) -> ConstraintBlock:
    """Concatenate blocks, preserving labels (prefixed with the source block name)."""
    eqs = [b.eq for b in blocks if b.n_eq]
    ineqs = [b.ineq for b in blocks if b.n_ineq]
    return ConstraintBlock(
        name=name,
        eq=ca.vertcat(*eqs) if eqs else None,
        ineq=ca.vertcat(*ineqs) if ineqs else None,
        eq_labels=[f"{b.name}/{lab}" for b in blocks for lab in b.eq_labels],
        ineq_labels=[f"{b.name}/{lab}" for b in blocks for lab in b.ineq_labels],
    )


def evaluate(block: ConstraintBlock, variables, values) -> dict[str, float]:
    """Numerically evaluate a block, returning {label: value}.

    The workhorse for the unit tests: it lets a test assert that a physically
    correct configuration satisfies a constraint and an incorrect one violates it,
    which is the only way to be confident the algebra matches the paper.
    """
    out: dict[str, float] = {}
    for expr, labels in ((block.eq, block.eq_labels), (block.ineq, block.ineq_labels)):
        if expr is None:
            continue
        fn = ca.Function("f", [variables] if not isinstance(variables, list) else variables, [expr])
        result = fn(*(values if isinstance(values, list) else [values]))
        for i, label in enumerate(labels):
            out[label] = float(result[i])
    return out
