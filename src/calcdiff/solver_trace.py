"""A differentiator that shows its work.

`derive(tree)` computes the derivative by structural recursion on the
expression tree -- one case per node type, exactly the rules a student
learns -- and records every rule it applies in a `Trace`. The trace is a
measurable proxy for the *procedural* work a problem demands: which rules,
how many times, and how deeply nested.

The derivative is returned as a tree in the same AST. It is deliberately not
simplified beyond trivial cleanup (x*1 -> x, x+0 -> x), so its size measures
the "expression swell" a student would have to write down.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction

from calcdiff.expr import Add, Const, Div, Fn, Mul, Node, Pow, Var

ZERO, ONE = Const(Fraction(0)), Const(Fraction(1))


@dataclass(frozen=True)
class Step:
    rule: str          # e.g. "PRODUCT", "CHAIN", "FN_sin"
    node: Node         # the subexpression this rule was applied to
    depth: int         # number of enclosing rule applications (recursion depth)
    chain_depth: int   # number of enclosing CHAIN applications


@dataclass
class Trace:
    steps: list[Step] = field(default_factory=list)

    def counts(self) -> Counter:
        return Counter(s.rule for s in self.steps)

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    @property
    def max_depth(self) -> int:
        return max((s.depth for s in self.steps), default=0)

    @property
    def max_chain_depth(self) -> int:
        """Longest chain of nested compositions, e.g. 2 for sin(e^(x^2))."""
        return max((s.chain_depth + (s.rule == "CHAIN") for s in self.steps), default=0)


def derive(tree: Node) -> tuple[Node, Trace]:
    """d/dx of `tree`, plus the trace of rules used to get there."""
    trace = Trace()
    return _d(tree, trace, 0, 0), trace


def _d(node: Node, tr: Trace, depth: int, chain: int) -> Node:
    def log(rule: str) -> None:
        tr.steps.append(Step(rule, node, depth, chain))

    def d(sub: Node, *, through_chain: bool = False) -> Node:
        return _d(sub, tr, depth + 1, chain + through_chain)

    match node:
        case Const():
            log("CONST")
            return ZERO

        case Var():
            log("VAR")
            return ONE

        case Pow(base, n):
            log("POWER")
            outer = _mul(Const(n), _pow(base, n - 1))      # n u^(n-1)
            if isinstance(base, Var):
                return outer
            log("CHAIN")
            return _mul(outer, d(base, through_chain=True))  # ... times u'

        case Add(terms):
            log("SUM")
            return _add(*(d(t) for t in terms))

        case Mul(factors):
            consts = [f for f in factors if isinstance(f, Const)]
            rest = [f for f in factors if not isinstance(f, Const)]
            c = _mul(*consts) if consts else ONE
            if consts:
                log("CONST_MULT")
            if len(rest) == 1:
                return _mul(c, d(rest[0]))
            # Generalized product rule: (f1...fk)' = sum_i f_i' prod_{j != i} f_j
            log("PRODUCT")
            terms = [_mul(*rest[:i], d(f), *rest[i + 1:]) for i, f in enumerate(rest)]
            return _mul(c, _add(*terms))

        case Div(num, den):
            log("QUOTIENT")
            dn, dd = d(num), d(den)
            top = _add(_mul(dn, den), _mul(Const(Fraction(-1)), num, dd))  # u'v - uv'
            return Div(top, _pow(den, Fraction(2)))

        case Fn(name, arg):
            log(f"FN_{name}")
            outer = _fn_derivative(name, arg)  # g'(u)
            if isinstance(arg, Var):
                return outer
            log("CHAIN")
            return _mul(outer, d(arg, through_chain=True))  # g'(u) u'

    raise TypeError(node)


def _fn_derivative(name: str, u: Node) -> Node:
    """g'(u) for each elementary g, written the way a textbook writes it."""
    match name:
        case "sin":
            return Fn("cos", u)
        case "cos":
            return _mul(Const(Fraction(-1)), Fn("sin", u))
        case "tan":
            return Div(ONE, Pow(Fn("cos", u), Fraction(2)))
        case "exp":
            return Fn("exp", u)
        case "ln":
            return Div(ONE, u)
        case "sqrt":
            return Div(ONE, _mul(Const(Fraction(2)), Fn("sqrt", u)))
    raise ValueError(name)


# ---------------------------------------------------------------------------
# Smart constructors: only the cleanup any person would do automatically.
# ---------------------------------------------------------------------------

def _mul(*factors: Node) -> Node:
    flat: list[Node] = []
    for f in factors:
        flat.extend(f.factors if isinstance(f, Mul) else (f,))
    c = Fraction(1)
    rest = []
    for f in flat:
        if isinstance(f, Const):
            c *= f.value
        else:
            rest.append(f)
    if c == 0:
        return ZERO
    if not rest:
        return Const(c)
    if c == 1:
        return rest[0] if len(rest) == 1 else Mul(tuple(rest))
    return Mul((Const(c), *rest))


def _add(*terms: Node) -> Node:
    flat: list[Node] = []
    for t in terms:
        flat.extend(t.terms if isinstance(t, Add) else (t,))
    flat = [t for t in flat if t != ZERO]
    if not flat:
        return ZERO
    return flat[0] if len(flat) == 1 else Add(tuple(flat))


def _pow(base: Node, n: Fraction) -> Node:
    if n == 0:
        return ONE
    if n == 1:
        return base
    return Pow(base, n)
