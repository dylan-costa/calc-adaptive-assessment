"""Expression trees for derivative problems.

We keep our own small AST instead of using SymPy objects directly because
SymPy canonicalizes on construction: x/(x+1) becomes x*(x+1)**-1, x*x becomes
x**2, and so on. That would erase the "as written" structure of a problem,
which is exactly what determines how hard it is for a student (a quotient
looks and feels different from a product with a negative power).

SymPy is still used as the ground truth for the mathematics: every tree can be
converted with `to_sympy` and differentiated with `sympy.diff`.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterator

import sympy as sp

X_SYM = sp.Symbol("x")

FUNCTIONS = ("sin", "cos", "tan", "exp", "ln", "sqrt")
_SYMPY_FN = {
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "exp": sp.exp,
    "ln": sp.log,
    "sqrt": sp.sqrt,
}


class Node:
    """Base class. All nodes are immutable and hashable."""

    def children(self) -> tuple[Node, ...]:
        return ()


@dataclass(frozen=True)
class Const(Node):
    value: Fraction


@dataclass(frozen=True)
class Var(Node):
    """The variable x."""


@dataclass(frozen=True)
class Pow(Node):
    base: Node
    exp: Fraction

    def children(self) -> tuple[Node, ...]:
        return (self.base,)


@dataclass(frozen=True)
class Add(Node):
    terms: tuple[Node, ...]

    def children(self) -> tuple[Node, ...]:
        return self.terms


@dataclass(frozen=True)
class Mul(Node):
    factors: tuple[Node, ...]

    def children(self) -> tuple[Node, ...]:
        return self.factors


@dataclass(frozen=True)
class Div(Node):
    num: Node
    den: Node

    def children(self) -> tuple[Node, ...]:
        return (self.num, self.den)


@dataclass(frozen=True)
class Fn(Node):
    name: str
    arg: Node

    def __post_init__(self) -> None:
        if self.name not in FUNCTIONS:
            raise ValueError(f"unknown function {self.name!r}")

    def children(self) -> tuple[Node, ...]:
        return (self.arg,)


X = Var()


# ---------------------------------------------------------------------------
# Tree statistics
# ---------------------------------------------------------------------------

def walk(node: Node) -> Iterator[Node]:
    """Pre-order traversal."""
    yield node
    for c in node.children():
        yield from walk(c)


def size(node: Node) -> int:
    return sum(1 for _ in walk(node))


def depth(node: Node) -> int:
    kids = node.children()
    return 1 + (max(depth(c) for c in kids) if kids else 0)


# ---------------------------------------------------------------------------
# Conversion to SymPy (the mathematical ground truth)
# ---------------------------------------------------------------------------

def to_sympy(node: Node) -> sp.Expr:
    match node:
        case Const(v):
            return sp.Rational(v.numerator, v.denominator)
        case Var():
            return X_SYM
        case Pow(b, e):
            return to_sympy(b) ** sp.Rational(e.numerator, e.denominator)
        case Add(ts):
            return sp.Add(*(to_sympy(t) for t in ts))
        case Mul(fs):
            return sp.Mul(*(to_sympy(f) for f in fs))
        case Div(n, d):
            return to_sympy(n) / to_sympy(d)
        case Fn(name, a):
            return _SYMPY_FN[name](to_sympy(a))
    raise TypeError(node)


# ---------------------------------------------------------------------------
# LaTeX printing that preserves the as-written form
# ---------------------------------------------------------------------------

def _frac_tex(q: Fraction) -> str:
    if q.denominator == 1:
        return str(q.numerator)
    sign = "-" if q < 0 else ""
    return f"{sign}{abs(q.numerator)}/{q.denominator}"


def _is_negative(node: Node) -> bool:
    """Does this term print with a leading minus sign?"""
    match node:
        case Const(v):
            return v < 0
        case Mul(fs):
            return isinstance(fs[0], Const) and fs[0].value < 0
    return False


def _negate(node: Node) -> Node:
    match node:
        case Const(v):
            return Const(-v)
        case Mul(fs):
            c = -fs[0].value  # type: ignore[union-attr]
            rest = fs[1:]
            if c == 1:
                return rest[0] if len(rest) == 1 else Mul(rest)
            return Mul((Const(c),) + rest)
    raise ValueError(node)


def _needs_parens_as_factor(node: Node) -> bool:
    return isinstance(node, Add) or _is_negative(node)


def _needs_parens_as_base(node: Node) -> bool:
    return not isinstance(node, (Var, Const)) or (isinstance(node, Const) and node.value < 0)


def to_latex(node: Node) -> str:
    match node:
        case Const(v):
            if v.denominator == 1:
                return str(v.numerator)
            sign = "-" if v < 0 else ""
            return rf"{sign}\frac{{{abs(v.numerator)}}}{{{v.denominator}}}"
        case Var():
            return "x"
        case Pow(b, e):
            # Textbook convention: sin^2(x) rather than (sin x)^2.
            if isinstance(b, Fn) and b.name in ("sin", "cos", "tan") and e.denominator == 1 and e > 0:
                return rf"\{b.name}^{{{e.numerator}}}{_fn_arg(b.arg)}"
            base = to_latex(b)
            if _needs_parens_as_base(b):
                base = rf"\left({base}\right)"
            return f"{base}^{{{_frac_tex(e)}}}"
        case Add(ts):
            out = to_latex(ts[0])
            for t in ts[1:]:
                if _is_negative(t):
                    out += " - " + to_latex(_negate(t))
                else:
                    out += " + " + to_latex(t)
            return out
        case Mul(fs):
            parts: list[str] = []
            rest = list(fs)
            if isinstance(rest[0], Const) and len(rest) > 1:
                c = rest.pop(0).value  # type: ignore[union-attr]
                if c == -1:
                    parts.append("-")
                elif c != 1:
                    parts.append(to_latex(Const(c)))
            for i, f in enumerate(rest):
                s = to_latex(f)
                if _needs_parens_as_factor(f):
                    s = rf"\left({s}\right)"
                # A digit immediately followed by a digit would be ambiguous.
                if parts and parts[-1] not in ("-",) and s[:1].isdigit():
                    s = r"\cdot " + s
                parts.append(s)
            return " ".join(parts).replace("- ", "-", 1) if parts[0] == "-" else " ".join(parts)
        case Div(n, d):
            return rf"\frac{{{to_latex(n)}}}{{{to_latex(d)}}}"
        case Fn(name, a):
            if name == "exp":
                return f"e^{{{to_latex(a)}}}"
            if name == "sqrt":
                return rf"\sqrt{{{to_latex(a)}}}"
            return rf"\{name}{_fn_arg(a)}"
    raise TypeError(node)


def _fn_arg(a: Node) -> str:
    # sin x, but sin(x^2 + 1)
    if isinstance(a, Var):
        return " x"
    return rf"\left({to_latex(a)}\right)"


# ---------------------------------------------------------------------------
# JSON-friendly serialization (for storing problems in the label database)
# ---------------------------------------------------------------------------

def to_dict(node: Node) -> dict:
    match node:
        case Const(v):
            return {"t": "const", "v": str(v)}
        case Var():
            return {"t": "x"}
        case Pow(b, e):
            return {"t": "pow", "base": to_dict(b), "exp": str(e)}
        case Add(ts):
            return {"t": "add", "terms": [to_dict(t) for t in ts]}
        case Mul(fs):
            return {"t": "mul", "factors": [to_dict(f) for f in fs]}
        case Div(n, d):
            return {"t": "div", "num": to_dict(n), "den": to_dict(d)}
        case Fn(name, a):
            return {"t": "fn", "name": name, "arg": to_dict(a)}
    raise TypeError(node)


def from_dict(d: dict) -> Node:
    match d["t"]:
        case "const":
            return Const(Fraction(d["v"]))
        case "x":
            return X
        case "pow":
            return Pow(from_dict(d["base"]), Fraction(d["exp"]))
        case "add":
            return Add(tuple(from_dict(t) for t in d["terms"]))
        case "mul":
            return Mul(tuple(from_dict(f) for f in d["factors"]))
        case "div":
            return Div(from_dict(d["num"]), from_dict(d["den"]))
        case "fn":
            return Fn(d["name"], from_dict(d["arg"]))
    raise ValueError(d)
