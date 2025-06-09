# mypy: allow-untyped-defs
import functools
import math
from typing import TYPE_CHECKING, Tuple, Union

import sympy
from sympy.core.numbers import equal_valued
from sympy.printing.precedence import PRECEDENCE

from .numbers import int_oo

if TYPE_CHECKING:
    from collections.abc import Iterable

# Portions of this file are adapted from the Sympy codebase, which was
# licensed as follows:
#
#   Copyright (c) 2006-2023 SymPy Development Team
#
#   All rights reserved.
#
#   Redistribution and use in source and binary forms, with or without
#   modification, are permitted provided that the following conditions are met:
#
#     a. Redistributions of source code must retain the above copyright notice,
#        this list of conditions and the following disclaimer.
#     b. Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#     c. Neither the name of SymPy nor the names of its contributors
#        may be used to endorse or promote products derived from this software
#        without specific prior written permission.
#
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#   ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
#   ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#   DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#   SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#   CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
#   LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
#   OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
#   DAMAGE.


def simple_floordiv_gcd(p: sympy.Basic, q: sympy.Basic) -> sympy.Basic:
    """
    Fast path for sympy.gcd, using a simple factoring strategy.

    We try to rewrite p and q in the form n*e*p1 + n*e*p2 and n*e*q0,
    where n is the greatest common integer factor and e is the largest
    syntactic common factor (i.e., common sub-expression) in p and q.
    Then the gcd returned is n*e, cancelling which we would be left with
    p1 + p2 and q0.

    Note that further factoring of p1 + p2 and q0 might be possible with
    sympy.factor (which uses domain-specific theories). E.g., we are unable
    to find that x*y + x + y + 1 is divisible by x + 1. More generally,
    when q is of the form q1 + q2 (instead of being already factored) it
    might be necessary to fall back on sympy.gcd.
    """

    def integer_coefficient(x: sympy.Basic) -> int:
        integer_coefficients: list[int] = [
            abs(int(arg)) for arg in sympy.Mul.make_args(x) if isinstance(arg, (int, sympy.Integer))
        ]
        return math.prod(integer_coefficients)

    def integer_factor(expr: sympy.Basic) -> int:
        integer_factors: Iterable[int] = map(integer_coefficient, sympy.Add.make_args(expr))
        return functools.reduce(math.gcd, integer_factors)

    gcd: int = math.gcd(integer_factor(p), integer_factor(q))
    p, q = p / gcd, q / gcd  # type: ignore[operator, assignment]  # remove in py3.12

    base_splits: list[tuple[sympy.Basic, ...]] = list(map(sympy.Mul.make_args, sympy.Add.make_args(p)))
    divisor_split: tuple[sympy.Basic, ...] = sympy.Mul.make_args(q)
    for x in divisor_split:
        if all(x in base_split for base_split in base_splits):
            gcd = gcd * x  # type: ignore[operator]  # remove in py3.12
    return gcd  # type: ignore[return-value]  # remove in py3.12


# It would be nice to have assertions on whether or not inputs is_integer
# However, with bugs like https://github.com/sympy/sympy/issues/26620 sympy
# sometimes inconsistently reports floats an integers.
#
# What we can assume from sympy is that if something is an int, it
# definitely is is_integer, but if it is a float it may or may not
# be is_integer.  So we are unable to do strong asserts that things
# are NOT integers.


# TODO: In Triton, // rounds to zero, but in Python, it is floor division.
# When we can prove both arguments are non-negative, we should just have a
# GenericFloorDiv (name pending) which can codegen efficiently in Python/C,
# and then PythonFloorDiv and CIntDiv which have the appropriate rounding
# semantics.
#
# Right now, FloorDiv de facto changes behavior if arguments are negative or
# not, this can potentially cause correctness issues.
class FloorDiv(sympy.Function):
    """
    We maintain this so that:
    1. We can use divisibility guards to simplify FloorDiv(a, b) to a / b.
    2. Printing out the expression is nicer (compared to say, representing a//b as (a - a % b) / b).

    NB: This is Python-style floor division, round to -Inf
    """

    nargs: Tuple[int, ...] = (2,)
    precedence: int = 35  # lower precedence than add
    is_integer: bool = True

    @property
    def base(self) -> sympy.Basic:
        return self.args[0]

    @property
    def divisor(self) -> sympy.Basic:
        return self.args[1]

    def _sympystr(self, printer: sympy.printing.StrPrinter) -> str:
        base = printer.parenthesize(self.base, PRECEDENCE["Atom"] - 0.5)
        divisor = printer.parenthesize(self.divisor, PRECEDENCE["Atom"] - 0.5)
        return f"({base}//{divisor})"

    # Automatic evaluation.
    # https://docs.sympy.org/latest/guides/custom-functions.html#best-practices-for-eval
    @classmethod
    def eval(cls, base: sympy.Integer, divisor: sympy.Integer) -> Union[sympy.Basic, None]:
        # python test/test_dynamic_shapes.py -k TestDimConstraints.test_dim_constraints_solve_full
        # Assert triggered by inequality solver
        # assert base.is_integer, base
        # assert divisor.is_integer, divisor

        # We don't provide the same error message as in Python because SymPy
        # makes it difficult to check the types.
        if divisor.is_zero:
            raise ZeroDivisionError("division by zero")
        if base in (int_oo, -int_oo, sympy.oo, -sympy.oo) and divisor in (
            int_oo,
            -int_oo,
            sympy.oo,
            -sympy.oo,
        ):
            return sympy.nan
        if base is sympy.nan or divisor is sympy.nan:
            return sympy.nan

        if base.is_zero:
            return sympy.S.Zero
        if base.is_integer and equal_valued(divisor, 1):
            return base
        if base.is_integer and equal_valued(divisor, -1):
            return sympy.Mul(base, -1)
        if (
            isinstance(base, sympy.Number)
            and isinstance(divisor, sympy.Number)
            and (base in (int_oo, -int_oo, sympy.oo, -sympy.oo) or divisor in (int_oo, -int_oo, sympy.oo, -sympy.oo))
        ):
            r = float(base) / float(divisor)
            if r == math.inf:
                return int_oo
            elif r == -math.inf:
                return -int_oo
            elif math.isnan(r):
                return sympy.nan
            else:
                return sympy.Integer(math.floor(r))
        if isinstance(base, sympy.Integer) and isinstance(divisor, sympy.Integer):
            return sympy.Integer(int(base) // int(divisor))
        if isinstance(base, FloorDiv):
            return FloorDiv(base.args[0], base.args[1] * divisor)

        # Expands (x + y) // b into x // b + y // b.
        # This only works if floor is an identity, i.e. x / b is an integer.
        if isinstance(divisor, sympy.Integer):
            quotients = 0
            terms = []
            for term in sympy.Add.make_args(base):
                quotient = term / divisor

                if quotient.is_integer:
                    terms.append(term)
                    quotients += quotient

            if len(terms) != 0:
                # Passing evaluate = False since expression will be optimized during the subtraction post its construction.
                return FloorDiv(base - sympy.Add(*terms, evaluate=False), divisor) + quotients

        try:
            gcd = simple_floordiv_gcd(base, divisor)
            if equal_valued(gcd, 1) and isinstance(divisor, sympy.Add):
                gcd = sympy.gcd(base, divisor)
            if not equal_valued(gcd, 1):
                return FloorDiv(sympy.simplify(base / gcd), sympy.simplify(divisor / gcd))
        except sympy.PolynomialError:
            pass  # https://github.com/pytorch/pytorch/issues/108276

        return None

    def _ccode(self, printer):
        base = printer.parenthesize(self.base, PRECEDENCE["Atom"] - 0.5)
        divisor = printer.parenthesize(self.divisor, PRECEDENCE["Atom"] - 0.5)
        return f"floor({base}/{divisor})"
