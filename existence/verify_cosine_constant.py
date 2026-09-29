#!/usr/bin/env python3
"""Exact certificate for the cosine-translation appendix; standard library only.

Run: python3 verify_cosine_constant.py
Every comparison uses Fraction. Printed decimals are for inspection only.
"""

from decimal import Decimal, localcontext
from fractions import Fraction as F
from math import factorial


def require(condition, description):
    if not condition:
        raise ArithmeticError(description)
    print("PASS:", description)


def arctan_bounds(denominator, terms=80):
    partial = sum(
        (F((-1) ** j, (2 * j + 1) * denominator ** (2 * j + 1))
         for j in range(terms)), F(0)
    )
    adjacent = partial + F(
        (-1) ** terms, (2 * terms + 1) * denominator ** (2 * terms + 1)
    )
    return min(partial, adjacent), max(partial, adjacent)


def pi_bounds():
    a_lo, a_hi = arctan_bounds(5)
    b_lo, b_hi = arctan_bounds(239)
    # A short rational enclosure prevents unnecessary denominator growth.
    lo, hi = 16 * a_lo - 4 * b_hi, 16 * a_hi - 4 * b_lo
    return outward(lo, hi)


def outward(lo, hi, digits=30):
    scale = 10**digits
    return F(lo * scale // 1, scale), F(-((-hi * scale) // 1), scale)


def series_bounds(z_lo, z_hi):
    # Integrating the degree-21 lower and degree-20 upper Taylor bounds
    # for exp(-z*x*x), with weight (1-x)^2, gives these sums.
    # Each signed monomial is enclosed separately, so no monotonicity
    # assertion about a truncated polynomial is needed.
    lower, upper = F(0), F(0)
    for j in range(22):
        denominator = factorial(j) * (2*j+1) * (2*j+2) * (2*j+3)
        if j % 2:
            term_lo, term_hi = -z_hi**j / denominator, -z_lo**j / denominator
        else:
            term_lo, term_hi = z_lo**j / denominator, z_hi**j / denominator
        lower += term_lo
        if j <= 20:
            upper += term_hi
    return lower, upper


def squared_g_bounds(c, pi_lo, pi_hi):
    z_lo, z_hi = outward(9*pi_lo**2 / (2*c**2), 9*pi_hi**2 / (2*c**2))
    s_lo, s_hi = series_bounds(z_lo, z_hi)
    require(0 < s_lo <= s_hi, f"positive series enclosure at C={c}")
    # G(pi/C) = (18*sqrt(2*pi)/C)*S, so squaring removes the radical.
    return 648*pi_lo*s_lo**2/c**2, 648*pi_hi*s_hi**2/c**2


def decimal_display(value):
    with localcontext() as context:
        context.prec = 24
        return str(Decimal(value.numerator) / Decimal(value.denominator))


def main():
    pi_lo, pi_hi = pi_bounds()
    require(F("3.14159265358979323846264338327") < pi_lo < pi_hi
            < F("3.14159265358979323846264338329"), "Machin enclosure for pi")
    for c, side in [(F(6), "above"), (F("6.9012574"), "above"),
                    (F("6.9012575"), "below"), (F("6.9013"), "below")]:
        lo, hi = squared_g_bounds(c, pi_lo, pi_hi)
        require(lo > 1 if side == "above" else hi < 1,
                f"G(pi/{decimal_display(c)}) is {side} one")
        if c == F("6.9013"):
            require(hi < F(99999, 100000), "G(pi/6.9013)^2 < 99999/100000")
        print("  squared G enclosure:", decimal_display(lo), decimal_display(hi))
    print("Certified: 6.9012574 < C_cos < 6.9012575 < 6.9013.")


if __name__ == "__main__":
    main()
