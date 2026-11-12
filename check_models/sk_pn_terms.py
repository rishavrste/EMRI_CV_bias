"""The PN terms of each SuperKludge 1PA function exactly as written in few/trajectory/ode/SKequatorialfluxes.py.

Every 1PA function there returns a sum of four terms, each v^k (1-e^2)^1.5 (polynomial in chi1, chi2, e), e.g.
    pdot1PA = (v^10 term) + (v^9 term) - (v^8 term) + (v^7 term)
This module reads the source, splits the return expression into its top-level addends (keeping their signs), and
compiles each addend with the function's own preamble (chi1 = a, v = 1/sqrt(p), pi, sqrt). Nothing is refitted.

    pn_terms(channel, a, p, e, chi2) -> (4, ...) array, rows ordered by increasing power of v:
        row 0 = leading 1PA term, rows 1, 2, 3 = +0.5PN, +1PN, +1.5PN beyond it
    LABELS[channel] -> the v power of each row
"""
import ast
import inspect
import re

import numpy as np
from few.trajectory.ode import SKequatorialfluxes as SK

FUNCS = {"pdot": "pdot1PA", "edot": "edot1PA", "Omega_phi": "OmegaPhi1PA", "Omega_r": "Omegar1PA"}
ORDER_NAMES = ("leading 1PA", "+0.5PN", "+1PN", "+1.5PN")


def _addends(node, sign=1):
    """Top-level terms of a +/- chain, as (sign, ast node)."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        left = _addends(node.left, sign)
        right = _addends(node.right, sign if isinstance(node.op, ast.Add) else -sign)
        return left + right
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return [(-s, n) for s, n in _addends(node.operand, sign)]
    return [(sign, node)]


def _v_power(src):
    powers = {int(k) for k in re.findall(r"\bv\s*\*\*\s*(\d+)", src)}
    if len(powers) != 1:
        raise RuntimeError(f"term does not have a single power of v: {src[:80]}...")
    return powers.pop()


def _split(func_name):
    """[(v power, compiled term function)] for one SK 1PA function, sorted by power."""
    tree = ast.parse(inspect.getsource(getattr(SK, func_name).py_func))
    fn = tree.body[0]
    preamble = [s for s in fn.body if isinstance(s, ast.Assign)]
    ret = [s for s in fn.body if isinstance(s, ast.Return)][0].value
    terms = []
    for sign, node in _addends(ret):
        src = ast.unparse(node)
        body = preamble + [ast.Return(value=ast.BinOp(left=ast.Constant(sign), op=ast.Mult(), right=node))]
        f = ast.FunctionDef(name="term", args=fn.args, body=body, decorator_list=[], returns=None, type_params=[])
        mod = ast.fix_missing_locations(ast.Module(body=[f], type_ignores=[]))
        ns = {"np": np}
        exec(compile(mod, f"<{func_name} v^{_v_power(src)}>", "exec"), ns)
        terms.append((_v_power(src), ns["term"]))
    terms.sort(key=lambda t: t[0])
    if len(terms) != 4 or [k for k, _ in terms] != list(range(terms[0][0], terms[0][0] + 4)):
        raise RuntimeError(f"{func_name}: expected four consecutive powers of v, got {[k for k, _ in terms]}")
    return terms


TERMS = {ch: _split(fn) for ch, fn in FUNCS.items()}
LABELS = {ch: [f"v^{k}" for k, _ in t] for ch, t in TERMS.items()}


def pn_terms(channel, a, p, e, chi2):
    """(4, ...) the four PN terms of a 1PA function as written, by increasing power of v; p, e may be arrays."""
    p, e = np.broadcast_arrays(np.asarray(p, float), np.asarray(e, float))
    return np.array([np.broadcast_to(f(a, p, e, chi2), p.shape) for _, f in TERMS[channel]])


def self_check(a, p, e, chi2, rtol=1e-12):
    """The four written terms add up to the full SK function in every channel."""
    for ch, fn in FUNCS.items():
        full = getattr(SK, fn)(a, p, e, chi2)
        if not np.isclose(pn_terms(ch, a, p, e, chi2).sum(), full, rtol=rtol, atol=0):
            raise RuntimeError(f"PN terms of {ch} do not add up at a={a} p={p} e={e} chi2={chi2}")
