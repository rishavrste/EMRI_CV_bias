"""Damped Levenberg-Marquardt (iterated Cutler-Vallisneri) climb.

chi2 = <s-h|s-h>; Gauss-Newton Hessian G_ij = <d_i h|d_j h>, gradient g_i = <d_i h|s-h>;
the damped step solves (G + lam diag G) delta = g, with Nielsen gain-ratio updates of lam.

With C.REGULARISE the damped matrix A = G + lam diag G is inverted explicitly and regularised,
    eps = ||A A^-1 - 1||_F,    Sigma_reg = A^-1 + eps diag(A^-1),    delta = Sigma_reg g,
so the added diagonal baseline grows with the inversion error.
"""
import numpy as np

import config as C


def project_physical(names, vec):
    out = np.array(vec, dtype=float)
    for i, nm in enumerate(names):
        out[i] = min(max(out[i], C.PHYS_LO.get(nm, -np.inf)), C.PHYS_HI.get(nm, np.inf))
    return out


def sigma_from_fisher(G):
    """1-sigma widths, inverting G in Jacobi-rescaled coordinates."""
    d = np.sqrt(np.abs(np.diag(G)))
    if np.any(d <= 0):
        return np.full(len(G), np.nan)
    try:
        C_hat = np.linalg.inv(G / np.outer(d, d))
    except np.linalg.LinAlgError:
        return np.full(len(G), np.nan)
    return np.sqrt(np.abs(np.diag(C_hat))) / d


def fisher_and_gradient(P, dH, r):
    inner, n = P["inner"], len(dH)
    G = np.array([[inner(dH[i], dH[j]) for j in range(n)] for i in range(n)])
    g = np.array([inner(dH[j], r) for j in range(n)])
    return G, g


def plain_delta(A, g):
    """Damped step from a linear solve; no regularisation (eps = 0)."""
    return np.linalg.solve(A, g), 0.0


def regularised_delta(A, g):
    """Damped step through the regularised inverse Sigma_reg = A^-1 + eps diag(A^-1)."""
    A_inv = np.linalg.inv(A)
    eps = float(np.linalg.norm(A @ A_inv - np.eye(len(A)), "fro"))
    sigma_reg = A_inv + eps * np.diag(np.diag(A_inv))
    return sigma_reg @ g, eps


def lm_step(P, cur, G, g, c0, lam):
    """One damped step: shrink lam on a gain (Nielsen), raise it on a loss; None if none helps.
    Also returns the eps of the accepted (or last tried) regularised inverse."""
    dvec = np.abs(np.diag(G)) + 1e-30
    step_delta = regularised_delta if C.REGULARISE else plain_delta
    eps = float("nan")
    for attempt in range(2):
        if attempt == 1:
            lam = C.LAMBDA0                 # one restart from a sane damping before giving up
        for _ in range(C.MAX_INNER):
            try:
                delta, eps = step_delta(G + lam * np.diag(dvec), g)
            except np.linalg.LinAlgError:
                lam = min(lam * C.LAMBDA_UP, C.LAMBDA_MAX)
                continue
            pred = (float(2.0 * delta @ g - delta @ G @ delta) if C.REGULARISE   # model decrease
                    else float(delta @ (g + lam * dvec * delta)))          # = same, exact solve
            cand = project_physical(P["names"], cur + delta)
            c1 = P["chi2r"](cand)
            gain = (c0 - c1) / pred if pred > 0 else -1.0
            if gain > 0.0:
                lam = max(lam * max(1.0 / 3.0, 1.0 - (2.0 * gain - 1.0) ** 3), C.LAMBDA_MIN)
                return cand, c1, lam, eps
            lam = min(lam * C.LAMBDA_UP, C.LAMBDA_MAX)
    return None, c0, lam, eps


def lm_climb(P, theta0, tag, on_iter=None):
    """Iterate lm_step from theta0; on_iter(cur, sigma, history) is called after every iteration
    so the caller can checkpoint. Returns final theta, sigma, info and the per-iteration history."""
    inner, s = P["inner"], P["s"]
    cur = project_physical(P["names"], theta0)
    lam, deltas, sigma = C.LAMBDA0, None, np.full(len(P["names"]), np.nan)
    history, n_accept, stop = [], 0, "max_iters"

    for it in range(C.LM_MAX_ITERS):
        if it % C.RECOMPUTE_DELTAS_EVERY == 0:
            deltas = None                   # re-scan SEF's stable steps
        dH, found = P["derivs"](cur, deltas)
        deltas = found if deltas is None else deltas
        h = P["make_h"](cur)
        r = P["residual"](h)
        G, g = fisher_and_gradient(P, dH, r)
        sigma = sigma_from_fisher(G)
        c0 = inner(r, r)
        ovc = inner(s, h) / np.sqrt(inner(s, s) * inner(h, h))
        rec = dict(it=it, theta=[float(v) for v in cur], overlap=float(ovc), chi2=float(c0),
                   lam=float(lam))
        if ovc > C.OVERLAP_TARGET:
            history.append(rec)
            stop = "overlap_target"
            break

        nxt, c1, lam, eps = lm_step(P, cur, G, g, c0, lam)
        rel = (c0 - c1) / c0 if nxt is not None else 0.0
        rec.update(rel=float(rel), lam_after=float(lam), accepted=nxt is not None, eps=eps)
        history.append(rec)
        print(f"    LM {tag} it {it:>3} lam={lam:.1e} ov={ovc:.12f} chi2={c0:.6e} rel={rel:.1e}"
              + (f" eps={eps:.2e}" if C.REGULARISE else ""), flush=True)
        if nxt is None:
            stop = "lambda_exhausted"
            break
        cur, n_accept = nxt, n_accept + 1
        if on_iter is not None:
            on_iter(cur, sigma, history)
        if rel < C.REL_TOL:
            stop = "rel_tol"
            break

    info = dict(n_iter=len(history), n_accept=n_accept, lam_final=float(lam), stop_reason=stop)
    return cur, sigma, info, history
