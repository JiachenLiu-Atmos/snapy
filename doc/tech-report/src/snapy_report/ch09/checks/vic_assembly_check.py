"""Executable check of the VIC block-tridiagonal assembly (chapter 7, _vic-assembly.qmd).

What is checked: the Roe flux decomposition (src/implicit/flux_decomposition_impl.h:37-126) and the assembly of
a, b, c (src/implicit/vic_assemble_full_impl.h:89-99, vic_assemble_partial_impl.h:110-120) at
snapy@6fbb39750f62b83df096058a4746e77d0a634bc5, ported in vic_port.py with grav = 0. Claims C1-C6; C6 is exact (sympy).
C4 is the frozen-m_t partial assembly (the kernel without du). C7 is the m_t fold of the partial assembly and the
right-hand-side term ForwardSweep subtracts (vic_assemble_partial_impl.h:174-214, forward_sweep_impl.h:34-40 at
snapy@c0c38593d3d3f37bebe204a590610fd7daa05a5f), ported in vic_port.fold_partial.

Run: python3 vic_assembly_check.py   (numpy and sympy; exits with the number of failed claims)
"""
import os
import sys

import numpy as np
import sympy as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import vic_port as vp  # noqa: E402


def report(tag, ok, label, numbers):
    print("[%s] %s %s: %s" % (tag, "PASS" if ok else "FAIL", label, numbers))
    return ok


rng = np.random.default_rng(7)


def random_state():
    return np.array([rng.uniform(0.2, 2.), *rng.uniform(-300., 300., 3), rng.uniform(1e4, 2e5)])


def check_C1_eigenvectors_invert():
    err = 0.
    for _ in range(200):
        g = rng.uniform(1.1, 1.67)
        p = vp.roe_average(g - 1., random_state(), random_state())
        R, Ri = vp.eigenvector(p, vp.sound_speed(p, g - 1.), g - 1.)
        err = max(err, np.abs(R @ Ri - np.eye(5)).max() / (np.abs(R).max() * np.abs(Ri).max()))
    return report("C1", err < 1e-14, "the coded left eigenvectors invert the right ones",
                  "max |R R^-1 - I| / (max|R| max|R^-1|) over 200 Roe states = %.1e" % err)


def check_C2_roe_matrix_and_roe_property():
    e1 = e2 = 0.
    for _ in range(200):
        g = rng.uniform(1.1, 1.67)
        wl, wr = random_state(), random_state()
        A = vp.abs_roe(g - 1., wl, wr, signed=True)
        J = vp.flux_jacobian(g - 1., vp.roe_average(g - 1., wl, wr))
        e1 = max(e1, np.abs(A - J).max() / np.abs(J).max())
        dF = vp.flux(wr, g) - vp.flux(wl, g)
        e2 = max(e2, np.abs(A @ (vp.cons(wr, g) - vp.cons(wl, g)) - dF).max() / np.abs(dF).max())
    return report("C2", e1 < 1e-12 and e2 < 1e-12,
                  "R Lambda R^-1 is the flux Jacobian at the Roe state, and A (q_R - q_L) = F_R - F_L for one gamma",
                  "max relative errors %.1e and %.1e over 200 pairs" % (e1, e2))


def column(n=6):
    W = np.array([random_state() for _ in range(n + 2)])
    W[:, vp.IVX] *= 0.2
    gam = rng.uniform(1.3, 1.45, n + 2)
    x = np.cumsum(rng.uniform(80., 120., n + 1))
    r = 6.0e5 + x                              # a spherical shell: areas grow outward
    area = r**2
    vol = (r[1:]**3 - r[:-1]**3) / 3.
    return W, gam, area, vol


def residual(Q, W, gam, area, vol, dt, Am, Ap):
    """Q/dt + (1/V)(A_{i+1} F_{i+1/2} - A_i F_{i-1/2}) for the interior, with the Roe dissipation frozen."""
    n = len(vol)
    P = [W[0]] + [vp.prim(Q[k], gam[k + 1]) for k in range(n)] + [W[-1]]
    q = [vp.cons(W[0], gam[0])] + list(Q) + [vp.cons(W[-1], gam[-1])]
    F = [vp.flux(P[i], gam[i]) for i in range(n + 2)]
    out = np.zeros((n, 5))
    for k in range(n):
        i = k + 1
        fm = 0.5 * (F[i - 1] + F[i]) - 0.5 * Am[k] @ (q[i] - q[i - 1])
        fp = 0.5 * (F[i] + F[i + 1]) - 0.5 * Ap[k] @ (q[i + 1] - q[i])
        out[k] = Q[k] / dt + (area[k + 1] * fp - area[k] * fm) / vol[k]
    return out


def check_C3_blocks_linearise_the_roe_divergence():
    W, gam, area, vol = column()
    n, dt = len(vol), 3.
    a, b, c = vp.assemble_full(W, gam, area, vol, dt, first_block=False, last_block=False)
    Am = [vp.abs_roe(0.5 * (gam[k] + gam[k + 1]) - 1., W[k], W[k + 1]) for k in range(n)]
    Ap = [vp.abs_roe(0.5 * (gam[k + 1] + gam[k + 2]) - 1., W[k + 1], W[k + 2]) for k in range(n)]
    Q0 = np.array([vp.cons(W[k + 1], gam[k + 1]) for k in range(n)])
    worst = 0.
    for k in range(n):
        for m in range(5):
            h = 1e-6 * max(abs(Q0[k, m]), 1.)
            Qp, Qm = Q0.copy(), Q0.copy()
            Qp[k, m] += h
            Qm[k, m] -= h
            col = (residual(Qp, W, gam, area, vol, dt, Am, Ap) - residual(Qm, W, gam, area, vol, dt, Am, Ap)) / (2 * h)
            for j, blk in ((k, a[k]), (k - 1, c[k - 1] if k > 0 else None), (k + 1, b[k + 1] if k + 1 < n else None)):
                if blk is None:
                    continue
                scale = np.abs(blk).max()
                worst = max(worst, np.abs(col[j] - blk[:, m]).max() / scale)
    return report("C3", worst < 1e-6,
                  "a, b, c are the Jacobian of Q/dt + the Roe-flux divergence with |A| frozen, on a spherical shell",
                  "max block error relative to the block size, central differences: %.1e" % worst)


def check_C4_partial_is_the_full_without_transverse_motion():
    W, gam, area, vol = column()
    n, dt = len(vol), 2.
    du = rng.normal(size=(n, 5)) * 1e-3
    errs = []
    for vt in (0., 50.):
        Wt = W.copy()
        Wt[:, vp.IVY] = vt
        Wt[:, vp.IVZ] = 0.
        a, b, c = vp.assemble_full(Wt, gam, area, vol, dt)
        _, af, df = vp.forward_sweep(a, b, c, du / dt)
        full = vp.backward_substitute(af, df)
        a3, b3, c3 = (np.array([m[np.ix_(vp.PART, vp.PART)] for m in x]) for x in (a, b, c))
        _, ap, dp = vp.forward_sweep(a3, b3, c3, du[:, vp.PART] / dt)
        part = vp.backward_substitute(ap, dp)
        errs.append(np.abs(part - full[:, vp.PART]).max() / np.abs(full[:, vp.PART]).max())
    ok = errs[0] < 1e-12 and errs[1] > 1e-6
    return report("C4", ok, "the frozen-m_t partial (3 x 3) solve equals the full one restricted to rho, rho u, E when "
                  "v = w = 0, "
                  "and differs when v != 0", "relative difference %.1e at v = 0, %.1e at v = 50 m/s" % tuple(errs))


def check_C5_solid_cell_is_inert():
    W, gam, area, vol = column()
    n = len(vol)
    solid = np.zeros(n + 2, bool)
    solid[3] = True
    a, b, c = vp.assemble_full(W, gam, area, vol, 2., solid=solid)
    k = 2
    ok = np.array_equal(a[k], np.eye(5) / 2.) and not b[k].any() and not c[k].any()
    Wm = W.copy()
    Wm[3] = random_state()
    a2, _, _ = vp.assemble_full(Wm, gam, area, vol, 2., solid=solid)
    same = np.abs(a2[1] - a[1]).max() / np.abs(a[1]).max()
    return report("C5", ok and same < 1e-14,
                  "a solid cell has a = I/dt, b = c = 0, and its fluid neighbour sees a mirror whatever the solid holds",
                  "solid row exact; neighbour block change when the solid cell's state is replaced: %.1e" % same)


def check_C6_blocks_from_the_row_residual():
    """eq-ch09-vic-blocks: differentiate U_i/dt_c + (A_hi F_hi - A_lo F_lo)/V_i - Phi U_i with A^+- frozen, on
    generic 2 x 2 blocks (the Euler flux is homogeneous of degree 1, F(U) = J(U) U)."""
    def mat(n):
        return sp.Matrix(2, 2, sp.symbols(n + "0:4"))

    def vec(n):
        return sp.Matrix(sp.symbols(n + "0:2"))

    Um, U0, Up = vec("um"), vec("u0"), vec("up")
    Am, Ap, Jm, J0, Jp, Phi = (mat(s) for s in ("am", "ap", "jm", "j0", "jp", "phi"))
    Alo, Ahi, V, dtc = sp.symbols("A_lo A_hi V dt_c", positive=True)
    F_lo = (Jm * Um + J0 * U0) / 2 - Am * (U0 - Um) / 2
    F_hi = (J0 * U0 + Jp * Up) / 2 - Ap * (Up - U0) / 2
    res = U0 / dtc + (Ahi * F_hi - Alo * F_lo) / V - Phi * U0
    stated = {  # eq-ch09-vic-blocks of _vic-assembly.qmd
        "A_i": (Alo * Am + Ahi * Ap + (Ahi - Alo) * J0) / (2 * V) - Phi + sp.eye(2) / dtc,
        "B_i": -Alo / (2 * V) * (Am + Jm),
        "C_i": -Ahi / (2 * V) * (Ap - Jp),
    }
    got = {"A_i": res.jacobian(U0), "B_i": res.jacobian(Um), "C_i": res.jacobian(Up)}
    ok = all((got[k] - stated[k]).applyfunc(sp.simplify) == sp.zeros(2) for k in stated)
    return report("C6", ok, "the derivatives of the row residual are the blocks of eq-ch09-vic-blocks, with I/dt_c on the "
                  "diagonal and the gravity Jacobian Phi subtracted", "A_i, B_i, C_i exact on generic 2 x 2 blocks")


def check_C7_folded_partial_carries_a_uniform_wind():
    """The m_t fold: with a uniform tangential wind and explicit increments that carry it (q_t = 0), the full solve
    keeps dm_t = v_t drho, so the folded partial solve equals the full one on rho, rho u, E; the frozen one does not."""
    errs = []
    for v2, v3 in ((50., 0.), (400., -30.)):
        W, gam, area, vol = column()
        n, dt = len(vol), 2.
        W[:, vp.IVY], W[:, vp.IVZ] = v2, v3
        du = rng.normal(size=(n, 5)) * 1e-3
        du[:, vp.IT] = W[1:-1][:, vp.IT] * du[:, [vp.IDN]]
        a, b, c = vp.assemble_full(W, gam, area, vol, dt)
        _, af, df = vp.forward_sweep(a, b, c, du / dt)
        full = vp.backward_substitute(af, df)
        keep = np.abs(full[:, vp.IT] - W[1:-1][:, vp.IT] * full[:, [vp.IDN]]).max() / np.abs(full[:, vp.IT]).max()
        a3, b3, c3 = (np.array([m[np.ix_(vp.PART, vp.PART)] for m in x]) for x in (a, b, c))
        _, ap, dp = vp.forward_sweep(a3, b3, c3, du[:, vp.PART] / dt)
        frozen = vp.backward_substitute(ap, dp)
        a3, b3, c3, corr = vp.fold_partial(a, b, c, W, du)
        _, ap, dp = vp.forward_sweep(a3, b3, c3, du[:, vp.PART] / dt - corr)
        fold = vp.backward_substitute(ap, dp)
        s = np.abs(full[:, vp.PART]).max()
        errs.append((keep, np.abs(fold - full[:, vp.PART]).max() / s, np.abs(frozen - full[:, vp.PART]).max() / s))
    ok = all(k < 1e-10 and f < 1e-10 and z > 1e-3 for k, f, z in errs)
    return report("C7", ok, "under a uniform tangential wind with q_t = 0 the full solve keeps dm_t = v_t drho and the "
                  "folded partial solve equals it on rho, rho u, E; the frozen-m_t partial solve does not",
                  "; ".join("v = (%g, %g) m/s: dm_t - v_t drho %.1e, folded %.1e, frozen %.1e relative"
                            % (v[0], v[1], *e) for v, e in zip(((50., 0.), (400., -30.)), errs)))


def main():
    checks = [check_C1_eigenvectors_invert, check_C2_roe_matrix_and_roe_property,
              check_C3_blocks_linearise_the_roe_divergence, check_C4_partial_is_the_full_without_transverse_motion,
              check_C5_solid_cell_is_inert,
              check_C6_blocks_from_the_row_residual, check_C7_folded_partial_carries_a_uniform_wind]
    sys.exit(sum(not c() for c in checks))


if __name__ == "__main__":
    main()
