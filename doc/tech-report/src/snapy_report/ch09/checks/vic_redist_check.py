"""Executable check of the VIC constituent redistribution (chapter 9, _vic-redistribute.qmd).

What is checked: vic_constituent_column and vic_redistribute_cell (src/implicit/vic_redistribute_impl.h:92-176,
205-225) and the clamp-residual diagnostic (src/implicit/implicit_hydro.cpp:583-607) at
snapy@73c45a7c8c991209924e1137992352c335575062, ported in vic_port.py. A column of 12 layers with dry gas and
two species. Claims C1-C6; C6 is exact (sympy). C7 is the tangential-momentum carry of the partial matrix,
vic_tangential_momentum_column (src/implicit/vic_redistribute_impl.h:186-201), ported in
vic_port.tangential_momentum_column.

Run: python3 vic_redist_check.py   (numpy and sympy; exits with the number of failed claims)
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


rng = np.random.default_rng(17)
N = 12


def column(scale=1e-3, dry_y=None):
    rho = np.exp(-np.linspace(0., 2., N))
    y = np.stack([0.02 * np.exp(-np.linspace(0., 4., N)), 0.005 + 0.004 * np.sin(np.linspace(0., 3., N))], 1)
    if dry_y is not None:
        y[:] = dry_y
    w = np.concatenate([rho[:, None], y], 1)                     # IDN = total density, then mass fractions
    vol = rng.uniform(80., 120., N)
    du = np.concatenate([rng.normal(size=(N, 1)) * scale * rho[:, None],
                         rng.normal(size=(N, 2)) * scale * 0.01 * rho[:, None]], 1)
    delta0 = du.sum(1) + rng.normal(size=N) * scale * rho        # the solved total-mass tendency
    return du, w, delta0, vol


def check_C1_top_face_telescopes_to_zero():
    worst = 0.
    for _ in range(50):
        du, w, d0, vol = column()
        M, mass, mark, moved, top, R = vp.constituent_column(du, w, d0, vol)
        worst = max(worst, abs(top) / np.abs(M).max())
    return report("C1", worst < 1e-13, "after the residual R is removed in proportion to cell mass, the top-face "
                  "transfer telescopes to zero", "max |M_top| / max |M| over 50 columns = %.1e" % worst)


def check_C2_every_constituent_is_conserved():
    worst = 0.
    for _ in range(50):
        du, w, d0, vol = column()
        M, mass, mark, moved, top, R = vp.constituent_column(du, w, d0, vol)
        tot = np.abs((mass * vol[:, None]).sum(0)) / (np.abs(mass) * vol[:, None]).sum(0)
        worst = max(worst, tot.max())
    return report("C2", worst < 1e-14, "dry gas and each species keep their column total: every face transfer is "
                  "added to one cell and taken from the other", "max |sum V dm| / sum V |dm| = %.1e" % worst)


def check_C3_cell_change_is_the_face_divergence():
    du, w, d0, vol = column()
    M, mass, mark, moved, top, R = vp.constituent_column(du, w, d0, vol)
    Mu = np.append(M[1:], 0.)
    cell = (mass * vol[:, None]).sum(1)
    res = np.abs(cell - (M - Mu)) / np.maximum(np.abs(M), np.abs(Mu)).clip(1e-300)
    phi = (d0 - du.sum(1)) * vol
    want = phi - R * w[:, 0] * vol / (w[:, 0] * vol).sum()
    err = np.abs((M - Mu) - want).max() / np.abs(want).max()
    ok = res.max() < 1e-12 and err < 1e-12 and not mark.any()
    return report("C3", ok, "without a clamp each cell's total change is M_i - M_(i+1) = phi_i - R m_i / sum m: "
                  "the clamp residual is round-off", "clamp residual %.1e; relative error of the face divergence %.1e"
                  % (res.max(), err))


def check_C4_drained_donor_stays_nonnegative():
    du, w, d0, vol = column(scale=1e-3)
    d0 = du.sum(1).copy()
    d0[0] -= 5. * w[0, 0]                          # a large upward push out of the bottom cell
    d0[1:] += 5. * w[0, 0] * vol[0] / vol[1:].sum()
    M, mass, mark, moved, top, R = vp.constituent_column(du, w, d0, vol)
    dry0 = w[:, 0] * (1 - w[:, 1:].sum(1)) + du[:, 0] + mass[:, 0]
    sp0 = w[:, :1] * w[:, 1:] + du[:, 1:] + mass[:, 1:]
    Mu = np.append(M[1:], 0.)
    cell = (mass * vol[:, None]).sum(1)
    res = (np.abs(cell - (M - Mu)) / np.maximum(np.abs(M), np.abs(Mu)).clip(1e-300)).max()
    ok = mark[0] == 1 and dry0.min() > -8 * np.finfo(float).eps * w[0, 0] and sp0.min() > 0 and res > 1e-3
    return report("C4", ok, "a face asking for more than the donor holds is clamped: the dry donor empties to round-off "
                  "(no margin), the species keep their margin, the dry clamp mark is set and the clamp residual "
                  "shows it", "mark at the bottom %d; min dry %.3g, "
                  "min species %.3g; clamp residual %.3g" % (mark[0], dry0.min(), sp0.min(), res))


def check_C5_species_keep_margin():
    keep64 = 1 - 4096 * np.finfo(np.float64).eps
    keep32 = 1 - np.float32(4096) * np.finfo(np.float32).eps
    ok = abs((1 - keep64) - 9.094947017729282e-13) < 1e-28 and abs(float(1 - keep32) - 0.00048828125) < 1e-12
    return report("C5", ok, "a species donor keeps 4096 ulp of its density: the margin is 1 - keep",
                  "double %.6g, float32 %.6g" % (1 - keep64, float(1 - keep32)))


def check_C6_telescoping_in_closed_form():
    """Derivation of _vic-redistribute.qmd: summing eq-ch09-vic-faces over the n cells gives
    T_top = -sum phi_i + DM sum m_i / sum m = 0, with phi_i the cell's extra mass and DM = sum phi_i."""
    n = 6
    phi = sp.symbols("phi0:%d" % n)
    m = sp.symbols("m0:%d" % n, positive=True)
    DM = sum(phi)
    T = sp.Integer(0)
    for i in range(n):
        T = T - (phi[i] - DM * m[i] / sum(m))
    ok = sp.simplify(T) == 0
    return report("C6", ok, "the face transfers of eq-ch09-vic-faces telescope: T at the top face is -DM + DM = 0 in "
                  "closed form", "6 cells, symbolic: T_top = %s" % sp.simplify(T))


def check_C7_tangential_momentum_rides_on_the_transfers():
    """Partial matrix (N = 3): every face moves its total transfer times the donor's tangential velocity. Each
    column sum is unchanged; a uniform wind stays uniform when the explicit increments carry it and the solve adds
    no net column mass (no clamp); without the carry it does not; a wind-free column is left bit for bit."""
    du, w, d0, vol = column()
    d0 = d0 - ((d0 - du.sum(1)) * vol).sum() / vol.sum()          # Delta M_col = 0
    M, mass, mark, moved, top, R = vp.constituent_column(du, w, d0, vol)
    rho = w[:, 0]
    v = np.array([400., -30.])
    du_t = du.sum(1)[:, None] * v                                  # explicit increments that carry the wind
    carried = vp.tangential_momentum_column(du_t.copy(), np.tile(v, (N, 1)), moved, vol)
    rho1 = rho + du.sum(1) + mass.sum(1)
    vel = (rho[:, None] * v + carried) / rho1[:, None]
    vel0 = (rho[:, None] * v + du_t) / rho1[:, None]
    e_vel = np.abs(vel - v).max() / np.abs(v).max()
    e_vel0 = np.abs(vel0 - v).max() / np.abs(v).max()
    flux = np.abs((carried - du_t) * vol[:, None]).sum(0)
    e_sum = (np.abs(((carried - du_t) * vol[:, None]).sum(0)) / flux).max()
    zero = np.full((N, 2), -0.)
    still = vp.tangential_momentum_column(zero.copy(), np.zeros((N, 2)), moved, vol)
    bitwise = np.array_equal(still, zero) and np.signbit(still).all()
    ok = mark.max() == 0 and e_vel < 1e-12 and e_sum < 1e-14 and e_vel0 > 1e-6 and bitwise
    return report("C7", ok, "the partial matrix's tangential momentum rides on the clamped face transfers at the donor's "
                  "velocity: column sums kept, a uniform wind kept, a wind-free column untouched",
                  "v = (400, -30) m/s, Delta M_col = 0, no clamp: velocity change %.1e relative with the carry, %.1e "
                  "without; column sum %.1e of the moved momentum; wind-free -0. kept: %s"
                  % (e_vel, e_vel0, e_sum, bitwise))


def main():
    checks = [check_C1_top_face_telescopes_to_zero, check_C2_every_constituent_is_conserved,
              check_C3_cell_change_is_the_face_divergence, check_C4_drained_donor_stays_nonnegative,
              check_C5_species_keep_margin,
              check_C6_telescoping_in_closed_form, check_C7_tangential_momentum_rides_on_the_transfers]
    sys.exit(sum(not c() for c in checks))


if __name__ == "__main__":
    main()
