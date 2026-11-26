"""A numpy port of the VIC kernels, shared by the part 7B checks of chapter 7.

Ported line for line from snapy@e894700ff7aee30b52882e5202b16461413780b0, direction dir = 0 (x1) only and no
gravity (grav = 0): the gravity rows of the operator belong to the gravity chapter.

  src/implicit/flux_decomposition_impl.h    RoeAverage, Eigenvalue, Eigenvector, FluxJacobian
  src/implicit/vic_assemble_full_impl.h     vic_assemble_full_impl (a, b, c and the column closure)
  src/implicit/vic_assemble_partial_impl.h  vic_assemble_partial_impl (the 3 x 3 rows IDN, IVX, IPR)
  src/implicit/forward_sweep_impl.h         ForwardSweep
  src/implicit/vic_redistribute_impl.h      vic_backward_substitute, vic_constituent_column
  at snapy@c0c38593d3d3f37bebe204a590610fd7daa05a5f, for the partial matrix (N = 3) only:
  src/implicit/vic_assemble_partial_impl.h  the m_t fold of vic_assemble_partial_impl (du and rhs_corr set)
  src/implicit/forward_sweep_impl.h         ForwardSweep's subtraction of that right-hand-side term
  src/implicit/vic_redistribute_impl.h      vic_tangential_momentum_column
  src/math/ludcmp.h                         ludcmp

Index layout as in snapy: IDN 0, IVX 1, IVY 2, IVZ 3, IPR 4; species ICY = 5 onward in the per-cell arrays.
"""
import numpy as np

IDN, IVX, IVY, IVZ, IPR = 0, 1, 2, 3, 4
PART = [IDN, IVX, IPR]                     # the rows and columns the partial (3 x 3) matrix keeps


def sound_speed(prim, gm1):
    return np.sqrt(prim[IPR] * (gm1 + 1.) / prim[IDN])


def roe_average(gm1, wl, wr):
    sl, sr = np.sqrt(wl[IDN]), np.sqrt(wr[IDN])
    isd = 1. / (sl + sr)
    prim = np.zeros(5)
    prim[IDN] = sl * sr
    for n in (IVX, IVY, IVZ):
        prim[n] = (sl * wl[n] + sr * wr[n]) * isd
    el = wl[IPR] / gm1 + 0.5 * wl[IDN] * (wl[IVX]**2 + wl[IVY]**2 + wl[IVZ]**2)
    er = wr[IPR] / gm1 + 0.5 * wr[IDN] * (wr[IVX]**2 + wr[IVY]**2 + wr[IVZ]**2)
    hbar = ((el + wl[IPR]) / sl + (er + wr[IPR]) / sr) * isd
    prim[IPR] = (hbar - 0.5 * (prim[IVX]**2 + prim[IVY]**2 + prim[IVZ]**2)) * gm1 / (gm1 + 1.) * prim[IDN]
    return prim


def eigenvalue(u, cs, signed=False):
    lam = np.array([u - cs, u, u + cs, u, u])
    return lam if signed else np.abs(lam)


def eigenvector(prim, cs, gm1):
    r, u, v, w, p = prim
    ke = 0.5 * (u * u + v * v + w * w)
    hp = (gm1 + 1.) / gm1 * p / r
    h = hp + ke
    R = np.array([[1., 1., 1., 0., 0.],
                  [u - cs, u, u + cs, 0., 0.],
                  [v, v, v, 1., 0.],
                  [w, w, w, 0., 1.],
                  [h - u * cs, ke, h + u * cs, v, w]])
    Ri = np.array([[(cs * ke + u * hp) / (2. * cs * hp), (-hp - cs * u) / (2. * cs * hp),
                    -v / (2. * hp), -w / (2. * hp), 1. / (2. * hp)],
                   [(hp - ke) / hp, u / hp, v / hp, w / hp, -1. / hp],
                   [(cs * ke - u * hp) / (2. * cs * hp), (hp - cs * u) / (2. * cs * hp),
                    -v / (2. * hp), -w / (2. * hp), 1. / (2. * hp)],
                   [-v, 0., 1., 0., 0.],
                   [-w, 0., 0., 1., 0.]])
    return R, Ri


def flux_jacobian(gm1, w):
    rho, v1, v2, v3, pres = w
    s2 = v1 * v1 + v2 * v2 + v3 * v3
    c1 = ((gm1 - 1) * s2 / 2 - (gm1 + 1) / gm1 * pres / rho) * v1
    c2 = (gm1 + 1) / gm1 * pres / rho + s2 / 2 - gm1 * v1 * v1
    return np.array([[0, 1., 0., 0., 0.],
                     [gm1 * s2 / 2 - v1 * v1, (2. - gm1) * v1, -gm1 * v2, -gm1 * v3, gm1],
                     [-v1 * v2, v2, v1, 0., 0.],
                     [-v1 * v3, v3, 0., v1, 0.],
                     [c1, c2, -gm1 * v2 * v1, -gm1 * v3 * v1, (gm1 + 1) * v1]])


def abs_roe(gm1, wl, wr, signed=False):
    """|A| = R |Lambda| R^-1 at the Roe average of wl, wr (signed=True gives A itself)."""
    prim = roe_average(gm1, wl, wr)
    cs = sound_speed(prim, gm1)
    R, Ri = eigenvector(prim, cs, gm1)
    return R @ np.diag(eigenvalue(prim[IVX], cs, signed)) @ Ri


def cons(w, gamma):
    """Conserved (rho, rho u, rho v, rho w, E) of a primitive (rho, u, v, w, p)."""
    rho, u, v, ww, p = w
    return np.array([rho, rho * u, rho * v, rho * ww, p / (gamma - 1) + 0.5 * rho * (u * u + v * v + ww * ww)])


def prim(q, gamma):
    rho = q[0]
    u, v, w = q[1:4] / rho
    return np.array([rho, u, v, w, (gamma - 1) * (q[4] - 0.5 * rho * (u * u + v * v + w * w))])


def flux(w, gamma):
    rho, u, v, ww, p = w
    E = cons(w, gamma)[4]
    return np.array([rho * u, rho * u * u + p, rho * u * v, rho * u * ww, u * (E + p)])


def assemble_full(W, gam, area, vol, dt, solid=None, first_block=True, last_block=True, periodic=False):
    """a, b, c for cells 1..n of a column with one ghost at each end: W has shape (n + 2, 5), gam (n + 2,),
    area (n + 1,) for the faces of the interior cells, vol (n,). grav = 0. The column closure of lines
    122-126 is applied at the first and the last interior cell."""
    n = W.shape[0] - 2
    solid = np.zeros(n + 2, bool) if solid is None else solid
    a, b, c = np.zeros((n, 5, 5)), np.zeros((n, 5, 5)), np.zeros((n, 5, 5))
    Bnd = np.diag([1., -1., 1., 1., 1.])
    for k in range(n):
        i = k + 1
        if solid[i]:
            a[k] = np.eye(5) / dt
            continue
        lo, up = solid[i - 1], solid[i + 1]
        wl, wr = W[i - 1].copy(), W[i].copy()
        if lo:
            wl = wr.copy(); wl[IVX] = -wr[IVX]
        gl = gam[i] if lo else gam[i - 1]
        dprev = flux_jacobian(gl - 1., wl)
        dcurr = flux_jacobian(gam[i] - 1., wr)
        Am = abs_roe(0.5 * (gl + gam[i]) - 1., wl, wr)
        wl, wr = W[i].copy(), W[i + 1].copy()
        if up:
            wr = wl.copy(); wr[IVX] = -wl[IVX]
        gu = gam[i] if up else gam[i + 1]
        dnext = flux_jacobian(gu - 1., wr)
        Ap = abs_roe(0.5 * (gam[i] + gu) - 1., wl, wr)
        h = 0.5 / vol[k]
        a[k] = (Am * area[k] + Ap * area[k + 1] + (area[k + 1] - area[k]) * dcurr) * h + np.eye(5) / dt
        b[k] = -(Am + dprev) * area[k] * h
        c[k] = -(Ap - dnext) * area[k + 1] * h
        if (k == 0 or lo) and first_block and not periodic:
            a[k] += b[k] @ Bnd
        if (k == n - 1 or up) and last_block and not periodic:
            a[k] += c[k] @ Bnd
    return a, b, c


def ludcmp(a, dtype=np.float64):
    """Crout LU with scaled partial pivoting and the 8 N eps pivot guard. Returns (d, lu, indx); d = 0 is a
    failure (singular, near-singular or nonfinite)."""
    a = np.array(a, dtype=dtype)
    N = a.shape[0]
    tol = dtype(8 * N) * np.finfo(dtype).eps
    indx = np.zeros(N, int)
    if not np.isfinite(a).all():
        return 0, a, indx
    d, vv, scale = 1, np.zeros(N, dtype), np.zeros(N, dtype)
    with np.errstate(all="ignore"):
        for i in range(N):
            big = np.abs(a[i]).max()
            if big == 0.:
                return 0, a, indx
            scale[i] = big
            vv[i] = dtype(1.) / big
            if not np.isfinite(vv[i]):
                return 0, a, indx
        for j in range(N):
            for i in range(j):
                s = a[i, j] - sum((a[i, k] * a[k, j] for k in range(i)), dtype(0))
                if not np.isfinite(s):
                    return 0, a, indx
                a[i, j] = s
            big, imax = dtype(0), j
            for i in range(j, N):
                s = a[i, j] - sum((a[i, k] * a[k, j] for k in range(j)), dtype(0))
                if not np.isfinite(s):
                    return 0, a, indx
                a[i, j] = s
                dum = vv[i] * abs(s)
                if dum >= big:
                    big, imax = dum, i
            if j != imax:
                a[[imax, j]] = a[[j, imax]]
                d = -d
                vv[imax] = vv[j]
                scale[[imax, j]] = scale[[j, imax]]
            indx[j] = imax
            if abs(a[j, j]) / scale[j] <= tol:
                return 0, a, indx
            if j != N - 1:
                a[j + 1:, j] *= dtype(1.) / a[j, j]
    if not np.isfinite(a).all():
        return 0, a, indx
    return d, a, indx


def forward_sweep(a, b, c, rhs):
    """ForwardSweep for one column: a, b, c of shape (n, N, N), rhs (n, N) (already divided by dt). Returns
    (ok, a', delta); on a failure every delta is NaN (vic_fail_column)."""
    n, N = rhs.shape
    a, delta = a.copy(), np.zeros((n, N))

    def fail():
        return False, a, np.full((n, N), np.nan)

    for i in range(n):
        if i > 0:
            a[i] = a[i] - b[i] @ a[i - 1]
        d, _, _ = ludcmp(a[i])
        if d == 0:
            return fail()
        r = rhs[i] - (b[i] @ delta[i - 1] if i > 0 else 0.)
        inv = np.linalg.inv(a[i])
        delta[i] = inv @ r
        a[i] = inv @ c[i]
        if not (np.isfinite(delta[i]).all() and np.isfinite(a[i]).all()):
            return fail()
    return True, a, delta


def backward_substitute(a, delta):
    delta = delta.copy()
    if not np.isfinite(delta[0]).all():
        return delta                       # a failed column is never back-substituted
    for i in range(len(delta) - 2, -1, -1):
        delta[i] -= a[i] @ delta[i + 1]
    return delta


def constituent_column(du, w, delta0, vol, dtype=np.float64):
    """vic_constituent_column for one column. du[n, ch] and w[n, ch] hold IDN in channel 0 and the species
    (mass fractions in w, density tendencies in du) in channels 1..ny; delta0[n] is the solved total-mass
    tendency delta_i(0). Returns face transfer M (n,), the dry and species increments mass (n, 1 + ny), the
    dry clamp marks (n,) and the total mass each face moved (n,)."""
    n, nch = du.shape
    ny = nch - 1
    M = np.zeros(n, dtype)
    mass = np.zeros((n, nch), dtype)
    mark = np.zeros(n, dtype)
    moved = np.zeros(n, dtype)
    phi = np.zeros(n, dtype)
    R = S = dtype(0)
    for i in range(n):                                     # pass 1
        phi[i] = (delta0[i] - du[i].sum()) * vol[i]
        R += phi[i]
        S += w[i, 0] * vol[i]
    m = dtype(0)
    for i in range(n):                                     # pass 2
        M[i] = m
        m -= phi[i] - R * (w[i, 0] * vol[i] / S)
    top = m
    dry = 1 - w[0, 1:].sum()                               # pass 3a
    avail = max((w[0, 0] * dry + du[0, 0]) * vol[0], 0)
    for i in range(n - 1):
        dry_up = 1 - w[i + 1, 1:].sum()
        avail_up = max((w[i + 1, 0] * dry_up + du[i + 1, 0]) * vol[i + 1], 0)
        q = M[i + 1] * dry if M[i + 1] > 0 else M[i + 1] * dry_up
        if q > avail:
            q, mark[i] = avail, 1
        if q < -avail_up:
            q, mark[i + 1] = -avail_up, 1
        mass[i, 0] -= q / vol[i]
        mass[i + 1, 0] += q / vol[i + 1]
        moved[i + 1] = q
        avail, dry = avail_up + q, dry_up
    keep = 1 - dtype(4096) * np.finfo(dtype).eps           # pass 3b
    for s in range(1, ny + 1):
        avail = max((w[0, 0] * w[0, s] * keep + du[0, s]) * vol[0], 0)
        for i in range(n - 1):
            avail_up = max((w[i + 1, 0] * w[i + 1, s] * keep + du[i + 1, s]) * vol[i + 1], 0)
            q = M[i + 1] * w[i, s] if M[i + 1] > 0 else M[i + 1] * w[i + 1, s]
            q = min(q, avail)
            q = max(q, -avail_up)
            mass[i, s] -= q / vol[i]
            mass[i + 1, s] += q / vol[i + 1]
            moved[i + 1] += q
            avail = avail_up + q
    return M, mass, mark, moved, top, R


IT = [IVY, IVZ]                            # the tangential momenta t = dir + 1, dir + 2 for dir = 0


def fold_partial(a, b, c, W, du):
    """vic_assemble_partial_impl with du and rhs_corr set, for the column of assemble_full (no solid cells).

    The kernel's aT, bT, cT are the m_t columns of the rows IDN, IVX, IPR of the same terms as a[i], b[i], c[i]
    (Phi and the 1/dt diagonal have none there), so they are the m_t columns of the 5 x 5 blocks of assemble_full;
    the wall mirror keeps m_t, so a face merged into a[i] is already in a's columns. Each block's rho column gains
    its m_t columns times v_t of the cell it multiplies, and rhs_corr[i] = aT q_i + bT q_{i-1} + cT q_{i+1} with
    q_t,j = du_t,j - v_t,j du_rho,j; ForwardSweep then solves with rhs du/dt - rhs_corr. Returns (a3, b3, c3,
    rhs_corr). tests/test_vic_partial_tangential.cpp checks the C++ kernel against this reduction."""
    n = len(a)
    v = W[1:-1][:, IT]
    q = du[:, IT] - v * du[:, [IDN]]
    a3, b3, c3 = (np.array([m[np.ix_(PART, PART)] for m in x]) for x in (a, b, c))
    corr = np.zeros((n, 3))
    for k in range(n):
        for blk, red, j in ((a[k], a3[k], k), (b[k], b3[k], k - 1), (c[k], c3[k], k + 1)):
            if not 0 <= j < n:
                continue                   # a merged wall face: cell k's v_t and q_t, already in a[k]
            tb = blk[np.ix_(PART, IT)]
            for t in range(2):
                if v[j, t] != 0:
                    red[:, 0] += tb[:, t] * v[j, t]
                if q[j, t] != 0:
                    corr[k] += tb[:, t] * q[j, t]
    return a3, b3, c3, corr


def tangential_momentum_column(du_t, w_t, moved, vol):
    """vic_tangential_momentum_column for one column: du_t[n, 2] the tangential momentum tendencies (updated in
    place and returned), w_t[n, 2] the tangential velocities, moved[i] the total clamped transfer through the face
    below cell i (the `moved` output of constituent_column), vol[n]."""
    n = len(vol)
    for i in range(n - 1):
        q = moved[i + 1]
        donor = i if q > 0 else i + 1
        for t in range(2):
            p = q * w_t[donor, t]
            if p == 0:
                continue
            du_t[i, t] -= p / vol[i]
            du_t[i + 1, t] += p / vol[i + 1]
    return du_t
