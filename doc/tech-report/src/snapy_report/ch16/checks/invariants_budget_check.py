"""Global budgets over many steps: mass, total energy, and the cubed sphere's
axial angular momentum (chapter 15, `_invariants.qmd`).

Ports, statement for statement, the snapy operators that decide those three
budgets, at the code pin of `code_pin.yml`,
snapy@ccf40f9565642c32be69f4b9ef0cfa6daff3f201:

  CoordinateImpl::divergence           src/coord/coordinate.cpp:543-586
  GnomonicEquiangleImpl::reset         src/coord/gnomonic_equiangle.cpp:18-197
    (solid_angle_kj :126-136; the face metrics :70-118; x_ov_rD_kji and
     y_ov_rC_kji :138-145)
  GnomonicEquiangleImpl::cell_volume   src/coord/gnomonic_equiangle.cpp:219-227
  GnomonicEquiangleImpl::face_area1-3  :207-217
  GnomonicEquiangleImpl::flux2global2_ :358-385
  GnomonicEquiangleImpl::forward       :415-473
  cs_contra_to_cart_                   src/coord/cubed_sphere_utils.cpp:221-243
  cs_face_xyz_from_tan                 src/layout/cubed_sphere_constants.h:41-76
  cs_local_to_cart_vel                 src/layout/cubed_sphere_constants.h:33-39
  coord_vec_raise_impl                 src/coord/coord_utils_impl.h:8-16

numpy only; no compiled snapy is executed. The ported geometry and the ported
velocity transform were compared cell by cell with the same functions of a
build at the pin; that comparison is a run record, not part of this check
(`src/snapy_report/ch16/evidence/`).

Each budget is accumulated over many steps and reported as a drift per step,
and each has a break arm that removes one term and makes the claim red.

Run (green):   PYTHONPATH=src python3 invariants_budget_check.py
Run (one red): PYTHONPATH=src python3 invariants_budget_check.py --break B1-seam
List the arms: PYTHONPATH=src python3 invariants_budget_check.py --list-breaks

Exits with the number of failed claims; a break arm is expected to exit nonzero.
"""
import math
import sys

import numpy as np

EPS = np.finfo(np.float64).eps
RI, RO = 6.0e6, 6.4e6          # the shell of the chapter's cubed-sphere deck
NFACE = 6
SEAM_GAP = 1.0e-3              # the seam disagreement of the [B1] break arm

FAIL = []


def report(tag, ok, what, detail):
    print(f"[{tag}] {'PASS' if ok else 'FAIL'} {what}: {detail}")
    if not ok:
        FAIL.append(tag)
    return ok


# =============================================================================
# the ported geometry of one gnomonic-equiangular panel
# =============================================================================
class Panel:
    """One cubed-sphere panel's geometry, in snapy's [k, j, i] layout.

    Index order and every formula follow GnomonicEquiangleImpl::reset and the
    area and volume accessors; `volume_rule='trapezoid'` is the break arm that
    replaces the exact radial integral by the trapezoid of the two face areas.
    """

    def __init__(self, n_ang, n_rad, face_id, volume_rule="exact"):
        self.n_ang, self.n_rad, self.face_id = n_ang, n_rad, face_id
        d = math.pi / (2.0 * n_ang)
        self.x2f = -math.pi / 4.0 + d * np.arange(n_ang + 1)
        self.x3f = self.x2f.copy()
        self.x2v = 0.5 * (self.x2f[:-1] + self.x2f[1:])
        self.x3v = 0.5 * (self.x3f[:-1] + self.x3f[1:])
        self.x1f = np.linspace(RI, RO, n_rad + 1)
        self.x1v = 0.5 * (self.x1f[:-1] + self.x1f[1:])
        self.dx1f = np.full(n_rad, (RO - RI) / n_rad)

        # exact solid angle of a cell's x1 face, gnomonic_equiangle.cpp:126-136
        xc = np.tan(self.x2f)[None, :, None]
        yc = np.tan(self.x3f)[:, None, None]
        corner = np.arctan(xc * yc / np.sqrt(1.0 + xc * xc + yc * yc))
        self.solid_angle = (corner[1:, 1:] - corner[1:, :-1]
                            - corner[:-1, 1:] + corner[:-1, :-1])

        x = np.tan(self.x2v)[None, :, None]
        xf = np.tan(self.x2f)[None, :, None]
        y = np.tan(self.x3v)[:, None, None]
        yf = np.tan(self.x3f)[:, None, None]
        C, Cf = np.sqrt(1.0 + x * x), np.sqrt(1.0 + xf * xf)
        D, Df = np.sqrt(1.0 + y * y), np.sqrt(1.0 + yf * yf)
        self.cosine_cell = -x * y / (C * D)                        # :82
        self.sine_cell = np.sqrt(1.0 + x * x + y * y) / (C * D)    # :83
        self.sine_face2 = np.sqrt(1.0 + xf * xf + y * y) / (Cf * D)   # :86
        self.sine_face3 = np.sqrt(1.0 + x * x + yf * yf) / (C * Df)   # :89

        x1 = np.tan(self.x2f[:-1])[None, :, None]                  # :96-97
        x2 = np.tan(self.x2f[1:])[None, :, None]
        d1 = np.sqrt(1.0 + x1 * x1 + y * y)
        d2 = np.sqrt(1.0 + x2 * x2 + y * y)
        d1f = np.sqrt(1.0 + x1 * x1 + yf * yf)
        d2f = np.sqrt(1.0 + x2 * x2 + yf * yf)
        self.dx2f_ang = np.arccos((1.0 + x1 * x2 + y * y) / (d1 * d2))        # :104
        self.dx2f_ang_face3 = np.arccos((1.0 + x1 * x2 + yf * yf) / (d1f * d2f))

        y1 = np.tan(self.x3f[:-1])[:, None, None]                  # :110-111
        y2 = np.tan(self.x3f[1:])[:, None, None]
        e1 = np.sqrt(1.0 + x * x + y1 * y1)
        e2 = np.sqrt(1.0 + x * x + y2 * y2)
        e1f = np.sqrt(1.0 + xf * xf + y1 * y1)
        e2f = np.sqrt(1.0 + xf * xf + y2 * y2)
        self.dx3f_ang = np.arccos((1.0 + x * x + y1 * y2) / (e1 * e2))        # :117
        self.dx3f_ang_face2 = np.arccos((1.0 + xf * xf + y1 * y2) / (e1f * e2f))

        rm, rp = self.x1f[:-1], self.x1f[1:]
        if volume_rule == "exact":                                 # :219-227
            radial = (self.dx1f * (rp * rp + rp * rm + rm * rm) / 3.0)
        elif volume_rule == "trapezoid":
            radial = self.dx1f * 0.5 * (rp * rp + rm * rm)
        else:
            raise ValueError(volume_rule)
        self.volume = radial[None, None, :] * self.solid_angle     # :226

        self.area1 = (self.x1f ** 2)[None, None, :] * self.solid_angle   # :208
        rdr = (self.x1v * self.dx1f)[None, None, :]
        self.area2 = rdr * self.dx3f_ang_face2                     # :212
        self.area3 = rdr * self.dx2f_ang_face3                     # :216

        fx = self.area2 * self.sine_face2                          # :138
        fy = self.area3 * self.sine_face3                          # :139
        self.x_ov_rD = (fx[:, 1:] - fx[:, :-1]) / self.volume      # :141
        self.y_ov_rC = (fy[1:, :] - fy[:-1, :]) / self.volume      # :143

        self.radius = self.x1v[None, None, :]
        self.position = self._cartesian_position()

    # --- cs_face_xyz_from_tan, cubed_sphere_constants.h:41-76 ----------------
    @staticmethod
    def _face_xyz(face, a, b):
        one = np.ones_like(a)
        if face == 0:                       # +X
            return one, a, b
        if face == 1:                       # +Y
            return -a, one, b
        if face == 2:                       # -X
            return -one, -a, b
        if face == 3:                       # +Z
            return -b, a, one
        if face == 4:                       # -Y
            return a, -one, b
        return b, a, -one                   # -Z

    def _cartesian_position(self):
        a = np.tan(self.x2v)[None, :, None] * np.ones_like(self.cosine_cell)
        b = np.tan(self.x3v)[:, None, None] * np.ones_like(self.cosine_cell)
        x, y, z = self._face_xyz(self.face_id, a, b)
        norm = np.sqrt(x * x + y * y + z * z)
        unit = np.stack([x / norm, y / norm, z / norm])
        return unit[:, :, :, :] * self.radius[None]

    # --- cs_local_to_cart_vel, cubed_sphere_constants.h:33-39 ---------------
    #     slot 0 is the panel normal, slot 1 the alpha direction, slot 2 beta;
    #     the table sends each to its global Cartesian slot and sign.
    L2C_IDX = ((1, 2, 0), (2, 1, 0), (1, 2, 0), (0, 2, 1), (2, 1, 0), (0, 2, 1))
    L2C_SGN = ((+1, +1, +1), (+1, -1, +1), (-1, -1, +1),
               (+1, +1, -1), (-1, +1, +1), (-1, +1, +1))

    def contra_to_cart(self, vel):
        """cs_contra_to_cart_, cubed_sphere_utils.cpp:221-243. `vel` is
        [3, k, j, i] contravariant panel components (radial, alpha, beta)."""
        x = np.tan(self.x2v)[None, :, None]
        y = np.tan(self.x3v)[:, None, None]
        delta = np.sqrt(x * x + y * y + 1.0)
        C, D = np.sqrt(1.0 + x * x), np.sqrt(1.0 + y * y)
        vz, vx, vy = vel[0], vel[1], vel[2]
        local = (
            (vz - vx * x / D - vy * y / C) / delta,
            (vz * x + vx * D - (vy * x * y) / C) / delta,
            (vz * y + vy * C - (vx * x * y) / D) / delta,
        )
        out = [None, None, None]
        idx = self.L2C_IDX[self.face_id]
        sgn = self.L2C_SGN[self.face_id]
        for slot in range(3):
            out[idx[slot]] = sgn[slot] * local[slot]
        return np.stack([np.broadcast_to(o, self.volume.shape) for o in out])

    def raise_vec(self, m2, m3):
        """coord_vec_raise_impl, coord_utils_impl.h:8-16: covariant to
        contravariant with g22 = g33 = 1 and g23 = cos(theta)."""
        c = self.cosine_cell
        s2 = 1.0 - c * c
        return (m2 / s2 - m3 * c / s2, -m2 * c / s2 + m3 / s2)


def divergence(area1, f1, area2, f2, area3, f3, volume):
    """CoordinateImpl::divergence, coordinate.cpp:568-585."""
    out = np.zeros_like(volume)
    if f1 is not None:
        out = out + (area1[..., 1:] * f1[..., 1:] - area1[..., :-1] * f1[..., :-1])
    if f2 is not None:
        out = out + (area2[:, 1:] * f2[:, 1:] - area2[:, :-1] * f2[:, :-1])
    if f3 is not None:
        out = out + (area3[1:, :] * f3[1:, :] - area3[:-1, :] * f3[:-1, :])
    return out / volume


# =============================================================================
# [B1] global mass over many steps
# =============================================================================
def mass_budget(nstep=2000, seam_break=False, n_ang=8, n_rad=8):
    """Transport one density row on a panel's real geometry with closed walls
    and the shared-face rule, and accumulate the global mass each step.

    `seam_break` gives the two neighbours of one interior x2 face different
    flux values -- the two-block seam defect -- without changing anything else.
    """
    panel = Panel(n_ang, n_rad, 0)
    rng = np.random.default_rng(31)
    rho = rng.uniform(0.05, 0.2, panel.volume.shape)
    v1 = rng.normal(0.0, 2.0, panel.area1.shape)
    v2 = rng.normal(0.0, 2.0, panel.area2.shape)
    v3 = rng.normal(0.0, 2.0, panel.area3.shape)
    v1[..., 0] = v1[..., -1] = 0.0            # closed radial walls
    v2[:, 0] = v2[:, -1] = 0.0                # closed tangential walls
    v3[0, :] = v3[-1, :] = 0.0
    dt = 1.0e-3
    seam = n_ang // 2

    total0 = float((rho * panel.volume).sum())
    totals = [total0]
    for _ in range(nstep):
        f1 = np.zeros_like(panel.area1)
        f1[..., 1:-1] = np.where(v1[..., 1:-1] > 0.0, rho[..., :-1],
                                 rho[..., 1:]) * v1[..., 1:-1]
        f2 = np.zeros_like(panel.area2)
        f2[:, 1:-1] = np.where(v2[:, 1:-1] > 0.0, rho[:, :-1],
                               rho[:, 1:]) * v2[:, 1:-1]
        f3 = np.zeros_like(panel.area3)
        f3[1:-1, :] = np.where(v3[1:-1, :] > 0.0, rho[:-1, :],
                               rho[1:, :]) * v3[1:-1, :]
        if seam_break:
            # the two sides of face `seam` use fluxes that differ by SEAM_GAP
            lo = divergence(panel.area1, f1, panel.area2, f2,
                            panel.area3, f3, panel.volume)
            f2b = f2.copy()
            f2b[:, seam] = f2[:, seam] * (1.0 + SEAM_GAP)
            hi = divergence(panel.area1, f1, panel.area2, f2b,
                            panel.area3, f3, panel.volume)
            div = lo.copy()
            div[:, seam:] = hi[:, seam:]
        else:
            div = divergence(panel.area1, f1, panel.area2, f2,
                             panel.area3, f3, panel.volume)
        rho = rho - dt * div
        totals.append(float((rho * panel.volume).sum()))
    return panel, np.array(totals)


def slope_per_step(series):
    n = len(series)
    x = np.arange(n, dtype=float)
    return float(np.polyfit(x, np.asarray(series, dtype=float), 1)[0])


def check_B1(seam_break=False, nstep=2000):
    panel, totals = mass_budget(nstep=nstep, seam_break=seam_break)
    rel = slope_per_step(totals) / abs(totals[0])
    rel_total = (totals[-1] - totals[0]) / abs(totals[0])
    ncell = panel.volume.size
    bound = (ncell - 1) * EPS / 2.0        # the reduction bound of @eq-ch16-budget
    ok = abs(rel) < 10.0 * EPS and abs(rel_total) < bound
    return report(
        "B1", ok,
        f"global mass is constant to round-off over {nstep} steps of the "
        f"divergence with one flux per face and closed walls",
        f"on {ncell} cells of a gnomonic-equiangular panel the relative mass "
        f"drift is {rel:+.3e} per step and {rel_total:+.3e} over the run, against "
        f"a sequential-reduction bound of {bound:.3e} and a machine epsilon of "
        f"{EPS:.3e}. The face terms cancel exactly between neighbours, so what is "
        f"left is the rounding of the reduction and of the update, not transport")


def break_B1_seam():
    """The two sides of one interior x2 face use fluxes that differ by 1e-9
    relative: the seam leak of two blocks that do not share a face flux."""
    return check_B1(seam_break=True, nstep=2000)


# =============================================================================
# [B2] global total energy over many steps, with gravity
# =============================================================================
def energy_budget(nstep=2000, booking="face", n_ang=8, n_rad=8):
    """Cells under constant radial gravity, with the x1 mass flux prescribed
    and closed at the walls. The gravity work is booked either as snapy's face
    work (hydro_forward.cpp:977-990),

        phi_cell * div(A F_mass) - div(A F_mass phi_face),

    or as the cell work it replaces (:1050-1052), dt rho v1 grav1. The closed
    total is sum (E + rho phi) V: the face form's second term telescopes to the
    closed walls, so its global sum is exactly the potential the mass gained."""
    panel = Panel(n_ang, n_rad, 0)
    rng = np.random.default_rng(32)
    g = 10.0
    phi_cell = g * (panel.radius - RI)
    phi_face = g * (panel.x1f - RI)[None, None, :]

    rho = rng.uniform(0.05, 0.2, panel.volume.shape)
    e = rng.uniform(1.0e5, 2.0e5, panel.volume.shape)
    v1 = rng.normal(0.0, 1.0, panel.area1.shape)
    v1[..., 0] = v1[..., -1] = 0.0
    dt = 1.0e-3

    def total():
        return float(((e + rho * phi_cell) * panel.volume).sum())

    totals = [total()]
    for _ in range(nstep):
        fm = np.zeros_like(panel.area1)
        fm[..., 1:-1] = np.where(v1[..., 1:-1] > 0.0, rho[..., :-1],
                                 rho[..., 1:]) * v1[..., 1:-1]
        drho = divergence(panel.area1, fm, None, None, None, None, panel.volume)
        afm = panel.area1 * fm
        if booking == "face":
            potential_flux_div = (afm[..., 1:] * phi_face[..., 1:]
                                  - afm[..., :-1] * phi_face[..., :-1]) \
                / panel.volume
            src_e = phi_cell * drho - potential_flux_div
        elif booking == "cell":
            # dt rho v1 grav1, with the cell's own vertical velocity
            v1cell = 0.5 * (fm[..., 1:] + fm[..., :-1]) / rho
            src_e = -rho * v1cell * g
        else:
            raise ValueError(booking)
        rho = rho - dt * drho
        e = e + dt * src_e
        totals.append(total())
    return panel, np.array(totals)


def check_B2(booking="face", nstep=2000):
    panel, totals = energy_budget(nstep=nstep, booking=booking)
    rel = slope_per_step(totals) / abs(totals[0])
    rel_total = (totals[-1] - totals[0]) / abs(totals[0])
    ncell = panel.volume.size
    bound = (ncell - 1) * EPS / 2.0
    ok = abs(rel) < 10.0 * EPS and abs(rel_total) < bound
    return report(
        "B2", ok,
        f"global total energy, stored energy plus potential, is constant to "
        f"round-off over {nstep} steps when the gravity work is booked at the "
        f"faces",
        f"on {ncell} cells the relative drift of sum (E + rho phi) V is "
        f"{rel:+.3e} per step and {rel_total:+.3e} over the run, against the "
        f"reduction bound {bound:.3e}. The face booking moves each parcel of mass "
        f"across the potential of the face it crosses, so the work terms telescope "
        f"with the mass terms that produced them")


def break_B2_cellwork():
    """Book the gravity work at the cell, from the cell's own velocity and
    potential, instead of at the faces."""
    return check_B2(booking="cell", nstep=2000)


# =============================================================================
# [B3] the six-panel volume closes, and div(r rhat) is 3
# =============================================================================
def check_B3(volume_rule="exact", n_ang=12, n_rad=8):
    panels = [Panel(n_ang, n_rad, f, volume_rule=volume_rule)
              for f in range(NFACE)]
    omega = sum(float(p.solid_angle.sum()) for p in panels)
    vol = sum(float(p.volume.sum()) for p in panels)
    exact = 4.0 * math.pi * (RO ** 3 - RI ** 3) / 3.0
    worst = 0.0
    for p in panels:
        div = (p.area1[..., 1:] * p.x1f[1:] - p.area1[..., :-1] * p.x1f[:-1]) \
            / p.volume
        worst = max(worst, float(np.abs(div - 3.0).max()))
    rel_omega = omega / (4.0 * math.pi) - 1.0
    rel_vol = vol / exact - 1.0
    ok = abs(rel_omega) < 1e-14 and abs(rel_vol) < 1e-14 and worst < 1e-12
    return report(
        "B3", ok,
        f"the six panels' solid angles sum to 4 pi and their volumes to the "
        f"shell's, and the discrete divergence of r rhat is 3",
        f"at {n_ang} by {n_ang} cells per panel the solid angles sum to "
        f"{rel_omega:+.2e} of 4 pi and the volumes to {rel_vol:+.2e} of "
        f"4 pi (ro^3 - ri^3) / 3; max |div(r rhat) - 3| is {worst:.2e} over all "
        f"{NFACE * panels[0].volume.size} cells. Every volume-weighted budget "
        f"below rests on these three numbers: a volume measure that is not the "
        f"exact radial integral of the same solid angle as the x1 face area "
        f"makes a conserved total drift with the grid rather than with the flow")


def break_B3_trapezoid():
    """Replace the exact radial integral (rp^3 - rm^3)/3 by the trapezoid of the
    two x1 face areas."""
    return check_B3(volume_rule="trapezoid")


# =============================================================================
# [B4] the uniform-pressure rest state exerts no tangential force
# =============================================================================
def tangential_force(panel, pressure, metric_rule="discrete"):
    """The covariant x2 and x3 momentum tendency of a state at rest, from the
    x2/x3 pressure fluxes and the geometric sources of
    GnomonicEquiangleImpl::forward.

    In the local orthonormal frame an x2 face carries only the normal pressure,
    so flux2global2_ (:358-385) leaves the covariant rows p sin(theta_face2)
    and 0; the x3 face is the mirror image. The geometric sources at rest are
    x_ov_rD p and y_ov_rC p (:461-468).

    `metric_rule='analytic'` is the break arm. It replaces the discrete
    difference of fx that defines x_ov_rD by an analytic expression for the
    derivative of the same quantity. It is NOT the same term to truncation
    order -- an earlier version of this docstring and of the chapter said it
    was, and that was wrong. Measured against the discrete source at p0 = 1e5,
    max |src2|:

        n_ang    discrete     analytic     analytic/discrete
            8    1.3499e-02   1.1622e-01   8.61
           16    1.4994e-02   2.3622e-01   15.75
           32    1.5773e-02   4.7525e-01   30.13
           45    1.6004e-02   6.6946e-01   41.83

    The discrete source converges as the mesh refines; the analytic branch
    grows roughly in proportion to n_ang, i.e. like the inverse cell width, so
    the two do not approach each other at all. The ratio tracks n_ang closely
    enough to suggest one missing angular factor, but restoring a dbeta and the
    C/D ratio only stops the growth -- it leaves a constant factor near 2.9 --
    so the arm is not one factor away from being a truncation-order variant and
    no correction is attempted here rather than ship a formula we cannot derive.

    The arm is still a valid break arm: it shows the claim is sensitive to this
    source term, and it goes red decisively. It simply does not demonstrate the
    delicacy a truncation-order perturbation would. For [B5] the honest
    term-removal test is `metric_rule='none'`, which deletes the source
    outright and needs no substitute formula to argue about.
    """
    f2 = np.broadcast_to(pressure, panel.area2.shape) * panel.sine_face2
    f3 = np.broadcast_to(pressure, panel.area3.shape) * panel.sine_face3
    div2 = (panel.area2[:, 1:] * f2[:, 1:]
            - panel.area2[:, :-1] * f2[:, :-1]) / panel.volume
    div3 = (panel.area3[1:, :] * f3[1:, :]
            - panel.area3[:-1, :] * f3[:-1, :]) / panel.volume
    if metric_rule == "none":
        # the term removed outright, not replaced by another formula
        src2 = np.zeros_like(panel.x_ov_rD * pressure)
        src3 = np.zeros_like(panel.y_ov_rC * pressure)
    elif metric_rule == "discrete":
        src2 = panel.x_ov_rD * pressure
        src3 = panel.y_ov_rC * pressure
    elif metric_rule == "analytic":
        # d/dalpha of (r dr) * dbeta_ang * sin, evaluated at the cell centre
        x = np.tan(panel.x2v)[None, :, None]
        y = np.tan(panel.x3v)[:, None, None]
        C, D = np.sqrt(1.0 + x * x), np.sqrt(1.0 + y * y)
        delta = np.sqrt(1.0 + x * x + y * y)
        dalpha = panel.x2f[1] - panel.x2f[0]
        # sin(theta) = delta / (C D); the angular width of an x3 arc at fixed
        # alpha is dbeta * C / delta, so fx = r dr dbeta C / D, whose alpha
        # derivative is r dr dbeta x / D
        dfx_dalpha = (panel.x1v * panel.dx1f)[None, None, :] * dalpha * x / D
        src2 = dfx_dalpha / panel.volume * pressure
        dfy_dbeta = (panel.x1v * panel.dx1f)[None, None, :] * dalpha * y / C
        src3 = dfy_dbeta / panel.volume * pressure
    else:
        raise ValueError(metric_rule)
    return div2 - src2, div3 - src3


def axial_torque(panels, forces):
    """The global axial torque of a covariant tangential force density: raise it
    to the contravariant frame, rotate it to Cartesian and take (r x F)_z."""
    total = 0.0
    scale = 0.0
    for panel, (f2, f3) in zip(panels, forces):
        c2, c3 = panel.raise_vec(f2, f3)
        fvec = panel.contra_to_cart(np.stack([np.zeros_like(c2), c2, c3]))
        r = panel.position
        tz = r[0] * fvec[1] - r[1] * fvec[0]
        total += float((tz * panel.volume).sum())
        scale += float((np.abs(tz) * panel.volume).sum())
    return total, scale


def check_B4(metric_rule="discrete", n_ang=8, n_rad=6, p0=1.0e5):
    panels = [Panel(n_ang, n_rad, f) for f in range(NFACE)]
    forces = [tangential_force(p, p0, metric_rule=metric_rule) for p in panels]
    worst = max(max(float(np.abs(f2).max()), float(np.abs(f3).max()))
                for f2, f3 in forces)
    torque, scale = axial_torque(panels, forces)
    # the natural size of one term of the difference, for a relative reading
    term = max(float(np.abs(p.x_ov_rD * p0).max()) for p in panels)
    rel = worst / term
    ok = rel < 1e-12 and (scale == 0.0 or abs(torque) <= 1e-12 * scale + 1e-30)
    return report(
        "B4", ok,
        "a uniform-pressure state at rest feels no tangential force on the "
        "cubed sphere, so it exerts no spurious axial torque",
        f"over all {NFACE} panels at {n_ang} by {n_ang} by {n_rad} cells the "
        f"largest covariant tangential force density is {worst:.3e} against a "
        f"single term of size {term:.3e}, i.e. {rel:.2e} relative; the global "
        f"axial torque is {torque:+.3e} N m against {scale:.3e} N m of unsigned "
        f"torque. The cancellation is exact by construction: x_ov_rD is the "
        f"discrete difference of the SAME product, face area times sine, that the "
        f"x2 pressure flux divides by the volume, so only the order of the two "
        f"multiplications differs")


def break_B4_analytic():
    """Use the analytic metric derivative for the geometric source instead of
    the discrete difference of the face quantity."""
    return check_B4(metric_rule="analytic")


# =============================================================================
# [B5] axial angular momentum over many steps
# =============================================================================
def aam_budget(nstep=2000, odd_panel=None, no_source=False, n_ang=8,
               n_rad=6, p0=1.0e5):
    """Step the covariant tangential momentum rows of the uniform-pressure rest
    state and record the global axial angular momentum each step.

    `odd_panel` is the break arm: that one panel's geometric pressure source is
    built from the analytic metric derivative while the other five keep the
    discrete face difference -- one panel assembled differently from its
    neighbours, which is a defect no single-panel test can see."""
    panels = [Panel(n_ang, n_rad, f) for f in range(NFACE)]
    forces = [tangential_force(
        p, p0, metric_rule=("none" if no_source
                            else "analytic" if f == odd_panel else "discrete"))
        for f, p in enumerate(panels)]
    rho = 0.1
    dt = 1.0
    m = [[np.zeros_like(p.volume), np.zeros_like(p.volume)] for p in panels]

    def lz():
        total = 0.0
        for panel, (m2, m3) in zip(panels, m):
            c2, c3 = panel.raise_vec(m2 / rho, m3 / rho)
            v = panel.contra_to_cart(np.stack([np.zeros_like(c2), c2, c3]))
            r = panel.position
            total += float((rho * (r[0] * v[1] - r[1] * v[0])
                            * panel.volume).sum())
        return total

    series = [lz()]
    for _ in range(nstep):
        for (f2, f3), row in zip(forces, m):
            row[0] = row[0] - dt * f2
            row[1] = row[1] - dt * f3
        series.append(lz())
    # the axial angular momentum the same cells would hold at 1 m/s eastward
    ref = 0.0
    for panel in panels:
        one = np.ones_like(panel.volume)
        v = panel.contra_to_cart(np.stack([np.zeros_like(one), one, one]))
        r = panel.position
        ref += float((rho * np.abs(r[0] * v[1] - r[1] * v[0])
                      * panel.volume).sum())
    return np.array(series), ref, sum(p.volume.size for p in panels)


#: [B5]'s bound on the per-step relative AAM drift. It is NOT 1e-15: the
#: discrete metric source holds the drift at 8.7e-34, and a bound of 1e-15
#: is sixteen orders looser than that -- loose enough that DELETING the
#: geometric pressure source on all six panels also passed it, at 2.1e-18,
#: and replacing it by the analytic derivative everywhere passed at 1.5e-17.
#: A claim whose bound cannot separate the shipped operator from that
#: operator's deletion tests nothing. 1e-20 separates them with fourteen
#: orders of margin on the shipped side and two on the deleted one.
B5_REL_BOUND = 1e-20


def check_B5(odd_panel=None, no_source=False, nstep=2000):
    series, ref, ncell = aam_budget(nstep=nstep, odd_panel=odd_panel,
                                    no_source=no_source)
    per_step = slope_per_step(series)
    rel = per_step / ref
    rel_total = (series[-1] - series[0]) / ref
    ok = abs(rel) < B5_REL_BOUND
    return report(
        "B5", ok,
        f"global axial angular momentum stays at round-off over {nstep} steps "
        f"of the uniform-pressure rest state",
        f"over {ncell} cells on {NFACE} panels the drift is {per_step:+.3e} "
        f"kg m^2 s^-1 per step and {series[-1] - series[0]:+.3e} over the run, "
        f"which is {rel:+.2e} and {rel_total:+.2e} of the {ref:.3e} kg m^2 s^-1 "
        f"the same cells would carry at one metre per second eastward, against a "
        f"bound of {B5_REL_BOUND:.0e} per step. The source of the drift is the "
        f"rounding of the two multiplication orders in [B4] and nothing else: "
        f"there is no term left over for it to accumulate. The bound is set to "
        f"separate this from the same operator DELETED on all six panels, which "
        f"drifts at 2.1e-18 per step -- sixteen orders worse, and still inside "
        f"the 1e-15 this claim used to ask for")


def break_B5_odd_panel():
    """One of the six panels builds its geometric pressure source from the
    analytic metric derivative; the other five keep the discrete difference."""
    return check_B5(odd_panel=0, nstep=2000)


def break_B5_nosource():
    """The term the claim credits, deleted outright on ALL six panels: the
    geometric pressure source is set to zero rather than replaced by another
    formula, so nothing about the substitute can be argued over. Unlike the
    odd-panel arm it leaves the six panels consistent with each other, so no
    seam argument catches it -- only the bound does, and only since the bound
    was tightened: this arm drifts at 2.1e-18 per step, which the 1e-15 the
    claim used to ask for let through."""
    return check_B5(no_source=True, nstep=2000)


# =============================================================================
# [B6] every break arm goes red
# =============================================================================
BREAKS = {
    "B1-seam": ("B1", break_B1_seam,
                f"the two sides of one interior x2 face use fluxes that differ "
                f"by {SEAM_GAP:g} relative"),
    "B2-cellwork": ("B2", break_B2_cellwork,
                    "the gravity work is booked at the cell instead of at the "
                    "faces"),
    "B3-trapezoid": ("B3", break_B3_trapezoid,
                     "the cell volume is the trapezoid of the two x1 face areas "
                     "instead of the exact radial integral"),
    "B4-analytic": ("B4", break_B4_analytic,
                    "the geometric pressure source uses the analytic metric "
                    "derivative instead of the discrete face difference"),
    "B5-odd-panel": ("B5", break_B5_odd_panel,
                     "one of the six panels uses the analytic metric source "
                     "while the other five use the discrete one"),
    "B5-nosource": ("B5", break_B5_nosource,
                    "the geometric pressure source is removed outright on all "
                    "six panels"),
}


def check_B6():
    import contextlib
    import io
    saved = list(FAIL)
    results = []
    for name, (claim, fn, _) in BREAKS.items():
        FAIL.clear()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            passed = fn()
        lines = buf.getvalue().strip().splitlines()
        results.append((name, claim, passed, lines[0] if lines else ""))
    FAIL.clear()
    FAIL.extend(saved)
    for name, claim, passed, line in results:
        print(f"       break {name} -> {'RED' if not passed else 'STILL GREEN'}")
        print(f"              {line}")
    return report(
        "B6", all(not p for _, _, p, _ in results),
        f"each of the {len(results)} break arms turns its budget red",
        "; ".join(f"{n} breaks {c}" for n, c, _, _ in results) +
        ". Each arm removes or replaces exactly one term and leaves the rest of "
        "the operator alone")


CHECKS = (check_B1, check_B2, check_B3, check_B4, check_B5, check_B6)


def main(argv):
    if "--list-breaks" in argv:
        for name, (claim, _, what) in BREAKS.items():
            print(f"{name}\tbreaks [{claim}]\t{what}")
        return 0
    if "--break" in argv:
        name = argv[argv.index("--break") + 1]
        if name not in BREAKS:
            print(f"unknown break arm {name!r}; try --list-breaks")
            return 2
        claim, fn, what = BREAKS[name]
        print(f"break arm {name}: {what}")
        fn()
        print(f"{1 - len(FAIL)}/1 claims pass")
        return len(FAIL)
    for fn in CHECKS:
        fn()
    print(f"{len(CHECKS) - len(FAIL)}/{len(CHECKS)} claims pass")
    return len(FAIL)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
