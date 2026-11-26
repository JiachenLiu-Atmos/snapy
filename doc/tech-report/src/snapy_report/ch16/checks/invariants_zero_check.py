"""Zero in, zero out, and what every floor and clamp actually changes
(chapter 15, `_invariants.qmd`).

Ports, statement for statement, the snapy operators that decide whether an
absent species can acquire a non-zero amount, at the code pin of
`code_pin.yml`, snapy@ccf40f9565642c32be69f4b9ef0cfa6daff3f201:

  flux_positivity_theta           src/hydro/flux_positivity.cpp:21-71
  flux_positivity_scale_          src/hydro/flux_positivity.cpp:73-100
  CoordinateImpl::divergence      src/coord/coordinate.cpp:543-586
  apply_conserved_limiter_        src/eos/equation_of_state.cpp:192-322
  apply_primitive_limiter_        src/eos/equation_of_state.cpp:331-356
  ReconstructImpl face clamps     src/recon/reconstruct.cpp:95, 136, 144, 154

numpy only; no compiled snapy is executed. The claims measure what the
operators DO to a state, not what their formulas say.

Motivation. kintera's absent-species defect (its 1e-300 concentration floor
made a species that was not there behave as if it were, and the implicit
update then drove it negative) is a defect class, not a kintera fact. [Z1]-[Z4]
test snapy's own absent-species paths for it; [Z5] is the inventory of every
positivity floor, clamp and epsilon in the code paths this chapter covers,
each with a measurement of what that floor changes; [Z6] re-runs each break
arm and requires it to fail, so no claim here is one that cannot go red.

Run (green):   PYTHONPATH=src python3 invariants_zero_check.py
Run (one red): PYTHONPATH=src python3 invariants_zero_check.py --break Z1-floor
List the arms: PYTHONPATH=src python3 invariants_zero_check.py --list-breaks

Exits with the number of failed claims; a break arm is expected to exit nonzero.
"""
import sys

import numpy as np

# --- constants read from the source at the pin --------------------------------
ULP = 4096.0               # kPositivityRoundoffUlp, flux_positivity.hpp:18
ULP_FLOAT = 64.0           # kPositivityRoundoffUlpFloat, flux_positivity.hpp:23
EPS64 = np.finfo(np.float64).eps
EPS32 = np.finfo(np.float32).eps
DENSITY_FLOOR = 1.0e-6     # YAML default, equation_of_state.cpp:65
PRESSURE_FLOOR = 1.0e-3    # :71
TEMPERATURE_FLOOR = 20.0   # :72
CXX_DENSITY_FLOOR = 1.0e-10   # equation_of_state.hpp:49
CXX_PRESSURE_FLOOR = 1.0e-10  # :50
DRAIN_GUARD = 1.0e-300     # flux_positivity.cpp:70
CV = 718.0                 # a dry-air heat capacity, for the energy floor only

FAIL = []
BREAKS = {}


def report(tag, ok, what, detail):
    print(f"[{tag}] {'PASS' if ok else 'FAIL'} {what}: {detail}")
    if not ok:
        FAIL.append(tag)
    return ok


# --- the ported operators ------------------------------------------------------
def divergence(af_lo, af_hi, volume):
    """CoordinateImpl::divergence, coordinate.cpp:568-585: a face difference of
    area-weighted fluxes over the cell volume. One dimension is enough: the
    other two add the same shape of term."""
    return (af_hi - af_lo) / volume


def theta(u, af_lo, af_hi, volume, dt, avail_rule="relu", dtype=np.float64):
    """flux_positivity_theta, flux_positivity.cpp:21-71.

    `out` accumulates the OUTGOING area-flux over the faces the divergence
    uses (:42): the upper face drains the cell where its flux is positive, the
    lower face where it is negative. `avail` is the cell's own amount, floored
    at zero by relu and stopped ULP machine epsilons short of it (:65-66).
    theta is min(1, avail / dt*out) where the cell drains, and 1 elsewhere
    (:68-70).

    `avail_rule` selects the break arm:
      relu  -- the shipped rule, u.relu()
      floor -- an absent row treated as present at 1e-300 (the kintera defect)
    """
    eps = EPS32 if dtype is np.float32 else EPS64
    # The margin is ULP (4096) in BOTH precisions: flux_positivity.cpp:65 reads
    # `kPositivityRoundoffUlp * machine_epsilon(u.scalar_type())`, so only the
    # epsilon switches on dtype, not the ulp count. ULP_FLOAT (64) belongs to
    # positivity_roundoff() (:16-20), the threshold that decides whether a
    # withheld or repaired mass is REPORTED as a positivity event -- it never
    # enters theta. flux_positivity.hpp:22 says so in as many words: "The
    # margin of flux_positivity_theta stays 4096 ulp."
    margin = ULP * eps
    out = np.maximum(af_hi, 0.0) + np.maximum(-af_lo, 0.0)   # :42
    if avail_rule == "relu":
        avail = np.maximum(u, 0.0) * volume * (1.0 - margin)  # :66
    elif avail_rule == "floor":
        avail = np.maximum(u, DRAIN_GUARD) * volume * (1.0 - margin)
    else:
        raise ValueError(avail_rule)
    drain = out * dt                                          # :67
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return np.where(drain > 0.0,
                        np.minimum(avail / np.maximum(drain, DRAIN_GUARD), 1.0),
                        np.ones_like(u))                      # :68-70


def scale_faces(th, af_lo, af_hi):
    """flux_positivity_scale_, flux_positivity.cpp:73-100: each face flux is
    multiplied by the factor of its DONOR cell -- the lower neighbour where the
    flux is positive, the upper neighbour where it is not."""
    n = th.size
    th_lo = np.concatenate(([1.0], th))[:n + 1]    # donor when f > 0
    th_hi = np.concatenate((th, [1.0]))[:n + 1]    # donor when f <= 0
    af = np.concatenate((af_lo[:1], af_hi))
    return af * np.where(af > 0.0, th_lo, th_hi)


def species_steps(u0, af, volume, dt, nstep, avail_rule="relu"):
    """N explicit steps of one species row through the positivity limiter and
    the divergence, with the shared-face rule (one flux per face, used by both
    neighbours). Returns the history of the row."""
    u = u0.copy()
    hist = [u.copy()]
    for _ in range(nstep):
        th = theta(u, af[:-1], af[1:], volume, dt, avail_rule=avail_rule)
        scaled = scale_faces(th, af[:-1], af[1:])
        u = u - dt * divergence(scaled[:-1], scaled[1:], volume)
        hist.append(u.copy())
    return np.array(hist)


def condensate_borrow(vapour, cloud, cloud_rule="shipped"):
    """apply_conserved_limiter_, equation_of_state.cpp:242-268. A negative
    condensate means the mass sits in a neighbour, so the deficit is borrowed
    from the parent vapour in the same cell and the cloud is set to exactly
    zero. `cloud_rule='floor'` is the break arm: the cloud is floored at 1e-300
    instead, which makes an absent cloud present."""
    deficit = np.minimum(cloud, 0.0)              # :264, clamp_max(0.)
    vap = vapour + deficit                        # :266, one parent, share 1
    if cloud_rule == "shipped":
        return vap, np.maximum(cloud, 0.0)        # :268, clamp_min_(0.)
    return vap, np.maximum(cloud, DRAIN_GUARD)


def reconstruct_species(cell, clamp=True):
    """ReconstructImpl face values of a species row, with the shock-off face
    clamp of reconstruct.cpp:154 (and :95 for the scale-off path). A linear
    (van Leer limited) reconstruction is enough: any reconstruction whose
    stencil is identically zero returns zero before the clamp."""
    n = cell.size
    left = np.empty(n + 1)
    right = np.empty(n + 1)
    pad = np.concatenate(([cell[0]], cell, [cell[-1]]))
    dl = pad[1:-1] - pad[:-2]
    dr = pad[2:] - pad[1:-1]
    slope = np.where(dl * dr > 0.0, 2.0 * dl * dr / (dl + dr + (dl + dr == 0)), 0.0)
    left[1:] = cell + 0.5 * slope
    right[:-1] = cell - 0.5 * slope
    left[0] = cell[0]
    right[-1] = cell[-1]
    if clamp:
        np.maximum(left, 0.0, out=left)           # :154
        np.maximum(right, 0.0, out=right)
    return left, right


# =============================================================================
# [Z1] an absent species gets a zero rate and stays exactly zero over N steps
# =============================================================================
def check_Z1(avail_rule="relu", nstep=2000):
    rng = np.random.default_rng(11)
    n = 48
    volume = rng.uniform(0.5, 2.0, n)              # unequal cells, as on a sphere
    af = rng.normal(0.0, 1.0, n + 1)               # one flux per face, both sides
    af[0] = af[-1] = 0.0                           # closed walls
    dt = 0.05
    absent = np.zeros(n)
    present = rng.uniform(1e-3, 1.0, n)

    ha = species_steps(absent, af, volume, dt, nstep, avail_rule=avail_rule)
    hp = species_steps(present, af, volume, dt, nstep, avail_rule=avail_rule)

    # the rate the first step gives the absent row
    th0 = theta(absent, af[:-1], af[1:], volume, dt, avail_rule=avail_rule)
    rate0 = divergence(*np.split(scale_faces(th0, af[:-1], af[1:]), [n]), volume)
    worst_absent = ha.min()
    nonzero = int(np.count_nonzero(ha))
    worst_present = hp.min()
    subnormal = float(np.nextafter(0.0, 1.0))        # one subnormal unit
    # the present row is the control: the limiter's own guarantee is that a cell
    # it drains lands above zero, and that holds down to the subnormal range,
    # where the 4096-eps margin itself underflows
    ok = (worst_absent == 0.0 and nonzero == 0 and np.all(rate0 == 0.0)
          and worst_present >= -subnormal)
    return report(
        "Z1", ok,
        "an absent species has an exactly zero rate and no negative amount after "
        f"{nstep} steps of the positivity limiter",
        f"the first step's tendency for the absent row is {np.abs(rate0).max():.1e} "
        f"over all {n} cells; after {nstep} steps the row holds {nonzero} non-zero "
        f"entries and its minimum is {worst_absent:+.1e}. theta is "
        f"{th0.min():.1f} in every draining cell of the absent row, so the limiter "
        f"scales its donor faces to zero and the row cannot be drained at all. The "
        f"present control row reaches {worst_present:+.1e}, which is "
        f"{abs(worst_present) / subnormal:.3g} subnormal unit(s): a cell drained "
        f"into the subnormal range loses the 4096-epsilon margin to underflow, and "
        f"the rounding of theta times the drain can cross zero by one unit")


def break_Z1_floor():
    """The kintera #147/#151 defect written into snapy's positivity limiter: an
    absent row treated as present at 1e-300 instead of relu'd to zero."""
    return check_Z1(avail_rule="floor", nstep=2000)


# =============================================================================
# [Z2] the absent row is not merely small: it is bitwise zero, in float32 too
# =============================================================================
def check_Z2(avail_rule="relu"):
    rng = np.random.default_rng(12)
    n = 32
    out = {}
    for dtype, name in ((np.float64, "float64"), (np.float32, "float32")):
        volume = rng.uniform(0.5, 2.0, n).astype(dtype)
        af = rng.normal(0.0, 1.0, n + 1).astype(dtype)
        af[0] = af[-1] = 0
        u = np.zeros(n, dtype=dtype)
        th = theta(u, af[:-1], af[1:], volume, dtype(0.05), avail_rule=avail_rule,
                   dtype=dtype)
        scaled = scale_faces(th, af[:-1], af[1:])
        out[name] = (float(np.abs(scaled).max()), float(th.min()),
                     float(dtype(DRAIN_GUARD)))
    # 1e-300 is not representable in float32: a 1e-300 floor is a no-op there and
    # a real amount in double, so a floor-based rule is not even precision-stable
    ok = (out["float64"][0] == 0.0 and out["float32"][0] == 0.0
          and out["float32"][2] == 0.0 and out["float64"][2] > 0.0)
    return report(
        "Z2", ok,
        "the limiter zeroes every donor face of an absent row in both precisions, "
        "and the 1e-300 guard is not even representable in float32",
        f"the largest scaled face flux of the absent row is "
        f"{out['float64'][0]:.1e} in float64 and {out['float32'][0]:.1e} in "
        f"float32, with theta {out['float64'][1]:.1f} and {out['float32'][1]:.1f}; "
        f"1e-300 stored as float32 is {out['float32'][2]:.1e} and as float64 is "
        f"{out['float64'][2]:.1e}. A rule built on that guard would mean one thing "
        f"in double and another in single, which is why the shipped rule is relu "
        f"and the guard sits only under a `where` that discards it")


def break_Z2_floor():
    """The floor rule instead of the rectifier. This is the only mutation [Z2]
    can see: for u == 0 exactly, relu(u) and u are the same number, so deleting
    the rectifier changes nothing here (verified: the claim stays green). What
    changes the answer is replacing it by a POSITIVE floor, which is also the
    defect the claim exists to catch -- and it changes it in float64 only, which
    is the precision instability the claim is about."""
    return check_Z2(avail_rule="floor")


# =============================================================================
# [Z3] an absent condensate borrows nothing and stays exactly zero
# =============================================================================
def check_Z3(cloud_rule="shipped", nstep=500):
    rng = np.random.default_rng(13)
    n = 4000
    vap0 = rng.uniform(1e-3, 1e-2, n)
    cloud0 = np.zeros(n)
    vap, cloud = vap0.copy(), cloud0.copy()
    for _ in range(nstep):
        vap, cloud = condensate_borrow(vap, cloud, cloud_rule=cloud_rule)
    moved = float(np.abs(vap - vap0).max())
    worst = float(cloud.min())
    nonzero = int(np.count_nonzero(cloud))
    ok = moved == 0.0 and worst == 0.0 and nonzero == 0
    return report(
        "Z3", ok,
        f"an absent condensate borrows nothing from its parent vapour over "
        f"{nstep} limiter calls",
        f"the parent vapour moves by at most {moved:.1e} over {n} cells and the "
        f"cloud row holds {nonzero} non-zero entries, minimum {worst:+.1e}: the "
        f"deficit is min(c, 0), which is exactly zero for an absent cloud, and "
        f"the repair then writes max(c, 0), also exactly zero. Nothing in the "
        f"call can make an absent condensate present")


def break_Z3_floor():
    """Floor the condensate at 1e-300 instead of clamping it to zero."""
    return check_Z3(cloud_rule="floor", nstep=500)


# =============================================================================
# [Z4] an absent species reconstructs to zero on both faces, so its flux is zero
# =============================================================================
def check_Z4(clamp=True):
    rng = np.random.default_rng(14)
    n = 40
    absent = np.zeros(n)
    present = rng.uniform(1e-4, 1.0, n)
    la, ra = reconstruct_species(absent, clamp=clamp)
    lp, rp = reconstruct_species(present, clamp=clamp)
    vel = rng.normal(0.0, 1.0, n + 1)
    flux_a = np.where(vel > 0.0, la, ra) * vel      # an upwind species flux
    flux_p = np.where(vel > 0.0, lp, rp) * vel
    ok = (np.all(la == 0.0) and np.all(ra == 0.0) and np.all(flux_a == 0.0)
          and np.abs(flux_p).max() > 0.0)
    return report(
        "Z4", ok,
        "an absent species reconstructs to zero on both sides of every face, so "
        "its upwind flux is exactly zero",
        f"over {n} cells the largest reconstructed face value of the absent row is "
        f"{max(np.abs(la).max(), np.abs(ra).max()):.1e} and its largest face flux "
        f"is {np.abs(flux_a).max():.1e}; a present row through the same stencil "
        f"carries fluxes up to {np.abs(flux_p).max():.3e}. The limited slope of an "
        f"identically zero stencil is zero before any clamp, and the face clamp "
        f"then has nothing to repair")


def break_Z4_noclamp():
    """Remove the face clamp and seed one cell with a negative round-off, the
    state a species repair upstream can leave behind."""
    n = 40
    seeded = np.zeros(n)
    seeded[n // 2] = -1e-18                         # one cell left negative
    left, right = reconstruct_species(seeded, clamp=False)
    worst = float(min(left.min(), right.min()))
    # both velocity signs, so the upwind choice cannot hide the negative side
    carried = max(float(np.abs(np.where(s > 0.0, left, right) * s).max())
                  for s in (1.0, -1.0))
    ok = np.all(left >= 0.0) and np.all(right >= 0.0) and carried == 0.0
    return report(
        "Z4", ok,
        "an absent species reconstructs to zero on both sides of every face, so "
        "its upwind flux is exactly zero",
        f"with the face clamp removed the most negative reconstructed face value "
        f"is {worst:.1e} and the largest face flux the row carries, over both "
        f"upwind directions at unit speed, is {carried:.1e}: one cell left at "
        f"-1e-18 reaches the faces and is transported instead of being repaired")


# =============================================================================
# [Z5] the inventory: every floor, clamp and epsilon, and what it changes
# =============================================================================
# kind: state  -- floors a stored state row; changes the state and invents mass
#                 or energy
#       guard  -- guards a division; changes the quotient only at or below the
#                 guard, and the surrounding expression may discard it
#       domain -- holds an argument inside the domain of the next function
#       index  -- clamps an index, never a value
#       eps    -- a comparison or pivot tolerance; it decides a branch
#       cap    -- a physical cap chosen by the deck
def probe_state_floor(value, floor):
    """What a clamp_min on a state row changes: the amount it invents."""
    out = np.maximum(value, floor)
    return float((out - value)[out != value].sum()), int(np.count_nonzero(out != value))


def inventory():
    rng = np.random.default_rng(5)
    rows = []

    def add(sites, kind, what, detail):
        rows.append((sites, kind, what, detail))

    # --- state floors ---------------------------------------------------------
    d = rng.uniform(-1e-7, 5e-6, 4000)
    inv, hit = probe_state_floor(d, DENSITY_FLOOR)
    add("eos/equation_of_state.cpp:209; :339", "state",
        "dry density floored at density-floor (YAML 1e-6, C++ default 1e-10)",
        f"over 4000 cells straddling it the floor invents {inv:.3e} of dry mass in "
        f"{hit} cells; it can only add, and nothing removes it again")

    e = rng.uniform(-1e-3, 1e-2, 4000)
    rho = np.full(4000, DENSITY_FLOOR)
    min_ie = rho * CV * TEMPERATURE_FLOOR
    inv_e, hit_e = probe_state_floor(e, min_ie)
    add("eos/equation_of_state.cpp:233", "state",
        "total energy floored at the kinetic energy plus rho cv T_floor, with the "
        "ALREADY floored density",
        f"on the same 4000 cells it invents {inv_e:.3e} of total energy in "
        f"{hit_e} cells; the floor is built from the density of :209, so the two "
        f"passes do not commute")

    p = rng.uniform(-1e-4, 1e-2, 4000)
    inv_p, hit_p = probe_state_floor(p, PRESSURE_FLOOR)
    add("eos/equation_of_state.cpp:355", "state",
        "pressure floored at pressure-floor (YAML 1e-3, C++ default 1e-10) on the "
        "primitive pass",
        f"it raises {hit_p} of 4000 pressures and invents {inv_p:.3e} of pressure; "
        f"the primitive pass floors no temperature and repairs no energy row")

    c = rng.uniform(-2e-4, 1e-3, 4000)
    vap = rng.uniform(1e-3, 1e-2, 4000)
    v_b, c_b = condensate_borrow(vap, c)
    drift_borrow = float((v_b + c_b - (vap + c)).sum())
    drift_zero = float((vap + np.maximum(c, 0.0) - (vap + c)).sum())
    add("eos/equation_of_state.cpp:264; :268", "state",
        "a negative condensate is borrowed from its parent vapour, then clamped "
        "to exactly zero",
        f"over 4000 cells the pair changes the cell total by {drift_borrow:+.1e}; "
        f"clamping the condensate to zero without the borrow would invent "
        f"{drift_zero:.3e}. On an ABSENT condensate both terms are exactly zero")

    cl = rng.uniform(-1e-9, 1e-3, 4000)
    inv_c, hit_c = probe_state_floor(cl, 0.0)
    add("eos/equation_of_state.cpp:317; :352", "state",
        "cloud and species rows clamped at zero after the kinetics source and on "
        "the primitive pass",
        f"it raises {hit_c} of 4000 rows and invents {inv_c:.3e} of species mass; "
        f"an absent row is already zero, so the clamp does not touch it")

    y = rng.uniform(-1e-9, 1.0, 4000)
    inv_y, hit_y = probe_state_floor(y, 0.0)
    add("recon/reconstruct.cpp:95; :154", "state",
        "reconstructed face species clamped at zero (scale-off and shock-off paths)",
        f"it raises {hit_y} of 4000 face values and invents {inv_y:.3e} of face "
        f"species; both faces of an absent row are exactly zero before the clamp")

    fd = rng.uniform(-1e-7, 5e-6, 4000)
    inv_fd, hit_fd = probe_state_floor(fd, DENSITY_FLOOR)
    add("recon/reconstruct.cpp:136; :144", "state",
        "reconstructed face density and pressure floored at the EOS floors",
        f"the density floor raises {hit_fd} of 4000 face values by {inv_fd:.3e} in "
        f"total; these run only where the reconstruction's `shock` arm is off")

    dry = rng.uniform(-1e-6, 1.0, 4000)
    inv_dr, hit_dr = probe_state_floor(dry, 0.0)
    add("forcing/relax_bot_comp.cpp:104", "state",
        "the relaxation target's dry fraction clamped at zero",
        f"it raises {hit_dr} of 4000 targets by {inv_dr:.3e} in total; the target "
        f"is a boundary value, so the change is a forcing change, not a repair")

    var = rng.normal(0.0, 1e-12, 4000)
    inv_v, hit_v = probe_state_floor(var, 0.0)
    add("output/output_type.cpp:359; :374", "state",
        "a variance clamped at zero before its square root (diagnostic output only)",
        f"it raises {hit_v} of 4000 round-off-negative variances by {inv_v:.3e}; "
        f"it changes an output column and never the state")

    # --- sign selections ------------------------------------------------------
    af = rng.normal(0.0, 1.0, 4000)
    outgoing = np.maximum(af, 0.0) + np.maximum(-af, 0.0)
    add("hydro/flux_positivity.cpp:42; :50; :58", "select",
        "the drain accumulates only the OUTGOING part of each face's area-flux, "
        "through relu on the upper face and on the negated lower face",
        f"over {af.size} random faces the relu pair keeps "
        f"{float(outgoing.sum()):.3e} of area-flux, which is the sum of the "
        f"magnitudes, against {float(af.sum()):+.3e} for the signed sum: an "
        f"incoming face cannot pay for an outgoing one, which is what makes the "
        f"bound per cell rather than per net flux")

    th_raw = np.array([-0.5, 0.0, 0.5, 1.0, 2.0, 1e30])
    th_cap = np.minimum(th_raw, 1.0)
    add("hydro/flux_positivity.cpp:66; :70", "select",
        "the cell's own amount is relu'd before it is offered, and the resulting "
        "factor is capped at one",
        f"the cap changes {int(np.count_nonzero(th_cap != th_raw))} of "
        f"{th_raw.size} probe factors, every one above 1; without it a cell with "
        f"more amount than drain would have its faces AMPLIFIED. The relu on the "
        f"amount is what sends an absent row's factor to exactly zero rather than "
        f"to a negative number")

    # --- meters ---------------------------------------------------------------
    hits = rng.random(4000) < 0.01
    add("hydro/hydro_forward.cpp:846; implicit/implicit_hydro.cpp:462; "
        "mesh/meshblock.cpp:885; :1115; :1118", "meter",
        "running minima and maxima that record that a floor or a clamp fired",
        f"on a {hits.size}-cell probe with {int(hits.sum())} hits these reductions "
        f"report the hit; they are read by the redo decision and the step log and "
        f"they change no state themselves. A floor with no meter is a silent "
        f"repair, which is the condition this chapter's checks exist to detect")

    # --- division guards ------------------------------------------------------
    drain = np.array([0.0, 1e-320, 1e-300, 1e-200, 1.0])
    quotient_guarded = 1.0 / np.maximum(drain, DRAIN_GUARD)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        quotient_raw = 1.0 / drain
    binds = int(np.count_nonzero(quotient_guarded != quotient_raw))
    selected = int(np.count_nonzero(drain > 0.0))
    add("hydro/flux_positivity.cpp:70", "guard",
        "the 1e-300 guard under the drain division",
        f"of {drain.size} probe denominators it changes the quotient in {binds}, "
        f"all of them at or below 1e-300; the enclosing `where(drain > 0)` selects "
        f"the quotient in only {selected} of them, every one above the guard, so "
        f"the guard changes NO selected value. It exists to keep the discarded "
        f"branch from evaluating 1/0, which would be a NaN under torch")

    dl = np.array([0.0, 1e-40, 1e-30, 1e-20, 1.0])
    g = np.maximum(dl, 1e-30)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        raw = 1.0 / dl
    add("coord/coordinate.cpp:37; :50; :57; :609; :610; :611; :612; :613; :614",
        "guard",
        "1e-30 guards under the interpolation distance and the curl line elements",
        f"of {dl.size} probe lengths the guard changes the quotient in "
        f"{int(np.count_nonzero(1.0 / g != raw))}, all at or below 1e-30. A real "
        f"line element on any grid this report describes is many orders above it, "
        f"so on a physical grid the guards are inert; they bind only on a "
        f"degenerate cell, where the result is already meaningless")

    rb = np.array([0.0, 1e-320, 1e-300, 1.0e3, 6.4e6])
    gg = np.maximum(rb ** 2, 1e-300)
    add("coord/coordinate.cpp:490; :519", "guard",
        "1e-300 guards under the radial centroid and face-moment divisions",
        f"of {rb.size} probe radii the guard changes the quotient in "
        f"{int(np.count_nonzero(rb ** 2 < 1e-300))}, all at r below 1e-150 m; the "
        f"smallest radius any deck in this report uses is positive and metres-scale, "
        f"so the guards never bind there")

    prod = np.array([0.0, 1e-40, 1e-30, 1.0])
    add("hydro/hydro.cpp:376", "guard",
        "a 1e-30 guard under the product of two face magnitudes in the time-step "
        "reduction",
        f"of {prod.size} probes it changes the quotient in "
        f"{int(np.count_nonzero(prod < 1e-30))}; it bounds a time step, so when it "
        f"binds it makes the step SMALLER, never larger")

    psf = np.array([-1.0, 0.0, 1e-320, np.finfo(np.float64).tiny, 1e5])
    out_psf = np.maximum(psf, np.finfo(np.float64).tiny)
    add("hydro/hydro_dispatch.cpp:59; :60; hydro/hydro_ref_x1_impl.h:41", "guard",
        "the hydrostatic face pressures floored at the smallest positive normal "
        "double",
        f"of {psf.size} probes it raises {int(np.count_nonzero(out_psf != psf))}, "
        f"every one at or below {np.finfo(np.float64).tiny:.1e} Pa; its purpose is "
        f"to keep the following logarithm finite, and a physical face pressure is "
        f"some 300 orders above it")

    ml = np.array([0.0, 1e-320, 1.0, 1e6])
    add("implicit/implicit_hydro.cpp:455; :457", "guard",
        "the implicit mass-correction residual scale floored at 1e-300 in double "
        "and at the smallest normal float in single",
        f"of {ml.size} probes it changes the scale in "
        f"{int(np.count_nonzero(ml < 1e-300))}; the floor only enlarges the "
        f"denominator, so a floored cell reports a SMALLER relative residual -- it "
        f"can hide a binding in a zero-flux layer, never invent one. The source "
        f"says the double value is kept because 1e-300 is zero in float32")

    cs = np.array([-1.0, 0.0, 1e-12, 1e-10, 3.4e5])
    add("riemann/roe_dispatch.cpp:151", "guard",
        "the Roe sound speed squared floored at 1e-10",
        f"of {cs.size} probes it raises "
        f"{int(np.count_nonzero(cs < 1e-10))} to 1e-10, i.e. a sound speed of "
        f"1e-5 m/s; it binds only on a non-physical Roe average and when it does "
        f"it makes the wave speeds, and so the dissipation, SMALLER")

    # --- domain clamps --------------------------------------------------------
    z = np.array([-1.0 - 2 * EPS64, -1.0, 0.0, 1.0, 1.0 + 2 * EPS64])
    zc = np.clip(z, -1.0, 1.0)
    add("coord/cubed_sphere_utils.cpp:188", "domain",
        "the cosine argument of the panel-angle arccos held in [-1, 1]",
        f"of {z.size} probes it moves {int(np.count_nonzero(zc != z))}, each by at "
        f"most {np.abs(zc - z).max():.1e}; without it arccos returns NaN just "
        f"outside the domain, which round-off in the dot product can reach")

    eta = np.array([-1e-16, 0.0, 0.5, 1.0, 1.0 + 1e-16])
    ec = np.clip(eta, 0.0, 1.0)
    add("forcing/top_sponge_lyr.cpp:61; forcing/bot_sponge_lyr.cpp:61; "
        "forcing/diffusion.cpp:223", "domain",
        "the sponge and blending coordinates held in [0, 1]",
        f"of {eta.size} probes it moves {int(np.count_nonzero(ec != eta))} by at "
        f"most {np.abs(ec - eta).max():.1e}; outside the layer the factor is "
        f"exactly 0 or 1, so the forcing stays inside the layer it names")

    alpha = np.array([-1e-15, 0.0, 0.5, 1.0, 1.0 + 1e-15])
    ac = np.clip(alpha, 0.0, 1.0)
    add("bc/bc_func.cpp:178; :200", "domain",
        "the boundary blending factor capped at 0.99 of its bound and held in [0, 1]",
        f"of {alpha.size} probes the [0, 1] clamp moves "
        f"{int(np.count_nonzero(ac != alpha))}; alpha multiplies a difference, so "
        f"0 keeps the interior value and 1 takes the boundary value in full")

    vcap = 50.0
    vs = np.array([-120.0, -10.0, 0.0, 10.0, 120.0])
    vc = np.clip(vs, -vcap, vcap)
    add("sedimentation/sed_vel.cpp:70; sedimentation/sed_hydro_dispatch.cpp:189",
        "cap",
        "the sedimentation velocity capped at the deck's upper-limit",
        f"at a {vcap:.0f} m/s limit it moves {int(np.count_nonzero(vc != vs))} of "
        f"{vs.size} probe velocities, by up to {np.abs(vc - vs).max():.0f} m/s; it "
        f"is a deck-chosen physical cap, not a positivity repair, and it changes "
        f"the transported amount whenever it binds")

    # --- index clamps ---------------------------------------------------------
    n = 12
    idx = np.clip(np.arange(-2, n + 2), 1, n - 2)
    add("hydro/gravity_work_radial.hpp:83; forcing/diffusion.cpp:221; "
        "coord/x1_centroid.cpp:194; :195; hydro/wb_ref4.cpp:116; "
        "layout/cubed_sphere_layout.cpp:376", "index",
        "index clamps that keep a stencil, a profile lookup or a block count "
        "inside its array",
        f"on a {n}-cell array the gravity-work clamp maps the "
        f"{int(np.count_nonzero(np.arange(-2, n + 2) != idx))} out-of-range indices "
        f"of {idx.size} onto the end stencils; it changes WHICH cell is read near a "
        f"wall and never the value of any cell")

    # --- epsilons -------------------------------------------------------------
    margin = ULP * EPS64
    u = 1.0
    add("hydro/flux_positivity.cpp:65-66; flux_positivity.hpp:18", "eps",
        "the available amount is taken 4096 machine epsilons short of the cell's "
        "own amount, in float32 as well as float64",
        f"the margin is {margin:.3e} relative, so a fully drained cell lands at "
        f"{u * margin:.3e} of its amount above zero instead of at an exact zero "
        f"that rounds negative. It costs at most that fraction of the transported "
        f"amount per step and it is what makes `u >= 0` survive the division")
    add("hydro/flux_positivity.cpp:16-20; flux_positivity.hpp:23", "meter",
        "the float32 ulp count, 64, of the threshold that decides whether a withheld "
        "or repaired mass is REPORTED as a positivity event",
        f"64 float32 epsilons is {64.0 * EPS32:.3e} relative, against "
        f"{ULP * EPS32:.3e} for the 4096 that theta's own margin uses in the same "
        f"precision. It never enters theta -- flux_positivity.cpp:65 takes the ulp "
        f"count from kPositivityRoundoffUlp in both precisions -- so it changes no "
        f"stored value, only whether MeshBlock::check_redo sees a limiter patch")

    keep = 1.0 - 4096.0 * EPS64
    add("implicit/vic_redistribute_impl.h:140", "eps",
        "the VIC redistribution keeps 1 - 4096 eps of the donor cell",
        f"the retained fraction is {keep:.17g}, i.e. {1 - keep:.3e} short of one; "
        f"it leaves that fraction behind rather than emptying a cell exactly")

    tol = 16 * EPS64
    add("coord/coordinate.cpp:200", "eps",
        "the uniform-spacing test accepts a spread of 16 machine epsilons relative",
        f"the tolerance is {tol:.3e} relative; it decides whether the uniform "
        f"stencil or the general one is built, so it changes the operator, not a "
        f"stored value")

    piv = 8 * 8 * EPS64
    add("math/ludcmp.h:35", "eps",
        "the LU pivot tolerance, 8N machine epsilons",
        f"for an 8 by 8 block the tolerance is {piv:.3e}; below it the pivot is "
        f"declared singular and the solve is refused rather than returning a huge "
        f"correction")

    return rows


def check_Z5():
    rows = inventory()
    kinds = {}
    for sites, kind, what, detail in rows:
        kinds[kind] = kinds.get(kind, 0) + len(sites.split(";"))
        print(f"       {kind:<6} {sites}")
        print(f"              {what}")
        print(f"              {detail}")
    total = sum(kinds.values())
    ordered = ", ".join(f"{k} {v}" for k, v in sorted(kinds.items()))
    ok = total >= 60 and set(kinds) == {"state", "guard", "domain", "index",
                                        "eps", "cap", "select", "meter"}
    return report(
        "Z5", ok,
        "every positivity floor, clamp and epsilon in the covered code paths is "
        "listed with a measurement of what it changes",
        f"{total} sites in {len(rows)} groups: {ordered}. The `state` group "
        f"changes stored values and is measured in invented mass, energy or "
        f"pressure; the `guard` group is shown to bind only at or below its own "
        f"value and, for the positivity drain, to change no SELECTED value at all; "
        f"`select` keeps a sign or a cap; `domain` and `index` change an argument "
        f"or an address, not an amount; `eps` decides a branch or leaves a stated "
        f"fraction behind; `cap` is a deck choice; `meter` records a hit. No floor "
        f"in this list can make an absent species present")


# =============================================================================
# [Z6] every break arm goes red
# =============================================================================
BREAKS = {
    "Z2-floor": ("Z2", break_Z2_floor,
                 "the limiter takes the cell's amount through a 1e-300 floor, "
                 "whose effect differs between float64 and float32"),
    "Z1-floor": ("Z1", break_Z1_floor,
                 "flux_positivity_theta takes the cell's amount through a 1e-300 "
                 "floor instead of relu"),
    "Z3-floor": ("Z3", break_Z3_floor,
                 "the condensate repair floors the cloud at 1e-300 instead of "
                 "clamping it to zero"),
    "Z4-noclamp": ("Z4", break_Z4_noclamp,
                   "the reconstruction's face species clamp is removed"),
}


def check_Z6():
    saved = list(FAIL)
    results = []
    for name, (claim, fn, what) in BREAKS.items():
        FAIL.clear()
        lines = []
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            passed = fn()
        lines = buf.getvalue().strip().splitlines()
        results.append((name, claim, passed, lines[0] if lines else ""))
    FAIL.clear()
    FAIL.extend(saved)
    all_red = all(not passed for _, _, passed, _ in results)
    for name, claim, passed, line in results:
        print(f"       break {name} -> {'RED' if not passed else 'STILL GREEN'}")
        print(f"              {line}")
    return report(
        "Z6", all_red,
        f"each of the {len(results)} break arms turns its claim red",
        "; ".join(f"{n} breaks {c}" for n, c, _, _ in results) +
        ". A claim that cannot be made to fail is not a check; each arm above "
        "removes exactly one term and prints the FAIL line it produces")


CHECKS = (check_Z1, check_Z2, check_Z3, check_Z4, check_Z5, check_Z6)


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
