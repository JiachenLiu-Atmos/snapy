"""Executable check of body heating, top cooling and bottom heating (chapter 9, _heating.qmd).

What is checked: BodyHeatImpl::forward (src/forcing/body_heat.cpp:43-55),
TopCoolImpl::forward (src/forcing/top_cool.cpp:42-56) and BotHeatImpl::forward
(src/forcing/bot_heat.cpp:42-56) at snapy@6fbb39750f62b83df096058a4746e77d0a634bc5, ported
line for line on one Cartesian column. Claims C1-C5 of the Derivation layer.

C4 and C5 port the `thickness` branch the two boundary fluxes gained later,
src/forcing/top_cool.cpp:60-74@68f4e066974ede87e4379285c4fba895b970b7f1 (and its mirror
image src/forcing/bot_heat.cpp:60-74@68f4e066974ede87e4379285c4fba895b970b7f1); the
`depth` branch they port above is unchanged there but for its line numbers,
src/forcing/top_cool.cpp:76-85@68f4e066974ede87e4379285c4fba895b970b7f1.

Run: python3 heating_check.py   (numpy only; exits with the number of failed claims)
"""
import sys

import numpy as np


def report(tag, ok, label, numbers):
    print("[%s] %s %s: %s" % (tag, "PASS" if ok else "FAIL", label, numbers))
    return ok


def top_cool(dx1f, flux, depth, dt):  # top_cool.cpp:51-54, interior cells only
    dE = np.zeros_like(dx1f)
    dz = dx1f[-1]
    dE[len(dx1f) - depth:] += dt * flux / (dz * depth)
    return dE


def check_C1_column_power_uniform():
    dx1f = np.full(20, 50.)
    dE = top_cool(dx1f, -30., 4, 1.)
    power = (dE * dx1f).sum()
    return report("C1", abs(power + 30.) < 1e-12, "uniform cells: the column loses exactly F per unit area",
                  "sum dE dz = %.12f W m^-2 for F = -30, depth 4" % power)


def check_C2_column_power_stretched():
    dx1f = 50. * 1.1**np.arange(20)  # cells growing upward by 10 %
    dE = top_cool(dx1f, -30., 4, 1.)
    power = (dE * dx1f).sum()
    ratio = power / -30.
    want = dx1f[-4:].sum() / (4. * dx1f[-1])
    return report("C2", abs(ratio - want) < 1e-14,
                  "stretched cells: the delivered power is F times mean(dz)/dz_top",
                  "ratio %.6f for 10 %% growth, depth 4 (= %.6f)" % (ratio, want))


def top_cool_thickness(x1f, flux, thickness, dt):  # top_cool.cpp:62-73 at the later sha
    hi = x1f[-1]
    lo = hi - thickness
    overlap = np.clip(np.minimum(x1f[1:], hi) - np.maximum(x1f[:-1], lo), 0., None)
    return dt * flux / thickness * overlap / np.diff(x1f)


def check_C4_thickness_column_power():
    """The layer form delivers exactly F per unit area on any x1 grid."""
    grids = {"uniform, 20 x 50 m": np.arange(21) * 50.,
             "growing 10 % upward": np.concatenate([[0.], np.cumsum(50. * 1.1**np.arange(20))]),
             "shrinking 10 % upward": np.concatenate([[0.], np.cumsum(50. * 0.9**np.arange(20))])}
    worst, detail = 0., []
    for name, x1f in grids.items():
        for thickness in (230., 7., x1f[-1] - x1f[0]):
            dE = top_cool_thickness(x1f, -30., thickness, 0.25)
            power = (dE * np.diff(x1f)).sum()
            worst = max(worst, abs(power - 0.25 * -30.))
        detail.append("%s: %.12f" % (name, power))
    return report("C4", worst < 1e-13,
                  "thickness: the column receives exactly F dt on uniform and stretched cells",
                  "max |sum dE dz - F dt| = %.1e over three grids and thicknesses 230 m, 7 m and the "
                  "whole column (F = -30, dt = 0.25, so F dt = -7.5); full-column totals %s; threshold "
                  "1e-13, round-off of a 20-term sum of order-1 numbers" % (worst, "; ".join(detail)))


def check_C5_thickness_overlap_share():
    """Each cell takes the fraction of the layer that lies inside it."""
    x1f = np.concatenate([[0.], np.cumsum(50. * 1.1**np.arange(20))])
    dz = np.diff(x1f)
    # a layer shorter than the top cell heats that cell alone
    short = top_cool_thickness(x1f, -30., 0.5 * dz[-1], 1.)
    only_top = np.count_nonzero(short) == 1 and short[-1] != 0.
    # a layer that ends inside a cell gives it its overlap fraction
    thickness = dz[-1] + 0.25 * dz[-2]
    part = top_cool_thickness(x1f, -30., thickness, 1.)
    want = -30. / thickness * (0.25 * dz[-2]) / dz[-2]
    ok = only_top and abs(part[-2] - want) < 1e-15 and part[-3] == 0.
    return report("C5", ok, "thickness: a cell takes its own overlap with the layer and no more",
                  "a layer of half the top cell heats one cell (%d nonzero); a layer reaching a quarter "
                  "into the next cell gives it %.9e against the exact %.9e, and the cell below exactly "
                  "zero" % (np.count_nonzero(short), part[-2], want))


def check_C3_body_heat_is_isochoric():
    rho, cv, dTdt, dt = 1.1, 717.5, 2.e-5, 10.
    dE = dt * dTdt * rho * cv  # body_heat.cpp:52-53
    dT = dE / (rho * cv)  # at fixed rho and constant cv
    return report("C3", abs(dT - dTdt * dt) < 1e-18, "body heat raises T by dTdt dt at fixed volume",
                  "dT = %.3e K for dTdt dt = %.3e K" % (dT, dTdt * dt))


def main():
    checks = [check_C1_column_power_uniform, check_C2_column_power_stretched,
              check_C3_body_heat_is_isochoric, check_C4_thickness_column_power,
              check_C5_thickness_overlap_share]
    sys.exit(sum(not c() for c in checks))


if __name__ == "__main__":
    main()
