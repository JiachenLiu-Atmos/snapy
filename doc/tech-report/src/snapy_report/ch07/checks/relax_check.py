"""Executable check of the bottom relaxations (chapter 9, _relax.qmd).

What is checked: the at-face target and gain of RelaxBotTempImpl::forward
(src/forcing/relax_bot_temp.cpp:80-104) and the mass bookkeeping of RelaxBotCompImpl::forward
(src/forcing/relax_bot_comp.cpp:113-117) at snapy@6fbb39750f62b83df096058a4746e77d0a634bc5,
ported line for line. Claims C1-C4.

Run: python3 relax_check.py   (numpy only; exits with the number of failed claims)
"""
import sys

import numpy as np


def report(tag, ok, label, numbers):
    print("[%s] %s %s: %s" % (tag, "PASS" if ok else "FAIL", label, numbers))
    return ok


def at_face(x1f_il, x1v_il, x1v_il1, T0, T1):  # relax_bot_temp.cpp:98-100
    a = (x1v_il - x1f_il) / (x1v_il1 - x1v_il)
    return (1. + a) * T0 - a * T1, 1. / (1. + a), a


def check_C1_linear_exact_any_spacing():
    rng = np.random.default_rng(8)
    worst = 0.
    for _ in range(100):
        w0, w1 = 0.5 + rng.random(2)
        x1f, x1v0, x1v1 = 0., w0 / 2., w0 + w1 / 2.
        c, b = 250. + 50. * rng.random(), rng.standard_normal()
        target, _, _ = at_face(x1f, x1v0, x1v1, c + b * x1v0, c + b * x1v1)
        worst = max(worst, abs(target - c))
    return report("C1", worst < 1e-12, "the at-face target is exact for a linear T",
                  "max error %.1e K over 100 random spacings" % worst)


def check_C2_uniform_weights():
    target, gain, a = at_face(0., 0.5, 1.5, 1., 0.)
    ok = a == 0.5 and target == 1.5 and abs(gain - 2. / 3.) < 1e-16
    return report("C2", ok, "uniform cells: a = 1/2, target 1.5 T0 - 0.5 T1, gain 2/3",
                  "a = %.2f, target(T0=1, T1=0) = %.2f, gain = %.6f" % (a, target, gain))


def check_C3_face_relaxes_at_one_over_tau():
    tau, btemp = 50., 300.
    T0, T1, dt = 290., 289., 1e-3
    target, gain, a = at_face(0., 0.5, 1.5, T0, T1)
    dT0 = dt / tau * gain * (btemp - target)  # heating / (rho cv)
    new_face, _, _ = at_face(0., 0.5, 1.5, T0 + dT0, T1)
    rate = (new_face - target) / dt
    want = (btemp - target) / tau
    return report("C3", abs(rate / want - 1.) < 1e-9, "with the gain, the face T relaxes at 1/tau",
                  "dT_face/dt = %.6f K/s, (btemp - T_face)/tau = %.6f K/s" % (rate, want))


def check_C4_composition_swap_conserves_mass():
    rng = np.random.default_rng(6)
    rho = 1. + rng.random()
    y_now = np.array([0.02, 0.005])
    y_target = np.array([0.03, 0.0])
    delta_dry = -rho * (y_target - y_now).sum()  # relax_bot_comp.cpp:114
    delta_y = rho * (y_target - y_now)  # :115
    total = delta_dry + delta_y.sum()
    return report("C4", abs(total) < 1e-17, "relax-bot-comp moves mass between dry air and species only",
                  "dry %.4e + species %.4e = %.1e" % (delta_dry, delta_y.sum(), total))


def main():
    checks = [check_C1_linear_exact_any_spacing, check_C2_uniform_weights,
              check_C3_face_relaxes_at_one_over_tau, check_C4_composition_swap_conserves_mass]
    sys.exit(sum(not c() for c in checks))


if __name__ == "__main__":
    main()
