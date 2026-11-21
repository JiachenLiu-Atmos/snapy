"""The closed-box gravity column of the (removed, V3 X5) gravity-work fixer
test, kept for the tests that build on it: a moving isentropic column (gamma
1.4, Rd 287, g 9.8, Ts 300 K, density ratio e^3) between reflecting x1 walls.
"""
import math

GAMMA, RD, GRAV, TS, PS = 1.4, 287.0, 9.8, 300.0, 1.0e5
CP = GAMMA * RD / (GAMMA - 1.0)
LZ = TS * (1.0 - math.exp(-3.0 * (GAMMA - 1.0))) * CP / GRAV  # rho_top / rho_bot = e^-3
NZ, MACH = 32, 1.0e-2
TOL = 1.0e-12


def config(gravity, x1bc="reflecting", scheme=0):
    rgas = 8.31446261815324
    return {
        "geometry": {"type": "cartesian",
                     "bounds": {"x1min": 0.0, "x1max": LZ, "x2min": 0.0, "x2max": 1.0,
                                "x3min": 0.0, "x3max": 1.0},
                     "cells": {"nx1": NZ, "nx2": 1, "nx3": 1, "nghost": 3}},
        "dynamics": {
            "equation-of-state": {"type": "ideal-gas", "gammad": GAMMA, "weight": rgas / RD,
                                  "density-floor": 1.e-12, "pressure-floor": 1.e-12,
                                  "temperature-floor": 1.e-6, "limiter": True},
            "reconstruct": {"vertical": {"type": "weno5", "scale": True, "shock": False},
                            "horizontal": {"type": "weno5", "scale": True, "shock": False}},
            "riemann-solver": {"type": "lmars"}},
        "boundary-condition": {"external": {"x1-inner": x1bc, "x1-outer": x1bc,
                                            "x2-inner": "periodic", "x2-outer": "periodic",
                                            "x3-inner": "periodic", "x3-outer": "periodic"}},
        "integration": {"type": "rk3", "cfl": 0.4, "implicit-scheme": scheme, "nlim": -1, "tlim": 1.e9},
        # face, the one gravity-work form since V3 I-7
        "forcing": {"const-gravity": dict({"grav1": -GRAV, "gravity-work": "face"}, **gravity)},
    }
