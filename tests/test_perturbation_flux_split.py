#!/usr/bin/env python3
"""M3: reuse coupled scalar/species seam checks with anomaly HLLC/WENO5."""
import argparse
import torch
import test_scalar_cov_x1_split as seam

original_config = seam.config
precision = "float64"


def config(nb1):
    cfg = original_config(nb1)
    cfg["dynamics"].update({"precision": precision, "anomaly": True, "perturbation-flux": True,
                            "riemann-solver": {"type": "hllc"},
                            "reconstruct": {"vertical": {"type": "weno5", "scale": False},
                                            "horizontal": {"type": "weno5", "scale": False}}})
    # Keep the scalar and the hydro tracer reconstruction identical.
    cfg["scalar"]["reconstruct"] = {"type": "weno5"}
    return cfg


def main():
    global precision
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", choices=("float32", "float64"), default="float64")
    precision = parser.parse_args().precision
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    seam.config = config
    for profile in seam.PROFILES:
        one = seam.run(1, profile, "cpu")
        four = seam.run(4, profile, "cpu")
        for key in ("s", "vapor", "rho", "p"):
            error = seam.rel(four[key], one[key])
            print(profile, key, error)
            assert error < (2.e-5 if precision == "float32" else 3.e-12)
        for result in (one, four):
            assert result["finite"]
            assert result["c"] < (2.e-6 if precision == "float32" else 2.e-13)
    print("perturbation split-column conservation passed")


if __name__ == "__main__":
    main()
