#!/usr/bin/env python3
"""FaceFloor counts on the FP32 split path (V3 C5): the floors reconstruct_sweep
applies itself to the total face pressure and density count into the monitor's
face_floor.p and face_floor.rho.

The anomaly-storage deck, dry, gravity-free, perturbation-flux, x2 the only
active flux direction. With the pressure floor raised to 5e4 Pa, above the
upper column's pressure, the FP32 sweeps floor the upper faces; one stage must
count them. With the deck's own floor (1e-10 Pa) nothing is floored and
nothing is counted. The FP64 run of the same block counts through the
reconstruction (C5 part 2) and is printed for reference.
"""
import argparse
import os
import sys
import tempfile

import torch
import yaml
from snapy import MeshBlock, MeshBlockOptions, kIV1

from test_anomaly_storage import initial_state

# the monitor's counters, in the order HydroImpl::reset requests them: the
# seven ef.* names, then face_floor.rho, face_floor.p, face_floor.species,
# riemann.pstar_clamp and riemann.llf
FACE_FLOOR_RHO, FACE_FLOOR_P = 7, 8


def block(precision, pressure_floor, device):
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "test_anomaly_storage.yaml")) as f:
        cfg = yaml.safe_load(f)
    cfg["dynamics"].update({"anomaly": True, "perturbation-flux": True,
                            "precision": precision, "flux-covariance": False,
                            "riemann-solver": {"type": "hllc"},
                            "disable-flux-x1": True, "disable-flux-x3": True})
    cfg["dynamics"]["equation-of-state"]["pressure-floor"] = pressure_floor
    cfg["geometry"]["cells"]["nx3"] = 4
    cfg["geometry"]["bounds"]["x3max"] = 4000.0
    cfg["boundary-condition"]["external"].update({"x3-inner": "periodic",
                                                  "x3-outer": "periodic"})
    cfg["forcing"]["const-gravity"]["grav1"] = 0.0
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(cfg, f)
        path = f.name
    try:
        b = MeshBlock(MeshBlockOptions.from_yaml(path))
        b.to(torch.device(device))
        return b
    finally:
        os.unlink(path)


def counts(precision, pressure_floor, device):
    b = block(precision, pressure_floor, device)
    w = initial_state(b)["hydro_w"]
    w[5:].zero_()
    w[kIV1:kIV1 + 3].zero_()
    state, _ = b.initialize({"hydro_w": w})
    b.forward(state, 0.01, 0)
    c = dict(b.named_buffers())["hydro.monitor_counters"].cpu()
    assert c.numel() >= 12, c.numel()
    return int(c[FACE_FLOOR_RHO]), int(c[FACE_FLOOR_P])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = ap.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("SKIP: CUDA unavailable")
        return 125
    os.environ["SNAP_TEST_DEVICE"] = args.device
    torch.set_num_threads(1)
    _, p_deck = counts("float32", 1.0e-10, args.device)
    _, p_raised = counts("float32", 5.0e4, args.device)
    _, p_fp64 = counts("float64", 5.0e4, args.device)
    print(f"FP32 face_floor.p: {p_deck} at the deck's floor, {p_raised} at 5e4 Pa"
          f" (FP64 at 5e4 Pa: {p_fp64})")
    failures = []
    if p_deck != 0:
        failures.append(f"{p_deck} floors counted at the deck's floor")
    if p_raised <= 0:
        failures.append("the FP32 split path floors faces at 5e4 Pa but "
                        "face_floor.p is 0")
    if failures:
        print("FAIL: " + "; ".join(failures))
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
