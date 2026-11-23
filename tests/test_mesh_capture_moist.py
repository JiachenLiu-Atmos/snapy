"""V3 B3: a moist multi-block Mesh captures and equals eager bitwise.

test_mesh_capture covers multi-block capture on a dry deck. This is the
moist case: the deck of test_sk3_capture_moist (H2O vapor and cloud with a
nucleation reaction, so the last stage's saturation adjustment condenses in
the cold bubble and aloft), split into two blocks along x2 in one process
(Mesh, blocks_per_process 2). Strict cuda-graph against an eager Mesh, FP64
and FP32, 10 steps with dt changing every step, bitwise after every step,
with flux-covariance on (the default: the hydro positivity limiter's theta
and species syncs and the scalar's theta sync exchange across the blocks
inside the stage, three cuts per block and stage) and off (no cut). Prints
max |graph - eager| per variable and row, and the condensate.

  python test_mesh_capture_moist.py
"""
import os
import sys
import tempfile

import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
QVAPOR = 0.005
NSTEPS = 10
KEYS = ("hydro_u", "hydro_w", "scalar_s", "scalar_r")


def deck(precision, execution, strict, flux_covariance):
    root = os.path.dirname(HERE)
    cfg = yaml.safe_load(open(os.path.join(root, "examples/straka.yaml")))
    moist = yaml.safe_load(open(os.path.join(HERE, "test_anomaly_storage.yaml")))
    cfg["reference-state"] = moist["reference-state"]
    cfg["species"] = moist["species"]
    cfg["reactions"] = moist["reactions"]
    cfg["distribute"].update({"layout": "slab", "nb2": 2, "nb3": 1,
                              "blocks_per_process": 2, "verbose": False})
    cfg["outputs"] = []
    cfg["dynamics"].update({"anomaly": True, "perturbation-flux": True,
                            "precision": precision,
                            "flux-covariance": flux_covariance,
                            "riemann-solver": {"type": "hllc"}})
    cfg["dynamics"]["equation-of-state"] = {
        "type": "ideal-moist", "limiter": False,
        "density-floor": 1.0e-10, "pressure-floor": 1.0e-10}
    cfg["integration"]["ncycle_out"] = 0
    cfg["integration"]["execution"] = execution
    cfg["integration"]["execution-strict"] = strict
    return cfg


def make(precision, execution, strict, flux_covariance):
    from snapy import Mesh, MeshOptions, kICY, kIDN, kIPR

    os.environ["DEVICE"] = "cuda"
    os.environ["DEVICE_ID"] = "0"
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(deck(precision, execution, strict, flux_covariance), f)
    try:
        opts = MeshOptions.from_yaml(f.name)
        opts.blocks_per_process(2)
        mesh = Mesh(opts)
    finally:
        os.unlink(f.name)
    blocks = list(mesh.blocks)
    for b in blocks:
        b.to(torch.device("cuda", 0))
    mvars = []
    for b in blocks:
        thermo = b.module("hydro.eos.thermo")
        rd = 8.31446261815324 * thermo.buffer("inv_mu")[0].item()
        cp = 3.5 * rd
        z = b.module("coord").buffer("x1v").view(1, 1, -1).double()
        x = b.module("coord").buffer("x2v").view(1, -1, 1).double()
        w = b.module("hydro").buffer("D").double().clone().zero_()
        temp = 300.0 - 9.8 * z / cp
        w[kIPR] = 1.0e5 * (temp / 300.0) ** (cp / rd)
        radius = ((x / 4000.0) ** 2 + ((z - 3000.0) / 2000.0) ** 2).sqrt()
        temp = temp + torch.where(radius <= 1, -7.5 * (torch.cos(torch.pi * radius) + 1.0), 0.0)
        w[kIDN] = w[kIPR] / (rd * temp)
        w[kICY] = QVAPOR  # vapor; the cloud row stays zero: the adjustment makes it
        ratio = torch.zeros_like(w[:1]) + (z / 6400.0).clamp(0.0, 1.0)
        mvars.append({"hydro_w": w, "scalar_r": ratio})
    mvars, _ = mesh.initialize(mvars)
    return mesh, blocks, mvars


def check(precision, flux_covariance):
    from snapy import kICY

    graph, gb, gv = make(precision, "cuda-graph", True, flux_covariance)
    eager, _, ev = make(precision, "eager", False, flux_covariance)
    for i, b in enumerate(gb):
        why = b.execution_capability(gv[i])["eager_reasons"]
        assert not why, (precision, flux_covariance, i, why)
    dt0 = eager.max_time_step(ev)
    maxabs = {}
    for n in range(NSTEPS):
        dt = 0.5 * dt0 * (1.0 + 0.1 * ((3 * n) % 4))
        for s in range(3):
            graph.forward(gv, dt, s)
            eager.forward(ev, dt, s)
        for i in range(len(gb)):
            for k in KEYS:
                d = (gv[i][k].double() - ev[i][k].double()).abs()
                for j in range(d.shape[0]):
                    name = f"{k}[{j}]"
                    maxabs[name] = max(maxabs.get(name, 0.0), d[j].max().item())
                assert torch.equal(gv[i][k], ev[i][k]), (
                    precision, flux_covariance, "step", n, "block", i, k, d.max().item())
    r = graph.execution_report()
    nb = len(gb)
    assert r["captures"] == 3 * nb and r["replays"] == 3 * nb * NSTEPS, r
    assert r["eager"] == 0, r
    assert (r["cuts"] > 0) == flux_covariance, r
    cloud = sum(int((v["hydro_w"][kICY + 1] > 0).sum().item()) for v in ev)
    assert cloud > 0, "no condensate: the saturation adjustment did nothing"
    worst = max(maxabs.values())
    print(precision, "flux-covariance" if flux_covariance else "no flux-covariance",
          ": graph equals eager over", NSTEPS, "steps, max |g - e| =", worst,
          "(per row:", {k: v for k, v in maxabs.items()}, "), cloud cells:", cloud,
          ", report:", {k: r[k] for k in ("captures", "replays", "cuts", "eager")})


def main():
    if not torch.cuda.is_available() or os.environ.get("SNAPY_BUILD_CUDA", "1") == "0":
        print("SKIP: CUDA unavailable")
        return
    for precision in ("float64", "float32"):
        for flux_covariance in (True, False):
            check(precision, flux_covariance)
    print("PASSED")


if __name__ == "__main__":
    sys.exit(main())
