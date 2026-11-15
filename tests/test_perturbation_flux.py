#!/usr/bin/env python3
"""M3 integration: full RHS, unbalanced rest, heating, conversion and VIC."""
import argparse
import os
import tempfile
import torch
import yaml
from test_anomaly_storage import initial_state, interior, run_steps
from snapy import MeshBlock, MeshBlockOptions, kIDN, kIPR, kIV1, balance_column


test_covariance = False

def make(split, order=2, scheme=0, nh=1., gravity=True, outputs=None, stretched=False, sediment=False, anomaly=True, solver="hllc", precision="float64", one_column=False, implicit_precision=None, outflow=False):
    with open(os.path.join(os.path.dirname(__file__), "test_anomaly_storage.yaml")) as f:
        cfg = yaml.safe_load(f)
    cfg["dynamics"].update({"anomaly": anomaly, "perturbation-flux": split,
                            "x1-mass-covariance": test_covariance,
                            "conversion-order": order, "precision": precision, "riemann-solver": {"type": solver}})
    if stretched:
        cfg["geometry"]["cells"]["x1rat"] = 1.02
    cfg["geometry"]["cells"]["nx3"] = 4
    cfg["geometry"]["bounds"]["x3max"] = 4000.
    cfg["boundary-condition"]["external"].update({"x3-inner": "periodic", "x3-outer": "periodic"})
    if one_column:
        cfg["geometry"]["cells"].update({"nx2": 1, "nx3": 1})
    if outflow:
        cfg["boundary-condition"]["external"].update({"x1-inner": "outflow", "x1-outer": "outflow"})
    cfg["integration"]["implicit-scheme"] = scheme
    if implicit_precision is not None:
        cfg["integration"]["implicit-precision"] = implicit_precision
    if sediment:
        cfg["sedimentation"] = {"radius": {}, "density": {}, "const-vsed": {"H2O(l)": -2.}}
    if outputs is not None:
        cfg["outputs"] = outputs
    cfg["forcing"]["const-gravity"]["non-hydrostatic"] = nh
    if not gravity:
        cfg["forcing"]["const-gravity"]["grav1"] = 0.
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(cfg, f)
        path = f.name
    try:
        return MeshBlock(MeshBlockOptions.from_yaml(path))
    finally:
        os.unlink(path)


def run(split, order, scheme, nh, kind, gravity=True, emulate=False):
    block = make(split, order, scheme, nh, gravity, stretched=kind == "stretched", sediment=kind == "sediment")
    w = initial_state(block)["hydro_w"]
    # Exercise the third direction too.
    w[kIV1+2] = .05 * torch.sin(block.module("coord").buffer("x3v").view(-1, 1, 1) / 4000. * 2.*torch.pi)
    if kind == "sediment":
        w[6].fill_(.001)
    if kind in ("rest", "balanced"):
        w[kIV1:kIV1+3].zero_()
    if kind == "balanced":
        # Dry rest removes condensation as a physical source of motion.
        w[5:].zero_()
        col = w[:, :1, :1, 3:-3].contiguous()
        wb = balance_column(col, block.module("coord").buffer("dx1f")[3:-3].contiguous(), 10.44, rtol=1.e-13, max_iter=400)[0]
        w[..., 3:-3] = wb
    bv, _ = block.initialize({"hydro_w": w})
    before = interior(block, block.total_u(bv)).clone()
    if kind == "heat":
        # Local finite energy injection in stored anomaly space.
        bv["hydro_u"][kIPR, :, :, 15:18] += 1.e-4
    for _ in range(20 if emulate else 4):
        if emulate:
            bv["hydro_u"].copy_(bv["hydro_u"].float().double())
        run_steps(block, bv, 1, dt=.05)
    after = interior(block, block.total_u(bv)).clone()
    assert torch.isfinite(after).all()
    return after, before


def restart():
    import glob
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            outputs = [{"type": "restart", "dt": 1.e9}]
            ref = make(True, outputs=outputs)
            bv, _ = ref.initialize(initial_state(ref))
            run_steps(ref, bv, 3, dt=.05)
            ref.make_outputs(bv, .15, True)
            path, = glob.glob("*.final.restart")
            run_steps(ref, bv, 3, dt=.05)
            cont = make(True, outputs=outputs)
            cv, _ = cont.initialize_from_restart(path)
            run_steps(cont, cv, 3, dt=.05)
            assert torch.equal(bv["hydro_u"], cv["hydro_u"])
            print("perturbation restart: bit-identical continuation")
        finally:
            os.chdir(cwd)


def main():
    global test_covariance
    parser = argparse.ArgumentParser()
    parser.add_argument("--x1-mass-covariance", action="store_true")
    test_covariance = parser.parse_args().x1_mass_covariance
    torch.set_num_threads(1)
    for invalid in ({"anomaly": False}, {"solver": "lmars"}):
        try:
            make(True, **invalid)
        except RuntimeError as error:
            assert "requires anomaly storage and HLLC" in str(error)
        else:
            raise AssertionError(f"unsupported configuration accepted: {invalid}")
    for order, scheme, nh, kind, gravity in [
        (2, 0, 1., "wind", True), (4, 0, 1., "wind", True),
        (2, 9, 1., "wind", True), (2, 0, .3, "wind", True),
        (2, 0, 1., "rest", True), (2, 0, 1., "balanced", True),
        (2, 0, 1., "heat", True), (2, 0, 1., "wind", False),
    ]:
        a, before = run(False, order, scheme, nh, kind, gravity)
        b, _ = run(True, order, scheme, nh, kind, gravity)
        scale = before.abs().amax(dim=(1, 2, 3), keepdim=True).clamp_min(1.)
        error = ((a-b).abs()/scale).max().item()
        print(f"order={order} scheme={scheme} nh={nh} kind={kind} gravity={gravity}: {error:.3e}")
        assert error < 2.e-10, error
        if kind == "rest":
            assert b[kIV1].abs().max() > 1.e-6, "unbalanced background force was suppressed"
        if kind == "balanced":
            assert b[kIV1].abs().max() < 2.e-9, b[kIV1].abs().max()
        # All walls are closed and horizontal axes periodic.
        mass0 = before[kIDN].sum() + before[5:].sum()
        mass1 = b[kIDN].sum() + b[5:].sum()
        assert abs((mass1-mass0)/mass0) < 2.e-13
        if not gravity:
            assert abs((b[kIPR].sum()-before[kIPR].sum())/before[kIPR].sum()) < 2.e-13
    a, before = run(False, 2, 0, 1., "sediment")
    b, _ = run(True, 2, 0, 1., "sediment")
    scale = before.abs().amax(dim=(1, 2, 3), keepdim=True).clamp_min(1.)
    error = ((a-b).abs()/scale).max().item()
    print(f"moist sedimentation: {error:.3e}")
    assert error < 2.e-10
    for order in (2, 4):
        a, before = run(False, order, 0, 1., "stretched")
        b, _ = run(True, order, 0, 1., "stretched")
        scale = before.abs().amax(dim=(1, 2, 3), keepdim=True).clamp_min(1.)
        error = ((a-b).abs()/scale).max().item()
        print(f"stretched order={order}: {error:.3e}")
        assert error < 2.e-10
    stored, _ = run(True, 2, 0, 1., "balanced", emulate=True)
    noise = stored[kIV1].abs().max().item()
    print(f"FP32 storage-emulated balanced rest, 20 steps: max momentum={noise:.3e}")
    assert noise < 2.e-9
    restart()
    print("perturbation flux integration passed")


if __name__ == "__main__":
    main()
