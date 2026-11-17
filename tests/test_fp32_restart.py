#!/usr/bin/env python3
"""Versioned anomaly restart, exact same-dtype and cross-precision continuation."""

import glob
import os
import tempfile

import torch
from snapy import load_restart, kIPR
from test_perturbation_flux import make
from test_anomaly_storage import initial_state, interior, run_steps
from test_fp32_jupiter import run as jupiter
from test_fp32_policy import init as initialize_balanced
import test_scalar_cov_x1_split as seam
import test_perturbation_flux_split as split


class Archive(torch.nn.Module):
    def __init__(self, data):
        super().__init__()
        for name, tensor in data.items():
            self.register_buffer(name, tensor.clone())

    def forward(self, x):
        return x


def save(data, path):
    # Same TorchScript tensor archive that kintera::save_tensors writes.
    torch.jit.trace(Archive(data), torch.zeros(1), check_trace=False).save(path)


def reject(data, message):
    save(data, "invalid.restart")
    block = make(True, precision="float32", scheme=9)
    try:
        block.initialize_from_restart("invalid.restart")
    except RuntimeError as exc:
        assert message in str(exc), str(exc)
    else:
        raise AssertionError(f"accepted invalid restart: {message}")


def precision_cases(outflow=False, rest=False):
    outputs = [{"type": "restart", "dt": 1.0e9}]
    for source in ("float32", "float64"):
        block = make(True, precision=source, scheme=9, outflow=outflow, outputs=outputs)
        bv = (
            initialize_balanced(block, rest=True)
            if rest
            else block.initialize(initial_state(block))[0]
        )
        if not rest:
            bv["hydro_u"][kIPR, :, :, 15:18] += 1.0e-4
        run_steps(block, bv, 5, dt=0.05)
        saved = bv["hydro_u"].clone()
        block.make_outputs(bv, 0.25, True)
        (path,) = glob.glob("*.final.restart")
        data = load_restart(path, 0)
        assert data["hydro_restart_metadata"].tolist() == [1, 32 if source == "float32" else 64]
        run_steps(block, bv, 4, dt=0.05)
        oracle = interior(block, block.total_u(bv))
        for target in ("float32", "float64"):
            resume = make(True, precision=target, scheme=9, outflow=outflow, outputs=outputs)
            rv, time = resume.initialize_from_restart(path)
            dtype = torch.float32 if target == "float32" else torch.float64
            assert time == 0.25 and rv["hydro_u"].dtype == dtype
            assert torch.equal(rv["hydro_u"], saved.to(dtype))
            assert "hydro_restart_metadata" not in rv
            assert not any(k.startswith("hydro_ref_x") for k in rv)
            for name, field in (("u0", "hydro_ref"), ("species0", "hydro_ref_species")):
                ref = dict(resume.named_buffers())["hydro.ref." + name]
                assert ref.dtype == torch.float64 and torch.equal(ref.cpu(), data[field])
            if outflow:
                assert rv["boundary_reference_w"].dtype == torch.float64
                assert torch.equal(rv["boundary_reference_w"].cpu(), data["boundary_reference_w"])
            run_steps(resume, rv, 4, dt=0.05)
            if source == target:
                assert torch.equal(rv["hydro_u"], bv["hydro_u"])
            total = interior(resume, resume.total_u(rv))
            scale = oracle.abs().amax((1, 2, 3), keepdim=True).clamp_min(1.0)
            error = ((total - oracle).abs() / scale).max().item()
            assert error < 2.0e-5, error
            print(f"restart {source}->{target}: normalized error={error:.3e}", flush=True)
        legacy = {
            k: v
            for k, v in data.items()
            if k != "hydro_restart_metadata" and not k.startswith("hydro_ref_x")
        }
        save(legacy, "legacy.restart")
        resume = make(True, precision=source, scheme=9, outflow=outflow)
        rv, _ = resume.initialize_from_restart("legacy.restart")
        assert torch.equal(rv["hydro_u"], saved)
        run_steps(resume, rv, 4, dt=0.05)
        assert torch.equal(rv["hydro_u"], bv["hydro_u"])
        changed = dict(data)
        changed["hydro_restart_metadata"] = torch.tensor([2, 32], dtype=torch.int64)
        reject(changed, "unsupported hydro restart metadata version")
        changed["hydro_restart_metadata"] = torch.tensor([1, 16], dtype=torch.int64)
        reject(changed, "dtype/shape disagrees")
        changed = dict(data)
        del changed["hydro_u_stored"]
        reject(changed, "metadata requires")
        changed = dict(data)
        changed["hydro_ref_x1f"] = changed["hydro_ref_x1f"] + 1.0
        reject(changed, "reference grid mismatch")
        changed = dict(legacy)
        del changed["hydro_ref_species"]
        reject(changed, "original hydro_ref")
        resume = make(True, precision=source, scheme=9, outflow=outflow)
        resume.module("coord").buffer("x1f").add_(1.0)
        try:
            resume.initialize_from_restart(path)
        except RuntimeError as exc:
            assert "reference grid mismatch" in str(exc)
        else:
            raise AssertionError("accepted a changed grid")
        os.unlink(path)


def main():
    torch.set_num_threads(1)
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            precision_cases()
            precision_cases(outflow=True)
            precision_cases(rest=True)
            original_config = seam.config
            seam.config = split.config
            split.precision = "float32"
            try:
                full = seam.run(4, "seam-step", os.environ.get("SNAP_TEST_DEVICE", "cpu"))
                resumed = seam.run(
                    4, "seam-step", os.environ.get("SNAP_TEST_DEVICE", "cpu"), restart_at=5
                )
                for key in ("stored", "s", "vapor", "rho", "p"):
                    assert torch.equal(full[key], resumed[key]), key
                print("four-block FP32 restart: bit-identical", flush=True)
                for path in glob.glob("*.final.restart"):
                    for rank in range(4):
                        assert load_restart(path, rank)["hydro_restart_metadata"].tolist() == [
                            1,
                            32,
                        ]
                    os.unlink(path)
            finally:
                seam.config = original_config
            for precision in ("float32", "float64"):
                full = jupiter(12, precision)
                resumed = jupiter(12, precision, restart_at=6)
                for field in full:
                    assert torch.equal(full[field], resumed[field]), (precision, field)
                print(
                    f"Jupiter hydro/scalar/chemistry restart {precision}: bit-identical", flush=True
                )
                for path in glob.glob("*.final.restart"):
                    os.unlink(path)
        finally:
            os.chdir(cwd)


if __name__ == "__main__":
    main()
