#!/usr/bin/env python3
"""A vapour-repair failure must be reported once and never resurface in a later check_redo.

The repairs inside a MeshBlock stage count their failures (on CUDA, on the device) and
check_redo reports them. Three arms, each ending with a check_redo on a healthy state
that must return 0 without raising:
  direct: eos.compute("U->W") on an unrepairable column raises at once;
  check:  check_redo on an unrepairable state redoes the step (#317), then the
          healthy state passes;
  stage:  a step whose stages hit an unrepairable column is abandoned before its
          check_redo (as when a driver stops on an error); the next step's
          stages and check_redo are healthy.

  python test_fix_vapor_stale_count.py [--device cuda]
"""
import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def setup(device, yaml_file):
    from snapy import MeshBlock, MeshBlockOptions, kICY, kIDN, kIPR

    block = MeshBlock(MeshBlockOptions.from_yaml(yaml_file))
    block.to(torch.device(device))
    eos = next(m for _, m in block.named_modules() if type(m).__name__ == "IdealMoist")
    w = dict(block.named_buffers())["hydro.D"].clone().zero_()
    w[kIDN] = 1.0
    w[kIPR] = 1.0e5
    w[kICY] = 1.0e-3
    block_vars, _ = block.initialize({"hydro_w": w})
    return block, eos, block_vars


def unrepairable(u):
    from snapy import kICY

    bad = u.clone()
    k, j = u.size(1) // 2, u.size(2) // 2
    bad[kICY, k, j, :] = -1.0e-6  # net deficit over the whole column
    return bad


def step(block, block_vars):
    dt = block.max_time_step(block_vars)
    for stage in range(len(block.intg.stages)):
        block.forward(block_vars, dt, stage)


def healthy_check(block, block_vars, what):
    try:
        err = block.check_redo(block_vars)
    except RuntimeError as e:
        return "healthy check_redo after %s raised: %s" % (what, str(e).splitlines()[0])
    if err != 0:
        return "healthy check_redo after %s returned %d" % (what, err)
    return None


def direct_arm(device, yaml_file):
    block, eos, block_vars = setup(device, yaml_file)
    try:
        eos.compute("U->W", [unrepairable(block_vars["hydro_u"])])
        return "eos.compute(U->W) on an unrepairable column did not raise"
    except RuntimeError as e:
        if "Failed to fix vapor" not in str(e):
            return "unexpected error: " + str(e).splitlines()[0]
    return healthy_check(block, block_vars, "a failed direct U->W")


def check_arm(device, yaml_file):
    block, _, block_vars = setup(device, yaml_file)
    u0 = block_vars["hydro_u"].clone()
    block_vars["hydro_u"].copy_(unrepairable(u0))
    try:
        err = block.check_redo(block_vars)
    except RuntimeError as e:
        return "check_redo on an unrepairable state raised: " + str(e).splitlines()[0]
    if err != 1:
        return "check_redo on an unrepairable state returned %d, not a redo" % err
    block_vars["hydro_u"].copy_(u0)
    return healthy_check(block, block_vars, "a reported failure")


def stage_arm(device, yaml_file):
    block, _, block_vars = setup(device, yaml_file)
    u0 = block_vars["hydro_u"].clone()
    w0 = block_vars["hydro_w"].clone()
    block_vars["hydro_u"].copy_(unrepairable(u0))
    try:
        step(block, block_vars)  # abandoned: no check_redo for this step
    except RuntimeError as e:
        if "Failed to fix vapor" not in str(e):
            return "unexpected error: " + str(e).splitlines()[0]
    block_vars["hydro_u"].copy_(u0)
    block_vars["hydro_w"].copy_(w0)
    step(block, block_vars)
    return healthy_check(block, block_vars, "an abandoned failed step")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--yaml", default=os.path.join(HERE, "test_fix_vapor_reports_failure.yaml"))
    args = ap.parse_args()
    if args.device.startswith("cuda") and (os.environ.get("SNAPY_BUILD_CUDA", "1") == "0" or not torch.cuda.is_available()):
        print("SKIP: cuda requested but not available")
        return 125

    failures = []
    for name, arm in (("direct", direct_arm), ("check", check_arm), ("stage", stage_arm)):
        msg = arm(args.device, args.yaml)
        print("%-6s %s" % (name, "PASS" if msg is None else "FAIL: " + msg))
        if msg is not None:
            failures.append(name)
    if failures:
        print("FAIL (%s): %s" % (args.device, ", ".join(failures)))
        return 1
    print("PASS (%s): every vapour failure reported once, none stale" % args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
