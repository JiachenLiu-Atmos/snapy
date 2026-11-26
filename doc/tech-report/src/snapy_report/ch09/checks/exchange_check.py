"""Executable check of where the ghost exchange sits in a stage (chapter 7, _exchange.qmd).

What is checked: the invariant of the section, that the ghost cells a stage reads equal the neighbour's interior
after every update that precedes the stage, for the two orders at
snapy@e894700ff7aee30b52882e5202b16461413780b0: exchange before the stage (MeshBlockImpl::forward,
src/mesh/meshblock.cpp:545-552) and exchange after it (MeshImpl::forward, src/mesh/mesh.cpp:331-365). A periodic
one-dimensional domain split into two blocks with one ghost cell each, a first-order upwind stage, and a host-side
source applied between steps stand in for the solver. Claims C1-C3.

Run: python3 exchange_check.py   (numpy only; exits with the number of failed claims)
"""
import sys

import numpy as np


def report(tag, ok, label, numbers):
    print("[%s] %s %s: %s" % (tag, "PASS" if ok else "FAIL", label, numbers))
    return ok


N, C, STEPS = 16, 0.4, 12          # cells per domain, Courant number, steps


def stage_single(u):
    """The reference: one periodic domain, first-order upwind for a positive velocity."""
    return u - C * (u - np.roll(u, 1))


def stage_block(ub):
    """One block with one ghost on each side; only the left ghost is read by the upwind stage."""
    out = ub.copy()
    out[1:-1] = ub[1:-1] - C * (ub[1:-1] - ub[:-2])
    return out


def exchange(blocks):
    a, b = blocks
    a[0], a[-1] = b[-2], b[1]
    b[0], b[-1] = a[-2], a[1]


def split(u):
    h = N // 2
    return [np.concatenate([[0.], u[:h], [0.]]), np.concatenate([[0.], u[h:], [0.]])]


def join(blocks):
    return np.concatenate([b[1:-1] for b in blocks])


def run(order, source_cell=None):
    x = np.arange(N)
    u = np.exp(-0.5 * ((x - 5.) / 2.) ** 2)
    ref = u.copy()
    blocks = split(u)
    exchange(blocks)                              # both drivers start from exchanged ghosts
    for _ in range(STEPS):
        if source_cell is not None:               # host-side operator-split source between steps
            ref[source_cell] += 0.1
            k, i = divmod(source_cell, N // 2)
            blocks[k][1 + i] += 0.1
        ref = stage_single(ref)
        if order == "before":
            exchange(blocks)
            blocks = [stage_block(b) for b in blocks]
        else:
            blocks = [stage_block(b) for b in blocks]
            exchange(blocks)
    return join(blocks), ref


def check_C1_both_orders_exact_without_host_sources():
    errs = [np.abs(np.subtract(*run(o))).max() for o in ("before", "after")]
    return report("C1", max(errs) == 0., "with no update between steps both orders give the single-domain result bit "
                  "for bit", "max difference: exchange-before %.1e, exchange-after %.1e" % tuple(errs))


def check_C2_host_source_at_the_seam():
    seam = N // 2 - 1                              # the last cell of block 0, read as the ghost of block 1
    eb = np.abs(np.subtract(*run("before", seam))).max()
    ea = np.abs(np.subtract(*run("after", seam))).max()
    ok = eb == 0. and ea > 1e-3
    return report("C2", ok, "a source applied between steps in the cell next to the seam: exchange-before still matches, "
                  "exchange-after reads a stale ghost", "max difference: before %.1e, after %.3e" % (eb, ea))


def check_C3_interior_source_is_harmless():
    ea = np.abs(np.subtract(*run("after", 3))).max()
    return report("C3", ea == 0., "the same source in a cell whose value is not a ghost of the other block changes "
                  "nothing for exchange-after", "max difference %.1e" % ea)


def main():
    checks = [check_C1_both_orders_exact_without_host_sources, check_C2_host_source_at_the_seam,
              check_C3_interior_source_is_harmless]
    sys.exit(sum(not c() for c in checks))


if __name__ == "__main__":
    main()
