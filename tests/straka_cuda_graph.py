"""Experimental graph executor for run_fp32_straka's fixed dry setup only.

CFL, redo and finite checks stay in the runner. Not a general MeshBlock API:
limiter host queues, chemistry, implicit solves and distributed exchange are
not supported. Construct again after configuration/reference changes.
"""

import time

import torch


class StrakaGraphStep:
    def __init__(self, block, state, dt, config):
        dynamics = config["dynamics"]
        if not (
            state["hydro_u"].is_cuda
            and config["geometry"]["type"] == "cartesian"
            and len(config["species"]) == 1
            and not config.get("reactions")
            and not dynamics["equation-of-state"].get("limiter", False)
            and dynamics["equation-of-state"]["type"] == "ideal-moist"
            and dynamics["riemann-solver"]["type"] == "hllc"
            and config["integration"].get("implicit-scheme", 0) == 0
            and config["integration"]["type"] == "rk3"
            and all(config["distribute"].get(k, 1) == 1 for k in ("nb1", "nb2", "nb3"))
            and set(config.get("forcing", {})) == {"const-gravity"}
            and "solid" not in state
        ):
            raise ValueError(
                "CUDA graph experiment supports only the dry explicit Straka configuration"
            )
        self.block, self.state, self.dt = block, state, dt
        self.graph = None
        self.setup_seconds = 0.0
        self.captures = 0
        self.replays = 0
        self._capture()

    def _eager(self, dt):
        for stage in range(len(self.block.intg.stages)):
            self.block.forward(self.state, dt, stage)

    def _signature(self):
        return {k: (v.data_ptr(), tuple(v.shape), v.dtype, v.device) for k, v in self.state.items()}

    def _capture(self):
        started = time.perf_counter()
        saved = {k: v.clone() for k, v in self.state.items()}
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                # Populate kernels/work buffers without advancing physical time
                # or changing the integrator's acceptance/retry counters.
                self._eager(0.0)
                if not torch.isfinite(self.state["hydro_u"]).all():
                    raise RuntimeError("Straka graph warmup produced nonfinite state")
            for k, v in self.state.items():
                v.copy_(saved[k])
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        # Separate TensorImpls keep the input storage alive when native RK
        # updates use set_()/set_data() on the caller's Tensor handles.
        self.inputs = {k: v.detach() for k, v in self.state.items()}
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            self._eager(self.dt)
        # Capture changes host handles but does not execute the captured work.
        for k, v in self.state.items():
            v.copy_(saved[k])
            self.inputs[k].copy_(saved[k])
        torch.cuda.synchronize()
        self.signature = self._signature()
        self.captures += 1
        self.setup_seconds += time.perf_counter() - started

    def __call__(self, dt):
        if dt != self.dt or self._signature() != self.signature:
            # A changed dt or externally replaced state must never reuse stale
            # graph inputs. Rebuild using the current accepted state.
            self.dt = dt
            self._capture()
        # Include in-place caller edits and restored retry state, not only
        # the previous graph output. No frozen-input replay is permissible.
        for k, v in self.state.items():
            if self.inputs[k].data_ptr() != v.data_ptr():
                self.inputs[k].copy_(v)
        self.graph.replay()
        self.replays += 1
