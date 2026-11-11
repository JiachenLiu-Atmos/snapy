# Snapy

**Compressible Finite Volume Solver for Atmospheric Dynamics, Chemistry and Thermodynamics**

Snapy is the dynamic core for simulating atmospheric and planetary dynamics using PyTorch tensors and GPU acceleration.

[![PyPI version](https://badge.fury.io/py/snapy.svg)](https://badge.fury.io/py/snapy)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **GPU-Accelerated**: Built on PyTorch for efficient GPU computation
- **Flexible Interfaces**: Both Python and C++ APIs available
- **Compressible Flow**: Finite volume solver for atmospheric dynamics
- **Multi-platform**: Support for Linux and macOS
- **NetCDF Output**: Standard output format for scientific data

## Installation

### Quick Install (Python Interface)

The easiest way to get started is to install via pip:

```bash
pip install snapy
```

This will install the Python interface with pre-built binaries for Python 3.9-3.13 on Linux (x86_64) and macOS (ARM64).

### Parallel run
```
pd-run 6 ./test_exchange.release
```

### list listening port
```bash
lsof -i:29500
```

### kill listening port
```bash
pkill -9 XXXXX
```

**Requirements:**
- Python 3.9 or higher
- PyTorch 2.10.x
- NumPy
- kintera >= 2.6.9

### Build from Source (Advanced)

Building from source is recommended only for advanced users who need to:
- Modify the C++ core
- Use custom PyTorch versions
- Access the C++ interface directly
- Develop new features

**Prerequisites:**
- CMake 3.20+
- C++17 compatible compiler
- Autoconf, Automake, Libtool, and pkg-config on Linux (for fetched UCX)
- PyTorch 2.10.x with C++ libraries
- NetCDF C library
- kintera >= 2.6.9

**Build steps:**

1. Clone the repository:
```bash
git clone https://github.com/chengcli/snapy.git
cd snapy
```

2. Install dependencies:
```bash
pip install numpy "kintera>=2.6.9" torch==2.10.0
```

Kintera 2.6.9 introduces the perturbation-pressure helper required by this
branch. Until a compatible wheel is published, build and install tag `v2.6.9`
from source before building snapy, using the commands in
[FP32_PROGRESS.md](FP32_PROGRESS.md). CI builds the pinned source commit;
package releases still require a compatible published or preinstalled dependency.

3. Install NetCDF:
   - **Linux (Ubuntu/Debian):**
     ```bash
     sudo apt-get install libnetcdf-dev
     ```
   - **macOS:**
     ```bash
     brew install netcdf
     ```

UCX support is provided by the Python package ``commux``. Install snapy's
Python dependencies in the active environment before configuring; use
``-DUCX=OFF`` only when building a Gloo-only configuration.

4. Configure and build:
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DNETCDF=ON
cmake --build build --parallel 3
```

5. Install the Python package:
```bash
pip install .
```

## Examples

The `examples/` directory contains several working examples:

**Python Examples:**
- `shock.py` - Sod shock tube with internal boundary
- `straka.py` - Straka cold bubble convection test
- `robert.py` - Robert warm bubble convection test

**C++ Examples:**
- `shock.cpp` - Sod shock tube (C++)
- `straka.cpp` - Straka cold bubble (C++)

Run a Python example:
```bash
cd examples
python shock.py
```

Run a C++ example (after building):
```bash
cd build/examples
./shock
```

See `examples/README` for detailed documentation on the code structure and available examples.

## Configuration

Simulations are configured using YAML files that specify:
- Grid dimensions and domain size
- Time integration settings (RK stages, CFL number)
- Boundary conditions
- Output settings (frequency, variables, format)
- Equation of state and thermodynamics

Example configuration files (`.yaml`) are provided alongside the examples.

Scheme choices are deck keys under `dynamics:`, read per block; every block
of a run must make the same choice, which blocks built from one deck do:

| key | default | what it does |
|---|---|---|
| `flux-covariance` | `true` | the x2/x3 face-flux covariance correction (#289, `docs/derivations/289-covariance-x3-curved.md`) |
| `x1-mass-covariance` | `false` | subtract the x1 rho-w covariance from the reconstructed cell velocity; needs at least 3 x1 cells per block |

They replace the environment switches `SNAP_FLUX_COVARIANCE` and
`SNAP_X1_MASS_COVARIANCE`, which are no longer read. Defaults are the same
whether options come from a deck without the key or are built in code: EOS
floors `density-floor: 1.e-6`, `pressure-floor: 1.e-3`,
`temperature-floor: 20.`; a `riemann-solver` without `type` is `hllc`.

## Development

### Testing

Run tests after building:
```bash
cd build/tests
ctest --output-on-failure
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

- **Author**: Cheng Li
- **Email**: chengcli@umich.edu
- **GitHub**: [https://github.com/chengcli/snapy](https://github.com/chengcli/snapy)
