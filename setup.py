#!/usr/bin/env python
import os
import sys
import glob
import torch
import kintera
import platform
from pathlib import Path
from setuptools import setup
from torch.utils import cpp_extension
import sysconfig

def parse_library_names(libdir):
    library_names = []
    for file in os.listdir(libdir):
        path = os.path.join(libdir, file)
        if os.path.isfile(path) and file.endswith((".a", ".so", ".dylib")):
            library_names.append(file[3:].rsplit(".", 1)[0])

    # add system netcdf library
    library_names.extend(['netcdf'])

    # move current library name to first
    #current = [item for item in library_names if item.startswith('snap')]
    #other = [item for item in library_names if not item.startswith('snap')]

    # 1) non-cuda libs first (consumers)
    snap_non_cuda = [l for l in library_names if l.startswith("snap") and "cuda" not in l]
    # 2) cuda libs last (providers)
    snap_cuda = [l for l in library_names if l.startswith("snap") and "cuda" in l]
    # 3) everything else
    other = [l for l in library_names if not l.startswith("snap")]
    return snap_non_cuda + other + snap_cuda

site_dir = sysconfig.get_paths()["purelib"]
kintera_dir = Path(kintera.__file__).resolve().parent
try:
    import commux
except ImportError:
    commux_dir = None
else:
    commux_dir = Path(commux.__file__).resolve().parent

current_dir = os.getenv("WORKSPACE", Path().absolute())
include_dirs = [
    f"{current_dir}",
    f"{current_dir}/build",
    f"{current_dir}/build/_deps/fmt-src/include",
    f'{current_dir}/build/_deps/yaml-cpp-src/include',
    str(kintera_dir),
    f"{site_dir}/pyharp",
]
if commux_dir is not None:
    include_dirs.append(f"{commux_dir}/include")

# add homebrew directories if on MacOS
lib_dirs = [f"{current_dir}/build/lib"]
if commux_dir is not None:
    lib_dirs.append(f"{commux_dir}/lib")
if platform.system() == 'Darwin':
    lib_dirs.extend(['/opt/homebrew/lib'])
else:
    lib_dirs.extend(['/lib64/', '/usr/lib/x86_64-linux-gnu/'])
nc_home = os.environ.get("NC_HOME")
lib_dirs.append(f"{nc_home}/lib")

libraries = parse_library_names(f"{current_dir}/build/lib")

if sys.platform == "darwin":
    extra_link_args = [
        "-Wl,-rpath,@loader_path/lib",
        "-Wl,-rpath,@loader_path/../torch/lib",
        "-Wl,-rpath,@loader_path/../pydisort/lib",
        "-Wl,-rpath,@loader_path/../pyharp/lib",
        "-Wl,-rpath,@loader_path/../kintera/lib",
        "-Wl,-rpath,@loader_path/../commux/lib",
    ]
else:
    cuda_linker = []
    cuda_libraries = [lib for lib in libraries if "cuda" in lib]
    cuda_runtime_dirs = [
        Path("/opt/nvidia/hpc_sdk/Linux_x86_64/26.3/cuda/13.1/targets/x86_64-linux/lib")
    ]

    if cuda_libraries:
        for lib in cuda_libraries:
            libraries.remove(lib)
        cuda_linker = (
            ["-Wl,--no-as-needed"]
            + [f"-l{lib}" for lib in cuda_libraries]
            + ["-Wl,--as-needed"]
            )

    extra_link_args = [
        "-Wl,-rpath,$ORIGIN/lib",
        "-Wl,-rpath,$ORIGIN/../torch/lib",
        "-Wl,-rpath,$ORIGIN/../pydisort/lib",
        "-Wl,-rpath,$ORIGIN/../pyharp/lib",
        "-Wl,-rpath,$ORIGIN/../kintera/lib",
        "-Wl,-rpath,$ORIGIN/../commux/lib",
    ]
    extra_link_args += [
        f"-Wl,-rpath,{path}" for path in cuda_runtime_dirs if path.exists()
    ]
    extra_link_args += cuda_linker

ext_module = cpp_extension.CppExtension(
    name='snapy.snapy',
    sources=glob.glob('python/csrc/*.cpp'),
    include_dirs=include_dirs,
    library_dirs=lib_dirs,
    libraries=libraries,
    extra_compile_args=['-Wno-attributes'],
    extra_link_args=extra_link_args,
    )

setup(
    package_dir={"snapy": "python"},
    ext_modules=[ext_module],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
