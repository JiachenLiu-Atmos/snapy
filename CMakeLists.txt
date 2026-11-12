cmake_minimum_required(VERSION 3.22)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BUILD_TESTS "Build tests" ON)
option(FULL_TESTS "Include additional reference and decomposition tests" OFF)
option(BUILD_EXAMPLES "Build examples" OFF)
option(CUDA "Enable CUDA support" OFF)
option(NETCDF "Build with netcdf support" ON)
if(APPLE)
  option(UCX "Build the UCX communication backend" OFF)
  option(PNETCDF "Build with pnetcdf support" OFF)
else()
  option(UCX "Build the UCX communication backend" ON)
  option(PNETCDF "Build with pnetcdf support" ON)
endif()

project(
  snapy
  LANGUAGES CXX C
  DESCRIPTION "finite volume method library for compressible fluids")

# Try to enable CUDA
if (CUDA)
  enable_language(CUDA)
endif()

# search for <Package>_ROOT
cmake_policy(SET CMP0074 NEW)

find_program(Python3_EXECUTABLE NAMES python)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))"
  RESULT_VARIABLE _ABI_RESULT
  OUTPUT_VARIABLE _ABI_OUTPUT
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT _ABI_RESULT EQUAL 0)
  message(FATAL_ERROR "Failed to detect Torch GLIBCXX ABI")
endif()

# Set the _GLIBCXX_USE_CXX11_ABI flag to 0 (use old ABI)
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=${_ABI_OUTPUT})

# Set MacOS deployment target
set(CMAKE_OSX_DEPLOYMENT_TARGET "15.0")

# make colorful output
string(ASCII 27 Esc)
set(ColorReset "${Esc}[m")
set(Green      "${Esc}[32m")
set(Blue       "${Esc}[34m")

message(STATUS "")
message(STATUS "${Green}==== ${PROJECT_NAME} configure begin ====${ColorReset}")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
  message(STATUS "Default Build Type = Release")
  message(STATUS "")
else()
  message(STATUS "Build Type = ${CMAKE_BUILD_TYPE}")
  message(STATUS "")
endif()

# load all modules
LIST(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules/)

# load all macros
file(GLOB _macro_files "${CMAKE_SOURCE_DIR}/cmake/macros/*.cmake")
foreach(_file ${_macro_files})
  message(STATUS "Include ${_file}")
  include(${_file})
endforeach()

# set output directories
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

message(STATUS "${Blue}${PROJECT_NAME}-1. Set up project compiler flags ...${ColorReset}")
message(STATUS "Include ${CMAKE_CURRENT_SOURCE_DIR}/cmake/compilers.cmake")
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/compilers.cmake)

message(STATUS "${Blue}${PROJECT_NAME}-2. Setting up project parameters ...${ColorReset}")
message(STATUS "Include ${CMAKE_CURRENT_SOURCE_DIR}/cmake/parameters.cmake")
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/parameters.cmake)

message(STATUS "${Blue}${PROJECT_NAME}-3. Setting up system libraries ...${ColorReset}")
set(BUILD_SHARED_LIBS OFF)
find_package(Torch REQUIRED)
find_package(Disort REQUIRED)
find_package(Harp REQUIRED)
# >= 2.5.8: SpeciesThermo names()/mu() and populate_thermo (kintera #121),
# read by the EOS cloud-parent cache and sedimentation (snapy #234)
# >= 2.5.13: ThermoY take_saturation_adjustment_failures() (kintera #131),
# read by check_redo
# >= 2.6.2: h2o_bryan beta of kintera #144, needed by examples/bryan.yaml (#295)
# >= 2.6.3: explicit same-call rate constants for the kinetics Jacobian at
# zero solid abundance, used by run_hydro and jupiter_evap_precip_1d
# >= 2.6.9: constant-cv perturbation pressure helper (kintera #155).
find_package(Kintera 2.6.9 REQUIRED)
#include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/kintera.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/fmt.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/gtest.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/yamlpp.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/eigen.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/ucx.cmake)

if (CUDA AND CMAKE_CUDA_COMPILER)
  find_package(CUDAToolkit REQUIRED)
else()
  set(CUDAToolkit_FOUND OFF)
endif()

if (CUDAToolkit_FOUND)
  set(CMAKE_CUDA_STANDARD 17)
  # 60: P100, 61: 1080Ti, 70: V100, 75: RTX 4000, 80: A100, 86: 3090, 89: 4090, 90: H100/H200, 120: 5090
  # Start with architectures that are broadly supported.
  set(_cuda_arch_list 60 61 70 75 80 86 89)

  if(DEFINED CUDAToolkit_VERSION)
    # nvcc 13+ stops support <75
    if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "13.0")
      list(REMOVE_ITEM _cuda_arch_list 60 61 70)
    endif()

    if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.0")
      list(APPEND _cuda_arch_list 90)
    endif()

    # nvcc supports sm_120 from 12.8
    if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.8")
      list(APPEND _cuda_arch_list 120)
    endif()
  elseif(DEFINED TORCH_CUDA_VERSION AND TORCH_CUDA_VERSION VERSION_GREATER_EQUAL "12.0")
    list(APPEND _cuda_arch_list 90)
    if(TORCH_CUDA_VERSION VERSION_GREATER_EQUAL "12.8")
      list(APPEND _cuda_arch_list 120)
    endif()
  endif()

  set(CMAKE_CUDA_ARCHITECTURES ${_cuda_arch_list})
endif()

message(STATUS "${Blue}${PROJECT_NAME}-4. Setting up project libraries ...${ColorReset}")
configure_file(${CMAKE_SOURCE_DIR}/configure.h.in configure.h @ONLY)
add_subdirectory(src)

if (BUILD_TESTS)
  message(STATUS "${Blue}${PROJECT_NAME}-5. Setting up unit tests ...${ColorReset}")
  enable_testing()
  add_subdirectory(tests)
endif()

#if (BUILD_EXAMPLES)
  message(STATUS "${Blue}${PROJECT_NAME}-6. Setting up examples ...${ColorReset}")
  add_subdirectory(examples)
#endif()
