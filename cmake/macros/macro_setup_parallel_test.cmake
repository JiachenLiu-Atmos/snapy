# A small macro used for setting up the build of a test.
#
# Usage: setup_parallel_test(name)

string(TOLOWER ${CMAKE_BUILD_TYPE} buildl)
string(TOUPPER ${CMAKE_BUILD_TYPE} buildu)

macro(setup_parallel_test namel cores)
  add_executable(${namel}.${buildl} ${namel}.cpp)

  set_target_properties(${namel}.${buildl}
                        PROPERTIES COMPILE_FLAGS ${CMAKE_CXX_FLAGS_${buildu}})

  target_include_directories(
    ${namel}.${buildl}
    PRIVATE ${CMAKE_BINARY_DIR}
            ${KINTERA_INCLUDE_DIR}
            ${SNAP_INCLUDE_DIR}
            ${HARP_INCLUDE_DIR}
            ${NETCDF_INCLUDES}
            ${TORCH_INCLUDE_DIR}
            ${TORCH_API_INCLUDE_DIR})

  target_link_libraries(${namel}.${buildl}
    PRIVATE snapy::snap
            $<IF:$<BOOL:${CUDAToolkit_FOUND}>,snapy::snap_cu,>
            gtest_main)

  # as in setup_test: the CUDA kernels register through static initializers
  # in snap_cu, which --as-needed drops when the test references none of its
  # symbols (then a CUDA dispatch has no kernel)
  if (UNIX AND NOT APPLE)
    target_link_options(${namel}.${buildl} PRIVATE -Wl,--no-as-needed)
  endif()

  add_test(NAME ${namel}.${buildl}
           COMMAND torchrun
                   --no-python
                   --nproc-per-node=${cores}
                   $<TARGET_FILE:${namel}.${buildl}>)
endmacro()
