# run_straka_redo.cmake: straka on one process at cfl 1.5 (shipped 0.9) to
# t = 30 s. Steps there go non-finite and are redone with a smaller dt; with
# gravity-work: cell and its default fixer on, they must reach the redo check and the
# run its time limit. (Runs to t = 60 s at this cfl end by chance on the redo limit
# near t = 58 s, with or without the fixer; a 1-ulp change flips the outcome.)

get_filename_component(build_dir "${CMAKE_CURRENT_BINARY_DIR}/.." ABSOLUTE)
set(work_dir "${CMAKE_CURRENT_BINARY_DIR}/straka_redo")
file(REMOVE_RECURSE "${work_dir}")
file(MAKE_DIRECTORY "${work_dir}")

file(READ "${build_dir}/bin/straka_single.yaml" config)
string(REPLACE "cfl: 0.9" "cfl: 1.5" config "${config}")
string(REPLACE "tlim: 900" "tlim: 30" config "${config}")
string(REPLACE "grav1: -9.8\n" "grav1: -9.8\n    gravity-work: cell\n" config "${config}")
if(NOT config MATCHES "cfl: 1.5" OR NOT config MATCHES "tlim: 30"
   OR NOT config MATCHES "gravity-work: cell")
  message(FATAL_ERROR "straka_single.yaml: no 'cfl: 0.9' / 'tlim: 900' / 'grav1: -9.8' to set")
endif()
file(WRITE "${work_dir}/straka.yaml" "${config}")

execute_process(
  COMMAND "${build_dir}/bin/straka.${buildl}" straka.yaml
  WORKING_DIRECTORY "${work_dir}"
  RESULT_VARIABLE res
  OUTPUT_VARIABLE out
  ERROR_VARIABLE out
)

if(NOT res EQUAL 0)
  message(FATAL_ERROR "straka at cfl 1.5 stopped with exit code ${res}:\n${out}")
endif()
if(NOT out MATCHES "Redoing the step")
  message(FATAL_ERROR "straka at cfl 1.5 redid no step: the case no longer tests the redo path")
endif()
