# =================
# Examples Problems
# =================

setup_problem(shock)
setup_problem(straka)
setup_problem(bryan)
setup_problem(jupiter_evap_precip_1d)
setup_problem(run_hydro)
# the command-line parser is used by these two drivers only
target_sources(jupiter_evap_precip_1d.${buildl} PRIVATE command_line.cpp)
target_sources(run_hydro.${buildl} PRIVATE command_line.cpp)
setup_problem(cubed_sphere_convection)
setup_problem(planar_convection)
setup_problem(spherical_convection)
setup_problem(shallow_xy)
setup_problem(shallow_splash)

file(GLOB inputs CONFIGURE_DEPENDS *.yaml *.py)
# create bin folder if does not exist
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
foreach(input ${inputs})
  get_filename_component(inp ${input} NAME)
  execute_process(COMMAND ln -sf ${input} ${CMAKE_BINARY_DIR}/bin/${inp})
endforeach()
