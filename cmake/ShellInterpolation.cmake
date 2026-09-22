# Reuse Dendro's implementation without changing its checkout. The Sobolev
# version snaps the upper z face to pt_min.z(), corrupting interpolation.
# A private namespace prevents differing template definitions in other TUs.
set(shell_compat "${CMAKE_CURRENT_BINARY_DIR}/shell_compat")
file(MAKE_DIRECTORY "${shell_compat}")
foreach(ext h tcc)
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
               "${DENDRO_dendrolib_DIR}/include/daUtils.${ext}")
  file(READ "${DENDRO_dendrolib_DIR}/include/daUtils.${ext}" source)
  string(REPLACE "namespace da {" "namespace shell_da {" source "${source}")
  string(REPLACE "SFCSORTBENCH_DAUTILS_H" "NLSM_SHELL_DAUTILS_H" source "${source}")
  string(REPLACE "if (fabs(domain_coord[2] - pt_max.z()) < 1e-6) domain_coord[2] = pt_min.z();"
                 "if (fabs(domain_coord[2] - pt_max.z()) < 1e-6) domain_coord[2] = pt_max.z();"
                 source "${source}")
  file(WRITE "${shell_compat}/daUtils.${ext}" "${source}")
endforeach()
foreach(target nlsmSolver solverScalingTest)
  target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
endforeach()
