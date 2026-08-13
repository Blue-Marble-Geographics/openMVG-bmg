
UNIT_TEST(openMVG rand_sampling "openMVG_testing")
UNIT_TEST(openMVG robust_estimator_lineKernel "openMVG_testing")
UNIT_TEST(openMVG robust_estimator_MaxConsensus "openMVG_testing")
UNIT_TEST(openMVG robust_estimator_Ransac "openMVG_testing")
#UNIT_TEST(openMVG robust_estimator_LMeds "openMVG_testing")
# ACRANSAC now reaches a runtime-dispatched NFA kernel in
# openMVG_robust_estimation, so anything instantiating ACRANSAC must link it.
UNIT_TEST(openMVG robust_estimator_ACRansac "openMVG_testing;openMVG_robust_estimation")

add_library(openMVG_robust_estimation
  gms_filter.hpp gms_filter.cpp
  robust_estimator_ACRansac_nfa_simd.hpp
  robust_estimator_ACRansac_nfa_simd.cpp
  robust_estimator_ACRansac_nfa_avx2.cpp
)
if (MSVC)
  set_target_properties(openMVG_robust_estimation PROPERTIES COMPILE_FLAGS "/bigobj")
endif()

# ------------------------------------------------------------------
# [ACRANSAC-PERF] AVX2 NFA kernel: per-source instruction set.
#
# robust_estimator_ACRansac_nfa_avx2.cpp is the ONLY file in the project
# compiled with AVX2 code generation. It is never included from a header and
# is reached exclusively through the runtime CPUID gate in
# robust_estimator_ACRansac_nfa_simd.cpp, so raising the instruction set here
# does not raise the baseline of the shipped binary: it still starts and runs
# correctly on pre-AVX2 CPUs.
#
# Deliberately independent of the project-wide USE_AVX2 option. USE_AVX2
# changes the baseline for *whole libraries* (see matching/CMakeLists.txt) and
# is OFF by default; this file must be built with AVX2 regardless so that the
# runtime dispatch has something to dispatch to.
#
# Do NOT add these flags to robust_estimator_ACRansac_nfa_simd.cpp -- that TU
# performs the capability check itself and must stay at the baseline.
#
# COMPILE_FLAGS (a string) rather than COMPILE_OPTIONS (a list) because the
# project declares cmake_minimum_required(VERSION 3.9) and source-level
# COMPILE_OPTIONS only arrived in 3.11.
# ------------------------------------------------------------------
if (MSVC)
  set_source_files_properties(robust_estimator_ACRansac_nfa_avx2.cpp
    PROPERTIES COMPILE_FLAGS "/arch:AVX2")
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|i.86|x86)$")
  set_source_files_properties(robust_estimator_ACRansac_nfa_avx2.cpp
    PROPERTIES COMPILE_FLAGS "-mavx2 -mfma")
endif()

target_compile_features(openMVG_robust_estimation INTERFACE ${CXX11_FEATURES})
target_include_directories(openMVG_robust_estimation
  PUBLIC
    $<INSTALL_INTERFACE:include>
    $<INSTALL_INTERFACE:include/openMVG>
)
target_link_libraries(openMVG_robust_estimation
  PUBLIC
    openMVG_numeric
)
set_property(TARGET openMVG_robust_estimation PROPERTY FOLDER OpenMVG/OpenMVG)
install(TARGETS openMVG_robust_estimation DESTINATION lib EXPORT openMVG-targets)
set_target_properties(
  openMVG_robust_estimation
  PROPERTIES SOVERSION ${OPENMVG_VERSION_MAJOR}
  VERSION "${OPENMVG_VERSION_MAJOR}.${OPENMVG_VERSION_MINOR}")

UNIT_TEST(openMVG gms_filter "openMVG_robust_estimation")
