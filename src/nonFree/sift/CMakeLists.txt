# libs should be static
set(BUILD_SHARED_LIBS OFF)

# use PIC code for link into shared lib
if(UNIX)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
endif(UNIX)

if(NOT USE_SSE2)
  add_definitions(-DVL_DISABLE_SSE2)
endif()

add_definitions(-DVL_DISABLE_THREADS)

include_directories(./vl)
set(FEATS
  vl/generic.c
  vl/imopv_sse2.c
  vl/sift.c
  vl/imopv.c
  vl/mathop_sse2.c
  vl/sift.c
  vl/host.c
  vl/mathop.c
  vl/siftAVX2Support.c
  vl/random.c)
set_source_files_properties(${FEATS} PROPERTIES LANGUAGE C)
add_library(vlsift ${FEATS})
install(TARGETS vlsift DESTINATION lib EXPORT openMVG-targets)
set_property(TARGET vlsift PROPERTY FOLDER OpenMVG/nonFree)
install(
  DIRECTORY .
  DESTINATION include/openMVG_dependencies/nonFree/sift
  COMPONENT headers
  FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h"
)

# ==============================================================================
# SIFT diagnostics test
# ==============================================================================
if (OpenMVG_BUILD_TESTS)
  add_executable(openMVG_test_sift_diagnostics sift_diagnostics_test.cpp)
  set_property(TARGET openMVG_test_sift_diagnostics PROPERTY FOLDER OpenMVG/test)
  target_include_directories(openMVG_test_sift_diagnostics
    PRIVATE
      ${CMAKE_SOURCE_DIR}
      ${CMAKE_SOURCE_DIR}/third_party)
  target_link_libraries(openMVG_test_sift_diagnostics
    PRIVATE
      vlsift
      openMVG_features
      openMVG_image
      openMVG_system
      openMVG_numeric
      CppUnitLite)
  if (MSVC)
    set_target_properties(openMVG_test_sift_diagnostics PROPERTIES COMPILE_FLAGS "/bigobj")
  endif()
  target_compile_definitions(openMVG_test_sift_diagnostics
    PRIVATE -DTHIS_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
  add_test(NAME openMVG_test_sift_diagnostics
    WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
    COMMAND $<TARGET_FILE:openMVG_test_sift_diagnostics>)
endif()
