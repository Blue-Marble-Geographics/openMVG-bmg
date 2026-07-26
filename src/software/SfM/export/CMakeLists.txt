###
# SfM export to X
# - PMVS + bundler data format
# - OpenMVS
# - CMPMVS
# - Meshlab
# - Agisoft Photoscan
# - MVE (File format v2)
###

# - Export a SfM openMVG scene to PMVS format + bundler data format
#
add_executable(openMVG_main_openMVG2PMVS main_openMVG2PMVS.cpp)
target_link_libraries(openMVG_main_openMVG2PMVS
  openMVG_system
  openMVG_image
  openMVG_features
  openMVG_sfm
  ${STLPLUS_LIBRARY}
)

# - Export a SfM openMVG scene to openMVS format
#
# The undistort inner loop (UndistortImageWithMap / BilinearInteriorRGB) is
# the hot function in this binary. The SIMD path uses pure SSE2 intrinsics
# only (which is the MSVC x64 baseline) so this TU compiles with the same
# /arch level as the rest of the project -- no per-target instruction-set
# elevation. If we want to add an AVX2 dispatch in the future, do it via a
# separate TU with runtime CPU-feature detection (the dead AVX2 helper in
# the .cpp is wired for that), not by raising /arch project-wide.
add_executable(openMVG_main_openMVG2openMVS main_openMVG2openMVS.cpp)
target_link_libraries(openMVG_main_openMVG2openMVS
  openMVG_system
  openMVG_image
  openMVG_features
  openMVG_sfm
  ${STLPLUS_LIBRARY}
)

# - Export a SfM openMVG scene to CMPMVS format
#
add_executable(openMVG_main_openMVG2CMPMVS main_openMVG2CMPMVS.cpp)
target_link_libraries(openMVG_main_openMVG2CMPMVS
  openMVG_system
  openMVG_image
  openMVG_features
  openMVG_sfm
  ${STLPLUS_LIBRARY}
)


# - Export a SfM openMVG scene to MVE(v2) format
#
add_executable(openMVG_main_openMVG2MVE2 main_openMVG2MVE2.cpp)
target_link_libraries(openMVG_main_openMVG2MVE2
  openMVG_system
  openMVG_image
  openMVG_features
  openMVG_sfm
  ${STLPLUS_LIBRARY}
)

# - Export a SfM openMVG scene to meshlab scene with rasters
# -
add_executable(openMVG_main_openMVG2MESHLAB main_openMVG2MESHLAB.cpp)
target_link_libraries(openMVG_main_openMVG2MESHLAB
  openMVG_system
  openMVG_image
  openMVG_features
  openMVG_sfm
  ${STLPLUS_LIBRARY}
)

# - Export a SfM openMVG scene to Agisoft Photoscan
# -
ADD_EXECUTABLE(openMVG_main_openMVG2Agisoft main_openMVG2Agisoft.cpp)
TARGET_LINK_LIBRARIES(openMVG_main_openMVG2Agisoft
  openMVG_system
  openMVG_image
  openMVG_features
  openMVG_sfm
  ${STLPLUS_LIBRARY}
  )

# - Export a SfM openMVG scene to mvs-texturing scene folder
# -
add_executable(openMVG_main_openMVG2MVSTEXTURING main_openMVG2MVSTEXTURING.cpp)
target_link_libraries(openMVG_main_openMVG2MVSTEXTURING
  openMVG_system
  openMVG_image
  openMVG_features
  openMVG_sfm
  ${STLPLUS_LIBRARY}
)

# - Export SfM openMVG scene to NVM (N-View Match) file
add_executable(openMVG_main_openMVG2NVM main_openMVG2NVM.cpp)
target_link_libraries( openMVG_main_openMVG2NVM
  openMVG_system
  openMVG_image
  openMVG_features
  openMVG_sfm
  ${STLPLUS_LIBRARY} )

# - Export SfM openMVG scene to Colmap format
add_executable(openMVG_main_openMVG2Colmap main_openMVG2Colmap.cpp)
target_link_libraries( openMVG_main_openMVG2Colmap
  openMVG_system
  openMVG_image
  openMVG_features
  openMVG_sfm
  ${STLPLUS_LIBRARY} )
  
# - Export SfM openMVG spherical scene to a scene with cubemap (6 faces per panorama)
add_executable(openMVG_main_openMVGSpherical2Cubic main_openMVGSpherical2Cubic.cpp)
target_link_libraries( openMVG_main_openMVGSpherical2Cubic
  openMVG_system
  openMVG_image
  openMVG_features
  openMVG_sfm
  ${STLPLUS_LIBRARY} )

# installation rules
set_property(TARGET openMVG_main_openMVG2PMVS PROPERTY FOLDER OpenMVG/software)
install(TARGETS openMVG_main_openMVG2PMVS DESTINATION bin/)
set_property(TARGET openMVG_main_openMVG2openMVS PROPERTY FOLDER OpenMVG/software)
install(TARGETS openMVG_main_openMVG2openMVS DESTINATION bin/)
# Deploy libjpeg(-turbo) runtime DLL next to the exe so the build-tree binary
# is directly runnable. No-op when OPENMVG_JPEG_RUNTIME_DLL is empty (i.e.
# internal static libjpeg or non-Windows).
if(OPENMVG_JPEG_RUNTIME_DLL)
  add_custom_command(TARGET openMVG_main_openMVG2openMVS POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${OPENMVG_JPEG_RUNTIME_DLL}"
      "$<TARGET_FILE_DIR:openMVG_main_openMVG2openMVS>"
    COMMENT "Deploying libjpeg runtime DLL")
endif()
set_property(TARGET openMVG_main_openMVG2CMPMVS PROPERTY FOLDER OpenMVG/software)
install(TARGETS openMVG_main_openMVG2CMPMVS DESTINATION bin/)
set_property(TARGET openMVG_main_openMVG2MESHLAB PROPERTY FOLDER OpenMVG/software)
install(TARGETS openMVG_main_openMVG2MESHLAB DESTINATION bin/)
set_property(TARGET openMVG_main_openMVG2Agisoft PROPERTY FOLDER OpenMVG/software)
install(TARGETS openMVG_main_openMVG2Agisoft DESTINATION bin/)
set_property(TARGET openMVG_main_openMVG2MVSTEXTURING PROPERTY FOLDER OpenMVG/software)
install(TARGETS openMVG_main_openMVG2MVSTEXTURING DESTINATION bin/)
set_property(TARGET openMVG_main_openMVG2MVE2 PROPERTY FOLDER OpenMVG/software)
install(TARGETS openMVG_main_openMVG2MVE2 DESTINATION bin/)
set_property(TARGET openMVG_main_openMVG2NVM PROPERTY FOLDER OpenMVG/software)
install(TARGETS openMVG_main_openMVG2NVM DESTINATION bin/)
set_property(TARGET openMVG_main_openMVG2Colmap PROPERTY FOLDER OpenMVG/software)
install(TARGETS openMVG_main_openMVG2Colmap DESTINATION bin/)
set_property(TARGET openMVG_main_openMVGSpherical2Cubic PROPERTY FOLDER OpenMVG/software)
install(TARGETS openMVG_main_openMVGSpherical2Cubic DESTINATION bin/)

###
# SfM tools to visualize feature tracking data
###

# - View extracted Keypoints
#
add_executable(openMVG_main_exportKeypoints main_exportKeypoints.cpp)
target_link_libraries(openMVG_main_exportKeypoints
  openMVG_system
  openMVG_features
  openMVG_sfm
  ${STLPLUS_LIBRARY}
)


# - View computed matches (putatives, geometrics) per image pair
#
add_executable(openMVG_main_exportMatches main_exportMatches.cpp)
target_link_libraries(openMVG_main_exportMatches
  openMVG_system
  openMVG_features
  openMVG_sfm
  ${STLPLUS_LIBRARY}
)

# - View tracks per image pair
#
add_executable(openMVG_main_exportTracks main_exportTracks.cpp)
target_link_libraries(openMVG_main_exportTracks
  openMVG_system
  openMVG_features
  openMVG_sfm
  ${STLPLUS_LIBRARY}
)

# - Export undistorted images related to a sfm_data file
#
add_executable(openMVG_main_ExportUndistortedImages main_ExportUndistortedImages.cpp)
target_link_libraries(openMVG_main_ExportUndistortedImages
  openMVG_system
  openMVG_image
  openMVG_features
  openMVG_sfm
  ${STLPLUS_LIBRARY}
)

# installation rules
set_property(TARGET openMVG_main_exportKeypoints PROPERTY FOLDER OpenMVG/software)
install(TARGETS openMVG_main_exportKeypoints DESTINATION bin/)
set_property(TARGET openMVG_main_exportMatches PROPERTY FOLDER OpenMVG/software)
install(TARGETS openMVG_main_exportMatches DESTINATION bin/)
set_property(TARGET openMVG_main_exportTracks PROPERTY FOLDER OpenMVG/software)
install(TARGETS openMVG_main_exportTracks DESTINATION bin/)
set_property(TARGET openMVG_main_ExportUndistortedImages PROPERTY FOLDER OpenMVG/software)
install(TARGETS openMVG_main_ExportUndistortedImages DESTINATION bin/)


# - Export SfM openMVG camera scene frustums as triangle meshes
# -
add_executable(openMVG_main_ExportCameraFrustums main_ExportCameraFrustums.cpp)
target_link_libraries(openMVG_main_ExportCameraFrustums
  openMVG_system
  openMVG_features
  openMVG_sfm
  ${STLPLUS_LIBRARY}
)
set_property(TARGET openMVG_main_ExportCameraFrustums PROPERTY FOLDER OpenMVG/software)
install(TARGETS openMVG_main_ExportCameraFrustums DESTINATION bin/)
