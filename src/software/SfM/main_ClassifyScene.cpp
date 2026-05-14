// This file is part of OpenMVG, an Open Multiple View Geometry C++ library.

// Copyright (c) 2024 Pierre MOULON.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

// Classify the scene as NADIR or OBLIQUE based on GPS priors.
// This runs before SfM so that downstream tools (matching, reconstruction)
// can adapt their parameters (e.g., angle thresholds).
//
// Pipeline order:
//   main_SfMInit_ImageListing -P  (creates sfm_data.json with GPS priors)
//   main_ComputeFeatures
//   main_ClassifyScene              <-- THIS TOOL
//   main_ListMatchingPairs / main_ComputeMatches
//   main_SfM

#include "openMVG/numeric/eigen_alias_definition.hpp"
#include "openMVG/sfm/sfm_data.hpp"
#include "openMVG/sfm/sfm_data_io.hpp"
#include "openMVG/sfm/sfm_view_priors.hpp"
#include "openMVG/system/logger.hpp"

#include "third_party/cmdLine/cmdLine.h"
#include "third_party/stlplus3/filesystemSimplified/file_system.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <string>
#include <vector>

using namespace openMVG;
using namespace openMVG::sfm;

int main(int argc, char **argv)
{
  CmdLine cmd;

  std::string sSfM_Data_Filename;
  std::string sOutputDir;

  cmd.add(make_option('i', sSfM_Data_Filename, "input_file"));
  cmd.add(make_option('o', sOutputDir, "output_dir"));

  try {
    if (argc == 1) throw std::string("Invalid parameter.");
    cmd.process(argc, argv);
  } catch (const std::string& s) {
    OPENMVG_LOG_INFO
      << "Usage: " << argv[0] << '\n'
      << "[-i|--input_file] path to a SfM_Data file (with GPS priors)\n"
      << "[-o|--output_dir] path where scene_type.txt and scene_diagnostics.txt will be written\n";
    OPENMVG_LOG_ERROR << s;
    return EXIT_FAILURE;
  }

  if (sOutputDir.empty())
  {
    OPENMVG_LOG_ERROR << "Invalid output directory";
    return EXIT_FAILURE;
  }

  if (!stlplus::folder_exists(sOutputDir))
  {
    if (!stlplus::folder_create(sOutputDir))
    {
      OPENMVG_LOG_ERROR << "Cannot create the output directory";
      return EXIT_FAILURE;
    }
  }

  // Load SfM_Data (only views + intrinsics — no poses or structure yet)
  SfM_Data sfm_data;
  if (!Load(sfm_data, sSfM_Data_Filename, ESfM_Data(VIEWS | INTRINSICS)))
  {
    OPENMVG_LOG_ERROR << "The input SfM_Data file \"" << sSfM_Data_Filename << "\" cannot be read.";
    return EXIT_FAILURE;
  }

  // Collect GPS prior positions from ViewPriors
  std::vector<Vec3> gps_positions;
  gps_positions.reserve(sfm_data.GetViews().size());
  for (const auto & view_it : sfm_data.GetViews())
  {
    const ViewPriors * prior = dynamic_cast<const ViewPriors *>(view_it.second.get());
    if (prior && prior->b_use_pose_center_)
    {
      gps_positions.push_back(prior->pose_center_);
    }
  }

  bool is_nadir = false;

  // --- Diagnostic values (written to scene_diagnostics.txt) ---
  // All default to 0 / empty so the file is always well-formed.
  size_t num_gps = gps_positions.size();
  size_t num_views = sfm_data.GetViews().size();
  size_t num_intrinsics = sfm_data.GetIntrinsics().size();
  double spread_major = 0, spread_mid = 0, spread_minor = 0;
  double planarity_ratio = 1.0, lateral_aspect = 0.0;
  double height_range = 0, lateral_range = 0;
  double height_to_lateral = 0;    // height_range / lateral_range
  double median_nn_dist = 0;       // median nearest-neighbor distance between cameras
  double mean_nn_dist = 0;         // mean nearest-neighbor distance
  double max_nn_dist = 0;          // max nearest-neighbor distance
  double nn_cv = 0;                // coefficient of variation of nn distances (regularity)
  double gps_noise_estimate = 0;   // estimated GPS noise (MAD of residual from PCA plane)
  double coverage_area = 0;        // approximate 2D footprint (spread_major * spread_mid)
  double camera_density = 0;       // cameras per unit area
  std::string flight_pattern = "UNKNOWN"; // GRID, CORRIDOR, CROSSHATCH, RADIAL, SPARSE
  std::string altitude_profile = "UNKNOWN"; // SINGLE, MULTI, VARIABLE

  if (gps_positions.size() < 3)
  {
    OPENMVG_LOG_WARNING
      << "Insufficient GPS priors (" << gps_positions.size()
      << ") for scene classification. Defaulting to OBLIQUE.\n"
      << "Run main_SfMInit_ImageListing with -P to populate GPS priors.";
  }
  else
  {
    // 1. Compute centroid
    Vec3 centroid = Vec3::Zero();
    for (const auto & p : gps_positions)
      centroid += p;
    centroid /= static_cast<double>(gps_positions.size());

    // 2. Build centered matrix and compute PCA via SVD
    Mat pts(3, gps_positions.size());
    for (size_t i = 0; i < gps_positions.size(); ++i)
      pts.col(i) = gps_positions[i] - centroid;

    const Eigen::JacobiSVD<Mat> svd(pts, Eigen::ComputeThinU);
    const auto & sv = svd.singularValues();

    // Principal axes
    const Vec3 axis_major = svd.matrixU().col(0).normalized(); // largest lateral spread
    const Vec3 axis_mid   = svd.matrixU().col(1).normalized(); // second lateral spread
    const Vec3 axis_minor = svd.matrixU().col(2).normalized(); // altitude / thin axis

    spread_major = sv(0);
    spread_mid   = sv(1);
    spread_minor = sv(2);

    planarity_ratio = (spread_major > 1e-8) ? spread_minor / spread_major : 1.0;
    lateral_aspect  = (spread_major > 1e-8) ? spread_mid / spread_major : 0.0;

    // Height range along the minor (altitude) axis
    double min_h = std::numeric_limits<double>::max();
    double max_h = std::numeric_limits<double>::lowest();
    for (const auto & p : gps_positions)
    {
      const double h = (p - centroid).dot(axis_minor);
      min_h = std::min(min_h, h);
      max_h = std::max(max_h, h);
    }
    height_range = max_h - min_h;

    // Lateral range along the major axis
    double min_lat = std::numeric_limits<double>::max();
    double max_lat = std::numeric_limits<double>::lowest();
    for (const auto & p : gps_positions)
    {
      const double d = (p - centroid).dot(axis_major);
      min_lat = std::min(min_lat, d);
      max_lat = std::max(max_lat, d);
    }
    lateral_range = max_lat - min_lat;
    height_to_lateral = (lateral_range > 1e-8) ? height_range / lateral_range : 0.0;

    // -------------------------------------------------------------------
    // 3. Nearest-neighbor distance statistics
    //    Tells us about overlap density and regularity of the flight grid.
    //    - Low median_nn + low CV = dense regular grid (easy)
    //    - High median_nn + low CV = sparse regular grid (harder)
    //    - High CV = irregular spacing (crosshatch, ad-hoc, gaps)
    // -------------------------------------------------------------------
    {
      std::vector<double> nn_dists;
      nn_dists.reserve(gps_positions.size());
      for (size_t i = 0; i < gps_positions.size(); ++i)
      {
        double best = std::numeric_limits<double>::max();
        for (size_t j = 0; j < gps_positions.size(); ++j)
        {
          if (i == j) continue;
          const double d = (gps_positions[i] - gps_positions[j]).norm();
          if (d < best) best = d;
        }
        nn_dists.push_back(best);
      }
      std::sort(nn_dists.begin(), nn_dists.end());
      median_nn_dist = nn_dists[nn_dists.size() / 2];
      max_nn_dist = nn_dists.back();

      double sum = 0;
      for (const double d : nn_dists) sum += d;
      mean_nn_dist = sum / static_cast<double>(nn_dists.size());

      double var_sum = 0;
      for (const double d : nn_dists) var_sum += (d - mean_nn_dist) * (d - mean_nn_dist);
      const double stddev = std::sqrt(var_sum / static_cast<double>(nn_dists.size()));
      nn_cv = (mean_nn_dist > 1e-8) ? stddev / mean_nn_dist : 0.0;
    }

    // -------------------------------------------------------------------
    // 4. GPS noise estimate (MAD of distance from PCA plane)
    //    High noise means GPS priors are less trustworthy; BA prior weights
    //    should be lower to avoid pulling cameras to noisy positions.
    // -------------------------------------------------------------------
    {
      std::vector<double> plane_dists;
      plane_dists.reserve(gps_positions.size());
      for (const auto & p : gps_positions)
      {
        const double d = std::abs((p - centroid).dot(axis_minor));
        plane_dists.push_back(d);
      }
      std::sort(plane_dists.begin(), plane_dists.end());
      gps_noise_estimate = plane_dists[plane_dists.size() / 2]; // MAD from plane
    }

    // -------------------------------------------------------------------
    // 5. Coverage area and camera density
    //    Tells us the GSD regime and whether the scene is over/under-sampled.
    // -------------------------------------------------------------------
    {
      // Project all positions onto the major/mid plane for 2D extent
      double min_u = std::numeric_limits<double>::max(), max_u = std::numeric_limits<double>::lowest();
      double min_v = std::numeric_limits<double>::max(), max_v = std::numeric_limits<double>::lowest();
      for (const auto & p : gps_positions)
      {
        const Vec3 d = p - centroid;
        const double u = d.dot(axis_major);
        const double v = d.dot(axis_mid);
        min_u = std::min(min_u, u); max_u = std::max(max_u, u);
        min_v = std::min(min_v, v); max_v = std::max(max_v, v);
      }
      coverage_area = (max_u - min_u) * (max_v - min_v);
      camera_density = (coverage_area > 1e-8)
        ? static_cast<double>(gps_positions.size()) / coverage_area
        : 0.0;
    }

    // -------------------------------------------------------------------
    // 6. Flight pattern classification
    //    Based on lateral_aspect and nn regularity.
    // -------------------------------------------------------------------
    {
      if (lateral_aspect < 0.10)
        flight_pattern = "CORRIDOR";       // essentially a line
      else if (lateral_aspect >= 0.10 && lateral_aspect < 0.35 && nn_cv < 0.5)
        flight_pattern = "CORRIDOR_WIDE";  // wide strip, few cross-tracks
      else if (nn_cv > 0.8)
        flight_pattern = "IRREGULAR";      // very uneven spacing (ad-hoc flight)
      else if (nn_cv > 0.5)
        flight_pattern = "CROSSHATCH";     // two grid orientations or mixed
      else
        flight_pattern = "GRID";           // regular grid pattern
    }

    // -------------------------------------------------------------------
    // 7. Altitude profile classification
    //    Based on height variation relative to median nn distance.
    // -------------------------------------------------------------------
    {
      if (median_nn_dist > 1e-8)
      {
        const double alt_ratio = height_range / median_nn_dist;
        if (alt_ratio < 0.5)
          altitude_profile = "SINGLE";     // all cameras ~same altitude
        else if (alt_ratio < 2.0)
          altitude_profile = "MULTI";      // 2-3 distinct altitudes
        else
          altitude_profile = "VARIABLE";   // continuous altitude variation
      }
    }

    // -------------------------------------------------------------------
    // 8. Final NADIR/OBLIQUE classification (same logic as before)
    // -------------------------------------------------------------------
    is_nadir = (planarity_ratio < 0.15)
            && (lateral_aspect > 0.15)
            && (lateral_range > 1e-8)
            && (height_range < lateral_range * 0.15);

    OPENMVG_LOG_INFO
      << "\n-----------------------------------------------------------"
      << "\n Scene classification (GPS-based):"
      << "\n   #views:             " << num_views
      << "\n   #GPS positions:     " << num_gps
      << "\n   #intrinsics:        " << num_intrinsics
      << "\n"
      << "\n   --- PCA spread ---"
      << "\n   spread (major):     " << spread_major
      << "\n   spread (mid):       " << spread_mid
      << "\n   spread (minor):     " << spread_minor
      << "\n   planarity_ratio:    " << planarity_ratio
      << "\n   lateral_aspect:     " << lateral_aspect
      << "\n   height_range:       " << height_range
      << "\n   lateral_range:      " << lateral_range
      << "\n   height/lateral:     " << height_to_lateral
      << "\n"
      << "\n   --- Camera spacing ---"
      << "\n   median_nn_dist:     " << median_nn_dist
      << "\n   mean_nn_dist:       " << mean_nn_dist
      << "\n   max_nn_dist:        " << max_nn_dist
      << "\n   nn_cv (regularity): " << nn_cv
      << "\n   coverage_area:      " << coverage_area
      << "\n   camera_density:     " << camera_density << " cams/unit²"
      << "\n"
      << "\n   --- Quality ---"
      << "\n   gps_noise_est:      " << gps_noise_estimate
      << "\n"
      << "\n   --- Patterns ---"
      << "\n   flight_pattern:     " << flight_pattern
      << "\n   altitude_profile:   " << altitude_profile
      << "\n   classification:     " << (is_nadir ? "NADIR" : "OBLIQUE")
      << "\n-----------------------------------------------------------";
  }

  // Write classification to disk
  {
    const std::string scene_type_path =
      stlplus::create_filespec(sOutputDir, "scene_type.txt");
    std::ofstream ofs(scene_type_path);
    if (ofs.is_open())
    {
      ofs << (is_nadir ? "NADIR" : "OBLIQUE") << std::endl;
      OPENMVG_LOG_INFO << "Wrote scene classification to: " << scene_type_path;
    }
    else
    {
      OPENMVG_LOG_ERROR << "Cannot write scene_type.txt to: " << scene_type_path;
      return EXIT_FAILURE;
    }
  }

  // Write detailed diagnostics to disk (machine-readable key=value format)
  {
    const std::string diag_path =
      stlplus::create_filespec(sOutputDir, "scene_diagnostics.txt");
    std::ofstream ofs(diag_path);
    if (ofs.is_open())
    {
      ofs << std::setprecision(8) << std::fixed;
      ofs << "# Scene diagnostics generated by openMVG_main_ClassifyScene\n";
      ofs << "# All distances are in the coordinate system of the GPS priors\n";
      ofs << "# (typically ECEF meters or UTM meters).\n";
      ofs << "#\n";
      ofs << "num_views="           << num_views << "\n";
      ofs << "num_gps="             << num_gps << "\n";
      ofs << "num_intrinsics="      << num_intrinsics << "\n";
      ofs << "classification="      << (is_nadir ? "NADIR" : "OBLIQUE") << "\n";
      ofs << "flight_pattern="      << flight_pattern << "\n";
      ofs << "altitude_profile="    << altitude_profile << "\n";
      ofs << "spread_major="        << spread_major << "\n";
      ofs << "spread_mid="          << spread_mid << "\n";
      ofs << "spread_minor="        << spread_minor << "\n";
      ofs << "planarity_ratio="     << planarity_ratio << "\n";
      ofs << "lateral_aspect="      << lateral_aspect << "\n";
      ofs << "height_range="        << height_range << "\n";
      ofs << "lateral_range="       << lateral_range << "\n";
      ofs << "height_to_lateral="   << height_to_lateral << "\n";
      ofs << "median_nn_dist="      << median_nn_dist << "\n";
      ofs << "mean_nn_dist="        << mean_nn_dist << "\n";
      ofs << "max_nn_dist="         << max_nn_dist << "\n";
      ofs << "nn_cv="               << nn_cv << "\n";
      ofs << "coverage_area="       << coverage_area << "\n";
      ofs << "camera_density="      << camera_density << "\n";
      ofs << "gps_noise_estimate="  << gps_noise_estimate << "\n";
      OPENMVG_LOG_INFO << "Wrote scene diagnostics to: " << diag_path;
    }
    else
    {
      OPENMVG_LOG_WARNING << "Cannot write scene_diagnostics.txt to: " << diag_path;
    }
  }

  return EXIT_SUCCESS;
}