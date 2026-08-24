// This file is part of OpenMVG, an Open Multiple View Geometry C++ library.

// Copyright (c) 2013 Pierre MOULON.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef OPENMVG_ROBUST_ESTIMATION_GUIDED_MATCHING_HPP
#define OPENMVG_ROBUST_ESTIMATION_GUIDED_MATCHING_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "openMVG/cameras/Camera_Intrinsics.hpp"
#include "openMVG/features/regions.hpp"
#include "openMVG/matching/indMatch.hpp"
#include "openMVG/numeric/numeric.h"

namespace openMVG{
namespace geometry_aware{

/// Guided Matching (features only):
///  Use a model to find valid correspondences:
///   Keep the best corresponding points for the given model under the
///   user specified distance.
template<
  typename ModelArg, // The used model type
  typename ErrorArg> // The metric to compute distance to the model
void GuidedMatching(
  const ModelArg & mod, // The model
  const Mat & xLeft,    // The left data points
  const Mat & xRight,   // The right data points
  double errorTh,       // Maximal authorized error threshold
  matching::IndMatches & vec_corresponding_index) // Ouput corresponding index
{
  assert(xLeft.rows() == xRight.rows());

  // Looking for the corresponding points that have
  //  the smallest distance (smaller than the provided Threshold)

  #pragma omp parallel for
  for (int i = 0; i < xLeft.cols(); ++i) {

    double min = std::numeric_limits<double>::max();
    matching::IndMatch match;
    for (size_t j = 0; j < xRight.cols(); ++j) {
      // Compute the geometric error: error to the model
      const double err = ErrorArg::Error(
        mod,  // The model
        xLeft.col(i), xRight.col(j)); // The corresponding points
      // if smaller error update corresponding index
      if (err < errorTh && err < min) {
        min = err;
        match = matching::IndMatch(i,j);
      }
    }
    if (min < errorTh)  {
      // save the best corresponding index
      #pragma omp critical
      vec_corresponding_index.push_back(match);
    }
  }

  // Remove duplicates (when multiple points at same position exist)
  matching::IndMatch::getDeduplicated(vec_corresponding_index);
}

// Struct to help filtering of correspondence according update of
//  two smallest distance.
// -> useful for descriptor distance ratio filtering
template <typename DistT>
struct distanceRatio
{
  DistT bd, sbd; // best and second best distance
  size_t idx; // best corresponding index

  distanceRatio():
    bd(std::numeric_limits<DistT>::max()),
    sbd(std::numeric_limits<DistT>::max()),
    idx(0)
  { }

  // Update match according the provided distance
  inline bool update(size_t index, DistT dist)
  {
    if (dist < bd) // best than any previous
    {
      idx = index;
      // update and swap
      sbd = dist;
      std::swap(bd, sbd);
      return true;
    }
    else if (dist < sbd)
    {
      sbd = dist;
      return true;
    }
    return false;
  }

  // Return if the ratio of distance is ok or not
  inline bool isValid(const double distRatio) const{
    // check:
    // - that two best distance have been found
    // - the distance ratio
    return
      (sbd != std::numeric_limits<DistT>::max()
      && bd < distRatio * sbd);
  }
};

/// Guided Matching (features + descriptors with distance ratio):
///  Use a model to find valid correspondences:
///   Keep the best corresponding points for the given model under the
///   user specified distance ratio.
template<
  typename ModelArg,    // The used model type
  typename ErrorArg,    // The metric to compute distance to the model
  typename DescriptorT, // The descriptor type
  typename MetricT >    // The metric to compare two descriptors
void GuidedMatching(
  const ModelArg & mod, // The model
  const Mat & xLeft,    // The left data points
  const std::vector<DescriptorT > & lDescriptors,
  const Mat & xRight,   // The right data points
  const std::vector<DescriptorT > & rDescriptors,
  double errorTh,       // Maximal authorized error threshold
  double distRatio,     // Maximal authorized distance ratio
  matching::IndMatches & vec_corresponding_index) // Ouput corresponding index
{
  assert(xLeft.rows() == xRight.rows());
  assert(xLeft.cols() == lDescriptors.size());
  assert(xRight.cols() == rDescriptors.size());

  MetricT metric;

  // Looking for the corresponding points that have to satisfy:
  //   1. a geometric distance below the provided Threshold
  //   2. a distance ratio between descriptors of valid geometric correspondencess

  for (size_t i = 0; i < xLeft.cols(); ++i) {

    distanceRatio<typename MetricT::ResultType > dR;
    for (size_t j = 0; j < xRight.cols(); ++j) {
      // Compute the geometric error: error to the model
      const double geomErr = ErrorArg::Error(
        mod,  // The model
        xLeft.col(i), xRight.col(j)); // The corresponding points
      if (geomErr < errorTh) {
        const typename MetricT::ResultType descDist =
          metric( lDescriptors[i].getData(), rDescriptors[j].getData(), DescriptorT::static_size );
        // Update the corresponding points & distance (if required)
        dR.update(j, descDist);
      }
    }
    // Add correspondence only iff the distance ratio is valid
    if (dR.isValid(distRatio))  {
      // save the best corresponding index
      vec_corresponding_index.push_back(matching::IndMatch(i,dR.idx));
    }
  }

  // Remove duplicates (when multiple points at same position exist)
  matching::IndMatch::getDeduplicated(vec_corresponding_index);
}

/// Guided Matching (features + descriptors with distance ratio):
///  Use a model to find valid correspondences:
///   Keep the best corresponding points for the given model under the
///   user specified distance ratio.
template<
  typename ModelArg,  // The used model type
  typename ErrorArg   // The metric to compute distance to the model
  >
void GuidedMatching(
  const ModelArg & mod, // The model
  const cameras::IntrinsicBase * camL, // Optional camera (in order to undistord on the fly feature positions, can be nullptr)
  const features::Regions & lRegions,  // regions (point features & corresponding descriptors)
  const cameras::IntrinsicBase * camR, // Optional camera (in order to undistord on the fly feature positions, can be nullptr)
  const features::Regions & rRegions,  // regions (point features & corresponding descriptors)
  double errorTh,       // Maximal authorized error threshold
  double distRatio,     // Maximal authorized distance ratio
  matching::IndMatches & vec_corresponding_index) // Ouput corresponding index
{
  // Looking for the corresponding points that have to satisfy:
  //   1. a geometric distance below the provided Threshold
  //   2. a distance ratio between descriptors of valid geometric correspondencess

  // Build region positions arrays (in order to un-distord on-demand point position once)
  std::vector<Vec2>
    lRegionsPos(lRegions.RegionCount()),
   rRegionsPos(rRegions.RegionCount());
  for (size_t i = 0; i < lRegions.RegionCount(); ++i) {
    lRegionsPos[i] = camL ? camL->get_ud_pixel(lRegions.GetRegionPosition(i)) : lRegions.GetRegionPosition(i);
  }
  for (size_t i = 0; i < rRegions.RegionCount(); ++i) {
    rRegionsPos[i] = camR ? camR->get_ud_pixel(rRegions.GetRegionPosition(i)) : rRegions.GetRegionPosition(i);
  }

  for (size_t i = 0; i < lRegions.RegionCount(); ++i) {

    distanceRatio<double> dR;
    for (size_t j = 0; j < rRegions.RegionCount(); ++j) {
      // Compute the geometric error: error to the model
      const double geomErr = ErrorArg::Error(
        mod,  // The model
        // The corresponding points
        lRegionsPos[i],
        rRegionsPos[j]);
      if (geomErr < errorTh) {
        // Update the corresponding points & distance (if required)
        dR.update(j, lRegions.SquaredDescriptorDistance(i, &rRegions, j));
      }
    }
    // Add correspondence only iff the distance ratio is valid
    if (dR.isValid(distRatio))  {
      // save the best corresponding index
      vec_corresponding_index.push_back(matching::IndMatch(i,dR.idx));
    }
  }

  // Remove duplicates (when multiple points at same position exist)
  matching::IndMatch::getDeduplicated(vec_corresponding_index);
}

/// Uniform bucket grid over a 2D point set, stored CSR style
/// (two flat arrays instead of a vector of vectors, so building it is a
/// couple of linear passes and no per-cell allocation).
/// Bounds come from the data rather than from the image size, so undistorted
/// positions falling outside the image frame are still indexed.
struct PointGrid2D
{
  double min_x = 0.0, min_y = 0.0;
  double cell_size = 1.0, inv_cell_size = 1.0;
  int nx = 0, ny = 0;
  std::vector<uint32_t> cell_begin; // size nx*ny+1
  std::vector<uint32_t> indices;    // size = number of points

  /// Build the grid. `min_cell_size` is a lower bound on the cell size (the
  /// caller passes the matching tolerance, since cells smaller than the
  /// tolerance only add traversal overhead).
  void Build(const std::vector<Vec2> & points, double min_cell_size)
  {
    nx = ny = 0;
    cell_begin.clear();
    indices.clear();
    if (points.empty())
      return;

    double max_x = points[0](0), max_y = points[0](1);
    min_x = max_x;
    min_y = max_y;
    for (const Vec2 & p : points)
    {
      min_x = std::min(min_x, p(0)); max_x = std::max(max_x, p(0));
      min_y = std::min(min_y, p(1)); max_y = std::max(max_y, p(1));
    }
    const double w = std::max(max_x - min_x, 1.0);
    const double h = std::max(max_y - min_y, 1.0);

    // Aim at ~1 point per cell: that keeps the number of cells visited along a
    // query and the number of points tested per cell in the same ballpark,
    // and bounds the grid memory at O(#points).
    const double density_cell = std::sqrt(w * h / static_cast<double>(points.size()));
    cell_size = std::max(std::max(min_cell_size, density_cell), 1e-3);
    inv_cell_size = 1.0 / cell_size;

    nx = static_cast<int>(w * inv_cell_size) + 1;
    ny = static_cast<int>(h * inv_cell_size) + 1;

    // Counting sort of the point indices into the cells.
    const size_t cell_count = static_cast<size_t>(nx) * static_cast<size_t>(ny);
    cell_begin.assign(cell_count + 1, 0);
    std::vector<uint32_t> cell_of_point(points.size());
    for (size_t i = 0; i < points.size(); ++i)
    {
      const uint32_t c = CellOf(points[i]);
      cell_of_point[i] = c;
      ++cell_begin[c + 1];
    }
    for (size_t c = 0; c < cell_count; ++c)
      cell_begin[c + 1] += cell_begin[c];

    indices.resize(points.size());
    std::vector<uint32_t> cursor(cell_begin.begin(), cell_begin.end() - 1);
    for (size_t i = 0; i < points.size(); ++i)
      indices[cursor[cell_of_point[i]]++] = static_cast<uint32_t>(i);
  }

  inline int ClampX(double x) const
  {
    const int gx = static_cast<int>(std::floor((x - min_x) * inv_cell_size));
    return std::min(std::max(gx, 0), nx - 1);
  }

  inline int ClampY(double y) const
  {
    const int gy = static_cast<int>(std::floor((y - min_y) * inv_cell_size));
    return std::min(std::max(gy, 0), ny - 1);
  }

  inline uint32_t CellOf(const Vec2 & p) const
  {
    return static_cast<uint32_t>(ClampY(p(1)) * nx + ClampX(p(0)));
  }

  /// Upper bounds of the gridded area (the lower bounds are min_x / min_y).
  inline double MaxX() const { return min_x + nx * cell_size; }
  inline double MaxY() const { return min_y + ny * cell_size; }

  /// Append the point indices of the cells in the inclusive row/column range.
  inline void Gather(int gx0, int gx1, int gy0, int gy1,
                     std::vector<uint32_t> & out) const
  {
    for (int gy = gy0; gy <= gy1; ++gy)
    {
      const size_t row = static_cast<size_t>(gy) * static_cast<size_t>(nx);
      const uint32_t begin = cell_begin[row + gx0];
      const uint32_t end   = cell_begin[row + gx1 + 1];
      out.insert(out.end(), indices.begin() + begin, indices.begin() + end);
    }
  }
};

/// Fill `positions` with the (optionally undistorted) region positions.
inline void RegionPositions(
  const cameras::IntrinsicBase * cam,
  const features::Regions & regions,
  std::vector<Vec2> & positions)
{
  const size_t count = regions.RegionCount();
  positions.resize(count);
  for (size_t i = 0; i < count; ++i)
  {
    positions[i] = cam ? cam->get_ud_pixel(regions.GetRegionPosition(i))
                       : regions.GetRegionPosition(i);
  }
}

/// Guided Matching for a fundamental matrix (features + descriptors with
/// distance ratio).
///
/// Same result as
///   GuidedMatching<Mat3, fundamental::kernel::EpipolarDistanceError>(...)
/// but it does not evaluate every left/right combination:
///  - the epipolar line of a left point is computed once (the generic version
///    recomputes F*x inside the inner loop, once per right point),
///  - only the right points bucketed near that line are tested, using a grid
///    walk along the line's dominant axis,
///  - the descriptor distances of the surviving candidates are computed in one
///    batched call, so the concrete Regions type is resolved once per left
///    point instead of once per candidate.
///
/// The candidate set is a strict superset of the points within `errorTh`, so no
/// match that the exhaustive version would report is lost.
inline void GuidedMatching_Fundamental_Grid(
  const Mat3 & F,       // The fundamental matrix
  const cameras::IntrinsicBase * camL, // Optional camera (undistort on the fly, can be nullptr)
  const features::Regions & lRegions,  // regions (point features & corresponding descriptors)
  const cameras::IntrinsicBase * camR, // Optional camera (undistort on the fly, can be nullptr)
  const features::Regions & rRegions,  // regions (point features & corresponding descriptors)
  double errorTh,       // Maximal authorized squared error threshold
  double distRatio,     // Maximal authorized squared distance ratio
  matching::IndMatches & vec_corresponding_index) // Output corresponding index
{
  if (lRegions.RegionCount() == 0 || rRegions.RegionCount() == 0)
    return;

  // Un-distort the positions once.
  std::vector<Vec2> lRegionsPos, rRegionsPos;
  RegionPositions(camL, lRegions, lRegionsPos);
  RegionPositions(camR, rRegions, rRegionsPos);

  // Half width of the epipolar band, in pixels.
  const double band = std::sqrt(std::max(errorTh, 0.0));

  PointGrid2D grid;
  grid.Build(rRegionsPos, 2.0 * band);
  if (grid.nx == 0 || grid.ny == 0)
    return;

  std::vector<uint32_t> cells_candidates, candidates;
  std::vector<double> descriptor_distances;

  for (size_t i = 0; i < lRegionsPos.size(); ++i)
  {
    // Epipolar line of the left point in the right image: hoisted out of the
    // candidate loop, unlike the generic implementation.
    const Vec3 line = F * lRegionsPos[i].homogeneous();
    const double a = line(0), b = line(1), c = line(2);
    const double norm2 = a * a + b * b;
    if (!(norm2 > 0.0))
      continue; // degenerate line: the exhaustive version rejects it too

    // Test |a.x + b.y + c|^2 < errorTh * (a^2 + b^2) instead of dividing.
    const double line_threshold = errorTh * norm2;
    const double norm = std::sqrt(norm2);

    // Walk the grid along the line's dominant axis and collect the cells the
    // band overlaps.
    cells_candidates.clear();
    if (std::abs(b) >= std::abs(a))
    {
      // |b| > 0 here: solve for y over each column's x range.
      const double dy = band * norm / std::abs(b); // band half height, in y
      for (int gx = 0; gx < grid.nx; ++gx)
      {
        const double x0 = grid.min_x + gx * grid.cell_size;
        const double x1 = x0 + grid.cell_size;
        const double y0 = -(a * x0 + c) / b;
        const double y1 = -(a * x1 + c) / b;
        const double lo = std::min(y0, y1) - dy;
        const double hi = std::max(y0, y1) + dy;
        if (hi < grid.min_y || lo > grid.MaxY())
          continue; // the band misses this column entirely
        grid.Gather(gx, gx, grid.ClampY(lo), grid.ClampY(hi), cells_candidates);
      }
    }
    else
    {
      // |a| > 0 here: solve for x over each row's y range.
      const double dx = band * norm / std::abs(a); // band half width, in x
      for (int gy = 0; gy < grid.ny; ++gy)
      {
        const double y0 = grid.min_y + gy * grid.cell_size;
        const double y1 = y0 + grid.cell_size;
        const double x0 = -(b * y0 + c) / a;
        const double x1 = -(b * y1 + c) / a;
        const double lo = std::min(x0, x1) - dx;
        const double hi = std::max(x0, x1) + dx;
        if (hi < grid.min_x || lo > grid.MaxX())
          continue; // the band misses this row entirely
        grid.Gather(grid.ClampX(lo), grid.ClampX(hi), gy, gy, cells_candidates);
      }
    }

    // Keep the candidates that really are within the band.
    candidates.clear();
    for (const uint32_t j : cells_candidates)
    {
      const Vec2 & p = rRegionsPos[j];
      const double signed_distance = a * p(0) + b * p(1) + c;
      if (signed_distance * signed_distance < line_threshold)
        candidates.push_back(j);
    }
    if (candidates.size() < 2)
      continue; // the distance ratio needs a best and a second best

    // The grid walk visits the right regions out of order. Restore ascending
    // index order so that equal descriptor distances break the same way they
    // do in the exhaustive version (distanceRatio::update keeps the first).
    std::sort(candidates.begin(), candidates.end());

    // One batched descriptor distance call per left point.
    descriptor_distances.resize(candidates.size());
    lRegions.SquaredDescriptorDistances(
      i, &rRegions, candidates.data(), candidates.size(),
      descriptor_distances.data());

    distanceRatio<double> dR;
    for (size_t k = 0; k < candidates.size(); ++k)
      dR.update(candidates[k], descriptor_distances[k]);

    // Add correspondence only iff the distance ratio is valid
    if (dR.isValid(distRatio))
      vec_corresponding_index.emplace_back(
        static_cast<IndexT>(i), static_cast<IndexT>(dR.idx));
  }

  // Remove duplicates (when multiple points at same position exist)
  matching::IndMatch::getDeduplicated(vec_corresponding_index);
}

/// Guided Matching for a homography (positions only).
///
/// Same result as
///   GuidedMatching<Mat3, homography::kernel::AsymmetricError>(...)
/// but a homography maps a left point to a single predicted right position, so
/// the candidates are just the grid cells within the error radius of that
/// prediction instead of every right point.
inline void GuidedMatching_Homography_Grid(
  const Mat3 & H,     // The homography
  const Mat & xLeft,  // The left data points
  const Mat & xRight, // The right data points
  double errorTh,     // Maximal authorized squared error threshold
  matching::IndMatches & vec_corresponding_index) // Output corresponding index
{
  assert(xLeft.rows() == xRight.rows());
  if (xLeft.cols() == 0 || xRight.cols() == 0)
    return;

  std::vector<Vec2> rPos(xRight.cols());
  for (Eigen::Index j = 0; j < xRight.cols(); ++j)
    rPos[j] = xRight.col(j).head<2>();

  const double radius = std::sqrt(std::max(errorTh, 0.0));

  PointGrid2D grid;
  grid.Build(rPos, 2.0 * radius);
  if (grid.nx == 0 || grid.ny == 0)
    return;

  for (Eigen::Index i = 0; i < xLeft.cols(); ++i)
  {
    const Vec3 projected = H * xLeft.col(i).head<2>().homogeneous();
    if (projected(2) == 0.0)
      continue; // point at infinity: the exhaustive version rejects it too
    const Vec2 p = projected.hnormalized();

    const int gx0 = grid.ClampX(p(0) - radius), gx1 = grid.ClampX(p(0) + radius);
    const int gy0 = grid.ClampY(p(1) - radius), gy1 = grid.ClampY(p(1) + radius);

    double min_error = std::numeric_limits<double>::max();
    uint32_t best = 0;
    for (int gy = gy0; gy <= gy1; ++gy)
    {
      const size_t row = static_cast<size_t>(gy) * static_cast<size_t>(grid.nx);
      const uint32_t begin = grid.cell_begin[row + gx0];
      const uint32_t end   = grid.cell_begin[row + gx1 + 1];
      for (uint32_t k = begin; k < end; ++k)
      {
        const uint32_t j = grid.indices[k];
        const double error = (rPos[j] - p).squaredNorm();
        // The `j < best` tie-break reproduces the exhaustive version, which
        // scans in ascending index order and keeps the first best.
        if (error < errorTh &&
            (error < min_error || (error == min_error && j < best)))
        {
          min_error = error;
          best = j;
        }
      }
    }
    if (min_error < errorTh)
      vec_corresponding_index.emplace_back(
        static_cast<IndexT>(i), static_cast<IndexT>(best));
  }

  // Remove duplicates (when multiple points at same position exist)
  matching::IndMatch::getDeduplicated(vec_corresponding_index);
}

/// Compute a bucket index from an epipolar point
///  (the one that is closer to image border intersection)
static inline unsigned int pix_to_bucket(const Vec2i &x, int W, int H)
{
  if (x(1) == 0) return x(0); // Top border
  if (x(0) == W-1) return W-1 + x(1); // Right border
  if (x(1) == H-1) return 2*W + H-3 - x(0); // Bottom border
  return 2*(W+H-2) - x(1); // Left border
}

/// Compute intersection of the epipolar line with the image border
static inline bool line_to_endPoints(const Vec3 & line, int W, int H, Vec2 & x0, Vec2 & x1)
{
  const double a = line(0), b = line(1), c = line(2);

  float r1, r2;
  // Intersection with Y axis (0 or W-1)
  if (b!=0)
  {
    double x = (b<0) ? 0 : W-1;
    double y = -(a*x+c)/b;
    if (y < 0) y = 0.;
    else if (y >= H) y = H-1;
    r1 = std::abs(a*x + b*y + c);
    x0 << x,y;
  }
  else  {
    return false;
  }

  // Intersection with X axis (0 or H-1)
  if (a!=0)
  {
    double y = (a<0) ? H-1 : 0;
    double x = -(b*y+c)/a;
    if (x < 0) x = 0.;
    else if (x >= W) x = W-1;
    r2 = std::abs(a*x + b*y + c);
    x1 << x,y;
  }
  else  {
    return false;
  }

  // Choose x0 to be as close as the intersection axis
  if (r1>r2)
    std::swap(x0,x1);

  return true;
}

/// Guided Matching (features + descriptors with distance ratio):
/// Cluster correspondences per epipolar line (faster than exhaustive search).
///   Keep the best corresponding points for the given model under the
///   user specified distance ratio.
/// Can be seen as a variant of geometry_aware method [1].
/// Note that implementation done here use a pixel grid limited to image border.
///
///  [1] Rajvi Shah, Vanshika Shrivastava, and P J Narayanan
///  Geometry-aware Feature Matching for Structure from Motion Applications.
///  WACV 2015.
template<
  typename ErrorArg> // The used model type
void GuidedMatching_Fundamental_Fast(
  const Mat3 & FMat,    // The fundamental matrix
  const Vec3 & epipole2,// Epipole2 (camera center1 in image plane2; must not be normalized)
  const cameras::IntrinsicBase * camL, // Optional camera (in order to undistord on the fly feature positions, can be nullptr)
  const features::Regions & lRegions,  // regions (point features & corresponding descriptors)
  const cameras::IntrinsicBase * camR, // Optional camera (in order to undistord on the fly feature positions, can be nullptr)
  const features::Regions & rRegions,  // regions (point features & corresponding descriptors)
  const int widthR, const int heightR,
  double errorTh,       // Maximal authorized error threshold (consider it's a square threshold)
  double distRatio,     // Maximal authorized distance ratio
  matching::IndMatches & vec_corresponding_index) // Ouput corresponding index
{
  // Looking for the corresponding points that have to satisfy:
  //   1. a geometric distance below the provided Threshold
  //   2. a distance ratio between descriptors of valid geometric correspondencess
  //
  // - Cluster left point according their epipolar line border intersection.
  // - For each right point, compute threshold limited bandwidth and compare only
  //   points that belong to this range (limited buckets).

  // Normalize F and epipole for (ep2->p2) line adequation
  Mat3 F = FMat;
  Vec3 ep2 = epipole2;
  if (ep2(2) > 0.0) {
    F = -F;
    ep2 = -ep2;
  }
  ep2 = ep2 / ep2(2);

  //--
  //-- Store point in the corresponding epipolar line bucket
  //--
  using Bucket_vec = std::vector<IndexT>;
  using Buckets_vec = std::vector<Bucket_vec>;
  const int nb_buckets = 2*(widthR + heightR-2);

  Buckets_vec buckets(nb_buckets);
  for (size_t i = 0; i < lRegions.RegionCount(); ++i) {

    // Compute epipolar line
    const Vec2 l_pt = camL ? camL->get_ud_pixel(lRegions.GetRegionPosition(i)) : lRegions.GetRegionPosition(i);
    const Vec3 line = F * Vec3(l_pt(0), l_pt(1), 1.);
    // If the epipolar line exists in Right image
    Vec2 x0, x1;
    if (line_to_endPoints(line, widthR, heightR, x0, x1))
    {
      // Find in which cluster the point belongs
      const int bucket = pix_to_bucket(x0.cast<int>(), widthR, heightR);
      buckets[bucket].push_back(i);
    }
  }

  // For each point in right image, find if there is good candidates.
  std::vector<distanceRatio<double >> dR(lRegions.RegionCount());
  for (size_t j = 0; j < rRegions.RegionCount(); ++j)
  {
    // According the point:
    // - Compute the epipolar line from the epipole
    // - Compute the range of possible bucket by computing
    //    the epipolar line gauge limitation introduced by the tolerated pixel error

    const Vec2 xR = camR ? camR->get_ud_pixel(rRegions.GetRegionPosition(j)) : rRegions.GetRegionPosition(j);
    const Vec3 l2 = ep2.cross(xR.homogeneous());
    const Vec2 n = l2.head<2>() * (sqrt(errorTh) / l2.head<2>().norm());

    const Vec3 l2min = ep2.cross(Vec3(xR(0) - n(0), xR(1) - n(1), 1.));
    const Vec3 l2max = ep2.cross(Vec3(xR(0) + n(0), xR(1) + n(1), 1.));

    // Compute corresponding buckets
    Vec2 x0, x1;
    if (!line_to_endPoints(l2min, widthR, heightR, x0, x1))
      continue;
    const int bucket_start = pix_to_bucket(x0.cast<int>(), widthR, heightR);

     if (!line_to_endPoints(l2max, widthR, heightR, x0, x1))
      continue;
    const int bucket_stop = pix_to_bucket(x0.cast<int>(), widthR, heightR);

    if (bucket_stop - bucket_start > 0) // test candidate buckets
    for (Buckets_vec::const_iterator itBs = buckets.begin() + bucket_start;
      itBs != buckets.begin() + bucket_stop; ++itBs)
    {
      const Bucket_vec & bucket = *itBs;
      for (const auto & i : bucket )
      {
        // Compute descriptor distance
        const double descDist = lRegions.SquaredDescriptorDistance(i, &rRegions, j);
        // Update the corresponding points & distance (if required)
        dR[i].update(j, descDist);
      }
    }
  }
  // Check distance ratio validity
  for (size_t i=0; i < dR.size(); ++i)
  {
    if (dR[i].isValid(distRatio))
    {
      // save the best corresponding index
      vec_corresponding_index.push_back(matching::IndMatch(i, dR[i].idx));
    }
  }
}

} // namespace geometry_aware
} // namespace openMVG

#endif // OPENMVG_ROBUST_ESTIMATION_GUIDED_MATCHING_HPP
