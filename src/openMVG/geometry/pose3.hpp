// This file is part of OpenMVG, an Open Multiple View Geometry C++ library.

// Copyright (c) 2015 Pierre MOULON.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef OPENMVG_GEOMETRY_POSE3_HPP
#define OPENMVG_GEOMETRY_POSE3_HPP

#include "openMVG/multiview/projection.hpp"

// [POSE3-PERF] Force-inline marker for the world-to-camera transform
// operator() below. Pose3::operator()(const Vec3&) is the innermost call
// in nearly every reprojection-residual loop in the codebase (BA cost
// functions, RemoveOutliers_*, EjectPosesByMedianResidual, triangulation,
// resection inlier counting). MSVC at /O2 will usually inline a plain
// `inline` member but does not guarantee it for templated callers and
// across-TU paths; __forceinline removes that uncertainty. GCC/Clang map
// to __attribute__((always_inline)). Output is bit-identical -- this is
// a code-generation hint only.
//
// Toggle: set OPENMVG_POSE3_FORCEINLINE to 0 to revert to plain `inline`.
#ifndef OPENMVG_POSE3_FORCEINLINE
#define OPENMVG_POSE3_FORCEINLINE 1
#endif
#if OPENMVG_POSE3_FORCEINLINE
  #if defined(_MSC_VER)
    #define OPENMVG_POSE3_INLINE __forceinline
  #elif defined(__GNUC__) || defined(__clang__)
    #define OPENMVG_POSE3_INLINE inline __attribute__((always_inline))
  #else
    #define OPENMVG_POSE3_INLINE inline
  #endif
#else
  #define OPENMVG_POSE3_INLINE inline
#endif

namespace openMVG
{
namespace geometry
{

/**
* @brief Defines a pose in 3d space
* [R|C] t = -RC
*/
class Pose3
{
  protected:

    /// Orientation matrix
    Mat3 rotation_;

    /// Center of rotation
    Vec3 center_;

  public:

    /**
    * @brief Constructor
    * @param r Rotation
    * @param c Center
    * @note Default (without args) defines an Identity pose.
    */
    Pose3
    (
      const Mat3& r = std::move(Mat3::Identity()),
      const Vec3& c = std::move(Vec3::Zero())
    )
    : rotation_( r ), center_( c ) {}

    /**
    * @brief Get Rotation matrix
    * @return Rotation matrix
    */
    const Mat3& rotation() const
    {
      return rotation_;
    }

    /**
    * @brief Get Rotation matrix
    * @return Rotation matrix
    */
    Mat3& rotation()
    {
      return rotation_;
    }

    /**
    * @brief Get center of rotation
    * @return center of rotation
    */
    const Vec3& center() const
    {
      return center_;
    }

    /**
    * @brief Get center of rotation
    * @return Center of rotation
    */
    Vec3& center()
    {
      return center_;
    }

    /**
    * @brief Get translation vector
    * @return translation vector
    * @note t = -RC
    */
    inline Vec3 translation() const
    {
      return -( rotation_ * center_ );
    }


    /**
    * @brief Apply pose
    * @param p Point
    * @return transformed point
    */
    template<typename T>
    inline typename T::PlainObject operator() (const T& p) const
    {
      return rotation_ * ( p.colwise() - center_ );
    }
    /// Specialization for Vec3
    OPENMVG_POSE3_INLINE typename Vec3::PlainObject operator() (const Vec3& p) const
    {
      return rotation_ * ( p - center_ );
    }


    /**
    * @brief Composition of poses
    * @param P a Pose
    * @return Composition of current pose and parameter pose
    */
    Pose3 operator * ( const Pose3& P ) const
    {
      return {rotation_ * P.rotation_,
              P.center_ + P.rotation_.transpose() * center_};
    }


    /**
    * @brief Get inverse of the pose
    * @return Inverse of the pose
    */
    Pose3 inverse() const
    {
      return {rotation_.transpose(),  -( rotation_ * center_ )};
    }

    /**
    * @brief Return the pose as a single Mat34 matrix [R|t]
    * @return The pose as a Mat34 matrix
    */
    inline Mat34 asMatrix() const
    {
      return (Mat34() << rotation_, translation()).finished();
    }

    /**
    * Serialization out
    * @param ar Archive
    */
    template <class Archive>
    inline void save( Archive & ar ) const;

    /**
    * @brief Serialization in
    * @param ar Archive
    */
    template <class Archive>
    inline void load( Archive & ar );
};
} // namespace geometry
} // namespace openMVG

#endif  // OPENMVG_GEOMETRY_POSE3_HPP
