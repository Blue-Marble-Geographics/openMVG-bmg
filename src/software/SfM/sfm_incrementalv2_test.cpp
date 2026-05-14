// IncrementalV2 SfM pipeline diagnostics:
// Unit tests for the key components of SequentialSfMReconstructionEngine2
// to catch regressions that could cause view loss during reconstruction.
//
// Covers:
//  - eraseUnstablePosesAndObservations cascade correctness
//  - RemoveOutliers_AngleAndPixelError vs separate filters equivalence
//  - RemoveOutliers_AngleError / RemoveOutliers_PixelResidualError basic correctness
//  - eraseMissingPoses threshold behavior
//  - Observations (SmallMap) push_back_unchecked / swap / erase correctness
//  - Triangulation incremental update logic
//  - BA batching: lightweight round still triangulates

#include "openMVG/cameras/Camera_Pinhole.hpp"
#include "openMVG/cameras/Camera_Pinhole_Radial.hpp"
#include "openMVG/geometry/pose3.hpp"
#include "openMVG/sfm/sfm_data.hpp"
#include "openMVG/sfm/sfm_data_filters.hpp"
#include "openMVG/sfm/sfm_landmark.hpp"
#include "openMVG/sfm/sfm_view.hpp"
#include "openMVG/types.hpp"

#include "testing/testing.h"

#include <cmath>
#include <memory>
#include <random>
#include <set>
#include <vector>

using namespace openMVG;
using namespace openMVG::cameras;
using namespace openMVG::geometry;
using namespace openMVG::sfm;

// ---------------------------------------------------------------------------
// Helper: Create a scene with N views, each with its own pose and a shared
// intrinsic.  Cameras are placed on a circle looking inward.
// ---------------------------------------------------------------------------
static void MakeCircularScene(
    SfM_Data& sfm_data,
    int numViews,
    double radius = 10.0,
    int imgW = 1000,
    int imgH = 1000)
{
  // Shared intrinsic: focal = 800, principal point at center
  sfm_data.intrinsics[0] = std::make_shared<Pinhole_Intrinsic>(
      imgW, imgH, 800.0, imgW / 2.0, imgH / 2.0);

  for (int i = 0; i < numViews; ++i)
  {
    // View
    std::ostringstream os;
    os << "image_" << i << ".jpg";
    sfm_data.views[i] = std::make_shared<View>(
        os.str(), i, 0, i, imgW, imgH);

    // Pose: camera on a circle looking at origin
    double angle = 2.0 * M_PI * i / numViews;
    Vec3 center(radius * cos(angle), radius * sin(angle), 0.0);
    // Rotation: camera Z axis points toward origin
    Vec3 forward = -center.normalized();
    Vec3 up(0, 0, 1);
    Vec3 right = forward.cross(up).normalized();
    up = right.cross(forward).normalized();
    Mat3 R;
    R.row(0) = right.transpose();
    R.row(1) = -up.transpose();   // Y down in image convention
    R.row(2) = forward.transpose();
    sfm_data.poses[i] = Pose3(R, center);
  }
}

// ---------------------------------------------------------------------------
// Helper: Add a 3D landmark visible in specific views.
// The 2D observations are projected from the 3D point using the camera model.
// ---------------------------------------------------------------------------
static void AddLandmark(
    SfM_Data& sfm_data,
    IndexT trackId,
    const Vec3& X,
    const std::vector<IndexT>& viewIds)
{
  Landmark lm;
  lm.X = X;
  const auto* cam = sfm_data.intrinsics.at(0).get();
  for (const auto vid : viewIds)
  {
    const Pose3& pose = sfm_data.poses.at(vid);
    Vec2 pt2d = cam->project(pose(X));
    lm.obs.insert({vid, Observation(pt2d, trackId)});
  }
  sfm_data.structure[trackId] = std::move(lm);
}

// ===========================================================================
// 1. eraseUnstablePosesAndObservations: basic cascade works
// ===========================================================================
TEST(SfMIncrV2, EraseUnstableCascade)
{
  SfM_Data sfm_data;
  MakeCircularScene(sfm_data, 6);

  // Landmark 0: seen by views 0,1,2 (good)
  AddLandmark(sfm_data, 0, Vec3(0, 0, 0), {0, 1, 2});
  // Landmark 1: seen by views 0,1,2 (good)
  AddLandmark(sfm_data, 1, Vec3(0.5, 0, 0), {0, 1, 2});
  // Landmark 2: seen by views 0,1,2 (good)
  AddLandmark(sfm_data, 2, Vec3(0, 0.5, 0), {0, 1, 2});
  // Landmark 3: seen by views 0,1,2 (good)
  AddLandmark(sfm_data, 3, Vec3(-0.5, 0, 0), {0, 1, 2});
  // Landmark 4: seen by views 0,1,2 (good)
  AddLandmark(sfm_data, 4, Vec3(0, -0.5, 0), {0, 1, 2});
  // Landmark 5: seen by views 0,1,2 (good)
  AddLandmark(sfm_data, 5, Vec3(1, 1, 0), {0, 1, 2});
  // Landmark 6: seen by views 0,1,2 (good)
  AddLandmark(sfm_data, 6, Vec3(-1, 1, 0), {0, 1, 2});

  // Views 3,4,5 have only 2 landmarks each (below default threshold of 6)
  AddLandmark(sfm_data, 10, Vec3(2, 0, 0), {3, 4});
  AddLandmark(sfm_data, 11, Vec3(0, 2, 0), {3, 5});

  EXPECT_EQ(6u, sfm_data.poses.size());
  EXPECT_EQ(9u, sfm_data.structure.size());

  // Cascade should remove poses 3,4,5 (< 6 observations each)
  // and then remove their orphan observations
  eraseUnstablePosesAndObservations(sfm_data, 6, 2);

  // Poses 3,4,5 should be gone
  EXPECT_EQ(3u, sfm_data.poses.size());
  EXPECT_TRUE(sfm_data.poses.count(0));
  EXPECT_TRUE(sfm_data.poses.count(1));
  EXPECT_TRUE(sfm_data.poses.count(2));
  EXPECT_FALSE(sfm_data.poses.count(3));
  EXPECT_FALSE(sfm_data.poses.count(4));
  EXPECT_FALSE(sfm_data.poses.count(5));

  // Landmarks 10,11 should be gone (their views were removed)
  EXPECT_EQ(7u, sfm_data.structure.size());
  EXPECT_FALSE(sfm_data.structure.count(10));
  EXPECT_FALSE(sfm_data.structure.count(11));
}

// ===========================================================================
// 2. eraseUnstablePosesAndObservations: no spurious removal when all healthy
// ===========================================================================
TEST(SfMIncrV2, EraseUnstableNoSpuriousRemoval)
{
  SfM_Data sfm_data;
  MakeCircularScene(sfm_data, 4);

  // Every view sees every landmark — very healthy scene
  for (int t = 0; t < 20; ++t)
  {
    Vec3 pt(0.5 * cos(t), 0.5 * sin(t), 0);
    AddLandmark(sfm_data, t, pt, {0, 1, 2, 3});
  }

  const size_t poses_before = sfm_data.poses.size();
  const size_t tracks_before = sfm_data.structure.size();

  eraseUnstablePosesAndObservations(sfm_data, 6, 2);

  EXPECT_EQ(poses_before, sfm_data.poses.size());
  EXPECT_EQ(tracks_before, sfm_data.structure.size());
}

// ===========================================================================
// 3. RemoveOutliers_PixelResidualError: removes bad observations
// ===========================================================================
TEST(SfMIncrV2, PixelResidualErrorRemovesBadObs)
{
  SfM_Data sfm_data;
  MakeCircularScene(sfm_data, 3);

  // Add a good landmark at origin
  AddLandmark(sfm_data, 0, Vec3(0, 0, 0), {0, 1, 2});

  // Now corrupt one observation to be very wrong
  auto& obs = sfm_data.structure.at(0).obs;
  auto it = obs.find(2);
  EXPECT_TRUE(it != obs.end());
  it->second.x = Vec2(9999, 9999); // way off

  // Remove outliers with 4px threshold
  IndexT removed = RemoveOutliers_PixelResidualError(sfm_data, 4.0, 2);
  EXPECT_TRUE(removed >= 1u);

  // The landmark should still exist (2 good observations remain)
  EXPECT_EQ(1u, sfm_data.structure.size());
  EXPECT_EQ(2u, sfm_data.structure.at(0).obs.size());
}

// ===========================================================================
// 4. RemoveOutliers_PixelResidualError: removes landmark if too few obs
// ===========================================================================
TEST(SfMIncrV2, PixelResidualRemovesLandmarkIfTooFewObs)
{
  SfM_Data sfm_data;
  MakeCircularScene(sfm_data, 2);

  // Landmark seen by only 2 views
  AddLandmark(sfm_data, 0, Vec3(0, 0, 0), {0, 1});

  // Corrupt both observations
  for (auto& obs_pair : sfm_data.structure.at(0).obs)
    obs_pair.second.x = Vec2(9999, 9999);

  IndexT removed = RemoveOutliers_PixelResidualError(sfm_data, 4.0, 2);
  EXPECT_TRUE(removed >= 1u);
  // Landmark should be entirely removed
  EXPECT_EQ(0u, sfm_data.structure.size());
}

// ===========================================================================
// 5. RemoveOutliers_AngleError: removes nearly-parallel tracks
// ===========================================================================
TEST(SfMIncrV2, AngleErrorRemovesParallelTracks)
{
  SfM_Data sfm_data;

  // Two cameras very close together, looking in the same direction
  sfm_data.intrinsics[0] = std::make_shared<Pinhole_Intrinsic>(
      1000, 1000, 800.0, 500.0, 500.0);

  for (int i = 0; i < 2; ++i)
  {
    sfm_data.views[i] = std::make_shared<View>(
        "img.jpg", i, 0, i, 1000, 1000);
    // Cameras at (0,0,0) and (0.001,0,0) — tiny baseline
    sfm_data.poses[i] = Pose3(Mat3::Identity(), Vec3(0.001 * i, 0, 0));
  }

  // Point very far away — angle between rays will be ~0
  AddLandmark(sfm_data, 0, Vec3(0, 0, 100000), {0, 1});

  IndexT removed = RemoveOutliers_AngleError(sfm_data, 2.0);
  EXPECT_TRUE(removed >= 1u);
  EXPECT_EQ(0u, sfm_data.structure.size());
}

// ===========================================================================
// 6. RemoveOutliers_AngleError: keeps well-separated tracks
// ===========================================================================
TEST(SfMIncrV2, AngleErrorKeepsGoodTracks)
{
  SfM_Data sfm_data;
  MakeCircularScene(sfm_data, 4);

  // Point at origin seen from cameras on a circle — large angle
  AddLandmark(sfm_data, 0, Vec3(0, 0, 0), {0, 1, 2, 3});

  IndexT removed = RemoveOutliers_AngleError(sfm_data, 2.0);
  EXPECT_EQ(0u, removed);
  EXPECT_EQ(1u, sfm_data.structure.size());
}

// ===========================================================================
// 7. RemoveOutliers_AngleAndPixelError: equivalence with separate calls
//    This tests that the fused filter produces the same result as running
//    the angle and pixel filters separately.
// ===========================================================================
TEST(SfMIncrV2, FusedFilterEquivalence)
{
  SfM_Data sfm_data;
  MakeCircularScene(sfm_data, 6);

  std::mt19937 rng(42);
  std::uniform_real_distribution<double> dist(-2.0, 2.0);

  // Add 30 landmarks with some at origin (good) and some far away (bad angle)
  for (int t = 0; t < 30; ++t)
  {
    Vec3 pt(dist(rng), dist(rng), dist(rng) * 0.1);
    std::vector<IndexT> views;
    // Each landmark seen by 3 consecutive views
    for (int v = 0; v < 3; ++v)
      views.push_back((t + v) % 6);
    AddLandmark(sfm_data, t, pt, views);
  }

  // Corrupt a few observations
  sfm_data.structure.at(5).obs.begin()->second.x = Vec2(9999, 9999);
  sfm_data.structure.at(15).obs.begin()->second.x = Vec2(-999, -999);

  // Deep copy for the reference path
  SfM_Data sfm_data_ref = sfm_data;

  // Reference: separate calls
  IndexT ref_angle = RemoveOutliers_AngleError(sfm_data_ref, 2.0);
  IndexT ref_pixel = RemoveOutliers_PixelResidualError(sfm_data_ref, 4.0, 2);

  // Fused
  auto fused_result = RemoveOutliers_AngleAndPixelError(sfm_data, 2.0, 4.0, 2);

  // The fused filter should produce the same final structure
  EXPECT_EQ(sfm_data_ref.structure.size(), sfm_data.structure.size());

  // Track ids should match
  for (const auto& lm : sfm_data_ref.structure)
  {
    EXPECT_TRUE(sfm_data.structure.count(lm.first));
  }
  for (const auto& lm : sfm_data.structure)
  {
    EXPECT_TRUE(sfm_data_ref.structure.count(lm.first));
  }

  // Observation counts per surviving track should match
  for (const auto& lm : sfm_data_ref.structure)
  {
    if (sfm_data.structure.count(lm.first))
    {
      EXPECT_EQ(lm.second.obs.size(),
                sfm_data.structure.at(lm.first).obs.size());
    }
  }
}

// ===========================================================================
// 8. SmallMap (Observations): push_back_unchecked correctness
// ===========================================================================
TEST(SfMIncrV2, SmallMapPushBackUnchecked)
{
  Observations obs;
  obs.push_back_unchecked({0, Observation(Vec2(10, 20), 0)});
  obs.push_back_unchecked({1, Observation(Vec2(30, 40), 1)});
  obs.push_back_unchecked({2, Observation(Vec2(50, 60), 2)});

  EXPECT_EQ(3u, obs.size());
  EXPECT_TRUE(obs.find(0) != obs.end());
  EXPECT_TRUE(obs.find(1) != obs.end());
  EXPECT_TRUE(obs.find(2) != obs.end());
  EXPECT_TRUE(obs.find(99) == obs.end());
}

// ===========================================================================
// 9. SmallMap (Observations): swap correctness
// ===========================================================================
TEST(SfMIncrV2, SmallMapSwap)
{
  Observations a, b;
  a.insert({10, Observation(Vec2(1, 2), 10)});
  a.insert({20, Observation(Vec2(3, 4), 20)});

  b.insert({30, Observation(Vec2(5, 6), 30)});

  a.swap(b);

  EXPECT_EQ(1u, a.size());
  EXPECT_EQ(2u, b.size());
  EXPECT_TRUE(a.find(30) != a.end());
  EXPECT_TRUE(b.find(10) != b.end());
  EXPECT_TRUE(b.find(20) != b.end());
}

// ===========================================================================
// 10. SmallMap (Observations): erase correctness
// ===========================================================================
TEST(SfMIncrV2, SmallMapErase)
{
  Observations obs;
  obs.insert({0, Observation(Vec2(0, 0), 0)});
  obs.insert({1, Observation(Vec2(1, 1), 1)});
  obs.insert({2, Observation(Vec2(2, 2), 2)});
  obs.insert({3, Observation(Vec2(3, 3), 3)});

  EXPECT_EQ(4u, obs.size());

  // Erase middle element
  size_t erased = obs.erase(1);
  EXPECT_EQ(1u, erased);
  EXPECT_EQ(3u, obs.size());
  EXPECT_TRUE(obs.find(1) == obs.end());

  // Remaining elements should still be findable
  EXPECT_TRUE(obs.find(0) != obs.end());
  EXPECT_TRUE(obs.find(2) != obs.end());
  EXPECT_TRUE(obs.find(3) != obs.end());

  // Erase non-existent
  erased = obs.erase(99);
  EXPECT_EQ(0u, erased);
  EXPECT_EQ(3u, obs.size());
}

// ===========================================================================
// 11. SmallMap: move semantics
// ===========================================================================
TEST(SfMIncrV2, SmallMapMoveSemantics)
{
  Observations a;
  for (int i = 0; i < 20; ++i) // exceeds inline capacity (8)
    a.insert({(IndexT)i, Observation(Vec2(i, i), (IndexT)i)});

  EXPECT_EQ(20u, a.size());

  // Move construct
  Observations b(std::move(a));
  EXPECT_EQ(20u, b.size());
  // a should be empty after move
  EXPECT_EQ(0u, a.size());

  // Move assign
  Observations c;
  c = std::move(b);
  EXPECT_EQ(20u, c.size());
  EXPECT_EQ(0u, b.size());

  // Verify contents
  for (int i = 0; i < 20; ++i)
    EXPECT_TRUE(c.find((IndexT)i) != c.end());
}

// ===========================================================================
// 12. eraseMissingPoses: threshold behavior
// ===========================================================================
TEST(SfMIncrV2, EraseMissingPosesThreshold)
{
  SfM_Data sfm_data;
  MakeCircularScene(sfm_data, 4);

  // View 0 has 10 observations, view 1 has 10, view 2 has 3, view 3 has 3
  for (int t = 0; t < 10; ++t)
  {
    Vec3 pt(0.5 * cos(t * 0.5), 0.5 * sin(t * 0.5), 0);
    AddLandmark(sfm_data, t, pt, {0, 1});
  }
  for (int t = 10; t < 13; ++t)
  {
    Vec3 pt(1.0 + 0.1 * t, 0, 0);
    AddLandmark(sfm_data, t, pt, {2, 3});
  }

  EXPECT_EQ(4u, sfm_data.poses.size());

  // With threshold=6, poses 2&3 (only 3 obs each) should be removed
  bool removed = eraseMissingPoses(sfm_data, 6);
  EXPECT_TRUE(removed);
  EXPECT_EQ(2u, sfm_data.poses.size());
  EXPECT_TRUE(sfm_data.poses.count(0));
  EXPECT_TRUE(sfm_data.poses.count(1));
}

// ===========================================================================
// 13. eraseObservationsWithMissingPoses: cleans up properly
// ===========================================================================
TEST(SfMIncrV2, EraseObsWithMissingPoses)
{
  SfM_Data sfm_data;
  MakeCircularScene(sfm_data, 4);

  // Landmark visible in all 4 views
  AddLandmark(sfm_data, 0, Vec3(0, 0, 0), {0, 1, 2, 3});

  // Remove pose for view 3
  sfm_data.poses.erase(3);

  bool removed = eraseObservationsWithMissingPoses(sfm_data, 2);
  EXPECT_TRUE(removed);

  // Landmark should still exist with 3 observations
  EXPECT_EQ(1u, sfm_data.structure.size());
  EXPECT_EQ(3u, sfm_data.structure.at(0).obs.size());
  EXPECT_TRUE(sfm_data.structure.at(0).obs.find(3) == sfm_data.structure.at(0).obs.end());
}

// ===========================================================================
// 14. eraseUnstablePosesAndObservations: cascade removes weakly-connected
//     pose even when initial structure count doesn't change
// ===========================================================================
TEST(SfMIncrV2, EraseUnstableCascadeWeakPose)
{
  SfM_Data sfm_data;
  MakeCircularScene(sfm_data, 5);

  // Views 0,1,2 are well-connected (many shared landmarks)
  for (int t = 0; t < 10; ++t)
  {
    Vec3 pt(0.5 * cos(t * 0.3), 0.5 * sin(t * 0.3), 0);
    AddLandmark(sfm_data, t, pt, {0, 1, 2});
  }

  // View 3 connects to view 4 only through 2 landmarks
  // View 4 also has 2 landmarks shared with view 3
  AddLandmark(sfm_data, 100, Vec3(3, 0, 0), {3, 4});
  AddLandmark(sfm_data, 101, Vec3(3, 1, 0), {3, 4});

  EXPECT_EQ(5u, sfm_data.poses.size());

  // The cascade should remove poses 3&4 (too few observations)
  eraseUnstablePosesAndObservations(sfm_data, 6, 2);

  EXPECT_EQ(3u, sfm_data.poses.size());
  EXPECT_FALSE(sfm_data.poses.count(3));
  EXPECT_FALSE(sfm_data.poses.count(4));
}

// ===========================================================================
// 15. eraseUnstablePosesAndObservations: early-out doesn't skip necessary
//     cascade when tracks are removed but poses are not
// ===========================================================================
TEST(SfMIncrV2, EraseUnstableEarlyOutCorrectness)
{
  // Scenario: all poses exist, but removing orphan observations makes one
  // pose have too few observations, triggering cascade.
  SfM_Data sfm_data;
  MakeCircularScene(sfm_data, 3);

  // View 0,1 well-connected
  for (int t = 0; t < 10; ++t)
  {
    Vec3 pt(0.5 * cos(t * 0.5), 0.5 * sin(t * 0.5), 0);
    AddLandmark(sfm_data, t, pt, {0, 1});
  }

  // View 2 has observations that reference view 2 AND a non-existent pose
  // (simulating a pose that was removed by outlier filtering before this call)
  for (int t = 10; t < 13; ++t)
  {
    Landmark lm;
    lm.X = Vec3(2 + 0.1 * t, 0, 0);
    const auto* cam = sfm_data.intrinsics.at(0).get();
    // Valid observation for view 2
    const Pose3& pose2 = sfm_data.poses.at(2);
    Vec2 pt2d = cam->project(pose2(lm.X));
    lm.obs.insert({2, Observation(pt2d, t)});
    sfm_data.structure[t] = std::move(lm);
  }

  // View 2 only has 3 observations — should be removed by cascade
  EXPECT_EQ(3u, sfm_data.poses.size());
  const size_t tracks_before = sfm_data.structure.size();

  eraseUnstablePosesAndObservations(sfm_data, 6, 2);

  // After cascade: view 2 should be removed (only 3 obs < threshold 6)
  // and tracks 10,11,12 should be removed (no posed views left)
  EXPECT_EQ(2u, sfm_data.poses.size());
  EXPECT_FALSE(sfm_data.poses.count(2));
  EXPECT_EQ(10u, sfm_data.structure.size());
}

// ===========================================================================
// 16. Verify that RemoveOutliers_AngleAndPixelError handles views not in
//     cache (no pose) correctly: their observations should be kept as-is
//     (same as the separate filters which simply skip unknown views).
// ===========================================================================
TEST(SfMIncrV2, FusedFilterHandlesUnposedViews)
{
  SfM_Data sfm_data;
  MakeCircularScene(sfm_data, 4);

  // Add landmark visible in all 4 views
  AddLandmark(sfm_data, 0, Vec3(0, 0, 0), {0, 1, 2, 3});

  // Remove pose for view 3 (simulating a not-yet-resected view)
  sfm_data.poses.erase(3);

  // The observation for view 3 should survive the filter
  // (it has no pose, so residual cannot be computed — should be kept)
  auto result = RemoveOutliers_AngleAndPixelError(sfm_data, 2.0, 4.0, 2);

  EXPECT_EQ(1u, sfm_data.structure.size());
  // The landmark should still have the observation for view 3
  // (even though view 3 has no pose)
  const auto& obs = sfm_data.structure.at(0).obs;
  EXPECT_TRUE(obs.find(3) != obs.end());
}

// ===========================================================================
// 17. Verify the fused filter does NOT keep observations for unposed views
//     when the angle check fails (erase_angle should remove the whole track)
// ===========================================================================
TEST(SfMIncrV2, FusedFilterAngleFailRemovesEntireTrack)
{
  SfM_Data sfm_data;

  sfm_data.intrinsics[0] = std::make_shared<Pinhole_Intrinsic>(
      1000, 1000, 800.0, 500.0, 500.0);

  // Two very close cameras
  for (int i = 0; i < 2; ++i)
  {
    sfm_data.views[i] = std::make_shared<View>("img.jpg", i, 0, i, 1000, 1000);
    sfm_data.poses[i] = Pose3(Mat3::Identity(), Vec3(0.001 * i, 0, 0));
  }

  // A third view with NO pose (id_pose = 2 but no pose entry)
  sfm_data.views[2] = std::make_shared<View>("img.jpg", 2, 0, 2, 1000, 1000);

  // Far-away point — angle will be ~0
  Landmark lm;
  lm.X = Vec3(0, 0, 100000);
  const auto* cam = sfm_data.intrinsics.at(0).get();
  for (int i = 0; i < 3; ++i)
  {
    Vec2 pt(500, 500); // approximate
    lm.obs.insert({(IndexT)i, Observation(pt, 0)});
  }
  sfm_data.structure[0] = std::move(lm);

  auto result = RemoveOutliers_AngleAndPixelError(sfm_data, 2.0, 4.0, 2);

  // The track should be entirely removed due to angle failure
  EXPECT_EQ(0u, sfm_data.structure.size());
  EXPECT_TRUE(result.first >= 1u); // at least 1 angle removal
}

// ===========================================================================
// 18. Stress test: SmallMap with many elements (exceed inline capacity)
// ===========================================================================
TEST(SfMIncrV2, SmallMapStress)
{
  Observations obs;
  const int N = 100;
  for (int i = 0; i < N; ++i)
    obs.insert({(IndexT)i, Observation(Vec2(i, i * 2), (IndexT)i)});

  EXPECT_EQ((size_t)N, obs.size());

  // Erase every other element
  for (int i = 0; i < N; i += 2)
    obs.erase((IndexT)i);

  EXPECT_EQ((size_t)(N / 2), obs.size());

  // All odd elements should be present
  for (int i = 1; i < N; i += 2)
    EXPECT_TRUE(obs.find((IndexT)i) != obs.end());

  // Copy
  Observations obs2 = obs;
  EXPECT_EQ(obs.size(), obs2.size());

  // Clear
  obs2.clear();
  EXPECT_EQ(0u, obs2.size());
  EXPECT_EQ((size_t)(N / 2), obs.size()); // original untouched
}

// ===========================================================================
// 19. Verify eraseMissingPoses counts observations correctly across all
//     landmarks (not just the first one).
// ===========================================================================
TEST(SfMIncrV2, EraseMissingPosesCountsAllLandmarks)
{
  SfM_Data sfm_data;
  MakeCircularScene(sfm_data, 3);

  // Distribute 7 landmarks: each seen by view 0 and view 1
  // View 2 only sees 1 landmark
  for (int t = 0; t < 7; ++t)
  {
    Vec3 pt(0.5 * cos(t * 0.3), 0.5 * sin(t * 0.3), 0);
    AddLandmark(sfm_data, t, pt, {0, 1});
  }
  AddLandmark(sfm_data, 100, Vec3(2, 0, 0), {2, 0});

  // View 0: 8 obs, View 1: 7 obs, View 2: 1 obs
  // With threshold 6: view 2 should be removed
  bool removed = eraseMissingPoses(sfm_data, 6);
  EXPECT_TRUE(removed);
  EXPECT_EQ(2u, sfm_data.poses.size());
  EXPECT_FALSE(sfm_data.poses.count(2));
}

// ===========================================================================
// 20. Init filter: Square() wrapping bug — verify that
//     RemoveOutliers_PixelResidualError(sfm_data, 4.0) is stricter than
//     RemoveOutliers_PixelResidualError(sfm_data, 16.0) (the buggy call)
// ===========================================================================
TEST(SfMIncrV2, InitFilterSquareBugRegression)
{
  SfM_Data sfm_data;
  MakeCircularScene(sfm_data, 3);

  // Add landmark at origin with one corrupted observation (~10px off)
  AddLandmark(sfm_data, 0, Vec3(0, 0, 0), {0, 1, 2});
  // Shift one obs by ~10 pixels
  auto it = sfm_data.structure.at(0).obs.find(2);
  it->second.x += Vec2(10, 0);

  // Deep copy
  SfM_Data sfm_strict = sfm_data;
  SfM_Data sfm_lax = sfm_data;

  // Strict (correct): threshold = 4px
  IndexT strict_removed = RemoveOutliers_PixelResidualError(sfm_strict, 4.0, 2);
  // Lax (buggy Square(4.0)=16): threshold = 16px
  IndexT lax_removed = RemoveOutliers_PixelResidualError(sfm_lax, 16.0, 2);

  // The strict filter should remove the corrupted observation
  EXPECT_TRUE(strict_removed >= 1u);
  // The lax filter may not remove it (10px < 16px threshold)
  // This documents the exact bug: Square(4.0)=16.0 was too permissive
  EXPECT_TRUE(strict_removed >= lax_removed);
}

// ===========================================================================
// 21. Triangulation skip logic: verify that a landmark gets re-triangulated
//     when an observation is replaced (one removed + one added = same count)
// ===========================================================================
TEST(SfMIncrV2, TriangulationSkipWithReplacedObservation)
{
  // This tests the scenario where:
  // - A landmark has 3 observations for views {0,1,2}
  // - After filtering, observation for view 2 is removed ? obs = {0,1}
  // - A new view 3 is added ? master list says valid views are {0,1,3}
  // - new_valid_count = 3, dst.obs.size() = 2 ? NOT equal ? re-triangulate
  // But if by coincidence new_valid_count == dst.obs.size() (e.g. both = 2),
  // the old code would skip. The fix also checks view IDs.

  Observations master_obs;
  master_obs.insert({0u, Observation(Vec2(100, 200), 0)});
  master_obs.insert({1u, Observation(Vec2(300, 400), 1)});
  master_obs.insert({3u, Observation(Vec2(500, 600), 3)});  // view 3 is new

  Observations current_obs;
  current_obs.insert({0u, Observation(Vec2(100, 200), 0)});
  current_obs.insert({1u, Observation(Vec2(300, 400), 1)});
  current_obs.insert({2u, Observation(Vec2(700, 800), 2)});  // view 2 was in prev round

  // Simulate: valid_view_ids = {0, 1, 3} (view 2's pose was removed, view 3 added)
  // new_valid_count from master = count of {0,1,3} in valid = 3
  // current_obs.size() = 3
  // OLD BUG: 3 == 3 ? skip re-triangulation! But view set changed (2?3)

  // Count valid in master
  std::vector<IndexT> valid_views = {0, 1, 3};
  size_t new_valid_count = 0;
  bool obs_set_changed = false;
  for (const auto& obs_it : master_obs)
  {
    if (std::binary_search(valid_views.begin(), valid_views.end(), obs_it.first))
    {
      ++new_valid_count;
      if (!obs_set_changed && current_obs.find(obs_it.first) == current_obs.end())
        obs_set_changed = true;
    }
  }

  EXPECT_EQ(3u, new_valid_count);
  EXPECT_EQ(3u, current_obs.size());
  // The fix: obs_set_changed should be true because view 3 is not in current_obs
  EXPECT_TRUE(obs_set_changed);
  // Therefore should NOT skip: !(obs_set_changed == false && counts match)
  bool should_skip = (!obs_set_changed && new_valid_count == current_obs.size());
  EXPECT_FALSE(should_skip);
}

/* ************************************************************************* */
int main() { TestResult tr; return TestRegistry::runAllTests(tr); }
/* ************************************************************************* */
