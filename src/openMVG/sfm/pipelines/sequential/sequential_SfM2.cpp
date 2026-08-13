// This file is part of OpenMVG, an Open Multiple View Geometry C++ library.

// Copyright (c) 2018 Pierre MOULON.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "openMVG/sfm/pipelines/sequential/sequential_SfM2.hpp"
#include "openMVG/sfm/pipelines/localization/SfM_Localizer.hpp"
#include "openMVG/sfm/pipelines/sfm_features_provider.hpp"
#include "openMVG/sfm/pipelines/sfm_matches_provider.hpp"
#include "openMVG/sfm/pipelines/sequential/SfmSceneInitializer.hpp"
#include "openMVG/sfm/pipelines/sequential/SfmSceneInitializerMaxPair.hpp"
#include "openMVG/sfm/sfm_data.hpp"
#include "openMVG/sfm/sfm_data_BA.hpp"
#include "openMVG/sfm/sfm_data_BA_ceres.hpp"
#include "openMVG/sfm/sfm_data_filters.hpp"
#include "openMVG/sfm/sfm_data_io.hpp"
#include "openMVG/sfm/sfm_data_triangulation.hpp"
#include "openMVG/stl/stl.hpp"
#include "openMVG/system/logger.hpp"

#include "third_party/histogram/histogram.hpp"
#include "third_party/htmlDoc/htmlDoc.hpp"

#include <array>
#include <ceres/types.h>
#include <chrono>
#include <cstdint>
#include <functional>
#include <iostream>

#ifdef OPENMVG_USE_OPENMP
#include <omp.h>
#endif

#ifdef _MSC_VER
#include <intrin.h>
#endif

namespace {

// popcount / count-trailing-zeros on a 64-bit word. std::popcount and
// std::countr_zero are C++20; openMVG builds as C++17, so use the
// compiler intrinsics (same ones cascade_hasher.hpp already relies on).
inline unsigned PopCount64(uint64_t v)
{
#ifdef _MSC_VER
  return static_cast<unsigned>(__popcnt64(v));
#else
  return static_cast<unsigned>(__builtin_popcountll(v));
#endif
}

// Precondition: v != 0.
inline unsigned CountTrailingZeros64(uint64_t v)
{
#ifdef _MSC_VER
  unsigned long index;
  _BitScanForward64(&index, v);
  return static_cast<unsigned>(index);
#else
  return static_cast<unsigned>(__builtin_ctzll(v));
#endif
}

} // namespace

// Toggle: relaxed angle thresholds for foliage / narrow-baseline scenes.
//   1 = current optimisation (1.5deg in-loop post-BA, 3.0deg post-init).
//   0 = S3 reference values (2.0deg in-loop, Square(2.0)deg post-init).
//   Defined here as a single switch so all four call sites flip together.
#ifndef OPENMVG_SFM2_RELAXED_ANGLE_THRESHOLDS
#define OPENMVG_SFM2_RELAXED_ANGLE_THRESHOLDS 0
#endif

// Minimum parallax (triangulation) angle, in DEGREES, required to KEEP a
// landmark in the post-Triangulation / pre-BA seed filter. Upstream hard-codes
// 4.0 deg here (Square(2.0)). That is fine for well-separated seed pairs but
// wipes the ENTIRE seed on low-parallax / near-nadir aerial pairs: the seed
// triangulates fine (thousands of points at <4px reprojection) yet every point
// has parallax < 4 deg, so the filter removes all of them and the engine stalls
// at 2 cameras / 0 tracks. Lowering the floor only affects datasets whose seed
// parallax is below the old 4 deg -- i.e. exactly the ones that were failing;
// high-parallax seeds have all points well above this and are unchanged. Keep
// this consistent with the MaxPair seed-validation gate
// (OPENMVG_MAXPAIR_SEED_MIN_PARALLAX_DEG) and the in-loop post-BA filter
// (1.5-2.0 deg).
#ifndef OPENMVG_SFM2_SEED_MIN_ANGLE_DEG
#define OPENMVG_SFM2_SEED_MIN_ANGLE_DEG 2.0
#endif

// Minimum parallax angle, in DEGREES, required to KEEP a landmark in the
// IN-LOOP post-BA outlier filter run each resection round (and the end-of-band
// flush). Same rationale as OPENMVG_SFM2_SEED_MIN_ANGLE_DEG but applied to
// growing structure: on low-parallax / near-nadir aerial scenes a 2.0 deg floor
// keeps pruning legitimate short-baseline tie points every round, so the final
// sparse cloud comes out thin (structure never accumulates). Lowering to
// 1.0-1.5 deg densifies such scenes; high-parallax datasets are unaffected
// (their kept structure is well above this). Leave at 2.0 to match the
// historical behaviour.
#ifndef OPENMVG_SFM2_TRACK_MIN_ANGLE_DEG
#define OPENMVG_SFM2_TRACK_MIN_ANGLE_DEG 2.0
#endif

// Toggle: freeze intrinsics on intermediate BAs (speed optimisation).
//   1 = intermediate BAs refine extrinsics+structure only; final STRICT BA
//       refines intrinsics end-to-end. ~5-15s faster per run.
//   0 = safe default (refine intrinsics every intermediate BA). Was disabled
//       to diagnose a 'ghost-layer' regression on a dataset with poor EXIF-
//       derived intrinsics; re-enable to test if your inputs are well-
//       calibrated.
#ifndef OPENMVG_SFM2_FREEZE_INTRINSICS_ON_INTERMEDIATE_BA
#define OPENMVG_SFM2_FREEZE_INTRINSICS_ON_INTERMEDIATE_BA 0
#endif

// A/B toggle: pretend no sparse linear-algebra library is available, even
// when Ceres reports otherwise. Forces every BundleAdjustment() call onto
// the DENSE_SCHUR path -- the exact behaviour of a Ceres build without
// SuiteSparse/EIGEN_SPARSE. Useful for measuring whether the SuiteSparse
// link is actually winning on a given scene without rebuilding Ceres.
//   0 = honour Ceres' runtime probe (production default)
//   1 = force DENSE_SCHUR everywhere
#ifndef OPENMVG_SFM2_FORCE_NO_SPARSE_BA
#define OPENMVG_SFM2_FORCE_NO_SPARSE_BA 0
#endif

// Pose count above which BundleAdjustment() switches from SPARSE_SCHUR to
// ITERATIVE_SCHUR + SCHUR_JACOBI. 0 DISABLES the branch entirely (SPARSE_SCHUR
// is then used for every scene above 100 poses).
//
// PERF ONLY, but NOT bit-identical: CG is an inexact solve, so the trust-region
// trajectory differs from a direct factorisation even though both target the
// same KKT system. Expect a different-but-equivalent reconstruction, not the
// same bytes.
//
// HISTORY: was a hard-coded `n_poses > 1500`, bumped there from 300 so the
// branch would not fire on the ~436-pose drone dataset, where ITERATIVE_SCHUR
// measured 22-36s against 11s for SPARSE_SCHUR (2-3.3x SLOWER). No crossover
// was ever observed at 1500 -- the threshold was placed above the test scene,
// not derived. The stated cache justification ("dense Schur block no longer
// fits in cache ~1500-2000 poses") does not hold: the block is (6*n_poses)^2
// doubles, which exceeds a 64MB L3 at ~470 poses, i.e. below the 436-pose
// measurement where direct factorisation still won by 3.3x.
//
// First run to actually cross 1500 poses (2000-image scene, 52.5M residuals)
// spent 50.9s of a 158.6s BA in the linear solve, and because ITERATIVE_SCHUR
// forces use_explicit_schur_complement it also materialised an 832MB dense
// Schur complement -- the very cost the branch claims to avoid.
//
// Defaulted to 0 pending a same-scene A/B. Set to a pose count to re-enable.
#ifndef OPENMVG_SFM2_ITERATIVE_SCHUR_MIN_POSES
#define OPENMVG_SFM2_ITERATIVE_SCHUR_MIN_POSES 0
#endif

// Diagnostic toggle: time every Triangulation() call (INERT -- logging only,
// no effect on output). Set to 1 to confirm whether triangulation is actually
// a wall-clock hotspot before committing to the incremental-triangulation
// rewrite. Each call logs its own duration plus a running cumulative total and
// the per-call structure size, so you can see how the cost scales as the scene
// matures. Gated off by default: zero overhead, byte-identical reconstruction.
//   0 = off (production default)
//   1 = log "[TRI-TIME] ..." once per Triangulation() call
#ifndef OPENMVG_SFM2_TIME_TRIANGULATION
#define OPENMVG_SFM2_TIME_TRIANGULATION 0
#endif

// Minimum number of observations a track must have to survive the initial
// TracksBuilder::Filter() pass in InitTracksAndLandmarks().
//
// CHANGES OUTPUT: this is a semantic knob, not a pure perf toggle.
//   2 = reference / safe default. Keeps every length-2+ track. Bit-identical
//       to upstream behaviour. Restore this value if you see ghost layers,
//       lost coverage, or fewer calibrated cameras after enabling the cut.
//   3 = drop length-2 tracks before triangulation. On drone scenes this
//       typically halves the track count and shrinks BA residual count by
//       30-50% with little or no loss of calibrated views, because the
//       short tracks dropped here are usually already pruned by the
//       post-Triangulation pixel/angle filter a few lines later anyway.
//       The wins propagate into every intermediate BA *and* the final
//       STRICT BA. Expected: -30 to -80 s on a 368-pose drone scene.
//   4+ = aggressive; only safe on very dense scenes.
//
// Flip this back to 2 if downstream MVS shows degraded output.
//
// HISTORY: briefly forced to 2 to confirm the cut was the cause of an
// intermittent non-convergence on the Randy dataset. Confirmed, then restored
// to 3 (the speed win) with a SOFTENED coverage guard below (MIN_CELL_SUPPORT
// 4->6, BORDER_PCT 12->20) so the load-bearing peripheral short tracks Randy
// needs are retained while the redundant interior ones are still dropped.
#ifndef OPENMVG_SFM2_TRACKS_MIN_LENGTH
#define OPENMVG_SFM2_TRACKS_MIN_LENGTH 3
#endif

// Coverage guard for the SELECTIVE short-track cut, measured by per-view
// IMAGE-SPACE occupancy. Strong (length>=MIN_LENGTH) observations are binned
// into a fixed-size pixel grid per view; a short track is dropped only if every
// one of its observations lands in a cell already holding at least
// OPENMVG_SFM2_TRACKS_MIN_CELL_SUPPORT strong observations. Short tracks in
// sparse cells (frame edges/corners with little redundant structure) are kept
// -- this is what prevents the right-edge blow-up a global cut produces.
//
//   0   = DISABLE the guard -> unconditional global cut. Equivalent to the old
//         tracksBuilder.Filter(MIN_LENGTH): fastest, but can splay sparsely-
//         covered edges. Only safe on uniformly dense scenes.
//   ~4  = balanced DEFAULT. Cells already packed with redundant strong
//         structure shed their short tracks for the speed win; sparse cells
//         keep theirs for stability.
//   higher = safer / less aggressive (fewer short tracks dropped). Raise this
//            if you still see edge drift; lower it toward 0 for more speed on
//            scenes you trust are dense.
//
// Tune per-dataset at build time, e.g.:
//   -DOPENMVG_SFM2_TRACKS_MIN_CELL_SUPPORT=6
//
// Raised 4->6 after the Randy non-convergence: a cell now needs 6 strong obs
// before its short tracks are considered redundant, so more peripheral short
// tracks survive and the BA conditioning margin stays wide enough to absorb
// the pipeline's run-to-run nondeterminism.
#ifndef OPENMVG_SFM2_TRACKS_MIN_CELL_SUPPORT
#define OPENMVG_SFM2_TRACKS_MIN_CELL_SUPPORT 6
#endif

// Pixel size of the square image-grid cell used by the coverage guard above.
// Larger cells = coarser coverage test = more short tracks judged redundant
// (more aggressive); smaller cells = finer, safer. ~1/20th of the image width
// is a reasonable starting point.
#ifndef OPENMVG_SFM2_TRACKS_COVERAGE_CELL_PX
#define OPENMVG_SFM2_TRACKS_COVERAGE_CELL_PX 96
#endif

// Frame-border margin, as a percentage of each image's min-dimension, for the
// coverage guard's border refinement. A short track is protected only if one
// of its sparse-cell observations also lies within this margin of a frame edge;
// sparse cells in the image interior do NOT protect (they are surrounded by
// structure and do not drive the scene-edge divergence). Smaller = more
// aggressive (only the very outermost rim protects); larger = safer. Set to
// >= 50 to disable the border test entirely (revert to "any sparse cell
// protects").
//
// Raised 12->20 alongside the Randy fix: a wider border band protects short
// tracks further in from the frame edge, covering the peripheral structure
// Randy depends on without protecting the dense interior.
#ifndef OPENMVG_SFM2_TRACKS_COVERAGE_BORDER_PCT
#define OPENMVG_SFM2_TRACKS_COVERAGE_BORDER_PCT 20
#endif

namespace openMVG {
namespace sfm {

using namespace openMVG::cameras;
using namespace openMVG::geometry;
using namespace openMVG::matching;

#if OPENMVG_SFM_PIPELINE_DIAG
// NOTE: kept at namespace scope (not in an anonymous namespace) so that
// Hash_Map<IndexT, ViewDiagRecord> can be named from member functions
// without IntelliSense / some compilers tripping on anon-namespace lookup.
// `static` on the free functions gives them internal linkage instead.

// Reasons a view may fail to be calibrated. Tracked per-view so we can give
// a definitive post-mortem for every input image at the end of Process().
enum class ResectionFailReason : uint8_t {
  Unknown                    = 0,
  NoVisibleTracks            = 1,  // view sees zero tracks at all
  NoReconstructedTracks      = 2,  // view sees tracks, but none are triangulated
  LowTrackRatio              = 3,  // 2D-3D ratio below track_inlier_ratio gate
  ResectionFailed            = 4,  // SfM_Localizer::Localize returned false
  LowInlierRatio             = 5,  // resection ran, inlier ratio <= 0.5
  RefinePoseFailed           = 6,  // post-resection BA refinement failed
  Calibrated                 = 7   // success (kept for symmetry / debugging)
};

static inline const char * to_string(ResectionFailReason r)
{
  switch (r) {
    case ResectionFailReason::NoVisibleTracks:       return "NO_VISIBLE_TRACKS";
    case ResectionFailReason::NoReconstructedTracks: return "NO_RECONSTRUCTED_TRACKS";
    case ResectionFailReason::LowTrackRatio:         return "LOW_TRACK_RATIO";
    case ResectionFailReason::ResectionFailed:       return "RESECTION_FAILED";
    case ResectionFailReason::LowInlierRatio:        return "LOW_INLIER_RATIO";
    case ResectionFailReason::RefinePoseFailed:      return "REFINE_POSE_FAILED";
    case ResectionFailReason::Calibrated:            return "CALIBRATED";
    default:                                         return "UNKNOWN";
  }
}

// One entry per input view, accumulated across the run. Updated under
// `#pragma omp critical` because AddingMissingView() is parallelized.
struct ViewDiagRecord {
  ResectionFailReason last_reason = ResectionFailReason::Unknown;
  uint32_t            attempts    = 0;
  uint32_t            best_2d3d   = 0;     // max 2D-3D matches seen
  uint32_t            last_inliers= 0;
  float               last_ratio  = 0.f;   // last inlier ratio
  float               best_track_ratio = 0.f;
};

// --- Checkpoint 1: input sanity ---------------------------------------------
static void DiagInputs(
  const SfM_Data & sfm_data,
  const Features_Provider * feats,
  const Matches_Provider * matches)
{
  const size_t n_views   = sfm_data.GetViews().size();
  const size_t n_intr    = sfm_data.GetIntrinsics().size();
  const size_t n_seed    = sfm_data.GetPoses().size();
  const size_t n_pairs   = matches ? matches->pairWise_matches_.size() : 0;
  const size_t n_feat_v  = feats   ? feats->feats_per_view.size()      : 0;

  size_t total_features = 0;
  size_t total_matches  = 0;
  size_t views_no_feats = 0;
  if (feats)
  {
    for (const auto & v : sfm_data.GetViews())
    {
      auto it = feats->feats_per_view.find(v.first);
      if (it == feats->feats_per_view.end() || it->second.empty())
        ++views_no_feats;
      else
        total_features += it->second.size();
    }
  }
  if (matches)
  {
    for (const auto & pw : matches->pairWise_matches_)
      total_matches += pw.second.size();
  }

  // Per-view pair count
  Hash_Map<IndexT, size_t> pv;
  for (const auto & v : sfm_data.GetViews()) pv[v.first] = 0;
  if (matches)
  {
    for (const auto & p : matches->pairWise_matches_)
    {
      ++pv[p.first.first];
      ++pv[p.first.second];
    }
  }
  size_t isolated = 0;
  std::vector<size_t> pv_counts; pv_counts.reserve(pv.size());
  for (const auto & kv : pv) { pv_counts.push_back(kv.second); if (kv.second <= 1) ++isolated; }
  std::sort(pv_counts.begin(), pv_counts.end());

  OPENMVG_LOG_INFO
    << "[PIPE-DIAG/INPUT]"
    << " views=" << n_views
    << " intrinsics=" << n_intr
    << " seed_poses=" << n_seed
    << " | features: views_with=" << n_feat_v
    << " views_without=" << views_no_feats
    << " total_keypoints=" << total_features
    << " | matches: pairs=" << n_pairs
    << " total=" << total_matches
    << " avg_pairs/view=" << (n_views ? (2.0 * n_pairs) / n_views : 0.0)
    << " | per-view pair count: min=" << (pv_counts.empty() ? 0 : pv_counts.front())
    << " median=" << (pv_counts.empty() ? 0 : pv_counts[pv_counts.size()/2])
    << " max=" << (pv_counts.empty() ? 0 : pv_counts.back())
    << " isolated(<=1)=" << isolated;

  if (n_views == 0)
    OPENMVG_LOG_ERROR << "[PIPE-DIAG/INPUT] FATAL: no views in SfM_Data.";
  if (n_intr == 0)
    OPENMVG_LOG_WARNING << "[PIPE-DIAG/INPUT] No intrinsics defined; resection will use DLT_6POINTS.";
  if (views_no_feats > 0)
    OPENMVG_LOG_WARNING << "[PIPE-DIAG/INPUT] " << views_no_feats
      << " view(s) have NO features. Those images cannot be calibrated."
      << " Re-run ComputeFeatures for the missing views.";
  if (n_pairs == 0)
    OPENMVG_LOG_ERROR << "[PIPE-DIAG/INPUT] FATAL: no pairwise matches loaded.";
  if (n_views > 0 && (2.0 * n_pairs) / n_views < 5.0)
    OPENMVG_LOG_WARNING << "[PIPE-DIAG/INPUT] Sparse match graph: only "
      << (2.0 * n_pairs) / n_views << " pairs/view (typical drone: 15-50)."
      << " Re-run ComputeMatches with more neighbors / EXHAUSTIVE pairing.";
  if (isolated > 0)
    OPENMVG_LOG_WARNING << "[PIPE-DIAG/INPUT] " << isolated
      << " view(s) have <=1 match pair; they almost certainly cannot be calibrated.";
}

// --- Checkpoint 2: per-round summary ----------------------------------------
static void DiagRound(
  IndexT round,
  float track_inlier_ratio,
  IndexT poses_before,
  IndexT poses_after,
  IndexT landmarks_after,
  IndexT remaining_views)
{
  OPENMVG_LOG_INFO
    << "[PIPE-DIAG/ROUND " << round << "]"
    << " ratio_gate=" << track_inlier_ratio
    << " poses: " << poses_before << " -> " << poses_after
    << " (+" << (poses_after - poses_before) << ")"
    << " | tracks=" << landmarks_after
    << " | remaining_views=" << remaining_views;
}

// --- Checkpoint 3: per-view post-mortem -------------------------------------
static void DiagOutputs(
  const SfM_Data & sfm_data,
  const Hash_Map<IndexT, ViewDiagRecord> & records)
{
  const size_t n_views = sfm_data.GetViews().size();
  const size_t n_poses = sfm_data.GetPoses().size();

  // Tally reasons across un-calibrated views
  std::map<ResectionFailReason, size_t> reason_counts;
  std::vector<IndexT> missing;
  missing.reserve(n_views > n_poses ? n_views - n_poses : 0);
  for (const auto & v : sfm_data.GetViews())
  {
    const View * view = v.second.get();
    if (sfm_data.GetPoses().count(view->id_pose) == 0)
    {
      missing.push_back(v.first);
      auto it = records.find(v.first);
      const ResectionFailReason r = (it != records.end())
        ? it->second.last_reason
        : ResectionFailReason::NoVisibleTracks;
      ++reason_counts[r];
    }
  }

  OPENMVG_LOG_INFO
    << "[PIPE-DIAG/OUTPUT]"
    << " input_views=" << n_views
    << " calibrated=" << n_poses
    << " missing=" << missing.size()
    << " (" << (n_views ? (100.0 * missing.size() / n_views) : 0.0) << "%)";

  if (!reason_counts.empty())
  {
    std::ostringstream os;
    os << "[PIPE-DIAG/OUTPUT] failure reasons:";
    for (const auto & rc : reason_counts)
      os << " " << to_string(rc.first) << "=" << rc.second;
    OPENMVG_LOG_INFO << os.str();
  }

  // Per-missing-view detail (cap output to avoid huge logs)
  const size_t max_detail = 60;
  if (!missing.empty())
  {
    std::ostringstream os;
    os << "[PIPE-DIAG/OUTPUT] missing view detail (showing up to "
       << max_detail << "):\n";
    size_t printed = 0;
    for (const IndexT vid : missing)
    {
      if (printed >= max_detail) { os << "    ... (" << (missing.size() - printed) << " more)\n"; break; }
      const auto & view = sfm_data.GetViews().at(vid);
      auto it = records.find(vid);
      os << "    view=" << vid
         << " img=" << view->s_Img_path;
      if (it != records.end())
      {
        const auto & r = it->second;
        os << " reason=" << to_string(r.last_reason)
           << " attempts=" << r.attempts
           << " best_2d3d=" << r.best_2d3d
           << " best_track_ratio=" << r.best_track_ratio
           << " last_inliers=" << r.last_inliers
           << " last_inlier_ratio=" << r.last_ratio;
      }
      else
      {
        os << " reason=NEVER_REACHED_RESECTION";
      }
      os << "\n";
      ++printed;
    }
    OPENMVG_LOG_INFO << os.str();
  }

  // Hard-failure heuristics
  if (n_poses == 0)
    OPENMVG_LOG_ERROR << "[PIPE-DIAG/OUTPUT] FATAL: zero cameras calibrated.";
  else if (missing.size() > n_views / 5)
    OPENMVG_LOG_WARNING << "[PIPE-DIAG/OUTPUT] "
      << missing.size() << "/" << n_views << " views uncalibrated."
      << " Inspect failure reasons above. Common fixes:"
      << " (a) NO_RECONSTRUCTED_TRACKS / LOW_TRACK_RATIO -> sparser match graph;"
      << " add more match neighbors or EXHAUSTIVE pairs."
      << " (b) RESECTION_FAILED -> intrinsic guess wrong; check camera_model."
      << " (c) LOW_INLIER_RATIO -> noisy matches; tighten geometric filter.";
}
#endif // OPENMVG_SFM_PIPELINE_DIAG

SequentialSfMReconstructionEngine2::SequentialSfMReconstructionEngine2(
  SfMSceneInitializer * scene_initializer,
  const SfM_Data & sfm_data,
  const std::string & soutDirectory,
  const std::string & sloggingFile)
  : ReconstructionEngine(sfm_data, soutDirectory),
    scene_initializer_(scene_initializer),
    sLogging_file_(sloggingFile),
    cam_type_(EINTRINSIC(PINHOLE_CAMERA_RADIAL3)),
    track_min_length_(OPENMVG_SFM2_TRACKS_MIN_LENGTH)
{
  if (!sLogging_file_.empty())
  {
    // setup HTML logger
    html_doc_stream_ = std::make_shared<htmlDocument::htmlDocumentStream>("SequentialReconstructionEngine SFM report.");
    html_doc_stream_->pushInfo(
      htmlDocument::htmlMarkup("h1", std::string("SequentialSfMReconstructionEngine")));
    html_doc_stream_->pushInfo("<hr>");

    html_doc_stream_->pushInfo( "Dataset info:");
    html_doc_stream_->pushInfo( "Views count: " +
      htmlDocument::toString( sfm_data.GetViews().size()) + "<br>");
    html_doc_stream_->pushInfo( "Poses count: " +
      htmlDocument::toString( sfm_data.GetPoses().size()) + "<br>");
  }
}

SequentialSfMReconstructionEngine2::~SequentialSfMReconstructionEngine2()
{
  if (!sLogging_file_.empty())
  {
    // Save the reconstruction Log
    std::ofstream htmlFileStream(sLogging_file_.c_str());
    htmlFileStream << html_doc_stream_->getDoc();
  }
}

void SequentialSfMReconstructionEngine2::SetFeaturesProvider(Features_Provider * provider)
{
  features_provider_ = provider;
}

void SequentialSfMReconstructionEngine2::SetMatchesProvider(Matches_Provider * provider)
{
  matches_provider_ = provider;
}

bool SequentialSfMReconstructionEngine2::Process() {

  //-------------------
  //-- Incremental reconstruction
  //-------------------
  //- 1. Init the reconstruction with a seed
  //- 2. While we can localize some cameras in the reconstruction
  //     a. Triangulation
  //     b. Bundle Adjustment and cleaning
  //- 3. Final bundle Adjustment
  //-------------------

  // -------------------------------------------------------------------------
  // Analytic Radial3 Jacobian self-test was validated (max err ~9e-12 vs
  // AutoDiff). Disabled in production runs; re-enable manually if the
  // analytic class is modified.
  // -------------------------------------------------------------------------
#if 0
  {
    const double err = RunSelfTest_AnalyticReprojectionCost_Radial3(1000);
    OPENMVG_LOG_INFO
        << "[Analytic-Radial3 self-test] 1000 trials, max err vs AutoDiff = "
        << err
        << "  (pass if < 1e-7)";
  }
#endif

#if OPENMVG_SFM_PIPELINE_DIAG
  // Per-view diagnostic accumulator. Built up by AddingMissingView() and
  // consumed at the end by DiagOutputs(). Kept on the stack of Process() so
  // it doesn't outlive the run.
  Hash_Map<IndexT, ViewDiagRecord> view_diag_records;
  // Checkpoint 1: validate inputs before doing anything.
  DiagInputs(sfm_data_, features_provider_, matches_provider_);
#endif

  //--
  //- 1. Init the reconstruction with a Seed
  //--
  {
    const bool primary_init_ok = scene_initializer_ && scene_initializer_->Process();
    if (primary_init_ok)
    {
      OPENMVG_LOG_INFO << "Initialization status : Success";
      sfm_data_.poses = scene_initializer_->Get_sfm_data().GetPoses();
    }
    else
    {
      // Fallback: the configured initializer produced no seed. This happens
      // with STELLAR on HIGH-feature / low-parallax aerial graphs, where too
      // few of the putative pod's edges yield a relative pose to form a
      // solvable star (only a scattered handful succeed -> largest star is a
      // single edge -> Stellar_Solver::Solve returns false). Failing here
      // would abort the engine and drop the surrounding pipeline to the GLOBAL
      // engine. Instead retry with MaxPair, which only needs one
      // well-conditioned pair and is robust on this data (lazy widest-baseline
      // seed + adaptive GPS prior). Its constructor clears any partial poses
      // the failed initializer left, and it writes directly into sfm_data_.
      OPENMVG_LOG_WARNING
        << "Primary scene initializer produced no seed; falling back to MaxPair.";
      SfMSceneInitializerMaxPair maxpair_fallback(
        sfm_data_, features_provider_, matches_provider_);
      if (!maxpair_fallback.Process())
      {
        OPENMVG_LOG_ERROR
          << "Initialization status: Failed (primary initializer and MaxPair"
          << " fallback both produced no seed).";
        return false;
      }
      OPENMVG_LOG_INFO << "Initialization status : Success (MaxPair fallback)";
      // MaxPair wrote the seed poses directly into sfm_data_ (shared by ref).
    }

#if OPENMVG_SFM_PIPELINE_DIAG
    OPENMVG_LOG_INFO
      << "[PIPE-DIAG/SEED] seed_poses=" << sfm_data_.GetPoses().size()
      << " of " << sfm_data_.GetViews().size() << " views";
    if (sfm_data_.GetPoses().empty())
      OPENMVG_LOG_ERROR << "[PIPE-DIAG/SEED] FATAL: scene initializer produced 0 poses.";
#endif

    if (!InitTracksAndLandmarks())
      return false;

#if OPENMVG_SFM_PIPELINE_DIAG
    {
      // Track-coverage diagnostic: how many input views are touched by tracks?
      std::set<uint32_t> covered;
      tracks::TracksUtilsMap::ImageIdInTracks(map_tracks_, covered);
      const size_t n_views = sfm_data_.GetViews().size();
      OPENMVG_LOG_INFO
        << "[PIPE-DIAG/TRACKS] tracks=" << map_tracks_.size()
        << " | views_in_tracks=" << covered.size()
        << " / " << n_views
        << " (" << (n_views ? (100.0 * covered.size() / n_views) : 0.0) << "%)";
      if (covered.size() < n_views)
      {
        size_t shown = 0;
        std::ostringstream os;
        os << "[PIPE-DIAG/TRACKS] views NOT touched by any track:";
        for (const auto & v : sfm_data_.GetViews())
        {
          if (covered.count(v.first) == 0)
          {
            if (shown++ < 30) os << " " << v.first;
            // Pre-record the failure so the post-mortem is complete.
            view_diag_records[v.first].last_reason =
              ResectionFailReason::NoVisibleTracks;
          }
        }
        if (shown > 30) os << " ... (+" << (shown - 30) << " more)";
        OPENMVG_LOG_WARNING << os.str();
      }
    }
#endif

    if (!sfm_data_.GetPoses().empty())
    {
      bool bTriangulation = Triangulation();

      Save(sfm_data_, stlplus::create_filespec(sOut_directory_, "Initialization", ".ply"), ESfM_Data(ALL));
      // Fused angle + pixel filter: one structure traversal, one cache build.
      // Semantically equivalent to the legacy upstream pair at this site:
      //   RemoveOutliers_AngleError(sfm_data_, Square(2.0));        // 4 deg
      //   RemoveOutliers_PixelResidualError(sfm_data_, Square(4.0));// 16 px
      // These deliberately-loose thresholds are correct *here* because the
      // structure was just triangulated and hasn't been BA'd yet -- residuals
      // are coarse, and tighter thresholds would over-prune before BA gets a
      // chance to refine. The in-loop post-BA filter uses 2.0 / 4.0.
      // NOTE: the angle floor is OPENMVG_SFM2_SEED_MIN_ANGLE_DEG (default
      // 2.0 deg), NOT the upstream 4.0 (Square(2.0)) -- 4 deg wipes the entire
      // seed on low-parallax aerial pairs (see the macro's comment).
#if OPENMVG_SFM2_RELAXED_ANGLE_THRESHOLDS // Relaxed angle threshold for foliage/narrow-baseline scenes
      RemoveOutliers_PixelAndAngleError(sfm_data_, Square(4.0), 3.0);
#else
      RemoveOutliers_PixelAndAngleError(sfm_data_, Square(4.0), OPENMVG_SFM2_SEED_MIN_ANGLE_DEG);
#endif

      // Auto-fallback for the selective short-track cut, evaluated on the
      // POST-FILTER structure. On thin-overlap scenes the seed pair's covisible
      // tracks are mostly length-2, which the cut (track_min_length_ > 2) drops;
      // the few length-3+ tracks that survive triangulation are then culled by
      // the loose outlier filter above, leaving 0 usable landmarks so the
      // resection loop can never bootstrap (ends at 2 cameras / 0 tracks).
      // Detect the empty structure here (checked AFTER the filter, since
      // triangulation itself may yield a handful the filter then removes) and
      // retry ONCE with the cut disabled (min_len=2 keeps every length-2 track),
      // redoing triangulation + the same filter. Dense scenes keep thousands of
      // landmarks and never enter this branch, so the fast cut stays the
      // default; only starved scenes pay the one-time re-init cost.
      if (sfm_data_.GetLandmarks().empty() && track_min_length_ > 2)
      {
        OPENMVG_LOG_WARNING
          << "[TRACKS-CUT] seed produced 0 usable landmarks with min_len="
          << track_min_length_ << " (short-track cut starved the seed)."
          << " Falling back to min_len=2 (keep all length-2 tracks) and"
          << " rebuilding tracks.";
        track_min_length_ = 2;
        if (!InitTracksAndLandmarks())
          return false;
        bTriangulation = Triangulation();
#if OPENMVG_SFM2_RELAXED_ANGLE_THRESHOLDS // Relaxed angle threshold for foliage/narrow-baseline scenes
        RemoveOutliers_PixelAndAngleError(sfm_data_, Square(4.0), 3.0);
#else
        RemoveOutliers_PixelAndAngleError(sfm_data_, Square(4.0), OPENMVG_SFM2_SEED_MIN_ANGLE_DEG);
#endif
      }

      //-- Display some statistics
      OPENMVG_LOG_INFO
        << "\n\n-------------------------------" << "\n"
        << "-- Starting Structure from Motion (statistics) with:\n"
        << "-- #Camera calibrated: " << sfm_data_.GetPoses().size()
        << " from " << sfm_data_.GetViews().size() << " input images.\n"
        << "-- #Tracks, #3D points: " << sfm_data_.GetLandmarks().size() << "\n"
        << "-------------------------------";

      if (!bTriangulation)
      {
        // We are not able to triangulate enough 3D points for the given poses.
        return false;
      }
    }
  }

  if (sfm_data_.GetPoses().empty())
  {
    return false;
  }

  //--
  //- 2. While we can localize some cameras in the reconstruction
  //     a. Triangulate the landmarks
  //     b. Perform Bundle Adjustment and cleaning
  //--
  IndexT resection_round = 0;

  // Incrementally estimate the pose of the cameras based on a confidence score.
  // The confidence score is based on the track_inlier_ratio.
  // First the camera with the most of 2D-3D overlap are added then we add
  // ones with lower confidence.
  const std::array<float, 2> track_inlier_ratios = {0.2, 0.0};

  // BA-skip threshold (intermediate BAs only).
  //   Minimum number of poses added since the last intermediate BA before
  //   we run BA again. Each BA pays a per-call setup overhead (~0.5-2.3s
  //   in the slow datasets) plus the minimizer cost; rounds that add 1-2
  //   views move geometry below the next-BA noise floor anyway. Skipping
  //   them lets the next round absorb the work cheaply, and the final
  //   STRICT BA still cleans up unconditionally. We always run BA on the
  //   final round of each track-inlier-ratio band so no skipped work
  //   leaks into the next phase or into the final BA.
  // Bumped from 2 -> 4 after [BA-PERF] showed each intermediate BA on the
  // slow dataset costs ~9-10s wall time (~4M residuals, 4-6 iters). The
  // end-of-band flush in the loop below still runs BA unconditionally on
  // the final pose of each track-inlier-ratio band, so accuracy is
  // protected; we only skip the truly small (1-3 new pose) calls.
  const IndexT kBAMinNewPoses = 4;
  IndexT poses_since_last_ba = 0;

  for (auto track_inlier_ratio = track_inlier_ratios.cbegin();
    track_inlier_ratio < track_inlier_ratios.cend(); ++track_inlier_ratio)
  {
    IndexT pose_before = sfm_data_.GetPoses().size();
    while (AddingMissingView(*track_inlier_ratio
#if OPENMVG_SFM_PIPELINE_DIAG
                              , &view_diag_records
#endif
                              ))
    {
      // Create new 3D points
      Triangulation();
      // Adjust the scene. Three presets are available:
      //   FAST     - aggressive speedup; safe only when no motion priors.
      //   BALANCED - moderate speedup; safe for motion-prior workflows.
      //   STRICT   - reference defaults; used for the final BA below.
      //
      // The fast preset's loose tolerances let LM exit before the motion
      // prior penalty fully balances reprojection, which on prior-bearing
      // scenes initialises the next resection from a slightly-wrong global
      // pose. The error compounds and the final strict BA can settle into
      // a parallel "ghost-layer" basin that no eject filter cleans up.
      // BALANCED gives prior workflows enough LM headroom to converge
      // while still skipping the last few strict-convergence iterations.
      const BAPreset intermediate_preset =
          !b_use_fast_intermediate_ba_ ? BAPreset::STRICT
        :  b_use_motion_prior_         ? BAPreset::BALANCED
        :                                BAPreset::FAST;

      // BA-skip gate. Track newly-added poses since the last successful
      // BA; run BA only when the accumulated delta crosses the threshold.
      // The Triangulation() above must always run (it's how new
      // landmarks are introduced for the next resection round); only the
      // BA itself is gated.
      const IndexT poses_now = sfm_data_.GetPoses().size();
      poses_since_last_ba += (poses_now > pose_before)
                           ? (poses_now - pose_before)
                           : IndexT(0);
      if (poses_since_last_ba >= kBAMinNewPoses)
      {
        BundleAdjustment(intermediate_preset);
        poses_since_last_ba = 0;
        // Remove unstable triangulations and camera poses (fused single pass).
        // Tied to the BA call: the residual statistics needed for outlier
        // pruning are only valid against the just-converged geometry.
#if OPENMVG_SFM2_RELAXED_ANGLE_THRESHOLDS // Relaxed angle threshold for foliage/narrow-baseline scenes
        RemoveOutliers_PixelAndAngleError(sfm_data_, 4.0, 1.5);
#else
        RemoveOutliers_PixelAndAngleError(sfm_data_, 4.0, OPENMVG_SFM2_TRACK_MIN_ANGLE_DEG);
#endif
        eraseUnstablePosesAndObservations(sfm_data_);
      }

      // Per-resection-round PLY snapshot. Diagnostic-only output;
      // disabled by default because each save serialises the full
      // sfm_data scene to disk (5-15s of I/O on multi-round runs)
      // and does not feed any subsequent SfM2 step. Re-enable by
      // flipping the flag below or wiring it to a runtime knob if
      // you need per-round snapshots for debugging.
#if defined(OPENMVG_SFM2_SAVE_PER_RESECTION_PLY)
      std::ostringstream os;
      os << std::setw(8) << std::setfill('0') << resection_round << "_Resection";
      Save(sfm_data_, stlplus::create_filespec(sOut_directory_, os.str(), ".ply"), ESfM_Data(ALL));
#endif

      const IndexT pose_after = sfm_data_.GetPoses().size();
      const IndexT remaining =
        static_cast<IndexT>(sfm_data_.GetViews().size()) - pose_after;
#if OPENMVG_SFM_PIPELINE_DIAG
      DiagRound(resection_round, *track_inlier_ratio, pose_before, pose_after,
                static_cast<IndexT>(sfm_data_.GetLandmarks().size()), remaining);
#endif
      ++resection_round;

      // Stop if no cameras have been added
      // Note: some cameras could have been removed due to instable camera positions.
      if (pose_before >= pose_after)
        break;
      pose_before = sfm_data_.GetPoses().size();
      // Since we have augmented our set of poses we can reset our track inlier ratio iterator
      track_inlier_ratio = track_inlier_ratios.cbegin();
    }
    // End-of-band flush: if we exit the while loop with pending un-baked
    // poses (BA-skip threshold not yet hit), run BA once now so the next
    // ratio band's resection scan reads from converged geometry instead
    // of the stale post-Triangulation parameters. The final STRICT BA
    // still runs unconditionally below; this flush only matters when we
    // still have a looser-threshold band ahead of us.
    if (poses_since_last_ba > 0)
    {
      const BAPreset intermediate_preset =
          !b_use_fast_intermediate_ba_ ? BAPreset::STRICT
        :  b_use_motion_prior_         ? BAPreset::BALANCED
        :                                BAPreset::FAST;
      BundleAdjustment(intermediate_preset);
#if OPENMVG_SFM2_RELAXED_ANGLE_THRESHOLDS // Relaxed angle threshold for foliage/narrow-baseline scenes
      RemoveOutliers_PixelAndAngleError(sfm_data_, 4.0, 1.5);
#else
      RemoveOutliers_PixelAndAngleError(sfm_data_, 4.0, OPENMVG_SFM2_TRACK_MIN_ANGLE_DEG);
#endif
      eraseUnstablePosesAndObservations(sfm_data_);
      poses_since_last_ba = 0;
    }
  }

  //--
  //- 3. Final bundle Adjustment (robust loop):
  //     BA -> eject bad poses (median-residual rule) -> if any ejected,
  //     filter outliers + re-BA. Otherwise we're done.
  //     Cost shape:
  //       - Clean runs: 1 BA + 1 cheap residual sweep (no sqrt) + early
  //         break -> ~same as the legacy single-BA path.
  //       - Bad runs: extra BAs only when poses actually need ejecting,
  //         and only those iterations pay for the O(K^2) angle filter.
  //     The filter is gated behind `ejected > 0` because its angle pass
  //     is O(track_length^2) per landmark over the entire post-BA scene,
  //     so running it unconditionally would force another full BA on
  //     every successful run for marginal benefit.
  //--
  {
    // Healthy reconstructions converge in iter 0 (no poses ejected ->
    // immediate break). The previous cap of 4 was a worst-case safety
    // net for pathological runs; in practice no observed dataset has
    // benefited from iters 2-3 on top of the rest of this pipeline.
    // Bit-identical to the cap=4 path on every run that breaks early.
    const int kMaxRobustIters = 2;
    BundleAdjustment(); // initial BA, same as legacy single-call path

    for (int iter = 0; iter < kMaxRobustIters; ++iter)
    {
      const IndexT prev_poses  = static_cast<IndexT>(sfm_data_.GetPoses().size());
      const IndexT prev_tracks = static_cast<IndexT>(sfm_data_.GetLandmarks().size());

      // EjectPosesByMedianResidual already calls
      // eraseObservationsWithMissingPoses internally for the poses it kills,
      // so we don't need a second eraseUnstablePosesAndObservations pass.
      const IndexT ejected_repro = EjectPosesByMedianResidual(sfm_data_,
                                                              /*k_factor=*/3.0,
                                                              /*abs_floor=*/2.0);
      // Pose-prior eject (EjectPosesByPriorResidual) is intentionally NOT
      // called here. With strict intermediate BA on prior workflows (see
      // resection-loop comment above), the ghost-layer/Z-drift mode that
      // motivated it does not occur, and any heuristic eject on a clean
      // scene only adds risk of culling good cameras. The function is kept
      // available in sfm_data_filters.* for diagnostic use.
      const IndexT ejected = ejected_repro;

      // Common clean-run path: nothing ejected -> done. No extra BA, no
      // expensive outlier filter.
      if (ejected == 0)
      {
        OPENMVG_LOG_INFO
          << "[ROBUST-BA iter " << iter << "]"
          << " ejected_poses=0 (converged) poses=" << prev_poses
          << " tracks=" << prev_tracks;
        break;
      }

      // A pose was ejected. Some observations near the ejected pose's
      // co-visibility may now be high-residual outliers in the surviving
      // scene; clean them before the corrective BA so the optimum isn't
      // pulled by stale features.
#if OPENMVG_SFM2_RELAXED_ANGLE_THRESHOLDS // Relaxed angle threshold for foliage/narrow-baseline scenes
      const IndexT removed = RemoveOutliers_PixelAndAngleError(sfm_data_, 4.0, 1.5);
#else
      const IndexT removed = RemoveOutliers_PixelAndAngleError(sfm_data_, 4.0, 2.0);
#endif

      const IndexT cur_poses  = static_cast<IndexT>(sfm_data_.GetPoses().size());
      const IndexT cur_tracks = static_cast<IndexT>(sfm_data_.GetLandmarks().size());

      OPENMVG_LOG_INFO
        << "[ROBUST-BA iter " << iter << "]"
        << " ejected_poses=" << ejected
        << " removed=" << removed
        << " poses=" << prev_poses << "->" << cur_poses
        << " tracks=" << prev_tracks << "->" << cur_tracks;

      // Something changed -> re-BA so the optimum reflects the cleaned scene.
      BundleAdjustment();
    }
  }

  //-- Reconstruction done.
  //-- Display some statistics
  OPENMVG_LOG_INFO
    << "\n-------------------------------" << "\n"
    << "-- Structure from Motion (statistics):\n"
    << "-- #Camera calibrated: " << sfm_data_.GetPoses().size()
    << " from " << sfm_data_.GetViews().size() << " input images.\n"
    << "-- #Tracks, #3D points: " << sfm_data_.GetLandmarks().size() << "\n"
    << "-- #Poses loop used: " << resection_round << "\n"
    << "-------------------------------";

#if OPENMVG_SFM_PIPELINE_DIAG
  // Checkpoint 3: post-mortem on every uncalibrated view.
  DiagOutputs(sfm_data_, view_diag_records);
#endif

  // [DIAG] Match-graph connectivity analysis. A healthy drone dataset has
  // hundreds of pairs per view and track median-length >= 4. If pairs/view
  // is below ~5 or median track length is 2, the upstream matching stage
  // (PairGenerator / ComputeMatches / GeometricFilter) is the bottleneck
  // - no amount of SfM tuning will recover the missing images.
  {
    const size_t n_views = sfm_data_.GetViews().size();
    const size_t n_pairs = matches_provider_->pairWise_matches_.size();
    const double pairs_per_view = n_views > 0 ? (2.0 * n_pairs) / n_views : 0.0;

    // Per-view pair count
    Hash_Map<IndexT, size_t> pairs_per_view_map;
    for (const auto & v : sfm_data_.GetViews())
      pairs_per_view_map[v.first] = 0;
    for (const auto & pw : matches_provider_->pairWise_matches_)
    {
      ++pairs_per_view_map[pw.first.first];
      ++pairs_per_view_map[pw.first.second];
    }
    std::vector<size_t> pv_counts;
    pv_counts.reserve(pairs_per_view_map.size());
    size_t isolated_views = 0; // views with 0 or 1 pair
    for (const auto & kv : pairs_per_view_map)
    {
      pv_counts.push_back(kv.second);
      if (kv.second <= 1) ++isolated_views;
    }
    std::sort(pv_counts.begin(), pv_counts.end());

    OPENMVG_LOG_INFO
      << "[MATCH-DIAG] n_views=" << n_views
      << " n_pairs=" << n_pairs
      << " avg_pairs/view=" << pairs_per_view
      << " | pair-count per view: min=" << (pv_counts.empty() ? 0 : pv_counts.front())
      << " median=" << (pv_counts.empty() ? 0 : pv_counts[pv_counts.size()/2])
      << " max=" << (pv_counts.empty() ? 0 : pv_counts.back())
      << " | isolated_views(<=1 pair)=" << isolated_views;

    if (pairs_per_view < 5.0)
    {
      OPENMVG_LOG_WARNING
        << "[MATCH-DIAG] CRITICAL: avg pairs/view=" << pairs_per_view
        << " is far below the ~15-50 typical for drone datasets."
        << " The match graph is too sparse for incremental SfM to"
        << " traverse the whole scene. Rerun matching with EXHAUSTIVE"
        << " pair generation or increase neighbor_count to >= 20.";
    }
  }

  return true;
}

// Coverage-aware short-track cut, measured by per-view IMAGE-SPACE occupancy
// with a FRAME-BORDER refinement.
//
// Covisibility metrics proved useless on dense drone match graphs: every
// length-2 track links a view pair that already shares many strong tracks, so
// any covisibility floor classifies ALL short tracks as redundant (the global
// cut, which blows up the scene edge). The true discriminator is spatial: the
// load-bearing short tracks sit in IMAGE REGIONS (frame edges/corners) that
// strong tracks do not cover. A length-2 observation in an otherwise-empty
// patch of its view is the only thing constraining that part of the frustum.
//
// Refinement: a SPARSE cell in the image *interior* is surrounded by strong
// structure and is not what diverges -- the right-edge blow-up happens at the
// image *border* (peripheral field of view -> scene periphery), where there is
// nothing beyond to constrain the frustum. So a short observation protects its
// track only when it is BOTH in a sparse cell AND near the frame border. Short
// tracks whose only sparse observations are interior are dropped too, which
// recovers most of the remaining speedup safely.
//
// Method: bin every strong (length >= min_len) observation into a fixed
// `cell_px`-pixel grid per view. A short track is KEPT iff at least one of its
// observations is (a) in a cell holding < `min_cell_support` strong obs AND
// (b) within `border_pct`% of the image min-dimension from a frame edge.
// Otherwise it is dropped. `min_cell_support == 0` reproduces the global cut;
// `border_pct >= 50` disables the border test (any sparse cell protects).
static void SelectiveShortTrackCut(
    tracks::STLMAPTracks & map_tracks,
    Features_Provider * features,
    const SfM_Data & sfm_data,
    const uint32_t min_len,
    const uint32_t min_cell_support,
    const uint32_t cell_px,
    const uint32_t border_pct)
{
  if (min_len <= 2)
    return; // nothing to cut: every length-2+ track is kept
  if (min_cell_support == 0 || features == nullptr || cell_px == 0)
  {
    // Degenerate: unconditional global length cut.
    size_t dropped = 0;
    for (auto it = map_tracks.begin(); it != map_tracks.end(); )
    {
      if (it->second.size() < min_len) { it = map_tracks.erase(it); ++dropped; }
      else                             { ++it; }
    }
    OPENMVG_LOG_INFO
      << "[TRACKS-CUT] global: min_len=" << min_len
      << " dropped=" << dropped << " kept_short=0";
    return;
  }

  // Pack (view, cell_x, cell_y) into a 64-bit key: view:32 | cx:16 | cy:16.
  const float inv_cell = 1.0f / static_cast<float>(cell_px);
  const auto cell_key =
    [inv_cell](uint32_t view, float x, float y) -> uint64_t {
      const uint32_t cx = static_cast<uint32_t>(x * inv_cell) & 0xFFFFu;
      const uint32_t cy = static_cast<uint32_t>(y * inv_cell) & 0xFFFFu;
      return (static_cast<uint64_t>(view) << 32) |
             (static_cast<uint64_t>(cx) << 16) |
              static_cast<uint64_t>(cy);
    };

  const auto obs_coords =
    [features](uint32_t view, uint32_t feat, float & x, float & y) -> bool {
      const auto vit = features->feats_per_view.find(view);
      if (vit == features->feats_per_view.end() ||
          feat >= vit->second.size())
        return false;
      const auto c = vit->second[feat].coords();
      x = static_cast<float>(c.x());
      y = static_cast<float>(c.y());
      return true;
    };

  // Per-view frame-border margin in pixels (border_pct% of the image
  // min-dimension, at least one cell). Returns false if dimensions unknown.
  const auto & views = sfm_data.GetViews();
  const auto near_border =
    [&views, border_pct, cell_px](uint32_t view, float x, float y) -> bool {
      const auto vit = views.find(view);
      if (vit == views.end() || !vit->second) return true; // unknown -> protect
      const float W = static_cast<float>(vit->second->ui_width);
      const float H = static_cast<float>(vit->second->ui_height);
      if (W <= 0.f || H <= 0.f) return true;
      if (border_pct >= 50) return true; // border test disabled
      const float margin =
        std::max(static_cast<float>(cell_px),
                 std::min(W, H) * (static_cast<float>(border_pct) / 100.f));
      return (x < margin) || (x > W - margin) ||
             (y < margin) || (y > H - margin);
    };

  // Pass 1: per-view, per-cell count of strong observations.
  std::unordered_map<uint64_t, uint32_t> cell_strong;
  for (const auto & trk : map_tracks)
  {
    if (trk.second.size() < min_len) continue;
    for (const auto & obs : trk.second)
    {
      float x, y;
      if (obs_coords(obs.first, obs.second, x, y))
        ++cell_strong[cell_key(obs.first, x, y)];
    }
  }

  // Pass 2: keep a short track only if at least one observation is in a sparse
  // cell AND near the frame border; otherwise drop it.
  size_t dropped = 0, kept_short = 0, kept_no_coords = 0;
  for (auto it = map_tracks.begin(); it != map_tracks.end(); )
  {
    if (it->second.size() >= min_len) { ++it; continue; }
    bool protective = false; // found a sparse + border observation
    bool had_coords = true;
    for (const auto & obs : it->second)
    {
      float x, y;
      if (!obs_coords(obs.first, obs.second, x, y)) { had_coords = false; break; }
      const auto s = cell_strong.find(cell_key(obs.first, x, y));
      const uint32_t c = (s == cell_strong.end()) ? 0u : s->second;
      const bool sparse = (c < min_cell_support);
      if (sparse && near_border(obs.first, x, y)) { protective = true; break; }
    }
    if (!had_coords)    { ++it; ++kept_no_coords; } // keep if coords unknown
    else if (protective){ ++it; ++kept_short; }
    else                { it = map_tracks.erase(it); ++dropped; }
  }
  OPENMVG_LOG_INFO
    << "[TRACKS-CUT] selective(coverage+border): min_len=" << min_len
    << " min_cell_support=" << min_cell_support
    << " cell_px=" << cell_px
    << " border_pct=" << border_pct
    << " dropped=" << dropped
    << " kept_short=" << kept_short
    << " kept_no_coords=" << kept_no_coords;
}

bool SequentialSfMReconstructionEngine2::InitTracksAndLandmarks()
{
  // Compute tracks from matches
  tracks::TracksBuilder tracksBuilder;
  {
    tracksBuilder.Build(matches_provider_->pairWise_matches_);
    // Keep every valid length-2+ track at the builder stage (this still removes
    // id-collision tracks). The short-track cut is applied SELECTIVELY below on
    // map_tracks_ so we can protect low-overlap / peripheral views; see the
    // top-of-file knobs OPENMVG_SFM2_TRACKS_MIN_LENGTH and
    // OPENMVG_SFM2_TRACKS_MIN_STRONG_SUPPORT.
    tracksBuilder.Filter(2);
    tracksBuilder.ExportToSTL(map_tracks_);
    SelectiveShortTrackCut(map_tracks_,
                           features_provider_,
                           sfm_data_,
                           track_min_length_,
                           OPENMVG_SFM2_TRACKS_MIN_CELL_SUPPORT,
                           OPENMVG_SFM2_TRACKS_COVERAGE_CELL_PX,
                           OPENMVG_SFM2_TRACKS_COVERAGE_BORDER_PCT);

    OPENMVG_LOG_INFO << "\n" << "Track stats";
    {
      std::ostringstream osTrack;
      //-- Display stats :
      //    - number of images
      //    - number of tracks
      std::set<uint32_t> set_imagesId;
      tracks::TracksUtilsMap::ImageIdInTracks(map_tracks_, set_imagesId);
      osTrack
        << "------------------" << "\n"
        << "-- Tracks Stats --" << "\n"
        << " Tracks number: " << tracksBuilder.NbTracks() << "\n"
        << " Images Id: " << "\n";
      std::copy(set_imagesId.cbegin(),
        set_imagesId.cend(),
        std::ostream_iterator<uint32_t>(osTrack, ", "));
      osTrack << "\n------------------" << "\n";

      std::map<uint32_t, uint32_t> map_Occurrence_TrackLength;
      tracks::TracksUtilsMap::TracksLength(map_tracks_, map_Occurrence_TrackLength);
      osTrack << "TrackLength, Occurrence" << "\n";
      for (const auto & it : map_Occurrence_TrackLength)  {
        osTrack << "\t" << it.first << "\t" << it.second << "\n";
      }
      osTrack << "\n";
      OPENMVG_LOG_INFO << osTrack.str();
    }
  }

  // Init the putative landmarks
  {
    // Reset the doomed-view filter set; populated alongside the landmarks
    // build below from the same per-observation walk we already do.
    views_with_tracks_.clear();
    views_with_tracks_.reserve(sfm_data_.GetViews().size());
    // Reset the per-view track-id cache. Built in track-id order via the
    // outer loop over `map_tracks_` (which is std::map<size_t, ...>, so
    // iteration order IS sorted track-id order). Each view's vector
    // therefore comes out naturally sorted with no explicit std::sort
    // pass needed -- a sorted track id is appended exactly once per
    // (view, track) pair, in increasing track-id order.
    view_track_ids_cache_.clear();
    view_track_ids_cache_.reserve(sfm_data_.GetViews().size());
    // Do NOT reserve() landmarks_. It is a Hash_Map<IndexT, Landmark>
    // (std::unordered_map); reserving changes the bucket count, which changes
    // iteration order -> sfm_data_.structure order (set in Triangulation) ->
    // Ceres residual-block insertion order -> FP summation order in the Schur
    // solve. That makes runs non-bit-identical and, on bimodal / ghost-layer-
    // prone scenes, tips the final BA into a worse basin (right-edge blow-up).
    // The few saved rehashes are not worth a divergent reconstruction.
    landmarks_.clear();

    // For every track add the observations:
    // - views and feature positions that see this landmark
    for ( const auto & iterT : map_tracks_ )
    {
      const IndexT track_id = static_cast<IndexT>(iterT.first);
      Observations obs;
      const size_t track_Length = iterT.second.size();
      for (const auto & track_ids : iterT.second) // {ViewId, FeatureId}
      {
        const auto & view_id = track_ids.first;
        const auto & feat_id = track_ids.second;
        const Vec2 x = features_provider_->feats_per_view[view_id][feat_id].coords().cast<double>();
        obs.insert({view_id, Observation(x, feat_id)});
        // Same per-observation walk; record that this view has at least
        // one track. unordered_set::insert is O(1) amortized and
        // idempotent so duplicate inserts are cheap.
        views_with_tracks_.insert(view_id);
        // Append this track id to the view's sorted-track-id list. Each
        // (track_id, view_id) pair appears exactly once in map_tracks_,
        // so no dedup needed; outer-loop iteration order over
        // map_tracks_ guarantees track_ids are appended in ascending
        // order, so per-view vectors are already sorted.
        view_track_ids_cache_[view_id].push_back(track_id);
      }
      landmarks_[iterT.first].obs = std::move(obs);
    }
  }

  // Initialize the shared track visibility helper
  shared_track_visibility_helper_.reset(new openMVG::tracks::SharedTrackVisibilityHelper(map_tracks_));

  // Reset the delta-cache state. Sized to max(view_id)+1 so that
  // resection_score_cache_[view_id] is a valid direct-indexed slot for
  // every view in sfm_data_; concurrent thread access in the OMP loop
  // touches only distinct slots and is data-race-free.
  IndexT max_view_id = 0;
  for (const auto & view_it : sfm_data_.GetViews())
    max_view_id = std::max(max_view_id, view_it.first);
  resection_score_cache_.assign(max_view_id + 1, ResectionScoreCache{});
  // Reconstructed-track membership bitmaps. `map_tracks_` is a std::map, so
  // rbegin() gives the largest track id in O(1); every landmark id in
  // sfm_data_.structure comes from a track, hence is <= that bound. Sized
  // once here and never reallocated during the resection loop, so the
  // per-call cost is a memset of (max_track_id/8) bytes.
  const IndexT max_track_id =
      map_tracks_.empty() ? 0 : static_cast<IndexT>(map_tracks_.rbegin()->first);
  const size_t track_bitmap_words = static_cast<size_t>(max_track_id) / 64 + 1;
  cur_reconstructed_track_bits_.assign(track_bitmap_words, uint64_t(0));
  prev_reconstructed_track_bits_.assign(track_bitmap_words, uint64_t(0));
  has_prev_reconstructed_tracks_ = false;
  track_bits_scratch_.clear();
  // Reset bitmask scratch (sized identically to resection_score_cache_).
  prev_view_with_no_pose_mask_.assign(max_view_id + 1, 0);
  cur_view_with_no_pose_mask_.assign(max_view_id + 1, 0);
  dirty_view_mask_.assign(max_view_id + 1, 0);
  return map_tracks_.size() > 0;
}

bool SequentialSfMReconstructionEngine2::Triangulation()
{
#if OPENMVG_SFM2_TIME_TRIANGULATION
  // INERT profiling probe (see OPENMVG_SFM2_TIME_TRIANGULATION toggle).
  // Function-local statics accumulate across the sequential resection loop;
  // Triangulation() is never called concurrently, so no synchronisation.
  static unsigned long long s_tri_call_count = 0;
  static double             s_tri_total_ms   = 0.0;
  const auto s_tri_t0 = std::chrono::steady_clock::now();
#endif

  sfm_data_.structure = landmarks_;

  //--
  // Triangulation
  //--
  // Clean the structure:
  //  - keep observations that are linked to valid pose and intrinsic data.

  const double max_reprojection_error = 4.0;
  const IndexT min_required_inliers = 2;
  const IndexT min_sample_index = 2;

  // One-time seed-stage breakdown: when only the 2 seed poses exist, log where
  // the seed structure is lost so a "2 cameras / 0 tracks" stall can be pinned
  // to either (a) no track spans the seed pair -> everything dies in
  // eraseObservationsWithMissingPoses (matching/track-graph problem), or
  // (b) tracks span the pair but triangulation rejects them (degenerate pose /
  // cheirality / reprojection). Init-only, so it never spams the resection loop.
  const bool seed_stage = (sfm_data_.GetPoses().size() <= 2);
  const std::size_t n_before_erase = seed_stage ? sfm_data_.structure.size() : 0;

  eraseObservationsWithMissingPoses(sfm_data_, min_sample_index);

  if (seed_stage)
  {
    const std::size_t n_after_erase = sfm_data_.structure.size();
    OPENMVG_LOG_INFO
      << "[SEED-TRI] poses=" << sfm_data_.GetPoses().size()
      << " landmarks=" << n_before_erase
      << " span_seed_pair(after_erase)=" << n_after_erase;
  }

  SfM_Data_Structure_Computation_Robust triangulation_engine(
      max_reprojection_error,
      min_required_inliers,
      min_sample_index,
      triangulation_method_);

  triangulation_engine.triangulate(sfm_data_);

  if (seed_stage)
  {
    OPENMVG_LOG_INFO
      << "[SEED-TRI] triangulated(after_robust)=" << sfm_data_.structure.size()
      << " (if span_seed_pair>0 but this=0 -> degenerate seed geometry;"
      << " if span_seed_pair=0 -> no track connects the seed views)";
  }

#if OPENMVG_SFM2_TIME_TRIANGULATION
  const auto s_tri_t1 = std::chrono::steady_clock::now();
  const double s_tri_ms =
    std::chrono::duration<double, std::milli>(s_tri_t1 - s_tri_t0).count();
  s_tri_total_ms += s_tri_ms;
  ++s_tri_call_count;
  OPENMVG_LOG_INFO
    << "[TRI-TIME] call=" << s_tri_call_count
    << " tracks=" << sfm_data_.structure.size()
    << " this_ms=" << s_tri_ms
    << " cumulative_ms=" << s_tri_total_ms;
#endif

  return !sfm_data_.structure.empty();
}

bool SequentialSfMReconstructionEngine2::AddingMissingView
(
  const float & track_inlier_ratio
#if OPENMVG_SFM_PIPELINE_DIAG
  , Hash_Map<IndexT, ViewDiagRecord> * diag_records
#endif
)
{
  if (sfm_data_.GetLandmarks().empty())
    return false;

  // Collect the views that does not have any 3D pose (sorted vector for
  // cache-friendly OMP iteration; keep deterministic order).
  //
  // Output-preserving pre-filter: views that appear in zero entries of
  // `map_tracks_` cannot be resected -- inside the parallel loop below,
  // shared_track_visibility_helper_->GetTracksInImages would return an
  // empty set, view_tracks_ids would be empty, and the resection branch
  // would never be entered. Skipping them up front saves the
  // GetTracksInImages call + set_intersection per iteration of the
  // outer resection loop. The diagnostic accumulator is updated below
  // so per-view records remain identical to the unfiltered behaviour.
  std::vector<IndexT> view_with_no_pose;
  std::vector<IndexT> view_no_tracks;
  {
    view_with_no_pose.reserve(sfm_data_.GetViews().size());
    view_no_tracks.reserve(sfm_data_.GetViews().size());
    for (const auto & view_it : sfm_data_.GetViews())
    {
      const View * v = view_it.second.get();
      const IndexT id_pose = v->id_pose;
      if (sfm_data_.GetPoses().count(id_pose) != 0)
        continue;
      if (views_with_tracks_.count(view_it.first))
        view_with_no_pose.push_back(view_it.first);
      else
        view_no_tracks.push_back(view_it.first);
    }
    std::sort(view_with_no_pose.begin(), view_with_no_pose.end());
  }

#if OPENMVG_SFM_PIPELINE_DIAG
  // Mirror what the inner-loop NoVisibleTracks branch would write: bump
  // attempts and tag last_reason. The two max(..., 0) calls in the
  // original branch are no-ops on the default-zero record so we omit
  // them. Done outside the parallel region; no critical needed.
  if (diag_records)
  {
    for (IndexT id : view_no_tracks)
    {
      auto & rec = (*diag_records)[id];
      rec.last_reason = ResectionFailReason::NoVisibleTracks;
      ++rec.attempts;
    }
  }
#endif

  const IndexT pose_before = sfm_data_.GetPoses().size();

  // Mark the track ids of the reconstructed landmarks in a membership
  // bitmap (bit t <=> track t has a landmark).
  //
  // This replaces the previous "extract every key into a vector, then
  // std::sort it" form. The sorted vector only ever existed so that the
  // two consumers below (the prev/cur diff and the per-view score) could
  // run merge-style algorithms; both are strictly cheaper on the bitmap,
  // so the O(n log n) sort is pure overhead and is gone. What remains is
  // one memset of the bitmap plus one bit-set per landmark -- and the
  // per-landmark step no longer touches a growing vector, so the loop is
  // bound only by the unordered_map node walk.
  //
  // Set-equivalent to the old vector by construction: unordered_map keys
  // are unique, so "sorted list of keys" and "bitmap of keys" carry the
  // same information, and every downstream consumer is rewritten below to
  // produce byte-identical results from it.
  //
  // The buffer is a class member so it is never reallocated; at the end of
  // this function it is swapped with `prev_reconstructed_track_bits_`.
  const Landmarks & reconstructed_landmarks = sfm_data_.GetLandmarks();
  const size_t reconstructed_count = reconstructed_landmarks.size();
  std::vector<uint64_t> & cur_track_bits = cur_reconstructed_track_bits_;
  const size_t n_track_words = cur_track_bits.size();
  // Bit capacity of the bitmap. Every landmark id is <= max track id (see
  // InitTracksAndLandmarks) so the bound test below never rejects a real
  // id; it is kept as a cheap, perfectly-predicted guard against a future
  // caller injecting structure that did not come from `map_tracks_`.
  const IndexT track_bit_capacity = static_cast<IndexT>(n_track_words * 64);

  bool bitmap_filled_in_parallel = false;

#ifdef OPENMVG_USE_OPENMP
  // Parallel fill threshold. Below this the OpenMP fork/join plus the
  // per-thread bitmap reduction costs more than the serial walk saves;
  // above it the walk is pure cache-miss latency on the unordered_map
  // nodes (each node holds a Landmark, so consecutive nodes are far
  // apart), which threads hide well by keeping several misses in flight.
  constexpr size_t kParallelBitmapFillMinLandmarks = 50000;
  if (reconstructed_count >= kParallelBitmapFillMinLandmarks
      && omp_get_max_threads() > 1)
  {
    // Bucket-partitioned fill. Each thread walks a static slice of the
    // hash table's buckets and ORs into its *private* bitmap, so the fill
    // needs no atomics and no locks; the merge is then partitioned by
    // word, a pure ALU pass over (#threads * bitmap) bytes that is
    // negligible next to the node walk. Both phases are order-
    // independent, so the result is identical to the serial fill --
    // including under a different thread count.
    const int n_bucket = static_cast<int>(reconstructed_landmarks.bucket_count());
    const int n_word = static_cast<int>(n_track_words);
    const int n_scratch = omp_get_max_threads();

    // Size the per-thread buffers here rather than inside the region:
    // an allocation inside would write the (adjacent) vector headers
    // from several threads at once -- correct, but needless false
    // sharing on the first call. After the first call this loop is a
    // no-op and the zeroing happens in parallel below.
    if (static_cast<int>(track_bits_scratch_.size()) < n_scratch)
      track_bits_scratch_.resize(n_scratch);
    for (int k = 0; k < n_scratch; ++k)
    {
      if (track_bits_scratch_[k].size() != n_track_words)
        track_bits_scratch_[k].assign(n_track_words, uint64_t(0));
    }

    #pragma omp parallel
    {
      const int thread_id = omp_get_thread_num();
      const int n_thread_actual = omp_get_num_threads();
      std::vector<uint64_t> & local_bits = track_bits_scratch_[thread_id];
      std::fill(local_bits.begin(), local_bits.end(), uint64_t(0));

      #pragma omp for schedule(static)
      for (int b = 0; b < n_bucket; ++b)
      {
        for (auto it = reconstructed_landmarks.begin(b),
                  it_end = reconstructed_landmarks.end(b); it != it_end; ++it)
        {
          const IndexT track_id = it->first;
          if (track_id < track_bit_capacity)
            local_bits[track_id >> 6] |= uint64_t(1) << (track_id & 63);
        }
      }
      // Implicit barrier above: every local_bits is complete here.
      // omp_get_thread_num() is in [0, n_thread_actual), so exactly the
      // buffers read below are the ones that were just written.

      #pragma omp for schedule(static)
      for (int w = 0; w < n_word; ++w)
      {
        uint64_t merged = 0;
        for (int k = 0; k < n_thread_actual; ++k)
          merged |= track_bits_scratch_[k][w];
        cur_track_bits[w] = merged;
      }
    }
    bitmap_filled_in_parallel = true;
  }
#endif

  if (!bitmap_filled_in_parallel)
  {
    std::fill(cur_track_bits.begin(), cur_track_bits.end(), uint64_t(0));
    for (const auto & landmark_it : reconstructed_landmarks)
    {
      const IndexT track_id = landmark_it.first;
      if (track_id < track_bit_capacity)
        cur_track_bits[track_id >> 6] |= uint64_t(1) << (track_id & 63);
    }
  }

  // === Resection-score delta cache ===
  //
  // The score for view V is `|view_tracks_ids(V) n reconstructed tracks|`
  // and the corresponding ratio. view_tracks_ids(V) is invariant after
  // InitTracksAndLandmarks (map_tracks_ never changes during the
  // resection loop). The reconstructed-track set changes between
  // successive AddingMissingView() calls via Triangulation() and
  // outlier-erase passes. We diff the current and previous bitmaps
  // to obtain the (added, removed) delta of track ids; only views whose
  // visible tracks intersect the delta need to be re-scored. All others
  // hold a still-valid cached intersection -- output bit-identical to
  // the unfiltered scan, just with the redundant set_intersection work
  // skipped.
  //
  // Two cases require forced recompute regardless of the delta:
  //   1. First call: prev state is empty, so we have no cache. Treated
  //      naturally by marking all current no-pose views dirty.
  //   2. View V was resected in some past round and later kicked back
  //      to view_with_no_pose (eraseUnstablePosesAndObservations).
  //      V's slot may reflect a much older reconstructed-track set.
  //      Detected via prev_view_with_no_pose_: if V is in current
  //      view_with_no_pose but wasn't last call, it wasn't scored last
  //      call, so we cannot validate its cache from one round of delta.
  //
  // Fast paths to avoid building dirty_views when it can't help:
  //   * First call (no prev state): every view is forced dirty via
  //     `was_scored_last_round=false`; skip the diff entirely.
  //   * Massive delta: if |added|+|removed| >= |reconstructed|/2 the
  //     reconstructed set has materially turned over and most views
  //     will end up dirty anyway. Walking every delta track's view
  //     list would cost more than just letting per-view recompute run.
  //
  // `dirty_view_mask_` is a reused bit-vector keyed by view_id
  // (replacing the previous std::unordered_set<IndexT>). Reset to all-
  // zero each call; O(1) mark, O(1) test, no hashing or bucket allocs.
  // Bounded by max_view_id+1 (== resection_score_cache_.size()).
  std::fill(dirty_view_mask_.begin(), dirty_view_mask_.end(), uint8_t(0));
  // `has_prev_reconstructed_tracks_` is false only on the first invocation
  // after InitTracksAndLandmarks; every later call published a bitmap into
  // prev_ at the end of this function.
  const bool first_call = !has_prev_reconstructed_tracks_;
  bool force_all_dirty = first_call;

  if (!force_all_dirty)
  {
    // Delta = symmetric difference of the two membership sets = XOR of the
    // two bitmaps, word by word. This subsumes both std::set_difference
    // calls: added and removed tracks feed the *same* mark_views_for_track,
    // so there is no reason to separate them, and no reason to materialise
    // them into vectors at all -- each delta bit is consumed the moment it
    // is found.
    //
    // The scan also computes |added|+|removed| incrementally and bails to
    // force_all_dirty the moment the running count crosses the turnover
    // threshold. That decision is identical to computing the full count
    // first (a running count only grows, so crossing early implies
    // crossing overall, and reaching the end means the running count *is*
    // the full count) -- but on the massive-delta path it stops scanning
    // instead of enumerating a delta it is about to throw away. Spurious
    // marks left in dirty_view_mask_ by the aborted scan are harmless:
    // force_all_dirty short-circuits the mask test in the loop below.
    const uint64_t * cur_words = cur_track_bits.data();
    const uint64_t * prev_words = prev_reconstructed_track_bits_.data();
    const IndexT mask_size = static_cast<IndexT>(dirty_view_mask_.size());
    auto mark_views_for_track = [&](IndexT track_id)
    {
      auto it = map_tracks_.find(track_id);
      if (it == map_tracks_.end()) return;
      for (const auto & vf : it->second)  // {ViewId, FeatureId}
      {
        const IndexT vid = vf.first;
        if (vid < mask_size) dirty_view_mask_[vid] = 1;
      }
    };

    size_t delta_size = 0;
    for (size_t w = 0; w < n_track_words; ++w)
    {
      uint64_t delta_word = cur_words[w] ^ prev_words[w];
      if (delta_word == 0)
        continue;
      delta_size += PopCount64(delta_word);
      if (delta_size * 2 >= reconstructed_count)
      {
        force_all_dirty = true;  // cheaper to just recompute everyone
        break;
      }
      const IndexT track_base = static_cast<IndexT>(w * 64);
      do
      {
        const unsigned bit = CountTrailingZeros64(delta_word);
        delta_word &= delta_word - 1;      // clear lowest set bit
        mark_views_for_track(track_base + bit);
      } while (delta_word);
    }
  }

  // Build current no-pose mask for next round's "re-entered" detection.
  // Reused member bit-vector (no per-call allocation). The previous
  // mask is still valid here (we read it below); we swap at function end.
  std::fill(cur_view_with_no_pose_mask_.begin(),
            cur_view_with_no_pose_mask_.end(), uint8_t(0));
  {
    const IndexT mask_size = static_cast<IndexT>(cur_view_with_no_pose_mask_.size());
    for (IndexT id : view_with_no_pose)
      if (id < mask_size) cur_view_with_no_pose_mask_[id] = 1;
  }

  // List the view that have a sufficient 2D-3D coverage for robust pose estimation
  //
  // The previous form was `#pragma omp parallel` over a range-based for with
  // `#pragma omp single nowait` inside the loop body. That made every thread
  // walk every iteration of the C++ for loop, with `single` electing one
  // thread per iteration to actually run the body. Even with `nowait`, the
  // lock contention on `single` and the redundant iteration walk cost more
  // than they saved -- the loop was only marginally parallel.
  //
  // Switch to a real `parallel for` over an index. MSVC's OpenMP 2.0 needs a
  // signed int loop counter, hence the explicit cast. Iteration body is
  // unchanged. All shared-state writes inside (sfm_data_.poses,
  // sfm_data_.intrinsics, diag_records) are already wrapped in
  // `#pragma omp critical` -- semantics preserved. Iteration order across
  // threads was already nondeterministic under `single nowait`; that
  // doesn't change. `schedule(dynamic)` matches the previous behaviour
  // (each thread grabs the next available iteration when free) and copes
  // well with the very uneven per-iteration cost (most iterations fail the
  // ratio gate cheaply; a few do a full Localize+RefinePose).
  const int n_no_pose = static_cast<int>(view_with_no_pose.size());
#ifdef OPENMVG_USE_OPENMP
  #pragma omp parallel for schedule(dynamic)
#endif
  for (int idx = 0; idx < n_no_pose; ++idx)
  {
    const IndexT view_id = view_with_no_pose[idx];
    {
      // Look up the precomputed sorted track-id vector for this view.
      // Built once in InitTracksAndLandmarks; never changes during the
      // resection loop because `map_tracks_` is invariant after init.
      // The cache replaces the per-iteration GetTracksInImages +
      // GetTracksIdVector pair *for the gate computation*. We still
      // call GetTracksInImages below, but only when the ratio gate
      // passes -- which is the rare case (most no-pose views fail
      // the gate every outer iteration).
      static const std::vector<IndexT> kEmptyTrackIds;
      const auto cache_it = view_track_ids_cache_.find(view_id);
      const std::vector<IndexT>& view_tracks_ids =
          (cache_it != view_track_ids_cache_.end())
              ? cache_it->second
              : kEmptyTrackIds;

      // Delta-cache check. The slot is only safely re-usable if V was
      // scored last round AND no track in V's view_tracks_ids was
      // added to / removed from the reconstructed-track set since.
      ResectionScoreCache & slot = resection_score_cache_[view_id];
      const bool was_scored_last_round =
          (view_id < prev_view_with_no_pose_mask_.size())
          && prev_view_with_no_pose_mask_[view_id] != 0;
      const bool is_dirty =
          force_all_dirty
          || !was_scored_last_round
          || (view_id < dirty_view_mask_.size()
              && dirty_view_mask_[view_id] != 0);

      if (is_dirty)
      {
        // Recompute the intersection by testing each of the view's track
        // ids against the reconstructed-track bitmap.
        //
        // Bit-identical to the previous std::set_intersection over the two
        // sorted id lists: view_tracks_ids is ascending and duplicate-free
        // (one append per (view, track) pair, in map_tracks_ order), so
        // filtering it in place emits exactly the common elements, exactly
        // once each, in exactly the same ascending order.
        //
        // The cost model is what changes. set_intersection is a merge: it
        // advances through the reconstructed list until it passes the
        // view's largest track id, so each dirty view streamed most of the
        // full reconstructed id array (megabytes, once per view). The
        // bitmap turns that into one random bit test per visible track --
        // O(#tracks in this view) against a working set of
        // (#tracks / 8) bytes, which stays resident in L2 across the whole
        // parallel loop instead of evicting it every iteration.
        slot.track_id_for_resection.clear();
        slot.track_id_for_resection.reserve(
            std::min(view_tracks_ids.size(), reconstructed_count));
        for (const IndexT track_id : view_tracks_ids)
        {
          if (track_id < track_bit_capacity
              && (cur_track_bits[track_id >> 6] >> (track_id & 63)) & uint64_t(1))
          {
            slot.track_id_for_resection.push_back(track_id);
          }
        }
        slot.track_ratio = slot.track_id_for_resection.size()
                         / static_cast<float>(view_tracks_ids.size() + 1);
      }
      // Read-only references for the rest of the iteration.  The slot
      // holds the same bytes the original lambda would have produced.
      const std::vector<IndexT> & track_id_for_resection =
          slot.track_id_for_resection;
      const double track_ratio = slot.track_ratio;

      // Per-view "ViewId: ...; #2D-3D matches: ..." log removed.
      // It fired unconditionally for every no-pose view in every resection
      // round (~14k records on 369-image runs), and OPENMVG_LOG_INFO takes
      // a global logger mutex -- inside an `#pragma omp parallel for` that
      // effectively serializes the loop. The gate-passed branch below still
      // logs each actually-resected view via "Robust Resection of camera
      // index: ..." so no observable diagnostic information is lost.

#if OPENMVG_SFM_PIPELINE_DIAG
      // Pre-classify failure reason; will be overwritten on success.
      ResectionFailReason reason = ResectionFailReason::NoReconstructedTracks;
      if (view_tracks_ids.empty())
        reason = ResectionFailReason::NoVisibleTracks;
      else if (track_id_for_resection.empty())
        reason = ResectionFailReason::NoReconstructedTracks;
      else if (track_ratio <= track_inlier_ratio)
        reason = ResectionFailReason::LowTrackRatio;
#endif

      if (!track_id_for_resection.empty() && track_ratio > track_inlier_ratio)
      {
        // Gate passed: now we genuinely need the STLMAPTracks for the
        // feature-index lookup below. Deferring this call until here
        // (instead of doing it unconditionally at the top of the loop
        // body) avoids the expensive GetTracksInImages work on the
        // overwhelming majority of view-iteration evaluations that fail
        // the gate.
        openMVG::tracks::STLMAPTracks view_tracks;
        shared_track_visibility_helper_->GetTracksInImages({view_id}, view_tracks);

        // Get feat_id for the 2D/3D associations
        std::vector<IndexT> feature_id_for_resection;
        tracks::TracksUtilsMap::GetFeatIndexPerViewAndTrackId(
          view_tracks,
          track_id_for_resection,
          view_id,
          &feature_id_for_resection);

        // Localize the image inside the SfM reconstruction
        Image_Localizer_Match_Data resection_data;
        resection_data.pt2D.resize(2, track_id_for_resection.size());
        resection_data.pt3D.resize(3, track_id_for_resection.size());

        // Look if the intrinsic data is known or not
        const View * view = sfm_data_.GetViews().at(view_id).get();
        std::shared_ptr<cameras::IntrinsicBase> intrinsic;
        if (sfm_data_.GetIntrinsics().count(view->id_intrinsic))
        {
          intrinsic = sfm_data_.GetIntrinsics().at(view->id_intrinsic);
        }

        // Collect the feature observation.
        // Hoist hash-map lookups (`feats_per_view.at(view_id)` and
        // `GetLandmarks()`) out of the per-point loop; both are O(1)
        // average but the constant-factor savings are noticeable when
        // a single view produces hundreds of 2D-3D pairs.
        Mat2X pt2D_original(2, track_id_for_resection.size());
        const auto & view_feats   = features_provider_->feats_per_view.at(view_id);
        const auto & landmarks    = sfm_data_.GetLandmarks();
        const bool   has_disto    = intrinsic && intrinsic->have_disto();
        auto track_it = track_id_for_resection.cbegin();
        auto feat_it = feature_id_for_resection.cbegin();
        for (size_t cpt = 0; cpt < track_id_for_resection.size(); ++cpt, ++track_it, ++feat_it)
        {
          resection_data.pt3D.col(cpt) = landmarks.at(*track_it).X;
          resection_data.pt2D.col(cpt) = pt2D_original.col(cpt) =
            view_feats[*feat_it].coords().cast<double>();
          // Handle image distortion if intrinsic is known (to ease the resection)
          if (has_disto)
          {
            resection_data.pt2D.col(cpt) = intrinsic->get_ud_pixel(resection_data.pt2D.col(cpt));
          }
        }

        geometry::Pose3 pose;
        const bool bResection = sfm::SfM_Localizer::Localize
        (
          intrinsic ? resection_method_ : resection::SolverType::DLT_6POINTS,
          {view->ui_width, view->ui_height},
          intrinsic ? intrinsic.get() : nullptr,
          resection_data,
          pose
        );
        resection_data.pt2D = std::move(pt2D_original); // restore original image domain points

        const float inlier_ratio = resection_data.vec_inliers.size()/static_cast<float>(feature_id_for_resection.size());
        OPENMVG_LOG_INFO
          << std::endl
          << "-------------------------------" << "\n"
          << "-- Robust Resection of camera index: <" << view_id << "> image: "
          <<  view->s_Img_path <<"\n"
          << "-- Threshold: " << resection_data.error_max << "\n"
          << "-- Resection status: " << (bResection ? "OK" : "FAILED") << "\n"
          << "-- Nb points used for Resection: " << feature_id_for_resection.size() << "\n"
          << "-- Nb points validated by robust estimation: " << resection_data.vec_inliers.size() << "\n"
          << "-- % points validated: "
          << inlier_ratio * 100 << "\n"
          << "-------------------------------";

#if OPENMVG_SFM_PIPELINE_DIAG
        if (!bResection)        reason = ResectionFailReason::ResectionFailed;
        else if (inlier_ratio <= 0.5f) reason = ResectionFailReason::LowInlierRatio;
#endif

        // Refine the pose of the found camera pose by using a BA and fix 3D points.
        if (bResection && inlier_ratio > 0.5)
        {
          // A valid pose has been found (try to refine it):
          // If no valid intrinsic as input:
          //  init a new one from the projection matrix decomposition
          // Else use the existing one and consider it as constant.
          if (!intrinsic)
          {
            // setup a default camera model from the found projection matrix
            Mat3 K, R;
            Vec3 t;
            KRt_From_P(resection_data.projection_matrix, &K, &R, &t);

            const double focal = (K(0,0) + K(1,1))/2.0;
            const Vec2 principal_point(K(0,2), K(1,2));

            // Create the new camera intrinsic
            switch (cam_type_)
            {
              case PINHOLE_CAMERA:
                intrinsic = std::make_shared<Pinhole_Intrinsic>
                  (view->ui_width, view->ui_height, focal, principal_point(0), principal_point(1));
              break;
              case PINHOLE_CAMERA_RADIAL1:
                intrinsic = std::make_shared<Pinhole_Intrinsic_Radial_K1>
                  (view->ui_width, view->ui_height, focal, principal_point(0), principal_point(1));
              break;
              case PINHOLE_CAMERA_RADIAL3:
                intrinsic = std::make_shared<Pinhole_Intrinsic_Radial_K3>
                  (view->ui_width, view->ui_height, focal, principal_point(0), principal_point(1));
              break;
              case PINHOLE_CAMERA_BROWN:
                intrinsic = std::make_shared<Pinhole_Intrinsic_Brown_T2>
                  (view->ui_width, view->ui_height, focal, principal_point(0), principal_point(1));
              break;
              case PINHOLE_CAMERA_FISHEYE:
                intrinsic = std::make_shared<Pinhole_Intrinsic_Fisheye>
                  (view->ui_width, view->ui_height, focal, principal_point(0), principal_point(1));
              break;
              default:
                OPENMVG_LOG_ERROR << "Try to create an unknown camera type.";
            }
          }
          const bool b_refine_pose = true;
          const bool b_refine_intrinsics = false;
          const bool refined = intrinsic && sfm::SfM_Localizer::RefinePose(
              intrinsic.get(), pose,
              resection_data, b_refine_pose, b_refine_intrinsics);
          if (refined)
          {
            // - intrinsic parameters (if the view has no intrinsic group add a new one)
            if (sfm_data_.intrinsics.count(sfm_data_.views.at(view_id)->id_intrinsic) == 0)
            {
              // Since the view have not yet an intrinsic group before, create a new one
              IndexT new_intrinsic_id = 0;
              if (!sfm_data_.GetIntrinsics().empty())
              {
                // Since some intrinsic Id already exists,
                //  we have to create a new unique identifier following the existing one.
                // Single-pass max over the existing keys; output bit-identical
                // to the previous std::set + rbegin() construction (both yield
                // max(existing) + 1) without the per-element red-black-tree
                // insert. GetIntrinsics() is unordered, so a linear scan is
                // the natural way to find the max.
                IndexT max_existing = 0;
                for (const auto & intr_kv : sfm_data_.GetIntrinsics())
                  max_existing = std::max(max_existing, intr_kv.first);
                new_intrinsic_id = max_existing + 1;
              }
              #pragma omp critical
              {
                sfm_data_.views.at(view_id)->id_intrinsic = new_intrinsic_id;
                sfm_data_.intrinsics[new_intrinsic_id] = intrinsic;
              }
            }

            // Update the found camera pose
            #pragma omp critical
            sfm_data_.poses[view->id_pose] = pose;

#if OPENMVG_SFM_PIPELINE_DIAG
            reason = ResectionFailReason::Calibrated;
#endif
          }
#if OPENMVG_SFM_PIPELINE_DIAG
          else
          {
            reason = ResectionFailReason::RefinePoseFailed;
          }
#endif
        }

#if OPENMVG_SFM_PIPELINE_DIAG
        if (diag_records)
        {
          #pragma omp critical
          {
            auto & rec = (*diag_records)[view_id];
            rec.last_reason = reason;
            ++rec.attempts;
            rec.best_2d3d = std::max<uint32_t>(rec.best_2d3d,
              static_cast<uint32_t>(track_id_for_resection.size()));
            rec.best_track_ratio = std::max(rec.best_track_ratio,
              static_cast<float>(track_ratio));
            rec.last_inliers = static_cast<uint32_t>(resection_data.vec_inliers.size());
            rec.last_ratio = inlier_ratio;
          }
        }
#endif
      }
#if OPENMVG_SFM_PIPELINE_DIAG
      else if (diag_records)
      {
        #pragma omp critical
        {
          auto & rec = (*diag_records)[view_id];
          rec.last_reason = reason;
          ++rec.attempts;
          rec.best_2d3d = std::max<uint32_t>(rec.best_2d3d,
            static_cast<uint32_t>(track_id_for_resection.size()));
          rec.best_track_ratio = std::max(rec.best_track_ratio,
            static_cast<float>(track_ratio));
        }
      }
#endif
    }
  }

  const IndexT pose_after = sfm_data_.GetPoses().size();

  // Persist delta-cache state for the next AddingMissingView() call.
  // Swap (no copy) the current track bitmap into prev_, and the current
  // no-pose mask into prev_. Done after the parallel loop so writes are
  // race-free. `cur_track_bits` aliases cur_reconstructed_track_bits_ and
  // must not be read past this point.
  prev_reconstructed_track_bits_.swap(cur_reconstructed_track_bits_);
  has_prev_reconstructed_tracks_ = true;
  prev_view_with_no_pose_mask_.swap(cur_view_with_no_pose_mask_);

  return (pose_after != pose_before);
}

bool SequentialSfMReconstructionEngine2::BundleAdjustment(BAPreset preset)
{
  Bundle_Adjustment_Ceres::BA_Ceres_options options;
  // Linear-solver selector:
  //   * < 100 poses                 -> DENSE_SCHUR (fastest at small scale).
  //   * > 100 poses (sparse lib)    -> SPARSE_SCHUR + JACOBI preconditioner
  //                                    (direct factorisation of Schur comp.).
  //   * ITERATIVE_SCHUR + SCHUR_JACOBI only above
  //     OPENMVG_SFM2_ITERATIVE_SCHUR_MIN_POSES, which defaults to 0 (= never).
  //
  // The only measurements we have on both solvers favour direct factorisation
  // (436 poses: 11s SPARSE_SCHUR vs 22-36s ITERATIVE_SCHUR; >1500 poses: 50.9s
  // in the linear solve plus an 832MB explicit Schur complement). See the
  // top-of-file knob for the full history before re-enabling the branch.
  const std::size_t n_poses = sfm_data_.GetPoses().size();
  const bool sparse_available =
#if OPENMVG_SFM2_FORCE_NO_SPARSE_BA
      false;  // A/B override: behave as if Ceres was built without sparse libs.
#else
       ceres::IsSparseLinearAlgebraLibraryTypeAvailable(ceres::SUITE_SPARSE)
    || ceres::IsSparseLinearAlgebraLibraryTypeAvailable(ceres::EIGEN_SPARSE);
#endif

  // 0 disables the iterative branch; see top-of-file knob.
  constexpr std::size_t kIterativeSchurMinPoses =
      OPENMVG_SFM2_ITERATIVE_SCHUR_MIN_POSES;

  if (kIterativeSchurMinPoses > 0 &&
      n_poses > kIterativeSchurMinPoses && sparse_available)
  {
    options.preconditioner_type_ = ceres::SCHUR_JACOBI;
    options.linear_solver_type_  = ceres::ITERATIVE_SCHUR;
  }
  else if (n_poses > 100 && sparse_available)
  {
    options.preconditioner_type_ = ceres::JACOBI;
    options.linear_solver_type_  = ceres::SPARSE_SCHUR;
  }
  else
  {
    options.linear_solver_type_  = ceres::DENSE_SCHUR;
  }

  // Preset for the intermediate BA. Solver / preconditioner / loss function
  // are unchanged -- only the stopping criteria and iteration caps move.
  // The final BA at the end of Process() runs with STRICT and cleans up
  // any slack left by intermediate BAs.
  //
  // All of these are pure Ceres stopping-criteria knobs; none change the
  // residual model, the parameter blocks, or the loss function.
  switch (preset)
  {
    case BAPreset::FAST:
      options.parameter_tolerance_              = 1e-6;   // strict: 1e-8
      options.gradient_tolerance_               = 1e-8;   // strict: 1e-10
      options.function_tolerance_               = 1e-4;   // strict: 1e-6
      options.max_num_iterations_               = 15;     // strict: 50
      options.max_num_consecutive_invalid_steps_= 2;      // strict: 5
      // Plateau early-stop: easy scenes settle in 4-6 iters; relative cost
      // change collapses below 5e-4 well before max_num_iterations on those
      // scenes, so we cut out without affecting hard scenes.
      options.plateau_relative_tolerance_       = 5e-4;
      options.plateau_min_iterations_           = 4;
      options.plateau_patience_                 = 2;
      // Inner iterations (Ceres' post-LM-step refinement subproblem) buy
      // accuracy at intermediate scale that the next intermediate BA or
      // the final STRICT BA would re-derive anyway. Disabling for FAST
      // saves ~6-10% of minimizer wall time per call without changing
      // the model being solved. Final BA keeps it on (STRICT default).
      options.use_inner_iterations_             = false;
      break;
    case BAPreset::BALANCED:
      // Tight enough for motion-prior LM convergence (the prior term needs
      // more iterations to balance reprojection than FAST allows), loose
      // enough to skip the last few strict-convergence iterations that
      // don't measurably change geometry. ~30% faster than STRICT on
      // prior-bearing scenes with no observed quality loss.
      options.parameter_tolerance_              = 1e-7;
      options.gradient_tolerance_               = 1e-9;
      options.function_tolerance_               = 1e-5;
      options.max_num_iterations_               = 30;
      options.max_num_consecutive_invalid_steps_= 4;
      // Plateau early-stop: tighter relative tol than FAST since BALANCED
      // runs on prior workflows where the last few iterations of
      // prior-vs-reprojection rebalancing matter. patience=2 makes the
      // detector robust to single-iteration relative-progress dips.
      options.plateau_relative_tolerance_       = 1e-4;
      options.plateau_min_iterations_           = 5;
      options.plateau_patience_                 = 2;
      // Same rationale as FAST: inner iterations are post-LM refinement
      // that intermediate BAs don't need to keep -- the final STRICT BA
      // refines the same parameters end-to-end. ~6-10% per-call savings.
      options.use_inner_iterations_             = false;
      break;
    case BAPreset::STRICT:
    default:
      // Use BA_Ceres_options defaults (matches reference engine).
      // plateau_relative_tolerance_ stays 0.0 -> callback not installed.
      break;
  }

  Bundle_Adjustment_Ceres bundle_adjustment_obj(options);
  // ---------------------------------------------------------------------
  // TEMPORARILY REVERTED for ghost-layer regression diagnosis.
  // ---------------------------------------------------------------------
  // Optimization "F" (freeze intrinsics on intermediate BAs) is suspected
  // of producing parallel ghost layers in DensifyPointCloud output on at
  // least one dataset whose input intrinsics (likely EXIF-derived) are
  // not accurate enough for the freeze-then-refine-once-at-the-end
  // strategy to land in the correct basin. With intrinsics frozen, every
  // intermediate BA fits poses+landmarks against a wrong focal length;
  // the final STRICT BA refines intrinsics from inside that basin and
  // can leave structure stranded at the old depth alongside structure at
  // the new depth -> ghost layers in the dense cloud.
  //
  // Toggle:
  //   kFreezeIntrinsicsOnIntermediateBA = true   (the speed optimization)
  //   kFreezeIntrinsicsOnIntermediateBA = false  (current; safe default
  //                                               while diagnosing)
  //
  // TODO: once the ghost-layer dataset is verified clean with the freeze
  // disabled, decide whether to:
  //   (a) re-enable the freeze unconditionally if the regression is
  //       traced to a different cause,
  //   (b) gate the freeze behind a "trusted intrinsics" engine flag so
  //       calibrated rigs get the speed win and EXIF-only datasets stay
  //       safe, or
  //   (c) leave it disabled permanently and absorb the ~5-15s cost.
  //
  // The original speed rationale (kept for reference):
  //   Intrinsic refinement on intermediate BAs is wasted work: focal
  //   length / principal-point / distortion drift well below 0.01 px
  //   after the first ~20 poses, while every intermediate BA pays full
  //   Jacobian-eval cost for the intrinsic block. The final STRICT BA at
  //   the end of Process() refines intrinsics end-to-end with the full
  //   reconstructed scene.
  static constexpr bool kFreezeIntrinsicsOnIntermediateBA =
      (OPENMVG_SFM2_FREEZE_INTRINSICS_ON_INTERMEDIATE_BA != 0);
  const cameras::Intrinsic_Parameter_Type intrinsic_refinement_for_ba =
      (preset == BAPreset::STRICT || !kFreezeIntrinsicsOnIntermediateBA)
          ? ReconstructionEngine::intrinsic_refinement_options_
          : cameras::Intrinsic_Parameter_Type::NONE;
  const Optimize_Options ba_refine_options
    ( intrinsic_refinement_for_ba,
      ReconstructionEngine::extrinsic_refinement_options_,
      Structure_Parameter_Type::ADJUST_ALL, // Adjust scene structure
      Control_Point_Parameter(),
      this->b_use_motion_prior_
    );
  return bundle_adjustment_obj.Adjust(sfm_data_, ba_refine_options);
}

} // namespace sfm
} // namespace openMVG
