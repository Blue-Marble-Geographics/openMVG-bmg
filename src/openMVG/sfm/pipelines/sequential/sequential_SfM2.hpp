// This file is part of OpenMVG, an Open Multiple View Geometry C++ library.

// Copyright (c) 2018 Pierre MOULON.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef OPENMVG_SFM_LOCALIZATION_SEQUENTIAL2_SFM_HPP
#define OPENMVG_SFM_LOCALIZATION_SEQUENTIAL2_SFM_HPP

#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#include "openMVG/sfm/pipelines/sfm_engine.hpp"
#include "openMVG/cameras/cameras.hpp"
#include "openMVG/multiview/solver_resection.hpp"
#include "openMVG/multiview/triangulation_method.hpp"
#include "openMVG/tracks/tracks.hpp"

// Pipeline-health diagnostics. Cheap per-checkpoint, but the per-view
// `#pragma omp critical` writes into `view_diag_records` inside the parallel
// resection loop serialize the loop on heavily threaded runs. Off by default
// for release; flip to 1 to triage pipeline regressions.
#ifndef OPENMVG_SFM_PIPELINE_DIAG
#define OPENMVG_SFM_PIPELINE_DIAG 0
#endif

// Toggle: use FAST/BALANCED presets on intermediate BAs vs. STRICT-everywhere.
//   1 = current optimisation (fewer iters, looser tols on intermediate BAs).
//   0 = S3 reference behaviour (every BA is STRICT). Use 0 to recover the
//       last ~5K tracks lost on dataset where intermediate BAs under-converge
//       and the post-BA outlier filter then ejects too many observations.
#ifndef OPENMVG_SFM2_FAST_INTERMEDIATE_BA
#define OPENMVG_SFM2_FAST_INTERMEDIATE_BA 1
#endif

namespace htmlDocument { class htmlDocumentStream; }

namespace openMVG {
namespace sfm {

struct Features_Provider;
struct Matches_Provider;
class SfMSceneInitializer;

#if OPENMVG_SFM_PIPELINE_DIAG
// Forward-declare at NAMESPACE scope so the .cpp's namespace-scope
// definition matches. The full definition is kept private to the .cpp.
struct ViewDiagRecord;
#endif

/// Sequential SfM Pipeline Reconstruction Engine.
/// This engine uses existing poses or starts from scratch the reconstruction
class SequentialSfMReconstructionEngine2 : public ReconstructionEngine
{
public:

  SequentialSfMReconstructionEngine2(
    SfMSceneInitializer * scene_initializer,
    const SfM_Data & sfm_data,
    const std::string & soutDirectory,
    const std::string & loggingFile = "");

  ~SequentialSfMReconstructionEngine2() override;

  virtual bool Process() override;

  void SetFeaturesProvider(Features_Provider * provider);
  void SetMatchesProvider(Matches_Provider * provider);

  /// Initialize tracks
  bool InitTracksAndLandmarks();

  /// Triangulate tracks
  bool Triangulation();

  /// Adding missing view (Try to find the pose of the missing camera)
#if OPENMVG_SFM_PIPELINE_DIAG
  bool AddingMissingView(
    const float & track_inlier_ratio,
    Hash_Map<IndexT, ViewDiagRecord> * diag_records = nullptr);
#else
  bool AddingMissingView(const float & track_inlier_ratio);
#endif

  /// BA preset selector for intermediate vs final BA passes.
  ///   STRICT   = reference defaults (50 iter, 1e-8 param tol, 1e-6 func tol, max_invalid=5).
  ///              Used for the final BA and historically for every intermediate BA.
  ///   BALANCED = 30 iter, 1e-7 param tol, 1e-5 func tol, max_invalid=4. Converges well
  ///              even when motion-prior penalties are active (the prior term needs more
  ///              LM iterations to balance reprojection than FAST allows). Roughly 30%
  ///              faster than STRICT on prior workflows with no observed quality loss.
  ///   FAST     = 15 iter, 1e-6 param tol, 1e-4 func tol, max_invalid=2. Fastest, but
  ///              under-converges on prior workflows -- never use when b_use_motion_prior_.
  enum class BAPreset { STRICT, BALANCED, FAST };

  /// Adjust intrinsics, landmark and extrinsics according the user config.
  bool BundleAdjustment(BAPreset preset = BAPreset::STRICT);

  /**
   * Set the default lens distortion type to use if it is declared unknown
   * in the intrinsics camera parameters by the previous steps.
   *
   * It can be declared unknown if the type cannot be deduced from the metadata.
   */
  void SetUnknownCameraType(const cameras::EINTRINSIC camType)
  {
    cam_type_ = camType;
  }

  /// Configure the 2view triangulation method used by the SfM engine
  void SetTriangulationMethod(const ETriangulationMethod method)
  {
    triangulation_method_ = method;
  }

  /// Configure the resetcion method method used by the Localization engine
  void SetResectionMethod(const resection::SolverType method)
  {
    resection_method_ = method;
  }

  /// Enable/disable the "fast" intermediate BA preset (looser Ceres stopping
  /// criteria, fewer iterations). The final BA at the end of Process() is
  /// unaffected and always runs with strict defaults. Default: true.
  /// Set to false to retest at the original (strict) quality level.
  void SetUseFastIntermediateBA(bool enable)
  {
    b_use_fast_intermediate_ba_ = enable;
  }

  bool GetUseFastIntermediateBA() const
  {
    return b_use_fast_intermediate_ba_;
  }

private:

  //----
  //-- Data
  //----

  // HTML logger
  std::shared_ptr<htmlDocument::htmlDocumentStream> html_doc_stream_;
  std::string sLogging_file_;

  // Parameter
  cameras::EINTRINSIC cam_type_; // The camera type for the unknown cameras
  // Runtime-adjustable minimum track length for the selective short-track cut
  // (SelectiveShortTrackCut). Initialized from OPENMVG_SFM2_TRACKS_MIN_LENGTH.
  // Process() auto-lowers this to 2 (cut disabled -> keep every length-2 track)
  // and re-inits if the seed triangulation is starved to 0 landmarks by the
  // cut, so dense scenes keep the fast cut while thin-overlap scenes (e.g.
  // Randy) recover automatically without a per-dataset flag or global slowdown.
  uint32_t track_min_length_;

  //-- Data provider
  Features_Provider * features_provider_;
  Matches_Provider  * matches_provider_;

  //-- Reconstruction Initialization
  SfMSceneInitializer * scene_initializer_;

  /// Putative landmark with view id visibility
  Landmarks landmarks_;
  /// Tracking (used to build landmark visibility and compute 2D-3D visibility)
  openMVG::tracks::STLMAPTracks map_tracks_;
  /// Helper to compute fast 2D-3D visibility
  std::unique_ptr<openMVG::tracks::SharedTrackVisibilityHelper> shared_track_visibility_helper_;
  /// Set of view ids that appear in at least one entry of `map_tracks_`.
  /// Built once in InitTracksAndLandmarks. Views not in this set have zero
  /// reconstructable tracks and would deterministically hit the
  /// NoVisibleTracks branch on every AddingMissingView() iteration; we
  /// skip them up front to avoid the per-view GetTracksInImages +
  /// set_intersection overhead. Output-preserving by construction.
  std::unordered_set<IndexT> views_with_tracks_;
  /// Per-view sorted vector of track ids visible in that view. Built
  /// once in InitTracksAndLandmarks from `map_tracks_`. Lets the
  /// resection-candidate scan in AddingMissingView() compute the
  /// 2D-3D ratio gate without calling shared_track_visibility_helper_
  /// ->GetTracksInImages() per (view, outer-iteration). The expensive
  /// helper call is then deferred until *after* the gate passes, when
  /// the full STLMAPTracks is genuinely needed for the feature-index
  /// lookup. Output-preserving: the cached vector contains exactly the
  /// same track ids GetTracksIdVector() would have extracted, in the
  /// same sorted order, so the downstream set_intersection produces a
  /// bit-identical track_id_for_resection.
  Hash_Map<IndexT, std::vector<IndexT>> view_track_ids_cache_;

  /// Per-view cached score (intersection vector + ratio) for the
  /// resection-candidate scan. Indexed by view_id directly so concurrent
  /// writes from different threads in the OMP-parallel scoring loop are
  /// data-race-free (each thread mutates only its own slot). The slot is
  /// re-used across AddingMissingView() invocations; we only recompute
  /// when the score could have changed (delta-cache invalidation below).
  struct ResectionScoreCache
  {
    std::vector<IndexT> track_id_for_resection; // sorted intersection
    double track_ratio = 0.0;
  };
  std::vector<ResectionScoreCache> resection_score_cache_;

  /// Snapshot of the sorted reconstructed-track-id list at the end of the
  /// previous AddingMissingView() call. Compared against the current list
  /// via std::set_difference to obtain the (added, removed) deltas, which
  /// determine which views' cached scores are stale.
  std::vector<IndexT> prev_reconstructed_track_ids_;

  /// Scratch buffer for the current call's sorted reconstructed-track-id
  /// list. Built fresh each AddingMissingView() invocation, then swapped
  /// with `prev_reconstructed_track_ids_` at the end (so next call's
  /// `prev_` is this call's `cur_`, with no copy). `clear()` preserves
  /// capacity across calls so the per-call alloc is amortised away after
  /// the first reconstruction round.
  std::vector<IndexT> cur_reconstructed_track_ids_;

  /// Snapshot of the set of view ids in `view_with_no_pose` at the end of
  /// the previous AddingMissingView() call. A view that re-enters
  /// view_with_no_pose (e.g. via eraseUnstablePosesAndObservations) was
  /// not scored last round, so its cache may not reflect intermediate
  /// changes to the reconstructed-track set; we force a recompute for
  /// such views regardless of the delta.
  ///
  /// Stored as a bit-vector keyed by view_id (same indexing scheme as
  /// `resection_score_cache_`). O(1) test, O(views) reset, no hashing or
  /// per-call bucket allocation -- replaces the previous
  /// `std::unordered_set<IndexT>`.
  std::vector<uint8_t> prev_view_with_no_pose_mask_;

  /// Scratch bit-vector used to build the next round's no-pose mask.
  /// Sized identically to `prev_view_with_no_pose_mask_`; at the end of
  /// AddingMissingView() the two are swapped so the freshly-built bits
  /// become `prev_` with no heap turnover. Reused across calls.
  std::vector<uint8_t> cur_view_with_no_pose_mask_;

  /// Scratch bit-vector marking views whose cached resection score is
  /// invalidated by this round's reconstructed-track-id delta. Sized to
  /// `max_view_id + 1` (same as `resection_score_cache_`). Reused across
  /// AddingMissingView() calls; cleared at the start of each call.
  std::vector<uint8_t> dirty_view_mask_;

  /// 2View triangulation method used in the robust triangulation engine
  ETriangulationMethod triangulation_method_ = ETriangulationMethod::DEFAULT;

  resection::SolverType resection_method_ = resection::SolverType::DEFAULT;

  /// If true, the BAs run inside the AddingMissingView() loop use a looser
  /// "fast" preset (see BundleAdjustment in the .cpp). The final BA at the
  /// end of Process() always runs with strict defaults regardless.
  ///
  /// Default controlled by OPENMVG_SFM2_FAST_INTERMEDIATE_BA (defined in
  /// sequential_SfM2.cpp). Set to 0 there for STRICT-on-every-BA (S3
  /// reference behaviour, ~5K more tracks observed on test scene).
  bool b_use_fast_intermediate_ba_ = OPENMVG_SFM2_FAST_INTERMEDIATE_BA;
};

} // namespace sfm
} // namespace openMVG

#endif // OPENMVG_SFM_LOCALIZATION_SEQUENTIAL_SFM_HPP
