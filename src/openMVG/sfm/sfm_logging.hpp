// This file is part of OpenMVG, an Open Multiple View Geometry C++ library.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef OPENMVG_SFM_SFM_LOGGING_HPP
#define OPENMVG_SFM_SFM_LOGGING_HPP

#include "openMVG/system/logger.hpp"

// ---------------------------------------------------------------------------
// SfM module logging switches.
//
// The SfM engines and the BA solver carry a large amount of *diagnostic*
// logging: per-resection blocks, per-BA timing lines, track-length histograms,
// seed-candidate traces. On a several-hundred-image run these produce tens of
// thousands of lines, and because OPENMVG_LOG_* serialises on a global logger
// the per-resection ones are emitted from inside an `#pragma omp parallel for`
// -- so they cost wall-clock as well as noise.
//
// Everything gated by the switches below is DIAGNOSTIC ONLY: no gated
// statement participates in control flow or mutates reconstruction state, so
// flipping these does not change the result of a run. The lines that report
// genuine pipeline outcomes (initialization status, the end-of-run statistics
// block, warnings about degraded input, and every error) stay unconditional.
//
// Enable at build time, e.g.:
//   -DOPENMVG_SFM_VERBOSE_LOGGING=1          (everything below)
//   -DOPENMVG_SFM_VERBOSE_RESECTION=1        (just the per-resection blocks)
//
// Related, and deliberately separate because it is not purely additive
// logging (it also threads a per-view record through AddingMissingView and
// takes an `omp critical` inside the parallel resection loop):
//   OPENMVG_SFM_PIPELINE_DIAG -- see sequential_SfM2.hpp. Also 0 by default.
// ---------------------------------------------------------------------------

/// Master switch. Every category below defaults to this value.
#ifndef OPENMVG_SFM_VERBOSE_LOGGING
#define OPENMVG_SFM_VERBOSE_LOGGING 0
#endif

/// Bundle-adjustment diagnostics: the `[BA-DIAG:*]`, `[BA-PERF]` and
/// `[BA-TUNE]` lines in sfm_data_BA_ceres.cpp, the `[ROBUST-BA iter ...]`
/// trace of the final robust-BA loop, and the pose-prior fitting statistics.
/// Note that the `[BA-DIAG:*]` blocks additionally require the pre-existing
/// runtime flag `BA_Ceres_options::bVerbose_`; this switch is a compile-time
/// gate on top of it, so the traversals they perform vanish when it is 0.
#ifndef OPENMVG_SFM_VERBOSE_BA
#define OPENMVG_SFM_VERBOSE_BA OPENMVG_SFM_VERBOSE_LOGGING
#endif

/// Per-resection logging: the "Robust Resection of camera index" block in the
/// SfM2 resection loop and the "Robust Resection statistics" block in
/// SfM_Localizer::Localize. These fire once per candidate view per resection
/// round -- the bulk of the output volume on large scenes.
#ifndef OPENMVG_SFM_VERBOSE_RESECTION
#define OPENMVG_SFM_VERBOSE_RESECTION OPENMVG_SFM_VERBOSE_LOGGING
#endif

/// Scene/track statistics dumps: track-length histograms, `[TRACKS-CUT]`
/// counters, `[SEED-TRI]` breakdowns, `[MATCH-DIAG]` match-graph analysis and
/// the per-candidate seed-pair trace. Several of these do non-trivial work
/// (extra traversals, sorts) purely to build the message, so the computation
/// is compiled out with the log.
#ifndef OPENMVG_SFM_VERBOSE_STATS
#define OPENMVG_SFM_VERBOSE_STATS OPENMVG_SFM_VERBOSE_LOGGING
#endif

// Stream-style loggers. These expand to the OPENMVG_LOG_*_IF form, whose
// condition folds to a compile-time constant: with the switch at 0 the
// message is never built and the streamed expressions are never evaluated,
// while still being type-checked so they cannot bit-rot.
#define OPENMVG_SFM_LOG_BA_INFO           OPENMVG_LOG_INFO_IF(OPENMVG_SFM_VERBOSE_BA)
#define OPENMVG_SFM_LOG_BA_WARNING        OPENMVG_LOG_WARNING_IF(OPENMVG_SFM_VERBOSE_BA)

#define OPENMVG_SFM_LOG_RESECTION_INFO    OPENMVG_LOG_INFO_IF(OPENMVG_SFM_VERBOSE_RESECTION)

#define OPENMVG_SFM_LOG_STATS_INFO        OPENMVG_LOG_INFO_IF(OPENMVG_SFM_VERBOSE_STATS)
#define OPENMVG_SFM_LOG_STATS_WARNING     OPENMVG_LOG_WARNING_IF(OPENMVG_SFM_VERBOSE_STATS)

#endif // OPENMVG_SFM_SFM_LOGGING_HPP
