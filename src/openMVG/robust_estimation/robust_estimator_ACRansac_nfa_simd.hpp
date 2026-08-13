// This file is part of OpenMVG, an Open Multiple View Geometry C++ library.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef OPENMVG_ROBUST_ESTIMATOR_ACRANSAC_NFA_SIMD_HPP
#define OPENMVG_ROBUST_ESTIMATOR_ACRANSAC_NFA_SIMD_HPP

#include <cstddef>
#include <cstdint>
#include <utility>

// ----------------------------------------------------------------
// [ACRANSAC-PERF] Runtime-dispatched SIMD kernel for the exhaustive NFA loop.
//
// WHY THIS IS NOT IMPLEMENTED IN THE ACRANSAC HEADER
//
// robust_estimator_ACRansac.hpp is included by ~17 translation units across
// openMVG_sfm, openMVG_matching_image_collection and the samples. Putting
// _mm256_* intrinsics in that header would force every one of those TUs --
// hence effectively the whole binary -- to be built with /arch:AVX2 (or
// -mavx2), which makes the result crash with an illegal instruction on any
// pre-Haswell CPU. Gating it on the project's compile-time USE_AVX2 option
// does not help either: that option is OFF in the default configuration, so
// the vector path would simply never be compiled.
//
// So the AVX2 body lives in exactly one TU
// (robust_estimator_ACRansac_nfa_avx2.cpp) which is the only file compiled
// with AVX2 code generation enabled, and it is reached through a CPUID check
// at run time. A single binary therefore runs the scalar path on old CPUs
// and the vector path on AVX2 CPUs, independently of USE_AVX2.
//
// This mirrors the existing convention in this tree: the SIFT AVX2 helpers
// are isolated in nonFree/sift/vl/siftAVX2Support.c and reached through a
// runtime `hasAVX2` flag.
//
// The kernel below is deliberately TYPE-ERASED (no Kernel template
// parameter) precisely so that it can live outside the template header.
// ----------------------------------------------------------------

// An AVX2 kernel only exists on x86 / x86-64. On any other architecture the
// declarations disappear and ACRANSAC uses the scalar loop unconditionally.
//
// Define OPENMVG_ACRANSAC_NFA_AVX2_KERNEL=0 project-wide to remove the vector
// path entirely (the kernel TU then compiles to nothing and ACRANSAC always
// takes the scalar loop). Useful for A/B timing, or for bisecting a suspected
// numerical difference. Must be defined consistently for every TU, since it
// controls both the declaration and the definition.
#ifndef OPENMVG_ACRANSAC_NFA_AVX2_KERNEL
#if defined(_M_X64) || defined(__x86_64__) || defined(_M_IX86) \
    || defined(__i386__)
#define OPENMVG_ACRANSAC_NFA_AVX2_KERNEL 1
#else
#define OPENMVG_ACRANSAC_NFA_AVX2_KERNEL 0
#endif
#endif

namespace openMVG {
namespace robust {
namespace acransac_nfa_internal {

#if OPENMVG_ACRANSAC_NFA_AVX2_KERNEL

/**
 * @brief Does this CPU support the instruction set used by
 *  ACRansacBestNFA_AVX2 (AVX2 + FMA3)?
 *
 * CPUID is executed once and the answer cached. Defined in
 * robust_estimator_ACRansac_nfa_simd.cpp, which MUST be compiled with the
 * project's baseline flags -- it runs before we know the CPU is capable, so
 * it may not itself contain AVX2 code.
 */
bool ACRansacNFA_HasAVX2();

/**
 * @brief Vectorized argmin of the ACRANSAC NFA score over a range of k.
 *
 * Evaluates, for every k in [k_begin, k_end] (both inclusive):
 *
 *   NFA(k) = nfa_base[k] + nfa_scale[k] * log10(sorted_residuals[k-1]
 *                                               + FLT_EPSILON)
 *
 * and returns {smallest NFA, its k}. Exact ties resolve to the smallest k,
 * matching the reference scalar loop's first-minimum-wins behaviour.
 *
 * @param[in] sorted_residuals residuals in ascending order, packed
 *  contiguously; element k-1 is read for each k. Every residual in the
 *  scanned range must be finite, normal and >= 0 (the caller guarantees this
 *  by clamping k_end, see ComputeNFA_and_inliers).
 * @param[in] nfa_base  per-k additive term,       indices [k_begin, k_end]
 * @param[in] nfa_scale per-k multiplicative term, indices [k_begin, k_end]
 * @param[in] k_begin first k to evaluate
 * @param[in] k_end   last k to evaluate (inclusive)
 *
 * @return {best NFA, best k}, or {+infinity, k_begin} if k_begin > k_end.
 *
 * @note Only call when ACRansacNFA_HasAVX2() is true.
 */
std::pair<double, uint32_t> ACRansacBestNFA_AVX2
(
  const double * sorted_residuals,
  const double * nfa_base,
  const double * nfa_scale,
  std::size_t k_begin,
  std::size_t k_end
);

#endif // OPENMVG_ACRANSAC_NFA_AVX2_KERNEL

} // namespace acransac_nfa_internal
} // namespace robust
} // namespace openMVG

#endif // OPENMVG_ROBUST_ESTIMATOR_ACRANSAC_NFA_SIMD_HPP
