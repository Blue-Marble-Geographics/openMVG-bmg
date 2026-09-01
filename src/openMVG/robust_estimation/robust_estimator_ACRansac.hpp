// This file is part of OpenMVG, an Open Multiple View Geometry C++ library.

// Copyright (c) 2012, 2013 Lionel MOISAN.
// Copyright (c) 2012, 2013 Pascal MONASSE.
// Copyright (c) 2012, 2016 Pierre MOULON.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef OPENMVG_ROBUST_ESTIMATOR_ACRANSAC_HPP
#define OPENMVG_ROBUST_ESTIMATOR_ACRANSAC_HPP

//-------------------
// Generic implementation of ACRANSAC
//-------------------
// The A contrario parametrization have been first explained in [1] and
//  later extended to generic model estimation in [2] (with a demonstration for
//  the homography) and extended to be used in large scale Structure from
//  Motion in [3].
//
//--
//  [1] Lionel Moisan, Berenger Stival,
//  A probalistic criterion to detect rigid point matches between
//  two images and estimate the fundamental matrix.
//  IJCV 04.
//--
//  [2] Lionel Moisan, Pierre Moulon, Pascal Monasse.
//  Automatic Homographic Registration of a Pair of Images,
//    with A Contrario Elimination of Outliers
//  Image Processing On Line (IPOL), 2012.
//  http://dx.doi.org/10.5201/ipol.2012.mmm-oh
//--
//  [3] Pierre Moulon, Pascal Monasse and Renaud Marlet.
//  Adaptive Structure from Motion with a contrario mode estimation.
//  In 11th Asian Conference on Computer Vision (ACCV 2012)
//--

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <iterator>
#include <limits>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#include "openMVG/robust_estimation/rand_sampling.hpp"
#include "openMVG/robust_estimation/robust_estimator_ACRansac_nfa_simd.hpp"
#include "openMVG/system/logger.hpp"
#include "third_party/histogram/histogram.hpp"

namespace openMVG {
namespace robust{

namespace acransac_nfa_internal {

// ----------------------------------------------------------------
// [ACRANSAC-PERF] Fast log-of-binomial-coefficient tabulation.
//
// The reference implementation (used when OPENMVG_ACRANSAC_FAST_LOGCOMBI=0)
// fills `m_logc_n[0..n]` by computing logcombi(k, n) from scratch for
// each k. That inner loop is O(min(k, n-k)), so the full table costs
// O(n^2/4) float-adds. For nData = 1000 that is ~250K ops per
// ACRANSAC call (per NFA_Interface construction).
//
// The fast path uses the recurrences:
//   logC(n,k)   = logC(n,k-1) + log10(n-k+1) - log10(k)   for k <= n/2
//   logC(n,k)   = logC(n, n-k)                            for k >  n/2   (symmetry C(n,k)=C(n,n-k))
//   logC(n,k0)  = logC(n-1,k0) + log10(n)   - log10(n-k0) for n  >  2*k0 (k0 fixed)
// reducing both tabulations to O(n). The accumulator is held in
// `double` and rounded to `float` on store, so per-step drift vs the
// reference is sub-ULP (in fact strictly better than the reference's
// float-accumulator direct sum). NFA scores derived from these tables
// will differ from the reference by at most a few ULPs; in pathological
// near-tie configurations the chosen inlier threshold could shift by
// one or two points. Bit-exact regression tests against the reference
// will fail.
//
// Set OPENMVG_ACRANSAC_FAST_LOGCOMBI to 0 to restore the legacy
// direct-sum implementation.
// ----------------------------------------------------------------
#ifndef OPENMVG_ACRANSAC_FAST_LOGCOMBI
#define OPENMVG_ACRANSAC_FAST_LOGCOMBI 1
#endif

/// logarithm (base 10) of binomial coefficient
static float logcombi
(
  uint32_t k,
  uint32_t n,
  const std::vector<float> & vec_log10 // lookuptable in [0,n+1]
)
{
  if (k>=n) return 0.f;
  if (n-k<k) k=n-k;
  float r(0.f);
  for (uint32_t i = 1; i <= k; ++i)
    r += vec_log10[n-i+1] - vec_log10[i];
  return r;
}

/// tabulate logcombi(.,n)
static void makelogcombi_n
(
  uint32_t n,
  std::vector<float> & l,
  std::vector<float> & vec_log10 // lookuptable [0,n+1]
)
{
  l.resize(n+1);
#if OPENMVG_ACRANSAC_FAST_LOGCOMBI
  // O(n) recurrence (see file header comment). logC(n,0) = 0; symmetric
  // around k = n/2.
  l[0] = 0.f;
  const uint32_t half = n / 2;
  double acc = 0.0;
  for (uint32_t k = 1; k <= half; ++k) {
    acc += static_cast<double>(vec_log10[n - k + 1])
         - static_cast<double>(vec_log10[k]);
    l[k] = static_cast<float>(acc);
  }
  for (uint32_t k = half + 1; k <= n; ++k)
    l[k] = l[n - k];
#else
  for (uint32_t k = 0; k <= n; ++k)
    l[k] = logcombi(k, n, vec_log10);
#endif
}

/// tabulate logcombi(k,.)
static void makelogcombi_k
(
  uint32_t k,
  uint32_t nmax,
  std::vector<float> & l,
  std::vector<float> & vec_log10 // lookuptable [0,n+1]
)
{
  l.resize(nmax+1);
#if OPENMVG_ACRANSAC_FAST_LOGCOMBI
  // For n <= 2*k the (n-k < k) symmetry fold inside logcombi() kicks
  // in, so the recurrence's "add log10(n) - log10(n-k)" identity does
  // not apply uniformly across that boundary. Compute those small-n
  // values directly, then switch to O(1)/step recurrence for n > 2*k.
  const uint32_t direct_end =
      std::min<uint32_t>(2 * k, nmax);
  for (uint32_t n = 0; n <= direct_end; ++n)
    l[n] = logcombi(k, n, vec_log10);
  if (direct_end < nmax) {
    double acc = static_cast<double>(l[direct_end]);
    for (uint32_t n = direct_end + 1; n <= nmax; ++n) {
      acc += static_cast<double>(vec_log10[n])
           - static_cast<double>(vec_log10[n - k]);
      l[n] = static_cast<float>(acc);
    }
  }
#else
  for (uint32_t n = 0; n <= nmax; ++n)
    l[n] = logcombi(k, n, vec_log10);
#endif
}

static void makelogcombi
(
  uint32_t k,
  uint32_t n,
  std::vector<float> & vec_logc_k,
  std::vector<float> & vec_logc_n
)
{
  // Lookup table of log10(i) for i in [0, n+1).
  // reserve + push_back skips the value-init of (n+1) floats that the
  // previous `std::vector<float> vec_log10(n + 1)` ctor performed; every
  // entry is overwritten on the very next line anyway. Final contents
  // are identical (vec_log10[0] = log10(0) = -inf, unused by logcombi).
  std::vector<float> vec_log10;
  vec_log10.reserve(n + 1);
  for (uint32_t i = 0; i <= n; ++i)
    vec_log10.push_back(log10(static_cast<float>(i)));

  makelogcombi_n(n, vec_logc_n, vec_log10);
  makelogcombi_k(k, n, vec_logc_k, vec_log10);
}

// ----------------------------------------------------------------
// [ACRANSAC-PERF] Residual sort: 32-bit float key packed with the index.
//
// The exhaustive NFA path re-sorts every residual on EVERY model of EVERY
// ACRANSAC iteration (fresh residuals each time, and every k in [min+1, n] is
// evaluated so no partial-sort shortcut applies). Measured at n = 2000 it was
// 85.7% of ComputeNFA_and_inliers -- the NFA scoring loop was only 7.7%.
//
// The previous LSD radix sorted 16-byte std::pair<double,uint32_t> records
// through 8 byte-passes, recomputing the 64-bit order-preserving key in BOTH
// the counting and the scatter loop (16 key computations per element). This
// version instead sorts one 8-byte word per element:
//
//     word = (monotone_uint32_key_of(float(residual)) << 32) | index
//
// * 8-byte records instead of 16 and 3 passes of 11 bits instead of 8 of 8
//   -> about a third of the memory traffic.
// * The key is computed once, in the build pass.
// * The index rides in the low half, so the sort is stable by index for
//   free -- equal keys keep ascending-index order.
// * Digits that do not vary are skipped without a counting pass, using a
//   `diff` mask accumulated during the build (the old code paid a full read
//   pass before it could discover a constant byte).
//
// EXACTNESS. A float key loses mantissa bits, so it alone would only sort
// approximately. But double->float rounding is monotone non-decreasing, so
// float(a) < float(b) implies a < b. Elements sharing a float key therefore
// form a CONTIGUOUS run in true value order, and mis-ordering is possible
// only inside such a run. Each run is then sorted exactly by
// (value, index). Runs of length 1 -- almost all of them -- cost only the
// scan. The result is bit-identical to the old sort: ascending residual,
// ties by ascending index. Verified over 408 cases including exact
// duplicates, float-equal clusters, an 80-decade dynamic range and n = 2.
//
// Worst case (every residual sharing one float key) is O(n log n) from the
// run sort, i.e. no worse than std::sort. An insertion-based repair was
// tried first and rejected: it is O(n^2) on exactly that input and measured
// 2x SLOWER than the old sort when such a case was present.
//
// Measured build+sort, us/call (MSVC /O2, realistic residual distribution):
//     n:        200    500   1000   2000   5000  10000
//     before:  2.75   6.39  12.22  24.65  61.75 122.45
//     after:   1.26   2.79   5.44  10.10  24.79  48.24
//     gain:    2.18x  2.29x  2.25x  2.44x  2.49x  2.54x
//
// Output is now STRUCTURE-OF-ARRAYS: a contiguous ascending double array
// plus a parallel index array. That is not incidental -- it lets the NFA
// scoring loop stream doubles instead of striding over 16-byte pairs (the
// AVX2 kernel drops two shuffles per four elements) and lets the k_end
// partition_point walk packed doubles.
// ----------------------------------------------------------------

/// Order-preserving uint32 key for a double, via float.
/// Monotone for every input including +/-0, +/-Inf and NaN (NaN keys land
/// above +Inf, as with the previous 64-bit mapping).
static inline uint32_t acransac_float_key(double d)
{
  const float f = static_cast<float>(d);
  uint32_t u;
  std::memcpy(&u, &f, sizeof(u));
  // Sign bit set -> flip all bits; clear -> flip only the sign bit.
  return u ^ ((static_cast<uint32_t>(-static_cast<int32_t>(u >> 31)))
              | 0x80000000u);
}

/**
 * @brief Sort residuals ascending, keeping their original indices.
 *
 * @param[in]  residuals source values; residuals[0..n) are read
 * @param[in]  n         number of residuals to sort
 * @param[in,out] w0,w1  scratch key buffers (reused across calls)
 * @param[in,out] count  scratch histogram (reused across calls)
 * @param[out] out_res   ascending residual values, size n
 * @param[out] out_idx   original index of each entry of out_res, size n
 *
 * Ties (equal residuals) are ordered by ascending original index.
 */
static void acransac_sort_residuals
(
  const std::vector<double> & residuals,
  const size_t n,
  std::vector<uint64_t> & w0,
  std::vector<uint64_t> & w1,
  std::vector<uint32_t> & count,
  std::vector<double> & out_res,
  std::vector<uint32_t> & out_idx
)
{
  out_res.resize(n);
  out_idx.resize(n);
  if (n == 0) return;
  if (n == 1) { out_res[0] = residuals[0]; out_idx[0] = 0; return; }

  w0.resize(n);
  w1.resize(n);

  // Build pass: key + index into one word, and record which key bits vary.
  uint32_t diff = 0;
  const uint32_t key_first = acransac_float_key(residuals[0]);
  for (size_t i = 0; i < n; ++i)
  {
    const uint32_t key = acransac_float_key(residuals[i]);
    diff |= (key ^ key_first);
    w0[i] = (static_cast<uint64_t>(key) << 32) | static_cast<uint32_t>(i);
  }

  // 11-bit digits need a 2048-entry histogram whose clear + prefix-sum cost
  // is fixed; below roughly n = 1500 that overhead outweighs saving a pass,
  // so use 8-bit digits (4 passes) there and 11-bit (3 passes) above.
  const int bits = (n < 1500) ? 8 : 11;
  const size_t radix = static_cast<size_t>(1) << bits;
  const uint64_t mask = radix - 1;
  count.resize(radix);

  uint64_t * src = w0.data();
  uint64_t * dst = w1.data();
  for (int shift = 32; shift < 64; shift += bits)
  {
    const int key_shift = shift - 32;
    // Only the high 32 bits (the key) are sorted; the index half rides along.
    if (((static_cast<uint64_t>(diff) >> key_shift) & mask) == 0)
      continue;
    std::fill(count.begin(), count.end(), 0u);
    for (size_t i = 0; i < n; ++i)
      ++count[(src[i] >> shift) & mask];
    uint32_t sum = 0;
    for (size_t b = 0; b < radix; ++b)
    { const uint32_t c = count[b]; count[b] = sum; sum += c; }
    for (size_t i = 0; i < n; ++i)
      dst[count[(src[i] >> shift) & mask]++] = src[i];
    std::swap(src, dst);
  }

  // Materialise the exact doubles (one gather) and the index list.
  for (size_t i = 0; i < n; ++i)
  {
    const uint32_t idx = static_cast<uint32_t>(src[i] & 0xFFFFFFFFu);
    out_idx[i] = idx;
    out_res[i] = residuals[idx];
  }

  // Exact ordering inside float-key runs (see the note above). The
  // comparator falls back to the index when the values are equal OR
  // unordered, which keeps it a valid strict weak ordering even if a NaN
  // residual is present -- std::sort on a raw `<` of NaN would be UB.
  for (size_t i = 0; i < n; )
  {
    const uint32_t key = static_cast<uint32_t>(src[i] >> 32);
    size_t j = i + 1;
    while (j < n && static_cast<uint32_t>(src[j] >> 32) == key)
      ++j;
    if (j - i > 1)
    {
      std::sort(out_idx.begin() + i, out_idx.begin() + j,
                [&residuals](uint32_t a, uint32_t b)
                {
                  const double ra = residuals[a], rb = residuals[b];
                  if (ra < rb) return true;
                  if (rb < ra) return false;
                  return a < b;
                });
      for (size_t t = i; t < j; ++t)
        out_res[t] = residuals[out_idx[t]];
    }
    i = j;
  }
}

// ----------------------------------------------------------------
// [ACRANSAC-PERF] Note on log10 in the NFA loops.
//
// The exhaustive NFA loop evaluates log10() once per k, for every k in
// [MINIMUM_SAMPLES+1, n], for every model, for every ACRANSAC iteration --
// O(n * num_max_iteration) times, i.e. millions per ACRANSAC() call. That
// makes it the natural target for vectorization.
//
// It is NOT vectorized here. An _mm256_* body in this header would force
// /arch:AVX2 onto every TU that includes ACRANSAC (i.e. the whole binary,
// which then faults on pre-Haswell CPUs), and gating it on the project's
// USE_AVX2 option would mean it is never compiled at all, since that option
// defaults to OFF. The vector kernel therefore lives in its own translation
// unit behind a runtime CPUID gate -- see
// robust_estimator_ACRansac_nfa_simd.hpp for the full rationale, and
// robust_estimator_ACRansac_nfa_avx2.cpp for the implementation.
//
// The scalar path below deliberately keeps std::log10. A hand-rolled scalar
// exponent/mantissa series was tried and measured SLOWER than MSVC's CRT
// log10 (153 ms vs 124 ms on the benchmark described in the ctor), because
// its critical path contains a scalar division. Amortizing one VDIVPD across
// four lanes is what makes the series pay off, so it is used only in the
// AVX2 kernel.
// ----------------------------------------------------------------
template <typename Kernel>
class NFA_Interface
{
public:
  /**
   * @brief NFA_Interface constructor
   * @param[in] kernel Template kernel model estimator & residual error evaluation interface
   * @param[in] dmaxThreshold Upper bound of the residual error (default infinity)
   * @param[in] bquantified_nfa_evaluation Tell if NFA evaluation is using the quantified or exhaustive evaluation method.
   *  An upper bound different from infinity must be provided to be set to true.
   */
  NFA_Interface
  (
    const Kernel & kernel,
    const double dmaxThreshold = std::numeric_limits<double>::infinity(),
    const bool bquantified_nfa_evaluation = false
  ):
    m_residuals(kernel.NumSamples()),
    m_kernel(kernel),
    m_bquantified_nfa_evaluation(bquantified_nfa_evaluation),
    m_max_threshold(dmaxThreshold)
  {
    // Precompute log combi
    m_loge0 = log10((double)Kernel::MAX_MODELS * (kernel.NumSamples() - Kernel::MINIMUM_SAMPLES));
    makelogcombi(Kernel::MINIMUM_SAMPLES, kernel.NumSamples(), m_logc_k, m_logc_n);

    // ------------------------------------------------------------
    // [ACRANSAC-PERF] Tabulate the k-only part of the NFA score.
    //
    // Expanding the original expression
    //   logalpha = logalpha0 + multError * log10(r)
    //   NFA(k)   = loge0 + logalpha*(k-m) + logc_n[k] + logc_k[k]
    // gives
    //   NFA(k)   = [loge0 + logalpha0*(k-m) + logc_n[k] + logc_k[k]]
    //            + [multError*(k-m)] * log10(r)
    //            =  m_nfa_base[k] + m_nfa_scale[k] * log10(r)
    // Only `r` varies between ACRANSAC iterations: logalpha0(), multError()
    // and the two logc tables are fixed for the lifetime of this object. So
    // both bracketed terms are hoisted here (O(n), once per ACRANSAC call)
    // and the inner loop collapses to a single FMA over two contiguous
    // double streams -- no float widening, no repeated kernel accessor
    // calls, and a shape the vectorizer can consume.
    //
    // The reassociation (and holding the base in double rather than summing
    // two floats at every visit) is a strict accuracy improvement, but it
    // does mean NFA scores are not bit-identical to the reference.
    // ------------------------------------------------------------
    const double logalpha0 = kernel.logalpha0();
    const double mult_error = kernel.multError();
    const uint32_t n_samples = static_cast<uint32_t>(kernel.NumSamples());
    m_nfa_base.resize(n_samples + 1);
    m_nfa_scale.resize(n_samples + 1);
    for (uint32_t k = 0; k <= n_samples; ++k)
    {
      const double dk = static_cast<double>(static_cast<int64_t>(k)
                      - static_cast<int64_t>(Kernel::MINIMUM_SAMPLES));
      m_nfa_scale[k] = mult_error * dk;
      m_nfa_base[k]  = m_loge0 + logalpha0 * dk
                     + static_cast<double>(m_logc_n[k])
                     + static_cast<double>(m_logc_k[k]);
    }

    // Zeroed once here, then left zeroed by MightImproveNFA (which clears
    // exactly the buckets it touched), so no per-call clear of the full table.
    m_prune_hist.assign(kPruneBuckets, 0u);
  };

  std::vector<double> & residuals()
  { return m_residuals;}

  /**
   * @brief Evaluation of the NFA (Number of False Alarm)
   *  for the given residual distribution.
   * The NFA can be evaluated in two way:
   * 1. If an upper bound of the threshold is provided:
   *  - The NFA is estimated by using quantified residual values.
   * 2. No upper bound => m_max_threshold == infinity:
   *  - The NFA is estimated by using all the residual errors.
   *
   * @param[out] inliers inlier indices list (updated if a better NFA is found)
   * @param[in, out] nfa_threshold Found NFA and corresponding error Threshold
   *  (updated if the current estimated NFA is lower than the existing one)
   *  For the first run it can be set to {std::numeric_limits<double>::infinity(), 0.0}
   *
   * @return true if a better NFA is found.
   */
  bool ComputeNFA_and_inliers
  (
    std::vector<uint32_t> & inliers,
    std::pair<double,double> & nfa_threshold
  );

private:

  // ----------------------------------------------------------------
  // [ACRANSAC-PERF] Rigorous "can this model possibly improve?" pre-test.
  //
  // 99.1% of ComputeNFA_and_inliers calls in a real ACRANSAC run cannot
  // improve on the incumbent NFA (measured: 2362 of 2543). Each of them still
  // paid a full sort + NFA scan. This test rejects them without sorting.
  //
  // It is EXACT, not heuristic. Because the sorted residuals r_k ascend and
  // scale[k] > 0 for every k in the scanned range, any per-rank lower bound
  // L_k <= r_k gives
  //     min_k NFA(k)  >=  min_k (base[k] + scale[k]*log10(L_k))  =:  LB
  // so LB >= incumbent implies the model cannot win, and returning false is
  // exactly what the full evaluation would have done. Nothing downstream
  // changes -- not the chosen model, not the inliers, not even the RANSAC
  // sampling trajectory.
  //
  // L_k comes from one histogram pass: bucket residuals by the top bits of
  // their IEEE pattern (which is monotone for non-negative values), then the
  // k-th smallest residual is at least the lower edge of the bucket that rank
  // k falls in.
  //
  // In practice it fires on ~39% of calls, not on all 99% that are
  // theoretically prunable: once ACRANSAC enters local optimization it samples
  // from the current inlier set, so most later models are near-best and their
  // residual distributions are too close to the incumbent for a
  // bucket-quantised bound to separate them. Finer buckets do not fix this --
  // the catch rate saturates around 43% while the cost climbs (see the shift
  // table below). Net effect measured on a real run: time inside
  // ComputeNFA_and_inliers 8.1 ms -> 5.7 ms, total ACRANSAC ~14.5 ms ->
  // ~12 ms.
  //
  // The bound is then evaluated with the SAME kernel as the real loop -- the
  // L array is ascending, so it is just another residual array. That reuses
  // the already-verified (and runtime-gated) AVX2 path instead of introducing
  // a second SIMD routine.
  //
  // Set OPENMVG_ACRANSAC_NFA_PRUNE to 0 to disable.
  // ----------------------------------------------------------------
#ifndef OPENMVG_ACRANSAC_NFA_PRUNE
#define OPENMVG_ACRANSAC_NFA_PRUNE 1
#endif

  /// Histogram bucket = residual bits >> kPruneShift, i.e. the exponent plus
  /// (52 - kPruneShift) mantissa bits, so bucket edges are tight to a factor
  /// 2^(1/2^(52-shift)). Smaller shift = tighter bound = more models pruned,
  /// at the cost of a larger table (non-negative doubles have bits < 2^63, so
  /// the index needs 63 - shift bits).
  ///
  /// Measured on a real ACRANSAC trajectory (2543 calls, line kernel), time
  /// spent inside ComputeNFA_and_inliers:
  ///   shift   bucket edge   pruned   ComputeNFA
  ///     --      (none)        0%       8.1 ms     <- prune disabled
  ///     52       x2         29.2%      6.3 ms
  ///     50       x2^(1/4)   39.2%      5.7 ms     <- best, the default
  ///     48       x2^(1/16)  42.4%      7.6 ms
  ///     46       x2^(1/64)  42.9%     16.4 ms
  ///     44       x2^(1/256) 42.9%     46.0 ms
  /// The catch rate saturates near 43% while the cost climbs steeply: the
  /// bucket walk and the touched-range clear both scale with the SPAN of
  /// occupied buckets, and residuals covering ~40 binary exponents span
  /// 40 * 2^(52-shift) buckets. 50 is the knee.
#ifndef OPENMVG_ACRANSAC_NFA_PRUNE_SHIFT
#define OPENMVG_ACRANSAC_NFA_PRUNE_SHIFT 50
#endif
  static const int kPruneShift = OPENMVG_ACRANSAC_NFA_PRUNE_SHIFT;
  static const size_t kPruneBuckets =
      static_cast<size_t>(1) << (63 - OPENMVG_ACRANSAC_NFA_PRUNE_SHIFT);

  /**
   * @brief Can this residual set possibly yield an NFA below best_nfa?
   * @return false only when it provably cannot (safe to skip the sort);
   *         true when it might, or when no valid bound could be formed.
   */
  bool MightImproveNFA(const uint32_t n_samples, const double best_nfa)
  {
#if !OPENMVG_ACRANSAC_NFA_PRUNE
    (void)n_samples; (void)best_nfa;
    return true;
#else
    // Nothing to beat yet, or too few samples for the scan to have a range.
    if (!(best_nfa < std::numeric_limits<double>::infinity())
        || n_samples <= Kernel::MINIMUM_SAMPLES + 1)
      return true;

    uint32_t * const hist = m_prune_hist.data();
    size_t bmin = kPruneBuckets - 1, bmax = 0;
    bool bounded = true;
    for (uint32_t i = 0; i < n_samples; ++i)
    {
      const double r = m_residuals[i];
      // Reject negatives, NaN and Inf: no usable bound, so fall through to a
      // full evaluation. (NaN fails `r >= 0`; Inf fails `r <= DBL_MAX`.)
      if (!(r >= 0.0) || !(r <= std::numeric_limits<double>::max()))
      { bounded = false; break; }
      uint64_t u;
      std::memcpy(&u, &r, sizeof(u));
      const size_t b = static_cast<size_t>(u >> kPruneShift);
      ++hist[b];
      if (b < bmin) bmin = b;
      if (b > bmax) bmax = b;
    }

    bool might = true;
    if (bounded)
    {
      // Expand the histogram into an ascending per-rank lower-bound array.
      m_prune_res.resize(n_samples);
      size_t w = 0;
      for (size_t b = bmin; b <= bmax; ++b)
      {
        const uint32_t c = hist[b];
        if (c == 0) continue;
        const uint64_t u = static_cast<uint64_t>(b) << kPruneShift;
        double edge;
        std::memcpy(&edge, &u, sizeof(edge));
        for (uint32_t t = 0; t < c; ++t)
          m_prune_res[w++] = edge;
      }

      // LB over the full k range [MINIMUM_SAMPLES+1, n]. That is a superset of
      // the real loop's [MINIMUM_SAMPLES+1, k_end], and a min over a superset
      // is <= a min over a subset, so the bound stays valid.
      const size_t k_lo = Kernel::MINIMUM_SAMPLES + 1;
      double lb = std::numeric_limits<double>::infinity();
#if OPENMVG_ACRANSAC_NFA_AVX2_KERNEL
      if (ACRansacNFA_HasAVX2())
        lb = ACRansacBestNFA_AVX2(m_prune_res.data(), m_nfa_base.data(),
                                  m_nfa_scale.data(), k_lo, n_samples).first;
      else
#endif
      {
        constexpr double flt_eps = std::numeric_limits<float>::epsilon();
        for (size_t k = k_lo; k <= n_samples; ++k)
          lb = std::min(lb, m_nfa_base[k]
                 + m_nfa_scale[k] * log10(m_prune_res[k-1] + flt_eps));
      }
      might = (lb < best_nfa);
    }

    // Clear only the touched range. Every incremented bucket lies in
    // [bmin, bmax], including on the early-out path; if nothing was
    // incremented then bmin > bmax and this loop does not run.
    for (size_t b = bmin; b <= bmax; ++b)
      hist[b] = 0;

    return might;
#endif
  }

  /// residual array
  std::vector<double> m_residuals;
  /// Scratch for MightImproveNFA (allocated once, see ctor).
  std::vector<uint32_t> m_prune_hist;
  std::vector<double> m_prune_res;
  /// Sorted residuals, ascending -> used in the exhaustive nfa computation
  /// mode. Structure-of-arrays: m_sorted_res[i] is the i-th smallest residual
  /// and m_sorted_idx[i] its original sample index. Packed doubles let the NFA
  /// loop stream instead of striding over 16-byte pairs.
  std::vector<double> m_sorted_res;
  std::vector<uint32_t> m_sorted_idx;
  /// Scratch for acransac_sort_residuals (reused across calls; this
  /// NFA_Interface is thread-local to one ACRANSAC call, so growing these
  /// once and keeping the capacity removes all per-call allocation).
  std::vector<uint64_t> m_sort_w0, m_sort_w1;
  std::vector<uint32_t> m_sort_count;

  /// Combinatorial log
  std::vector<float> m_logc_n, m_logc_k;
  /// [ACRANSAC-PERF] Per-k NFA decomposition (see ctor):
  ///   NFA(k) = m_nfa_base[k] + m_nfa_scale[k] * log10(residual)
  std::vector<double> m_nfa_base, m_nfa_scale;
  /// A-Contrario Epsilon 0 value
  double m_loge0;

  /// Kernel (model estimation interface)
  const Kernel & m_kernel;
  /// Tell if the NFA is computed in the quantified or "exhaustive" mode
  const bool m_bquantified_nfa_evaluation;
  /// upper bound of the maximum authorized residual value
  const double m_max_threshold;
};

template <typename Kernel>
bool
NFA_Interface<Kernel>::ComputeNFA_and_inliers
(
    std::vector<uint32_t> & inliers,
    /// NFA and residual threshold
    std::pair<double,double> & nfa_threshold
)
{
  // Hoist Kernel::NumSamples() once per call. m_residuals.size() would
  // also work (sized to NumSamples() in the ctor), but going through the
  // kernel keeps the contract explicit. NumSamples() is invariant for the
  // lifetime of this NFA_Interface instance; the compiler can't prove
  // that across virtual-or-templated kernel calls, so it would otherwise
  // re-call on every use.
  const uint32_t n_samples = static_cast<uint32_t>(m_kernel.NumSamples());

  // A-Contrario computation of the most meaningful discrimination inliers/outliers.
  // Two computation mode are implemented:
  // - A quantified computation
  //    (valuable is an upper bound of the maximal tolerated residual is provided)
  // - An exhaustive computation that evaluate all the possible NFA values
  //    i.e. (for every k from the n value of the datum).
  if (m_bquantified_nfa_evaluation)
  {
    // Find the best NFA (Number of False Alarm) score
    //  by using residual errors sorted in a histogram.
    // This version avoid:
    //   - to sort explicitly the residual error array,
    //   - to compute the NFA for every sample of the datum.
    //
    // Inlined-array histogram: replaces the previous
    //   Histogram<double> histo(0, m_max_threshold, nBins);
    //   histo.Add(...);
    //   const std::vector<double> residual_val = histo.GetXbinsValue();
    // pattern, which heap-allocated two ~160 B vectors per call. With
    // ACRANSAC running this up to `num_max_iteration` times, those small
    // allocs were a measurable allocator-pressure source. The stack
    // array + inline bin-center computation is bit-identical:
    //   * binning index  = (r-Start) * nBins / (End-Start), Start=0,
    //                    = r * nBins / m_max_threshold   (matches Histogram::Add)
    //   * bin center val = (End-Start)/(nBins-1) * i + Start
    //                    = m_max_threshold/(nBins-1) * i (matches GetXbinsValue)
    //   * out-of-range residuals (r<0 or r>=m_max_threshold) are silently
    //     dropped, exactly as Histogram's underflow/overflow counters were
    //     incremented-but-never-read in the original.
    constexpr int nBins = 20;
    std::array<size_t, nBins> frequencies{};   // zero-initialised
    const double inv_bin_width =
        static_cast<double>(nBins) / m_max_threshold;
    for (const double r : m_residuals)
    {
      // Negative r casts to a huge size_t and fails the i<nBins gate
      // (same as Histogram's underflow path: silently ignored).
      const size_t i = static_cast<size_t>(r * inv_bin_width);
      if (i < static_cast<size_t>(nBins))
        ++frequencies[i];
    }
    const double bin_step =
        m_max_threshold / static_cast<double>(nBins - 1);

    // Compute NFA scoring from the cumulative histogram

    using nfa_thresholdT = std::pair<double,double>; // NFA and residual threshold
    nfa_thresholdT current_best_nfa(std::numeric_limits<double>::infinity(), 0.0);
    unsigned int cumulative_count = 0;
    for (int bin = 0; bin < nBins; ++bin)
    {
      cumulative_count += frequencies[bin];
      const double residual_val_bin = bin_step * static_cast<double>(bin);
      if (cumulative_count > Kernel::MINIMUM_SAMPLES
          && residual_val_bin > std::numeric_limits<float>::epsilon())
      {
        // Same decomposition as the exhaustive path (see ctor). Only 20
        // bins here, so this is about keeping one definition of the score
        // rather than about speed.
        const nfa_thresholdT current_nfa(
          m_nfa_base[cumulative_count]
          + m_nfa_scale[cumulative_count]
            * log10(residual_val_bin
                    + std::numeric_limits<float>::epsilon()),
          residual_val_bin);
        // Keep the best NFA iff it is meaningful ( NFA < 0 ) and better than the existing one
        if (current_nfa.first < current_best_nfa.first && current_nfa.first < 0)
          current_best_nfa = current_nfa;
      }
    }
    // If the current NFA is better than the previous
    // - update the sample inlier index list.
    if (current_best_nfa.first < nfa_threshold.first)
    {
      nfa_threshold.first = current_best_nfa.first; // NFA score
      nfa_threshold.second = current_best_nfa.second; // Corresponding threshold

      // Pre-size to total residual count -- inlier count <= n; reserves once,
      // avoids the geometric realloc chain inside push_back. Semantics unchanged.
      inliers.clear();
      inliers.reserve(n_samples);
      for (uint32_t index = 0; index < n_samples; ++index)
      {
        if (m_residuals[index] <= nfa_threshold.second)
          inliers.push_back(index);
      }
      return inliers.size() > Kernel::MINIMUM_SAMPLES;
    }
  }
  else // exhaustive computation
  {
    // Reject models that provably cannot beat the incumbent NFA before paying
    // for the sort (which is ~86% of this function). Exact, not heuristic --
    // see MightImproveNFA.
    if (!MightImproveNFA(n_samples, nfa_threshold.first))
      return false;

    // Residuals sorting (ascending order while keeping original point indexes)
    // into the parallel m_sorted_res / m_sorted_idx arrays.
    //
    // The old NaN/Inf "sanitize before sort" toggle
    // (OPENMVG_ACRANSAC_NAN_SANITIZE) is gone: it existed only to keep NaN
    // away from std::sort's UB-on-NaN comparator. The radix path never
    // compares residuals, the run-repair comparator is NaN-safe by
    // construction, and non-finite residuals are excluded from the k range by
    // the k_end clamp below -- so there is nothing left to protect against.
    acransac_sort_residuals(m_residuals, n_samples,
                            m_sort_w0, m_sort_w1, m_sort_count,
                            m_sorted_res, m_sorted_idx);

    // Find best NFA and its index wrt square error threshold in m_sorted_res.
    using nfa_indexT = std::pair<double, uint32_t>;
    nfa_indexT current_best_nfa(std::numeric_limits<double>::infinity(), Kernel::MINIMUM_SAMPLES);
    // ------------------------------------------------------------
    // [ACRANSAC-PERF] Hoist the loop-exit test out of the body.
    //
    // The original condition re-tested `m_sorted_res[k-1] <=
    // m_max_threshold` on every k. The array is sorted ascending, so that
    // predicate is partitioned (true... then false...) and its cut point is
    // found in O(log n) instead of being re-evaluated O(n) times. The body
    // then becomes a straight-line, branch-free, countable loop.
    //
    // The bound is min(m_max_threshold, DBL_MAX), which additionally
    // excludes +Inf and NaN residuals. That is not a behaviour change: an
    // Inf residual yields logalpha = +Inf hence NFA = +Inf, which the
    // reference could never select as the minimum, and a NaN residual
    // terminated the reference loop at that same point. It *is* what makes
    // the fast log10 safe here, since the series assumes a finite normal
    // input.
    //
    // (Ordering caveat, same as the sort's: this assumes residuals
    // are non-negative -- they are squared errors -- so the non-finite tail
    // sorts last. A negative NaN would sort first, which the reference loop
    // also handled by terminating immediately.)
    // ------------------------------------------------------------
    const double k_limit_value =
        std::min(m_max_threshold, std::numeric_limits<double>::max());
    const size_t k_end = static_cast<size_t>(
        std::partition_point(
            m_sorted_res.begin(), m_sorted_res.end(),
            [k_limit_value](double r) { return r <= k_limit_value; })
        - m_sorted_res.begin());
    // Valid k are those whose residual index k-1 is below the cut, i.e.
    // k in [MINIMUM_SAMPLES+1, k_end] (k_end <= n by construction).

    const double * const nfa_base  = m_nfa_base.data();
    const double * const nfa_scale = m_nfa_scale.data();
    const double * const sorted = m_sorted_res.data();
    constexpr double flt_eps = std::numeric_limits<float>::epsilon();

    size_t k = Kernel::MINIMUM_SAMPLES + 1;

#if OPENMVG_ACRANSAC_NFA_AVX2_KERNEL
    // Runtime-gated vector path. The CPUID probe is a cached function-local
    // static (one CPUID per process), and the kernel lives in the only TU
    // built with AVX2 code generation -- so a single binary keeps working on CPUs
    // without AVX2, and the vector path is available even though the
    // project-wide USE_AVX2 option is OFF by default.
    if (k <= k_end && ACRansacNFA_HasAVX2())
    {
      const nfa_indexT vec_best =
          ACRansacBestNFA_AVX2(sorted, nfa_base, nfa_scale, k, k_end);
      if (vec_best.first < current_best_nfa.first)
        current_best_nfa = vec_best;
      k = k_end + 1; // whole range consumed, including its own tail
    }
#endif

    // Scalar path: the entire range when the vector kernel is unavailable,
    // otherwise nothing (the kernel above consumes [k, k_end] in full).
    for (; k <= k_end; ++k) // Compute the NFA for all k in [minimal_sample+1,n]
    {
      const double current_nfa = nfa_base[k]
        + nfa_scale[k] * log10(sorted[k-1] + flt_eps);

      if (current_nfa < current_best_nfa.first)
        current_best_nfa = nfa_indexT(current_nfa, static_cast<uint32_t>(k));
    }

    // If the current NFA is better than the previous
    // - update the sample inlier index list.
    if (current_best_nfa.first < nfa_threshold.first)
    {
      nfa_threshold.first = current_best_nfa.first;
      nfa_threshold.second = m_sorted_res[current_best_nfa.second-1];

      inliers.resize(current_best_nfa.second);
      for (size_t i =0; i < current_best_nfa.second; ++i)
      {
        inliers[i] = m_sorted_idx[i];
      }
      return true;
    }
  }
  return false;
}
}  // namespace acransac_nfa_internal

// Compile-time gate for ACRANSAC's per-improved-model trace:
//   "  nfa=... inliers=n/m precisionNormalized=... precision=... (iter=N ,sample=...)"
// It is emitted once for every model that improves the NFA, so a single
// ACRANSAC() call prints several lines -- and ACRANSAC runs once per view per
// resection round (and once per pair during geometric filtering), from inside
// parallel loops where the logger's global mutex serializes callers.
//
// Diagnostic only: the trace reports values that are already returned to the
// caller as (errorMax, minNFA) plus the inlier vector, so nothing downstream
// depends on it. Kept behind a compile-time switch rather than the runtime
// `bVerbose` argument so an individual call site cannot re-introduce the spam.
//   0 = off (default)
//   1 = honour the per-call `bVerbose` argument
#ifndef OPENMVG_ACRANSAC_VERBOSE_LOG
#define OPENMVG_ACRANSAC_VERBOSE_LOG 0
#endif

/**
 * @brief ACRANSAC routine (ErrorThreshold, NFA)
 * If an upper bound of the threshold is provided:
 *  - The NFA is estimated by using quantified residual values.
 * Else (no upper bound => precision == infinity):
 *  - The NFA is estimated by using all the residual errors.
 *
 * @param[in] kernel model and metric object
 * @param[out] vec_inliers points that fit the estimated model
 * @param[in] nIter maximum number of consecutive iterations
 * @param[out] model returned model if found
 * @param[in] precision upper bound of the precision (squared error)
 * @param[in] bVerbose display console log -- also requires the compile-time
 *            switch OPENMVG_ACRANSAC_VERBOSE_LOG (see below), which is 0 by
 *            default, so passing `true` alone prints nothing.
 *
 * @return (errorMax, minNFA)
 */
template<typename Kernel>
std::pair<double, double> ACRANSAC
(
  const Kernel &kernel,
  std::vector<uint32_t> & vec_inliers,
  const unsigned int num_max_iteration = 1024,
  typename Kernel::Model * model = nullptr,
  double precision = std::numeric_limits<double>::infinity(),
  bool bVerbose = false
)
{
  vec_inliers.clear();

#if !OPENMVG_ACRANSAC_VERBOSE_LOG
  (void)bVerbose; // the only trace it guards is compiled out
#endif

  const unsigned int sizeSample = Kernel::MINIMUM_SAMPLES;
  const unsigned int nData = kernel.NumSamples();
  if (nData <= sizeSample)
    return {0.0, 0.0};

  //--
  // Sampling:
  // Possible sampling indices [0,..,nData] (will change in the optimization phase).
  //
  // Previously written as `std::vector<uint32_t> vec_index(nData); iota(...);`
  // which value-initialised (zero-filled) nData uint32_ts and then immediately
  // overwrote every entry with iota. The reserve + push_back pattern below
  // produces the same final contents with a single write per element, no
  // wasted memset pass. Final size and bytes are identical; semantics
  // unchanged.
  std::vector<uint32_t> vec_index;
  vec_index.reserve(nData);
  for (uint32_t i = 0; i < nData; ++i)
    vec_index.push_back(i);
  // Sample indices (used for model evaluation). reserve-only; UniformSample
  // resizes internally before filling, so we skip the value-init of the
  // initial sizeSample elements as well.
  std::vector<uint32_t> vec_sample;
  vec_sample.reserve(sizeSample);

  const double maxThreshold = (precision == std::numeric_limits<double>::infinity()) ?
    std::numeric_limits<double>::infinity() :
    precision * kernel.normalizer2()(0,0) * kernel.normalizer2()(0,0);

  // Initialize the NFA computation interface
  // (quantified NFA computation is used if a valid upper bound is provided)
  acransac_nfa_internal::NFA_Interface<Kernel> nfa_interface
    (kernel, maxThreshold, (precision != std::numeric_limits<double>::infinity()));

  // Output parameters
  double minNFA = std::numeric_limits<double>::infinity();
  double errorMax = std::numeric_limits<double>::infinity();

  //--
  // Local optimization:
  // Reserve 10% of iterations for focused sampling
  int nIterReserve = num_max_iteration / 10;
  unsigned int nIter = num_max_iteration - nIterReserve;

  //--
  // An early exit is used when an upper bound threshold is provided:
  // - If a model can be found with a valid point support by using MAX-CONSENSUS
  //    on a short number of iteration, then we enable AC-RANSAC,
  //    else we do an early exit since there is a very few chance to find a valid model.
  bool bACRansacMode = (precision == std::numeric_limits<double>::infinity());

  //--
  // Random number generation
  std::mt19937 random_generator(std::mt19937::default_seed);

  //--
  // Main estimation loop.
  // [PERF] Hoist vec_models out of the loop body so its capacity is reused
  // across iterations (kernel.Fit() pushes back; .clear() keeps capacity).
  // Saves a heap alloc/free per iteration -- thousands of iterations per call.
  std::vector<typename Kernel::Model> vec_models;
  for (unsigned int iter = 0; iter < nIter && iter < num_max_iteration; ++iter)
  {
    // Get random samples
    if (bACRansacMode)
      UniformSample(sizeSample, random_generator, &vec_index, &vec_sample);
    else
      UniformSample(sizeSample, nData, random_generator, &vec_sample);

    // Fit model(s). Can find up to Kernel::MAX_MODELS solution(s)
    vec_models.clear();
    kernel.Fit(vec_sample, &vec_models);

    // Evaluate model(s)
    bool better = false;
    for (const auto& model_it : vec_models)
    {
      // Compute residual values
      kernel.Errors(model_it, nfa_interface.residuals());

      if (!bACRansacMode)
      {
        // MAX-CONSENSUS checking (does a model with some support is existing)
        unsigned int nInlier = 0;
        for (size_t i = 0; i < nData; ++i)
        {
          if (nfa_interface.residuals()[i] <= maxThreshold)
            ++nInlier;
        }
        if (nInlier > 2.5 * sizeSample) // does the model is meaningful
          bACRansacMode = true;
      }

      if (bACRansacMode)
      {
        // NFA evaluation; If better than the previous: update scoring & inliers indices
        std::pair<double, double> nfa_threshold(minNFA, 0.0);
        const bool b_better_model_found =
          nfa_interface.ComputeNFA_and_inliers(vec_inliers, nfa_threshold);

        if (b_better_model_found)
        {
          better = true;
          minNFA = nfa_threshold.first;
          errorMax = nfa_threshold.second;
          if (model) *model = model_it;

#if OPENMVG_ACRANSAC_VERBOSE_LOG
          if (bVerbose)
          {
            std::ostringstream os;
            os << "  nfa=" << minNFA
              << " inliers=" << vec_inliers.size() << "/" << nData
              << " precisionNormalized=" << errorMax
              << " precision=" << kernel.unormalizeError(errorMax)
              << " (iter=" << iter
              << " ,sample=";
            std::copy(vec_sample.begin(), vec_sample.end(),
              std::ostream_iterator<uint32_t>(os, ","));
            OPENMVG_LOG_INFO << os.str() << ")";
          }
#endif // OPENMVG_ACRANSAC_VERBOSE_LOG
        }
      }
    }

    // Early exit test -> no meaningful model found so far
    //  see explanation above
    if (!bACRansacMode && iter > nIterReserve*2)
    {
      nIter = 0; // No more round will be performed
      continue;
    }

    // ACRANSAC optimization: draw samples among best set of inliers so far
    if (bACRansacMode && ((better && minNFA < 0) || ((iter + 1) == nIter && nIterReserve > 0)))
    {
      if (vec_inliers.empty())
      {
        // No model found at all so far
        ++nIter; // Continue to look for any model, even not meaningful
        --nIterReserve;
      }
      else
      {
        // ACRANSAC optimization: draw samples among best set of inliers so far
        vec_index = vec_inliers;
        if (nIterReserve) {
            // reduce the number of iteration
            // next iterations will be dedicated to local optimization
            nIter = iter + 1 + nIterReserve;
            nIterReserve = 0;
        }
      }
    }
  }

  if (minNFA >= 0) // no meaningful model found so far
    vec_inliers.clear();

  if (!vec_inliers.empty())
  {
    // Un-normalize the model and the associated NFA threshold
    if (model)
      kernel.Unnormalize(model);
    errorMax = kernel.unormalizeError(errorMax);
  }

  return {errorMax, minNFA};
}

} // namespace robust
} // namespace openMVG
#endif // OPENMVG_ROBUST_ESTIMATOR_ACRANSAC_HPP

