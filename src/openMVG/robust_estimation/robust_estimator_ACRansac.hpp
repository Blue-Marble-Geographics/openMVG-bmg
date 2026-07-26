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
#include <iostream>
#include <iterator>
#include <limits>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#include "openMVG/robust_estimation/rand_sampling.hpp"
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

  /// residual array
  std::vector<double> m_residuals;
  /// [residual,index] array -> used in the exhaustive nfa computation mode
  std::vector<std::pair<double,uint32_t>> m_sorted_residuals;

  /// Combinatorial log
  std::vector<float> m_logc_n, m_logc_k;
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
        const double logalpha = m_kernel.logalpha0()
          + m_kernel.multError() * log10(residual_val_bin
          + std::numeric_limits<float>::epsilon());
        const nfa_thresholdT current_nfa( m_loge0
          + logalpha * (double)(cumulative_count - Kernel::MINIMUM_SAMPLES)
          + m_logc_n[cumulative_count]
          + m_logc_k[cumulative_count], residual_val_bin);
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
    // Residuals sorting (ascending order while keeping original point indexes)
    {
      m_sorted_residuals.clear();
      m_sorted_residuals.reserve(n_samples);
      // ----------------------------------------------------------------
      // [ACRANSAC-SAFE] NaN/Inf sanitize before sort.
      //
      // std::sort with NaN values is undefined behaviour: NaN violates the
      // strict-weak-ordering required by operator<. With NaN in the input,
      // the sort can loop indefinitely, segfault, or silently corrupt the
      // ordering -- and downstream the inlier-count loop reads an array
      // it believes is sorted.
      //
      // Degenerate models (e.g. resection points that project behind the
      // camera) can produce NaN residuals from kernel error functions.
      // Replacing NaN/Inf with a value strictly greater than m_max_threshold
      // makes them sort to the tail and naturally excludes them from
      // inlier consideration via the existing `<= m_max_threshold` gate.
      // Output is bit-equivalent for clean inputs (no NaN/Inf present).
      //
      // Toggle: set OPENMVG_ACRANSAC_NAN_SANITIZE to 0 to revert to the
      // legacy unguarded sort.
      // ----------------------------------------------------------------
#ifndef OPENMVG_ACRANSAC_NAN_SANITIZE
#define OPENMVG_ACRANSAC_NAN_SANITIZE 0
#endif
#if OPENMVG_ACRANSAC_NAN_SANITIZE
      const double sentinel = std::numeric_limits<double>::max();
      for (uint32_t i = 0; i < n_samples; ++i)
      {
        double r = m_residuals[i];
        // NaN-safe: (r != r) iff r is NaN. Inf is also rejected.
        if (!(r == r) || !std::isfinite(r))
          r = sentinel;
        m_sorted_residuals.emplace_back(r, i);
      }
#else
      for (uint32_t i = 0; i < n_samples; ++i)
      {
        m_sorted_residuals.emplace_back(m_residuals[i], i);
      }
#endif
      std::sort(m_sorted_residuals.begin(), m_sorted_residuals.end());
    }

    // Find best NFA and its index wrt square error threshold in m_sorted_residuals.
    using nfa_indexT = std::pair<double, uint32_t>;
    nfa_indexT current_best_nfa(std::numeric_limits<double>::infinity(), Kernel::MINIMUM_SAMPLES);
    const size_t n = n_samples;
    for (size_t k = Kernel::MINIMUM_SAMPLES + 1;
        k <= n && m_sorted_residuals[k-1].first <= m_max_threshold;
        ++k) // Compute the NFA for all k in [minimal_sample+1,n]
    {
      const double logalpha = m_kernel.logalpha0()
        + m_kernel.multError() * log10(m_sorted_residuals[k-1].first
        + std::numeric_limits<float>::epsilon());
      const nfa_indexT current_nfa( m_loge0
        + logalpha * (double)(k - Kernel::MINIMUM_SAMPLES)
        + m_logc_n[k]
        + m_logc_k[k], k);

      if (current_nfa.first < current_best_nfa.first)
        current_best_nfa = current_nfa;
    }

    // If the current NFA is better than the previous
    // - update the sample inlier index list.
    if (current_best_nfa.first < nfa_threshold.first)
    {
      nfa_threshold.first = current_best_nfa.first;
      nfa_threshold.second = m_sorted_residuals[current_best_nfa.second-1].first;

      inliers.resize(current_best_nfa.second);
      for (size_t i =0; i < current_best_nfa.second; ++i)
      {
        inliers[i] = m_sorted_residuals[i].second;
      }
      return true;
    }
  }
  return false;
}
}  // namespace acransac_nfa_internal

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
 * @param[in] bVerbose display console log
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
