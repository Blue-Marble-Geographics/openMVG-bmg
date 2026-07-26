// This file is part of OpenMVG, an Open Multiple View Geometry C++ library.

// Copyright (c) 2015 Pierre MOULON.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef OPENMVG_PATENTED_SIFT_SIFT_DESCRIBER_HPP
#define OPENMVG_PATENTED_SIFT_SIFT_DESCRIBER_HPP

#include "P2PUtils.h"

#include "openMVG/features/image_describer.hpp"
#include "openMVG/features/regions_factory.hpp"
#include "openMVG/image/image_container.hpp"

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <iostream>
#include <numeric>

extern "C" {
#include "nonFree/sift/vl/sift.h"
  extern int hasAVX2;
  extern int hasSSE41;
}

namespace openMVG {
namespace features {

// Bibliography:
// [1] R. Arandjelović, A. Zisserman.
// Three things everyone should know to improve object retrieval. CVPR2012.

  inline void siftDescToUChar(
    const vl_sift_pix descr[128],
    Descriptor<unsigned char, 128>& descriptor,
    bool brootSift = false)
  {
    if (brootSift) {
      /* SSE2 horizontal sum */
      __m128 vSum = _mm_setzero_ps();
      for (int k = 0; k < 128; k += 4)
        vSum = _mm_add_ps(vSum, _mm_loadu_ps(descr + k));
      /* horizontal reduce */
      vSum = _mm_add_ps(vSum, _mm_movehl_ps(vSum, vSum));
      vSum = _mm_add_ss(vSum, _mm_shuffle_ps(vSum, vSum, 1));
      const float sum = _mm_cvtss_f32(vSum);
      const float invSum = (sum > 1e-12f) ? 1.0f / sum : 0.0f;

      /* sqrt(descr[k] * invSum) * 512, 4 at a time */
      const __m128 v512 = _mm_set1_ps(512.0f);
      const __m128 vInvSum = _mm_set1_ps(invSum);
      for (int k = 0; k < 128; k += 4) {
        __m128 v = _mm_mul_ps(_mm_loadu_ps(descr + k), vInvSum);
        v = _mm_sqrt_ps(v);
        v = _mm_mul_ps(v, v512);
        /* clamp to [0, 255] and convert */
        __m128i vi = _mm_cvttps_epi32(v);
        vi = _mm_packs_epi32(vi, vi);     /* 4×int32 → 4×int16 */
        vi = _mm_packus_epi16(vi, vi);    /* 4×int16 → 4×uint8 */
        *(int*)(&descriptor[k]) = _mm_cvtsi128_si32(vi);
      }
    }
    else {
      const __m128 v512 = _mm_set1_ps(512.0f);
      for (int k = 0; k < 128; k += 4) {
        __m128 v = _mm_mul_ps(_mm_loadu_ps(descr + k), v512);
        __m128i vi = _mm_cvttps_epi32(v);
        vi = _mm_packs_epi32(vi, vi);
        vi = _mm_packus_epi16(vi, vi);
        *(int*)(&descriptor[k]) = _mm_cvtsi128_si32(vi);
      }
    }
  }

class SIFT_Image_describer : public Image_describer
{
public:

  using Regions_type = SIFT_Regions;

  struct Params
  {
    Params(
      int first_octave = 0,
      int num_octaves = 5, // Drop one from 6 negligible quality loss.
      int num_scales = 3,
      float edge_threshold = 10.0f,
      float peak_threshold = 0.04f,
      bool root_sift = true
    ):
      _first_octave(first_octave),
      _num_octaves(num_octaves),
      _num_scales(num_scales),
      _edge_threshold(edge_threshold),
      _peak_threshold(peak_threshold),
      _root_sift(root_sift) {}

    template<class Archive>
    inline void serialize( Archive & ar );

    // Parameters
    int _first_octave;      // Use original image, or perform an upscale if == -1
    int _num_octaves;       // Max octaves count
    int _num_scales;        // Scales per octave
    float _edge_threshold;  // Max ratio of Hessian eigenvalues
    float _peak_threshold;  // Min contrast
    bool _root_sift;        // see [1]
  };

  //--
  // Constructor
  //--
  SIFT_Image_describer
  (
    const Params & params = Params(),
    bool bOrientation = true
  ):Image_describer(), _params(params), _bOrientation(bOrientation)
  {
    vl_constructor();
  }

  ~SIFT_Image_describer()
  {
    vl_destructor();
  }

  bool Set_configuration_preset(EDESCRIBER_PRESET preset) override
  {
    switch (preset)
    {
    case NORMAL_PRESET:
      _params._peak_threshold = 0.04f;
    break;
    case HIGH_PRESET:
      _params._peak_threshold = 0.01f;
    break;
    case ULTRA_PRESET:
      _params._peak_threshold = 0.01f;
      _params._first_octave = 0; // was -1; saves 4× memory on first octave
      break;
    default:
      return false;
    }
    return true;
  }

  /**
  @brief Detect regions on the image and compute their attributes (description)
  @param image Image.
  @param mask 8-bit gray image for keypoint filtering (optional).
     Non-zero values depict the region of interest.
  @return regions The detected regions and attributes (the caller must delete the allocated data)
  */
  std::unique_ptr<Regions> Describe(
    const image::Image<unsigned char>& image,
    const image::Image<unsigned char>* mask = nullptr
  ) override
  {
    return DescribeSIFT(image, mask);
  }

  /**
  @brief Detect regions on the image and compute their attributes (description)
  @param image Image.
  @param mask 8-bit gray image for keypoint filtering (optional).
     Non-zero values depict the region of interest.
  @return regions The detected regions and attributes (the caller must delete the allocated data)
  */
  std::unique_ptr<Regions_type> DescribeSIFT(
      const image::Image<unsigned char>& image,
      const image::Image<unsigned char>* mask = nullptr
  )
  {
    const int w = image.Width(), h = image.Height();
    //Convert to float
    const image::Image<float> If(image.GetMat().cast<float>());

    // Optional runtime override of the first octave (o_min), for A/B testing
    // without recompiling. Default is unchanged: uses _params._first_octave
    // (normally 0 = no upscale, the fast path). Set
    //   OPENMVG_SIFT_FIRST_OCTAVE=-1  -> re-enable the 2x upscale octave
    //        (more fine-scale keypoints, ~2-3x slower, 4x first-octave memory)
    //   OPENMVG_SIFT_FIRST_OCTAVE=0   -> force no upscale (fast)
    // Read once; thread-safe under the per-image parallelism (C++11 static init).
    static const int s_first_octave_override = []() -> int {
      const char* v = std::getenv("OPENMVG_SIFT_FIRST_OCTAVE");
      return v ? std::atoi(v) : INT_MIN; // INT_MIN => not set
    }();
    const int first_octave = (s_first_octave_override != INT_MIN)
      ? s_first_octave_override
      : _params._first_octave;

    // Optional GLOBAL override of the peak threshold (min contrast), applied to
    // every image regardless of preset -- set it once in the environment, no
    // per-dataset tuning needed. Higher value -> fewer low-contrast keypoints
    // -> less descriptor work -> faster (mild recall trade). Unset => preset
    // value. Read once; thread-safe under the per-image parallelism.
    static const float s_peak_thresh_override = []() -> float {
      const char* v = std::getenv("OPENMVG_SIFT_PEAK_THRESHOLD");
      return v ? static_cast<float>(std::atof(v)) : -1.0f; // <0 => not set
    }();
    const float peak_threshold = (s_peak_thresh_override >= 0.0f)
      ? s_peak_thresh_override
      : _params._peak_threshold;

    VlSiftFilt *filt = vl_sift_new(w, h,
      _params._num_octaves, _params._num_scales, first_octave);
    if (_params._edge_threshold >= 0)
      vl_sift_set_edge_thresh(filt, _params._edge_threshold);
    if (peak_threshold >= 0)
      vl_sift_set_peak_thresh(filt, 255*peak_threshold/_params._num_scales);

    Descriptor<vl_sift_pix, 128> descr;
    Descriptor<unsigned char, 128> descriptor;

    // Process SIFT computation
    vl_sift_process_first_octave(filt, If.data());

    // Build alias to cached data
    auto regions = std::unique_ptr<Regions_type>(new Regions_type);

    // reserve some memory for faster keypoint saving
    const size_t estimatedKeypoints = (size_t)(w * h) / 400; // ~1 keypoint per 20x20 block
    regions->Features().reserve(estimatedKeypoints);
    regions->Descriptors().reserve(estimatedKeypoints);

    while (true) {
      vl_sift_detect(filt);

      VlSiftKeypoint const *keys  = vl_sift_get_keypoints(filt);
      const int nkeys = vl_sift_get_nkeypoints(filt);

#if 0 // Now gradient buffer free
      // Update gradient before launching parallel extraction
      vl_sift_update_gradient(filt);
#endif

      for (int i = 0; i < nkeys; ++i) {

        // Feature masking
        if (mask)
        {
          const image::Image<unsigned char> & maskIma = *mask;
          if (maskIma(keys[i].y, keys[i].x) == 0)
            continue;
        }

        double angles[4];
        int nangles = 1; // by default (1 upright feature)
        if (_bOrientation)
        { // compute from 1 to 4 orientations
          nangles = vl_sift_calc_keypoint_orientations(filt, angles, keys + i);
        }
        else
        {
          angles[0] = 0.0;
        }

        // Cache keypoint fields once — avoid re-reading the struct
        // through a pointer on each orientation iteration
        const float kx = keys[i].x;
        const float ky = keys[i].y;
        const float ksig = keys[i].sigma;

        for (int q=0 ; q < nangles ; ++q) {
          vl_sift_calc_keypoint_descriptor(filt, &descr[0], keys + i, angles[q]);

          siftDescToUChar(&descr[0], descriptor, _params._root_sift);

          regions->Descriptors().push_back(descriptor);
          regions->Features().emplace_back(kx, ky, ksig, static_cast<float>(angles[q]));
        }
      }
      if (vl_sift_process_next_octave(filt))
        break; // Last octave
    }
    vl_sift_delete(filt);

    return regions;
  }

  std::unique_ptr<Regions> Allocate() const override
  {
    return std::unique_ptr<Regions_type>(new Regions_type);
  }

  template<class Archive>
  inline void serialize( Archive & ar );

private:
  Params _params;
  bool _bOrientation;
};

} // namespace features
} // namespace openMVG

#endif // OPENMVG_PATENTED_SIFT_SIFT_DESCRIBER_HPP
