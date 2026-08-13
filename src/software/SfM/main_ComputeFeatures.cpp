// This file is part of OpenMVG, an Open Multiple View Geometry C++ library.

// Copyright (c) 2012, 2013 Pierre MOULON.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

// The <cereal/archives> headers are special and must be included first.
#include <cereal/archives/json.hpp>

#include "P2PUtils.h"

/* Have CF process a single image for testing */
#define TEST_CF_SINGLE_IMAGE         (0)
#define TEST_CF_NOTHREADING          (0)
#define TEST_CF_MAX_IMAGES           (0)

#include "openMVG/features/akaze/image_describer_akaze_io.hpp"

#include "openMVG/features/sift/SIFT_Anatomy_Image_Describer_io.hpp"
#include "openMVG/image/image_io.hpp"
#include "openMVG/features/regions_factory_io.hpp"
#include "openMVG/sfm/sfm_data.hpp"
#include "openMVG/sfm/sfm_data_io.hpp"
#include "openMVG/system/logger.hpp"
#include "openMVG/system/loggerprogress.hpp"
#include "openMVG/system/timer.hpp"

#include "third_party/cmdLine/cmdLine.h"
#include "third_party/stlplus3/filesystemSimplified/file_system.hpp"

#include "nonFree/sift/SIFT_describer_io.hpp"

#include <cereal/details/helpers.hpp>

#include <atomic>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#ifdef OPENMVG_USE_OPENMP
#include <omp.h>
#endif

#include <windows.h>

using namespace openMVG;
using namespace openMVG::image;
using namespace openMVG::features;
using namespace openMVG::sfm;

features::EDESCRIBER_PRESET stringToEnum(const std::string & sPreset)
{
  features::EDESCRIBER_PRESET preset;
  if (sPreset == "NORMAL")
    preset = features::NORMAL_PRESET;
  else
  if (sPreset == "HIGH")
    preset = features::HIGH_PRESET;
  else
  if (sPreset == "ULTRA")
    preset = features::ULTRA_PRESET;
  else
    preset = features::EDESCRIBER_PRESET(-1);
  return preset;
}

extern "C" int hasAVX2 = -1;
extern "C" int hasSSE41 = -1;

#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif

static int
CpuHasAVX2()
{
#if defined(_MSC_VER)
  int cpuInfo[4];
  __cpuid(cpuInfo, 0);
  if (cpuInfo[0] < 7) return 0;

  __cpuidex(cpuInfo, 7, 0);
  return (cpuInfo[1] & (1 << 5)) != 0;
#else
  unsigned eax, ebx, ecx, edx;
  if (!__get_cpuid_max(0, 0) || __get_cpuid_max(0, 0) < 7)
    return 0;

  __cpuid_count(7, 0, eax, ebx, ecx, edx);
  return (ebx & (1 << 5)) != 0;
#endif
}

static int
CpuHasSSE41(void)
{
  int info[4];
  __cpuid(info, 1);
  return (info[2] & (1 << 19)) != 0;  /* ECX bit 19 = SSE4.1 */
}

// Returns one affinity mask per physical core (a single logical processor per
// core, i.e. hyperthread siblings are dropped). Single processor-group only
// (<= 64 logical CPUs), which covers typical single-socket workstations.
static std::vector<DWORD_PTR>
GetPhysicalCoreMasks()
{
  std::vector<DWORD_PTR> masks;
  DWORD len = 0;
  GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
  if (len == 0)
    return masks;

  std::vector<char> buf(len);
  auto * first = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data());
  if (!GetLogicalProcessorInformationEx(RelationProcessorCore, first, &len))
    return masks;

  char * ptr = buf.data();
  while (ptr < buf.data() + len)
  {
    auto * cur = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(ptr);
    if (cur->Relationship == RelationProcessorCore)
    {
      // GroupMask[0].Mask holds all logical procs (HT siblings) of this core;
      // keep only the lowest set bit to pin to a single logical processor.
      const DWORD_PTR full = cur->Processor.GroupMask[0].Mask;
      masks.push_back(full & (~full + 1));
    }
    ptr += cur->Size;
  }
  return masks;
}

/// - Compute view image description (feature & descriptor extraction)
/// - Export computed data
int main(int argc, char **argv)
{
  hasAVX2 = CpuHasAVX2();
  hasSSE41 = CpuHasSSE41();

  CmdLine cmd;

  std::string sSfM_Data_Filename;
  std::string sOutDir = "";
  bool bUpRight = false;
  std::string sImage_Describer_Method = "SIFT";
  bool bForce = false;
  std::string sFeaturePreset = "";
#ifdef OPENMVG_USE_OPENMP
  int iNumThreads = 0;
#endif

  // required
  cmd.add( make_option('i', sSfM_Data_Filename, "input_file") );
  cmd.add( make_option('o', sOutDir, "outdir") );
  // Optional
  cmd.add( make_option('m', sImage_Describer_Method, "describerMethod") );
  cmd.add( make_option('u', bUpRight, "upright") );
  cmd.add( make_option('f', bForce, "force") );
  cmd.add( make_option('p', sFeaturePreset, "describerPreset") );

#ifdef OPENMVG_USE_OPENMP
  cmd.add( make_option('n', iNumThreads, "numThreads") );
#endif

  try {
      if (argc == 1) throw std::string("Invalid command line parameter.");
      cmd.process(argc, argv);
  } catch (const std::string& s) {
      OPENMVG_LOG_INFO
        << "Usage: " << argv[0] << '\n'
        << "[-i|--input_file] a SfM_Data file \n"
        << "[-o|--outdir path] \n"
        << "\n[Optional]\n"
        << "[-f|--force] Force to recompute data\n"
        << "[-m|--describerMethod]\n"
        << "  (method to use to describe an image):\n"
        << "   SIFT (default),\n"
        << "   SIFT_ANATOMY,\n"
        << "   AKAZE_FLOAT: AKAZE with floating point descriptors,\n"
        << "   AKAZE_MLDB:  AKAZE with binary descriptors\n"
        << "[-u|--upright] Use Upright feature 0 or 1\n"
        << "[-p|--describerPreset]\n"
        << "  (used to control the Image_describer configuration):\n"
        << "   NORMAL (default),\n"
        << "   HIGH,\n"
        << "   ULTRA: !!Can take long time!!\n"
#ifdef OPENMVG_USE_OPENMP
        << "[-n|--numThreads] number of parallel computations\n"
#endif
      ;

      OPENMVG_LOG_ERROR << s;
      return EXIT_FAILURE;
  }

  OPENMVG_LOG_INFO
    << " You called : " << "\n"
    << argv[0] << "\n"
    << "--input_file " << sSfM_Data_Filename << "\n"
    << "--outdir " << sOutDir << "\n"
    << "--describerMethod " << sImage_Describer_Method << "\n"
    << "--upright " << bUpRight << "\n"
    << "--describerPreset " << (sFeaturePreset.empty() ? "NORMAL" : sFeaturePreset) << "\n"
    << "--force " << bForce << "\n"
#ifdef OPENMVG_USE_OPENMP
    << "--numThreads " << iNumThreads << "\n"
#endif
    ;


  if (sOutDir.empty())
  {
    OPENMVG_LOG_ERROR << "\nIt is an invalid output directory";
    return EXIT_FAILURE;
  }

  // Create output dir
  if (!stlplus::folder_exists(sOutDir))
  {
    if (!stlplus::folder_create(sOutDir))
    {
      OPENMVG_LOG_ERROR << "Cannot create output directory";
      return EXIT_FAILURE;
    }
  }

  //---------------------------------------
  // a. Load input scene
  //---------------------------------------
  SfM_Data sfm_data;
  if (!Load(sfm_data, sSfM_Data_Filename, ESfM_Data(VIEWS|INTRINSICS))) {
    OPENMVG_LOG_ERROR
      << "The input file \""<< sSfM_Data_Filename << "\" cannot be read";
    return EXIT_FAILURE;
  }

  // b. Init the image_describer
  // - retrieve the used one in case of pre-computed features
  // - else create the desired one

  using namespace openMVG::features;
  std::unique_ptr<Image_describer> image_describer;

  const std::string sImage_describer = stlplus::create_filespec(sOutDir, "image_describer", "json");
  if (!bForce && stlplus::is_file(sImage_describer))
  {
    // Dynamically load the image_describer from the file (will restore old used settings)
    std::ifstream stream(sImage_describer.c_str());
    if (!stream)
      return EXIT_FAILURE;

    try
    {
      cereal::JSONInputArchive archive(stream);
      archive(cereal::make_nvp("image_describer", image_describer));
    }
    catch (const cereal::Exception & e)
    {
      OPENMVG_LOG_ERROR << e.what() << '\n'
        << "Cannot dynamically allocate the Image_describer interface.";
      return EXIT_FAILURE;
    }
  }
  else
  {
    // Create the desired Image_describer method.
    // Don't use a factory, perform direct allocation
    if (sImage_Describer_Method == "SIFT")
    {
      image_describer.reset(new SIFT_Image_describer
        (SIFT_Image_describer::Params(), !bUpRight));
    }
    else
    if (sImage_Describer_Method == "SIFT_ANATOMY")
    {
      image_describer.reset(
        new SIFT_Anatomy_Image_describer(SIFT_Anatomy_Image_describer::Params()));
    }
    else
    if (sImage_Describer_Method == "AKAZE_FLOAT")
    {
      image_describer = AKAZE_Image_describer::create
        (AKAZE_Image_describer::Params(AKAZE::Params(), AKAZE_MSURF), !bUpRight);
    }
    else
    if (sImage_Describer_Method == "AKAZE_MLDB")
    {
      image_describer = AKAZE_Image_describer::create
        (AKAZE_Image_describer::Params(AKAZE::Params(), AKAZE_MLDB), !bUpRight);
    }
    if (!image_describer)
    {
      OPENMVG_LOG_ERROR << "Cannot create the designed Image_describer:"
        << sImage_Describer_Method << ".";
      return EXIT_FAILURE;
    }
    else
    {
      if (!sFeaturePreset.empty())
      if (!image_describer->Set_configuration_preset(stringToEnum(sFeaturePreset)))
      {
        OPENMVG_LOG_ERROR << "Preset configuration failed.";
        return EXIT_FAILURE;
      }
    }

    // Export the used Image_describer and region type for:
    // - dynamic future regions computation and/or loading
    {
      std::ofstream stream(sImage_describer.c_str());
      if (!stream)
        return EXIT_FAILURE;

      cereal::JSONOutputArchive archive(stream);
      archive(cereal::make_nvp("image_describer", image_describer));
      auto regionsType = image_describer->Allocate();
      archive(cereal::make_nvp("regions_type", regionsType));
    }
  }

  // Feature extraction routines
  // For each View of the SfM_Data container:
  // - if regions file exists continue,
  // - if no file, compute features
  {
    system::Timer timer;
    Image<unsigned char> imageGray;

    system::LoggerProgress my_progress_bar(sfm_data.GetViews().size(), "- EXTRACT FEATURES -" );

    // Use a boolean to track if we must stop feature extraction
    std::atomic<bool> preemptive_exit(false);

    // Snapshot view pointers for O(1) indexed access. Iterating a std::map with
    // std::advance(begin(), i) inside the loop is O(i), i.e. O(n^2) overall.
    std::vector<const View*> vec_views;
    vec_views.reserve(sfm_data.views.size());
    for (const auto & view_it : sfm_data.views)
      vec_views.push_back(view_it.second.get());

#if (!TEST_CF_SINGLE_IMAGE) || (!TEST_CF_NOTHREADING)
#ifdef OPENMVG_USE_OPENMP
    // Physical-core affinity masks (used only for pinning below). Dropping
    // hyperthreads keeps one worker per core, reducing peak memory / HT contention.
    const std::vector<DWORD_PTR> core_masks = GetPhysicalCoreMasks();
    const unsigned int nb_physical =
        core_masks.empty() ? omp_get_max_threads()
                           : static_cast<unsigned int>(core_masks.size());

    // The caller passes -n as a physical-core count: it has already performed the
    // core/memory budgeting, so honor it directly (one pinned worker per core).
    // Only when -n is unspecified do we default to all detected physical cores.
    if (iNumThreads > 0) {
        omp_set_num_threads(iNumThreads);
    } else {
        omp_set_num_threads(static_cast<int>(nb_physical));
    }

    #pragma omp parallel for schedule(dynamic) private(imageGray)
#endif
#endif
#if (TEST_CF_MAX_IMAGES==0)
    for (int i = 0; i < static_cast<int>(sfm_data.views.size()); ++i)
#else
    for (int i = 0; i < std::min(static_cast<int>(sfm_data.views.size()), (int) TEST_CF_MAX_IMAGES); ++i)
#endif
    {
#ifdef OPENMVG_USE_OPENMP
      // Pin each OpenMP worker to a distinct physical core exactly once, to keep
      // its large SIFT pyramid buffers cache/NUMA-local and prevent migration.
      static thread_local bool pinned = false;
      if (!pinned && !core_masks.empty())
      {
        const int tid = omp_get_thread_num();
        SetThreadAffinityMask(GetCurrentThread(),
                              core_masks[tid % core_masks.size()]);
        pinned = true;
      }
#endif
      const View * view = vec_views[i];
      const std::string
        sView_filename = stlplus::create_filespec(sfm_data.s_root_path, view->s_Img_path),
        sFeat = stlplus::create_filespec(sOutDir, stlplus::basename_part(sView_filename), "feat"),
        sDesc = stlplus::create_filespec(sOutDir, stlplus::basename_part(sView_filename), "desc");

      // If features or descriptors file are missing, compute them
      if (!preemptive_exit && (bForce || !stlplus::file_exists(sFeat) || !stlplus::file_exists(sDesc)))
      {
        if (!ReadImage(sView_filename.c_str(), &imageGray))
          continue;

        //
        // Look if there is an occlusion feature mask
        //
        Image<unsigned char> * mask = nullptr; // The mask is null by default

        const std::string
          mask_filename_local =
            stlplus::create_filespec(sfm_data.s_root_path,
              stlplus::basename_part(sView_filename) + "_mask", "png"),
          mask_filename_global =
            stlplus::create_filespec(sfm_data.s_root_path, "mask", "png");

        Image<unsigned char> imageMask;
        // Try to read the local mask
        if (stlplus::file_exists(mask_filename_local))
        {
          if (!ReadImage(mask_filename_local.c_str(), &imageMask))
          {
            OPENMVG_LOG_ERROR
              << "Invalid mask: " << mask_filename_local << ';'
              << "Stopping feature extraction.";
            preemptive_exit = true;
            continue;
          }
          // Use the local mask only if it fits the current image size
          if (imageMask.Width() == imageGray.Width() && imageMask.Height() == imageGray.Height())
            mask = &imageMask;
        }
        else
        {
          // Try to read the global mask
          if (stlplus::file_exists(mask_filename_global))
          {
            if (!ReadImage(mask_filename_global.c_str(), &imageMask))
            {
              OPENMVG_LOG_ERROR
                << "Invalid mask: " << mask_filename_global << ';'
                << "Stopping feature extraction.";
              preemptive_exit = true;
              continue;
            }
            // Use the global mask only if it fits the current image size
            if (imageMask.Width() == imageGray.Width() && imageMask.Height() == imageGray.Height())
              mask = &imageMask;
          }
        }

        // Compute features and descriptors and export them to files
        auto regions = image_describer->Describe(imageGray, mask);
        // Release the image buffers before the disk write: they are no longer
        // needed and holding them during Save() needlessly inflates peak memory
        // (multiplied across all worker threads).
        imageGray = Image<unsigned char>();
        imageMask = Image<unsigned char>();
        mask = nullptr;
        if (regions && !image_describer->Save(regions.get(), sFeat, sDesc)) {
          OPENMVG_LOG_ERROR
            << "Cannot save regions for image: " << sView_filename << ';'
            << "Stopping feature extraction.";
          preemptive_exit = true;
          continue;
        }
      }
      ++my_progress_bar;
#if (TEST_CF_SINGLE_IMAGE) && (!TEST_CF_NOTHREADING)
      break;
#endif
    }
    OPENMVG_LOG_INFO << "Task done in (s): " << timer.elapsed();
  }
  return EXIT_SUCCESS;
}
