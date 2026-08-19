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

#include <algorithm>
#include <atomic>
#include <cstdint>
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

//------------------------------------------------------------------------------
// Machine topology + memory budgeting.
//
// Feature extraction is embarrassingly parallel per image, but each worker
// holds a large PRIVATE footprint (the float image plus the SIFT scale-space
// pyramid, tens of bytes per input pixel). So the useful worker count is
//    min( what the CPU can run , what RAM can hold )
// and neither term alone is a good answer: one physical core per worker leaves
// a big machine idle, while one logical processor per worker will thrash a
// 16 GB box on 24 MP imagery. Everything below exists to compute that min.
//------------------------------------------------------------------------------

static int
BitCount(DWORD_PTR mask)
{
  int n = 0;
  for (; mask; mask &= (mask - 1))
    ++n;
  return n;
}

// Affinity masks for the machine's physical cores. Single processor-group only
// (<= 64 logical CPUs), which covers typical single-socket workstations; on
// bigger hosts the vectors come back empty and pinning is simply skipped.
struct CoreTopology
{
  std::vector<DWORD_PTR> primary;   // one logical processor per physical core
  std::vector<DWORD_PTR> shared;    // all hyperthread siblings of that core
  int nb_logical = 0;               // logical processors in this group
};

static CoreTopology
GetCoreTopology()
{
  CoreTopology topo;

  DWORD len = 0;
  GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
  if (len != 0)
  {
    std::vector<char> buf(len);
    auto * first = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data());
    if (GetLogicalProcessorInformationEx(RelationProcessorCore, first, &len))
    {
      char * ptr = buf.data();
      while (ptr < buf.data() + len)
      {
        auto * cur = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(ptr);
        if (cur->Relationship == RelationProcessorCore)
        {
          // GroupMask[0].Mask holds all logical procs (HT siblings) of this
          // core; its lowest set bit pins to a single logical processor.
          const DWORD_PTR full = cur->Processor.GroupMask[0].Mask;
          topo.shared.push_back(full);
          topo.primary.push_back(full & (~full + 1));
          topo.nb_logical += BitCount(full);
        }
        ptr += cur->Size;
      }
    }
  }

  if (topo.nb_logical == 0)
  {
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    topo.nb_logical = static_cast<int>(si.dwNumberOfProcessors);
  }
  return topo;
}

// Bytes of physical RAM we are willing to hand to the extraction workers, minus
// headroom for the OS, the file cache and this process' own allocations.
// Returns 0 if the budget cannot be established (caller then ignores memory).
static uint64_t
FeatureMemoryBudget(uint64_t * total_out, uint64_t * avail_out)
{
  MEMORYSTATUSEX ms;
  ms.dwLength = sizeof(ms);
  const bool ok = (GlobalMemoryStatusEx(&ms) != 0);
  if (total_out) *total_out = ok ? ms.ullTotalPhys : 0;
  if (avail_out) *avail_out = ok ? ms.ullAvailPhys : 0;

  if (const char * env = std::getenv("OPENMVG_CF_MEM_BUDGET_MB"))
  {
    const long long mb = std::atoll(env);
    if (mb > 0)
      return static_cast<uint64_t>(mb) << 20;
  }

  if (!ok)
    return 0;

  // What we can actually spend. Windows counts only the free + standby lists as
  // "available": memory parked in ANOTHER process' working set is excluded even
  // when it is idle and the OS would trim it the moment we asked for it. Taking
  // ullAvailPhys literally therefore collapses a 16-core box to two workers
  // whenever the host application happens to be holding a large idle heap --
  // measured in the field: 3.2 GB reported available of 16 GB installed. So
  // treat half of installed RAM as spendable when the reported figure is below
  // that; above it, the reported figure is the honest constraint and wins.
  const uint64_t usable =
      std::max<uint64_t>(ms.ullAvailPhys, ms.ullTotalPhys / 2);

  // Headroom on top of that: 10% of RAM, floored at 1.5 GB so a small box stays
  // responsive and capped at 4 GB because the need is roughly absolute -- it
  // does not grow with RAM, and scaling it would park 12 GB on a 128 GB host.
  const uint64_t reserve =
      std::min<uint64_t>(4096ull << 20,
        std::max<uint64_t>(1536ull << 20, ms.ullTotalPhys / 10));

  // Under real pressure: report a token budget rather than 0, which the caller
  // reads as "unknown" and would answer with full concurrency. A token budget
  // instead collapses us to the single mandatory worker.
  return (usable > reserve) ? (usable - reserve) : 1;
}

// Per-worker peak footprint, expressed per pixel of the input image.
//
// vlfeat SIFT (num_scales = 3 => s_min = -1, s_max = 4) allocates, at the
// resolution of the FIRST octave and then reuses those buffers for the coarser
// ones, so this is the peak:
//    octave  : nel * (s_max - s_min + 1) = 6 float planes
//    dog+temp: nel * (s_max - s_min) + nel = 6 float planes
// = 12 float planes = 48 B/px, plus the uchar image (1 B/px), its float copy
// (4 B/px) and the regions vectors + allocator slack (~11 B/px of margin).
// OPENMVG_SIFT_FIRST_OCTAVE=-1 upsamples 2x first, i.e. 4x those planes.
static uint64_t
BytesPerPixelPerWorker(const Image_describer * describer)
{
  if (const char * env = std::getenv("OPENMVG_CF_BYTES_PER_PIXEL"))
  {
    const long long v = std::atoll(env);
    if (v > 0)
      return static_cast<uint64_t>(v);
  }

  // dynamic_cast (not the -m string) so a describer restored from
  // image_describer.json is classified correctly too.
  if (dynamic_cast<const SIFT_Image_describer *>(describer))
  {
    const char * fo = std::getenv("OPENMVG_SIFT_FIRST_OCTAVE");
    const bool upscale = (fo != nullptr && std::atoi(fo) < 0);
    return upscale ? 208 : 64;
  }
  if (dynamic_cast<const SIFT_Anatomy_Image_describer *>(describer))
    return 96;  // holds the whole Gaussian + DoG pyramid resident at once
  if (dynamic_cast<const AKAZE_Image_describer *>(describer))
    return 96;  // 4 slices x 4 float planes per octave, all octaves resident

  return 96;    // unknown describer: assume the expensive case
}

// How a worker is bound to a core, once the worker count is known.
enum class EPinMode
{
  NONE,               // topology unknown, or more workers than logical procs
  CORE_EXCLUSIVE,     // <= 1 worker per physical core: pin to one sibling
  CORE_SHARED         // oversubscribed: pin to a core, either sibling allowed
};

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
        << "[-n|--numThreads] upper bound on parallel image extractions\n"
        << "   (pass the logical thread count; 0 = use every logical\n"
        << "    processor). The effective worker count is this value clamped\n"
        << "    by the memory the largest images in the scene need.\n"
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

    // Largest images first. With schedule(dynamic) this keeps the long tasks
    // off the end of the run (where they would finish alone, all other workers
    // idle), and it makes the memory bound below exact rather than pessimistic:
    // the images resident at peak concurrency are precisely the k largest.
    std::sort(vec_views.begin(), vec_views.end(),
      [](const View * a, const View * b) {
        return static_cast<uint64_t>(a->ui_width) * a->ui_height >
               static_cast<uint64_t>(b->ui_width) * b->ui_height;
      });

    // Pixel count per view, descending.
    std::vector<uint64_t> view_pixels;
    view_pixels.reserve(vec_views.size());
    for (const View * v : vec_views)
      view_pixels.push_back(static_cast<uint64_t>(v->ui_width) * v->ui_height);

    if (!view_pixels.empty() && view_pixels.front() == 0)
    {
      // Scene carries no image dimensions (unusual). Probe a few headers --
      // cheap, no pixel decode -- and assume the worst of them for all views.
      uint64_t probed = 0;
      const size_t nb_probe = std::min<size_t>(vec_views.size(), 16);
      for (size_t k = 0; k < nb_probe; ++k)
      {
        ImageHeader hdr;
        const std::string path =
          stlplus::create_filespec(sfm_data.s_root_path, vec_views[k]->s_Img_path);
        if (ReadImageHeader(path.c_str(), &hdr))
          probed = std::max(probed,
                            static_cast<uint64_t>(hdr.width) * hdr.height);
      }
      if (probed == 0)
        probed = 40ull * 1000 * 1000;  // conservative fallback: 40 MP
      std::fill(view_pixels.begin(), view_pixels.end(), probed);
    }

#ifdef OPENMVG_USE_OPENMP
    const CoreTopology topo = GetCoreTopology();
    const int nb_physical =
        topo.primary.empty() ? topo.nb_logical
                             : static_cast<int>(topo.primary.size());

    // -n is a CEILING on concurrency, not the answer: the caller says how many
    // threads it is willing to give us (0 = every logical processor) and we
    // spend as many of them as RAM allows. OPENMVG_CF_THREADS overrides it, so
    // the ceiling can be retuned without touching the calling process.
    int thread_ceiling = iNumThreads;
    if (const char * env = std::getenv("OPENMVG_CF_THREADS"))
    {
      const int v = std::atoi(env);
      if (v != 0)
        thread_ceiling = v;
    }
    if (thread_ceiling <= 0)
      thread_ceiling = topo.nb_logical;   // hyperthreads included
    thread_ceiling = std::max(1, std::min(thread_ceiling, 4 * topo.nb_logical));

    // Admit workers largest-image-first while their summed footprint fits the
    // budget. Summing the actual top-k images beats (k * largest image) on the
    // mixed-resolution scenes we see in practice. At least one worker always
    // runs: a single huge image over budget must still be attempted.
    const uint64_t bytes_per_pixel = BytesPerPixelPerWorker(image_describer.get());
    uint64_t phys_total = 0, phys_avail = 0;
    const uint64_t mem_budget = FeatureMemoryBudget(&phys_total, &phys_avail);

    int nb_workers = thread_ceiling;
    if (mem_budget > 0 && !view_pixels.empty())
    {
      uint64_t acc = 0;
      int k = 0;
      while (k < thread_ceiling && k < static_cast<int>(view_pixels.size()))
      {
        const uint64_t need = view_pixels[k] * bytes_per_pixel;
        if (k > 0 && acc + need > mem_budget)
          break;
        acc += need;
        ++k;
      }
      nb_workers = std::max(1, k);
    }

    // Pin each worker to a core to keep its large pyramid buffers cache/NUMA
    // local. One worker per core gets a single sibling; when memory allowed us
    // to oversubscribe, workers share a core's siblings instead.
    const EPinMode pin_mode =
        topo.primary.empty()               ? EPinMode::NONE
      : (nb_workers <= nb_physical)        ? EPinMode::CORE_EXCLUSIVE
      : (nb_workers <= topo.nb_logical)    ? EPinMode::CORE_SHARED
                                           : EPinMode::NONE;

    OPENMVG_LOG_INFO
      << "Extraction concurrency: " << nb_workers << " worker(s)"
      << " (ceiling " << thread_ceiling << ", "
      << nb_physical << " physical / " << topo.nb_logical << " logical cores)"
      << "\n  RAM: " << (phys_total >> 20) << " MB installed, "
      << (phys_avail >> 20) << " MB reported available"
      << "\n  memory budget: " << (mem_budget >> 20) << " MB"
      << ", ~" << ((view_pixels.empty()
                     ? 0 : view_pixels.front() * bytes_per_pixel) >> 20)
      << " MB per worker on the largest image"
      << " (" << bytes_per_pixel << " B/px)";

    omp_set_dynamic(0);   // do not let the runtime hand us fewer threads
    omp_set_num_threads(nb_workers);
#endif

#if (!TEST_CF_SINGLE_IMAGE) || (!TEST_CF_NOTHREADING)
#ifdef OPENMVG_USE_OPENMP
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
      // Apply the affinity decided above, once per worker thread, to prevent
      // migration away from the cache holding its SIFT pyramid buffers.
      static thread_local bool pinned = false;
      if (!pinned && pin_mode != EPinMode::NONE)
      {
        const std::vector<DWORD_PTR> & masks =
          (pin_mode == EPinMode::CORE_EXCLUSIVE) ? topo.primary : topo.shared;
        const int tid = omp_get_thread_num();
        SetThreadAffinityMask(GetCurrentThread(), masks[tid % masks.size()]);
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
