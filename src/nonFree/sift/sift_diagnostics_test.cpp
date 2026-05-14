// SIFT diagnostics: cross-validation of VLFeat SIFT against the independent
// Anatomy-of-SIFT reference implementation.  Verifies keypoint count,
// determinism, descriptor quality, and guards against regressions.

#include "nonFree/sift/SIFT_describer.hpp"
#include "openMVG/features/sift/SIFT_Anatomy_Image_Describer.hpp"
#include "openMVG/features/regions_factory.hpp"
#include "openMVG/image/image_io.hpp"

#include "testing/testing.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif

// These globals are required by the VLFeat SIFT C code (sift.c).
extern "C" int hasAVX2 = -1;
extern "C" int hasSSE41 = -1;

static int CpuHasAVX2()
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

static int CpuHasSSE41()
{
#if defined(_MSC_VER)
  int info[4];
  __cpuid(info, 1);
  return (info[2] & (1 << 19)) != 0;
#else
  unsigned eax, ebx, ecx, edx;
  __cpuid(1, eax, ebx, ecx, edx);
  return (ecx & (1 << 19)) != 0;
#endif
}

using namespace openMVG;
using namespace openMVG::image;
using namespace openMVG::features;

// Path to a stable test image shipped with openMVG samples.
static const std::string kTestImage =
    std::string(THIS_SOURCE_DIR)
    + "/../../openMVG_Samples/imageData/StanfordMobileVisualSearch/Ace_0.png";

// ---------------------------------------------------------------------------
// 1. Empty image ? zero keypoints, no crash
// ---------------------------------------------------------------------------
TEST(VlSift_Diagnostics, EmptyImageReturnsZeroKeypoints)
{
  Image<unsigned char> empty;
  SIFT_Image_describer describer;
  auto regions = describer.Describe(empty);
  EXPECT_TRUE(regions != nullptr);
  EXPECT_EQ(0u, regions->RegionCount());
}

// ---------------------------------------------------------------------------
// 2. Real image must produce a reasonable number of keypoints
// ---------------------------------------------------------------------------
TEST(VlSift_Diagnostics, RealImageProducesKeypoints)
{
  Image<unsigned char> image;
  EXPECT_TRUE(ReadImage(kTestImage.c_str(), &image));

  SIFT_Image_describer describer;
  auto regions = describer.Describe(image);
  EXPECT_TRUE(regions != nullptr);
  EXPECT_TRUE(regions->RegionCount() > 100u);
}

// ---------------------------------------------------------------------------
// 3. Two identical runs must produce bit-identical output
// ---------------------------------------------------------------------------
TEST(VlSift_Diagnostics, DeterministicOutput)
{
  Image<unsigned char> image;
  EXPECT_TRUE(ReadImage(kTestImage.c_str(), &image));

  // Use a single describer instance to avoid double vl_constructor/destructor
  SIFT_Image_describer d;
  auto r1 = d.DescribeSIFT(image);
  auto r2 = d.DescribeSIFT(image);

  EXPECT_EQ(r1->RegionCount(), r2->RegionCount());

  for (size_t i = 0; i < r1->RegionCount(); ++i)
  {
    const auto& f1 = r1->Features()[i];
    const auto& f2 = r2->Features()[i];
    EXPECT_NEAR(f1.x(), f2.x(), 1e-6);
    EXPECT_NEAR(f1.y(), f2.y(), 1e-6);
    EXPECT_NEAR(f1.scale(), f2.scale(), 1e-6);
    EXPECT_NEAR(f1.orientation(), f2.orientation(), 1e-6);

    for (int k = 0; k < 128; ++k)
      EXPECT_EQ(r1->Descriptors()[i][k], r2->Descriptors()[i][k]);
  }
}

// ---------------------------------------------------------------------------
// 4. Descriptors must not be all-zero (broken gradient / descriptor path)
// ---------------------------------------------------------------------------
TEST(VlSift_Diagnostics, DescriptorNonZero)
{
  Image<unsigned char> image;
  EXPECT_TRUE(ReadImage(kTestImage.c_str(), &image));

  SIFT_Image_describer describer;
  auto regions = describer.DescribeSIFT(image);
  EXPECT_TRUE(regions->RegionCount() > 0u);

  int zeroDescriptors = 0;
  for (size_t i = 0; i < regions->RegionCount(); ++i)
  {
    const auto& d = regions->Descriptors()[i];
    int sum = 0;
    for (int k = 0; k < 128; ++k)
      sum += d[k];
    if (sum == 0)
      ++zeroDescriptors;
  }
  // Allow at most 1% all-zero descriptors
  EXPECT_TRUE(zeroDescriptors < (int)(regions->RegionCount() * 0.01 + 1));
}

// ---------------------------------------------------------------------------
// 5. Cross-validate keypoint count vs Anatomy-of-SIFT reference
// ---------------------------------------------------------------------------
TEST(VlSift_Diagnostics, CrossValidateKeypointCount)
{
  Image<unsigned char> image;
  EXPECT_TRUE(ReadImage(kTestImage.c_str(), &image));

  // VLFeat path
  SIFT_Image_describer vl_describer(
      SIFT_Image_describer::Params(0, 6, 3, 10.0f, 0.04f, true));
  auto vl_regions = vl_describer.DescribeSIFT(image);

  // Anatomy-of-SIFT path (independent reference)
  SIFT_Anatomy_Image_describer anat_describer(
      SIFT_Anatomy_Image_describer::Params(0, 6, 3, 10.0f, 0.04f, true));
  auto anat_regions = anat_describer.Describe_SIFT_Anatomy(image);

  const size_t vl_count   = vl_regions->RegionCount();
  const size_t anat_count = anat_regions->RegionCount();

  EXPECT_TRUE(vl_count > 0u);
  EXPECT_TRUE(anat_count > 0u);

  // The two implementations should agree within a factor of ~2.
  const double ratio = (double)vl_count / (double)anat_count;
  EXPECT_TRUE(ratio > 0.5);
  EXPECT_TRUE(ratio < 2.0);
}

// ---------------------------------------------------------------------------
// 6. Self-consistency: VLFeat descriptors are matchable across runs
//    with identical input (confirms descriptor pipeline is functional)
// ---------------------------------------------------------------------------
TEST(VlSift_Diagnostics, DescriptorSelfConsistency)
{
  Image<unsigned char> image;
  EXPECT_TRUE(ReadImage(kTestImage.c_str(), &image));

  // Use a single describer instance — multiple instances cause
  // double vl_constructor/vl_destructor which corrupts the heap.
  SIFT_Image_describer d(
      SIFT_Image_describer::Params(0, 6, 3, 10.0f, 0.04f, false));
  auto r1 = d.DescribeSIFT(image);
  auto r2 = d.DescribeSIFT(image);

  EXPECT_EQ(r1->RegionCount(), r2->RegionCount());
  EXPECT_TRUE(r1->RegionCount() > 100u);

  // Every descriptor from run 1 should have L2 distance 0 to
  // the corresponding descriptor from run 2 (bit-exact determinism).
  int perfect_matches = 0;
  for (size_t i = 0; i < r1->RegionCount(); ++i)
  {
    const auto& d1v = r1->Descriptors()[i];
    const auto& d2v = r2->Descriptors()[i];
    int dist2 = 0;
    for (int k = 0; k < 128; ++k)
    {
      int diff = (int)d1v[k] - (int)d2v[k];
      dist2 += diff * diff;
    }
    if (dist2 == 0)
      ++perfect_matches;
  }

  // 100% of descriptors should be identical across two runs
  EXPECT_EQ((size_t)perfect_matches, r1->RegionCount());
}

// ---------------------------------------------------------------------------
// 7. Regression guard: keypoint count on a fixed image stays in a known range
// ---------------------------------------------------------------------------
TEST(VlSift_Diagnostics, KeypointCountRegression)
{
  Image<unsigned char> image;
  EXPECT_TRUE(ReadImage(kTestImage.c_str(), &image));

  SIFT_Image_describer describer(
      SIFT_Image_describer::Params(0, 6, 3, 10.0f, 0.04f, true));
  auto regions = describer.DescribeSIFT(image);

  const size_t count = regions->RegionCount();

  // Wide bounds — tighten once you establish a baseline.
  EXPECT_TRUE(count > 200u);
  EXPECT_TRUE(count < 20000u);
}

/* ************************************************************************* */
int main() {
  hasAVX2 = CpuHasAVX2();
  hasSSE41 = CpuHasSSE41();
  TestResult tr; return TestRegistry::runAllTests(tr);
}
/* ************************************************************************* */
