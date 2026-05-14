// Matching diagnostics: verify accuracy and correctness of the matching
// pipeline components (Cascade Hashing, distance ratio, deduplication).
// Guards against regressions in the optimized matching code paths.

#include "openMVG/features/feature.hpp"
#include "openMVG/features/regions_factory.hpp"
#include "openMVG/matching/cascade_hasher.hpp"
#include "openMVG/matching/indMatch.hpp"
#include "openMVG/matching/indMatch_utils.hpp"
#include "openMVG/matching/indMatchDecoratorXY.hpp"
#include "openMVG/matching/matching_filters.hpp"
#include "openMVG/matching/metric.hpp"
#include "openMVG/matching/regions_matcher.hpp"
#include "openMVG/matching_image_collection/Cascade_Hashing_Matcher_Regions.hpp"
#include "openMVG/matching_image_collection/Matcher_Regions.hpp"
#include "openMVG/sfm/pipelines/sfm_regions_provider.hpp"
#include "openMVG/types.hpp"

#include "testing/testing.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <vector>

using namespace openMVG;
using namespace openMVG::features;
using namespace openMVG::matching;
using namespace openMVG::matching_image_collection;

// ---------------------------------------------------------------------------
// Test helper: Regions_Provider subclass that exposes cache_ for direct
// population without loading from disk.
// ---------------------------------------------------------------------------
struct TestRegionsProvider : public sfm::Regions_Provider
{
  void Insert(IndexT id, std::shared_ptr<Regions> r)
  {
    cache_[id] = std::move(r);
  }
  void SetRegionType(std::unique_ptr<Regions> rt)
  {
    region_type_ = std::move(rt);
  }
};

// ---------------------------------------------------------------------------
// Helper: deep-copy a SIFT_Regions via CopyRegion.
// ---------------------------------------------------------------------------
static std::shared_ptr<SIFT_Regions> DeepCopyRegions(const SIFT_Regions& src)
{
  auto dst = std::make_shared<SIFT_Regions>();
  for (size_t i = 0; i < src.RegionCount(); ++i)
    src.CopyRegion(i, dst.get());
  return dst;
}

// ---------------------------------------------------------------------------
// Helper: build a SIFT_Regions object with N synthetic descriptors.
// Descriptors are random but reproducible from seed.  Positions are on a grid.
// ---------------------------------------------------------------------------
static std::unique_ptr<SIFT_Regions> MakeSyntheticRegions(
    int N, unsigned seed, int imgW = 640, int imgH = 480)
{
  auto regions = std::make_unique<SIFT_Regions>();
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> dist(0, 255);

  int cols = (int)std::ceil(std::sqrt((double)N));
  float dx = (float)imgW / (cols + 1);
  float dy = (float)imgH / (cols + 1);

  for (int i = 0; i < N; ++i)
  {
    Descriptor<unsigned char, 128> d;
    for (int k = 0; k < 128; ++k)
      d[k] = (unsigned char)dist(rng);
    regions->Descriptors().push_back(d);

    float x = dx * (float)((i % cols) + 1);
    float y = dy * (float)((i / cols) + 1);
    regions->Features().emplace_back(x, y, 1.6f, 0.0f);
  }
  return regions;
}

// ---------------------------------------------------------------------------
// Helper: build a SIFT_Regions that is a copy of src with some noise added.
// ---------------------------------------------------------------------------
static std::unique_ptr<SIFT_Regions> MakeNoisyCopy(
    const SIFT_Regions& src, int nIdentical, unsigned noiseSeed)
{
  auto regions = std::make_unique<SIFT_Regions>();
  std::mt19937 rng(noiseSeed);
  std::uniform_int_distribution<int> noiseDist(-10, 10);

  for (size_t i = 0; i < src.RegionCount(); ++i)
  {
    Descriptor<unsigned char, 128> d = src.Descriptors()[i];
    if ((int)i >= nIdentical)
    {
      for (int k = 0; k < 128; ++k)
      {
        int v = (int)d[k] + noiseDist(rng);
        d[k] = (unsigned char)std::max(0, std::min(255, v));
      }
    }
    regions->Descriptors().push_back(d);
    regions->Features().push_back(src.Features()[i]);
  }
  return regions;
}

// ---------------------------------------------------------------------------
// Helper: count intersection of two IndMatch sets.
// ---------------------------------------------------------------------------
static size_t MatchIntersection(
    const IndMatches& a, const IndMatches& b)
{
  std::set<IndMatch> sa(a.begin(), a.end());
  std::set<IndMatch> sb(b.begin(), b.end());
  size_t count = 0;
  for (const auto& m : sa)
    if (sb.count(m))
      ++count;
  return count;
}

// ===========================================================================
// 1. NNdistanceRatio: basic correctness
// ===========================================================================
TEST(MatchingDiagnostics, NNdistanceRatioBasic)
{
  std::vector<float> distances = {10, 100,  50, 55,  80, 81,  1, 200};
  std::vector<int> passed;
  NNdistanceRatio(distances.begin(), distances.end(), 2, passed, 0.8f);

  EXPECT_EQ(2u, passed.size());
  EXPECT_EQ(0, passed[0]);
  EXPECT_EQ(3, passed[1]);
}

// ===========================================================================
// 2. NNdistanceRatio with squared metric
// ===========================================================================
TEST(MatchingDiagnostics, NNdistanceRatioSquared)
{
  std::vector<unsigned int> distances = {100, 10000, 2500, 3025};
  std::vector<int> passed;
  NNdistanceRatio(distances.begin(), distances.end(), 2, passed,
                  Square(0.8f));

  EXPECT_EQ(1u, passed.size());
  EXPECT_EQ(0, passed[0]);
}

// ===========================================================================
// 3. IndMatch deduplication removes exact duplicates
// ===========================================================================
TEST(MatchingDiagnostics, IndMatchDeduplication)
{
  IndMatches matches;
  matches.emplace_back(0, 5);
  matches.emplace_back(1, 6);
  matches.emplace_back(0, 5);
  matches.emplace_back(2, 7);
  matches.emplace_back(1, 6);

  IndMatch::getDeduplicated(matches);
  EXPECT_EQ(3u, matches.size());
}

// ===========================================================================
// 4. IndMatchDecorator removes matches with same (X,Y) coordinates
// ===========================================================================
TEST(MatchingDiagnostics, SpatialDeduplication)
{
  std::vector<PointFeature> featI;
  featI.emplace_back(100.0f, 200.0f);
  featI.emplace_back(100.0f, 200.0f);

  std::vector<PointFeature> featJ;
  featJ.emplace_back(300.0f, 400.0f);
  featJ.emplace_back(500.0f, 600.0f);

  IndMatches matches;
  matches.emplace_back(0, 0);
  matches.emplace_back(1, 1);

  IndMatchDecorator<float> decorator(matches, featI, featJ);
  decorator.getDeduplicated(matches);

  EXPECT_EQ(1u, matches.size());
}

// ===========================================================================
// 5. Empty regions: no crash, zero matches
// ===========================================================================
TEST(MatchingDiagnostics, EmptyRegionsNoCrash)
{
  auto regA = std::make_unique<SIFT_Regions>();
  auto regB = std::make_unique<SIFT_Regions>();

  IndMatches matches;
  DistanceRatioMatch(0.8f, BRUTE_FORCE_L2, *regA, *regB, matches);
  EXPECT_EQ(0u, matches.size());
}

// ===========================================================================
// 6. BruteForce self-match: every descriptor matches itself at distance 0
// ===========================================================================
TEST(MatchingDiagnostics, BruteForceSelfMatch)
{
  auto regions = MakeSyntheticRegions(50, /*seed=*/42);

  IndMatches matches;
  Match(BRUTE_FORCE_L2, *regions, *regions, matches);

  EXPECT_EQ(regions->RegionCount(), matches.size());
  for (const auto& m : matches)
    EXPECT_EQ(m.i_, m.j_);
}

// ===========================================================================
// 7. BruteForce distance-ratio: identical descriptors always pass ratio test
// ===========================================================================
TEST(MatchingDiagnostics, BruteForceDistRatioSelfMatch)
{
  auto regions = MakeSyntheticRegions(50, /*seed=*/123);

  IndMatches matches;
  DistanceRatioMatch(0.8f, BRUTE_FORCE_L2, *regions, *regions, matches);

  EXPECT_EQ(regions->RegionCount(), matches.size());
}

// ===========================================================================
// 8. CascadeHasher determinism: two runs produce identical matches
// ===========================================================================
TEST(MatchingDiagnostics, CascadeHasherDeterministic)
{
  auto regA = MakeSyntheticRegions(200, /*seed=*/10);
  auto regB = MakeSyntheticRegions(200, /*seed=*/20);

  using ScalarT = unsigned char;
  using BaseMat = Eigen::Matrix<ScalarT, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  const size_t dimension = 128;
  const ScalarT* tabA = reinterpret_cast<const ScalarT*>(regA->DescriptorRawData());
  const ScalarT* tabB = reinterpret_cast<const ScalarT*>(regB->DescriptorRawData());

  Eigen::Map<BaseMat> matA((ScalarT*)tabA, regA->RegionCount(), dimension);
  Eigen::Map<BaseMat> matB((ScalarT*)tabB, regB->RegionCount(), dimension);

  // Run 1
  CascadeHasher hasher1;
  hasher1.Init(128);
  Eigen::VectorXf zmA = CascadeHasher::GetZeroMeanDescriptor(matA);
  Eigen::VectorXf zmB = CascadeHasher::GetZeroMeanDescriptor(matB);
  Eigen::MatrixXf zmAll(2, 128);
  zmAll.row(0) = zmA.transpose();
  zmAll.row(1) = zmB.transpose();
  Eigen::VectorXf zm = CascadeHasher::GetZeroMeanDescriptor(zmAll);
  auto hashA1 = hasher1.CreateHashedDescriptions(matA, zm);
  auto hashB1 = hasher1.CreateHashedDescriptions(matB, zm);

  IndMatches idx1;
  std::vector<unsigned int> dist1;
  Eigen::MatrixXi chd1; Eigen::VectorXi nhd1(129);
  std::vector<std::pair<unsigned int, int>> ced1;
  hasher1.Match_HashedDescriptions<BaseMat, unsigned int>(
      *hashA1, matA, *hashB1, matB, &idx1, &dist1, 2, chd1, nhd1, ced1);

  // Run 2
  CascadeHasher hasher2;
  hasher2.Init(128);
  auto hashA2 = hasher2.CreateHashedDescriptions(matA, zm);
  auto hashB2 = hasher2.CreateHashedDescriptions(matB, zm);

  IndMatches idx2;
  std::vector<unsigned int> dist2;
  Eigen::MatrixXi chd2; Eigen::VectorXi nhd2(129);
  std::vector<std::pair<unsigned int, int>> ced2;
  hasher2.Match_HashedDescriptions<BaseMat, unsigned int>(
      *hashA2, matA, *hashB2, matB, &idx2, &dist2, 2, chd2, nhd2, ced2);

  EXPECT_EQ(idx1.size(), idx2.size());
  EXPECT_EQ(dist1.size(), dist2.size());
  for (size_t i = 0; i < idx1.size(); ++i)
  {
    EXPECT_EQ(idx1[i].i_, idx2[i].i_);
    EXPECT_EQ(idx1[i].j_, idx2[i].j_);
    EXPECT_EQ(dist1[i], dist2[i]);
  }
}

// ===========================================================================
// 9. CascadeHasher vs BruteForce accuracy (direct hasher API).
//    Bypasses the collection matcher to test the core hash+match quality.
// ===========================================================================
TEST(MatchingDiagnostics, CascadeHasherVsBruteForceAccuracy)
{
  // Image A: 500 unique random descriptors
  auto regA = MakeSyntheticRegions(500, /*seed=*/77, 2000, 2000);
  // Image B: 300 different random descriptors + 200 exact copies from A
  auto regB = MakeSyntheticRegions(300, /*seed=*/88, 2000, 2000);
  for (int i = 0; i < 200; ++i)
  {
    regB->Descriptors().push_back(regA->Descriptors()[i]);
    float x = 50.0f + 8.0f * (float)(i % 20);
    float y = 50.0f + 8.0f * (float)(i / 20);
    regB->Features().emplace_back(x, y, 1.6f, 0.0f);
  }

  using ScalarT = unsigned char;
  using BaseMat = Eigen::Matrix<ScalarT, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  const size_t dim = 128;

  const ScalarT* tabA = reinterpret_cast<const ScalarT*>(regA->DescriptorRawData());
  const ScalarT* tabB = reinterpret_cast<const ScalarT*>(regB->DescriptorRawData());
  Eigen::Map<BaseMat> matA((ScalarT*)tabA, regA->RegionCount(), dim);
  Eigen::Map<BaseMat> matB((ScalarT*)tabB, regB->RegionCount(), dim);

  // Compute zero-mean
  Eigen::VectorXf zmA = CascadeHasher::GetZeroMeanDescriptor(matA);
  Eigen::VectorXf zmB = CascadeHasher::GetZeroMeanDescriptor(matB);
  Eigen::MatrixXf zmAll(2, dim);
  zmAll.row(0) = zmA.transpose();
  zmAll.row(1) = zmB.transpose();
  Eigen::VectorXf zm = CascadeHasher::GetZeroMeanDescriptor(zmAll);

  CascadeHasher hasher;
  hasher.Init(128);
  auto hashA = hasher.CreateHashedDescriptions(matA, zm);
  auto hashB = hasher.CreateHashedDescriptions(matB, zm);

  // Match B queries against A database (same direction as collection matcher)
  IndMatches indices;
  std::vector<unsigned int> distances;
  Eigen::MatrixXi chd; Eigen::VectorXi nhd(129);
  std::vector<std::pair<unsigned int, int>> ced;
  hasher.Match_HashedDescriptions<BaseMat, unsigned int>(
      *hashB, matB, *hashA, matA,
      &indices, &distances, 2, chd, nhd, ced);

  // Apply NN distance ratio
  std::vector<int> ratio_idx;
  NNdistanceRatio(distances.begin(), distances.end(), 2, ratio_idx, Square(0.8f));

  IndMatches ch_matches;
  for (const auto& k : ratio_idx)
  {
    // indices[k*2] = IndMatch(query_in_B, match_in_A)
    ch_matches.emplace_back(indices[k * 2].j_, indices[k * 2].i_);
  }

  // BruteForce reference
  IndMatches bf_matches;
  DistanceRatioMatch(0.8f, BRUTE_FORCE_L2, *regA, *regB, bf_matches);

  std::cout << "  [DiagInfo] bf=" << bf_matches.size()
            << " ch=" << ch_matches.size()
            << " totalQueries=" << regB->RegionCount()
            << " returned=" << indices.size() / 2 << std::endl;

  // BF should find most of the 200 planted matches
  EXPECT_TRUE(bf_matches.size() >= 100u);

  if (!bf_matches.empty() && !ch_matches.empty())
  {
    size_t overlap = MatchIntersection(bf_matches, ch_matches);
    double recall = (double)overlap / (double)bf_matches.size();
    std::cout << "  [DiagInfo] overlap=" << overlap
              << " recall=" << recall << std::endl;
    // Cascade hashing with 500+500 descriptors should find at least 20%
    EXPECT_TRUE(recall > 0.2);
  }
}

// ===========================================================================
// 10. CascadeHasher self-match: most descriptors find themselves at dist 0
// ===========================================================================
TEST(MatchingDiagnostics, CascadeHasherSelfMatch)
{
  const int N = 100;
  auto regions = MakeSyntheticRegions(N, /*seed=*/55);

  using ScalarT = unsigned char;
  using BaseMat = Eigen::Matrix<ScalarT, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  const ScalarT* tab = reinterpret_cast<const ScalarT*>(regions->DescriptorRawData());
  Eigen::Map<BaseMat> mat((ScalarT*)tab, N, 128);

  CascadeHasher hasher;
  hasher.Init(128);
  Eigen::VectorXf zm = CascadeHasher::GetZeroMeanDescriptor(mat);
  auto hashed = hasher.CreateHashedDescriptions(mat, zm);

  IndMatches indices;
  std::vector<unsigned int> distances;
  Eigen::MatrixXi chd; Eigen::VectorXi nhd(129);
  std::vector<std::pair<unsigned int, int>> ced;
  hasher.Match_HashedDescriptions<BaseMat, unsigned int>(
      *hashed, mat, *hashed, mat, &indices, &distances, 2, chd, nhd, ced);

  int selfMatches = 0;
  for (size_t i = 0; i + 1 < indices.size(); i += 2)
  {
    if (indices[i].i_ == indices[i].j_ && distances[i] == 0)
      ++selfMatches;
  }
  std::cout << "  [DiagInfo] CascadeHasherSelfMatch: N=" << N
            << " totalPairs=" << indices.size() / 2
            << " selfMatches=" << selfMatches << std::endl;
  // Cascade hashing is approximate — with random descriptors many items
  // may not hash to the same bucket as themselves in all groups.
  // Require at least 50%.
  EXPECT_TRUE(selfMatches > N / 2);
}

// ===========================================================================
// 11. Match count regression: random descriptors with ratio 0.8
// ===========================================================================
TEST(MatchingDiagnostics, MatchCountRegression)
{
  auto regA = MakeSyntheticRegions(300, /*seed=*/1);
  auto regB = MakeSyntheticRegions(300, /*seed=*/2);

  IndMatches matches;
  DistanceRatioMatch(0.8f, BRUTE_FORCE_L2, *regA, *regB, matches);

  // Random 128-d descriptors rarely pass ratio 0.8
  EXPECT_TRUE(matches.size() < 100u);
}

/* ************************************************************************* */
int main() {
  TestResult tr; return TestRegistry::runAllTests(tr);
}
/* ************************************************************************* */
