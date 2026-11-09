// SPDX-License-Identifier: BSD-3-Clause

#define _USE_MATH_DEFINES

#include "colmap/retrieval/vote_and_verify.h"

#include "colmap/math/random.h"
#include "colmap/retrieval/geometry.h"

#include <cmath>

#include <gtest/gtest.h>

namespace colmap {
namespace retrieval {
namespace {

struct SyntheticData {
  FeatureGeometryTransform img2_from_img1;
  std::vector<FeatureGeometryMatch> matches;
};

SyntheticData SynthesizeData(size_t num_inliers, size_t num_outliers) {
  SyntheticData data;
  data.img2_from_img1.scale = RandomUniformReal<float>(0.1, 10);
  data.img2_from_img1.angle = RandomUniformReal<float>(0, 2 * M_PI);
  data.img2_from_img1.tx = RandomUniformReal<float>(-100, 100);
  data.img2_from_img1.ty = RandomUniformReal<float>(-100, 100);

  const float sin_angle = std::sin(data.img2_from_img1.angle);
  const float cos_angle = std::cos(data.img2_from_img1.angle);

  data.matches.resize(num_inliers + num_outliers);
  for (size_t i = 0; i < num_inliers; ++i) {
    FeatureGeometryMatch& match = data.matches[i];
    match.geometry1.scale = RandomUniformReal<float>(0.5, 2);
    match.geometry1.orientation = RandomUniformReal<float>(0, 2 * M_PI);
    match.geometry1.x = RandomUniformReal<float>(-100, 100);
    match.geometry1.y = RandomUniformReal<float>(-100, 100);
    match.geometry2.scale = data.img2_from_img1.scale * match.geometry1.scale;
    match.geometry2.orientation =
        data.img2_from_img1.angle + match.geometry1.orientation;
    match.geometry2.x =
        data.img2_from_img1.scale *
        (cos_angle * match.geometry1.x - sin_angle * match.geometry1.y);
    match.geometry2.y =
        data.img2_from_img1.scale *
        (sin_angle * match.geometry1.x + cos_angle * match.geometry1.y);
  }

  for (size_t i = 0; i < num_outliers; ++i) {
    FeatureGeometryMatch& match = data.matches[i + num_inliers];
    match.geometry1.scale = RandomUniformReal<float>(0.5, 2);
    match.geometry1.orientation = RandomUniformReal<float>(0, 2 * M_PI);
    match.geometry1.x = RandomUniformReal<float>(-100, 100);
    match.geometry1.y = RandomUniformReal<float>(-100, 100);
    match.geometry2.scale = RandomUniformReal<float>(0.5, 2);
    match.geometry2.orientation = RandomUniformReal<float>(0, 2 * M_PI);
    match.geometry2.x = RandomUniformReal<float>(-100, 100);
    match.geometry2.y = RandomUniformReal<float>(-100, 100);
  }

  return data;
}

TEST(VoteAndVerify, NoMatches) {
  EXPECT_EQ(VoteAndVerify(VoteAndVerifyOptions(), {}), 0);
}

TEST(VoteAndVerify, NoEffectiveInliers) {
  const size_t kNumInliers = 100;
  const size_t kNumOutliers = 50;
  const auto data = SynthesizeData(kNumInliers, kNumOutliers);
  VoteAndVerifyOptions options;
  options.eff_inlier_count = false;
  const int num_inliers = VoteAndVerify(options, data.matches);
  EXPECT_EQ(num_inliers, kNumInliers);
}

TEST(VoteAndVerify, EffectiveInliers) {
  const size_t kNumInliers = 100;
  const size_t kNumOutliers = 50;
  const auto data = SynthesizeData(kNumInliers, kNumOutliers);
  VoteAndVerifyOptions options;
  options.eff_inlier_count = true;
  const int num_inliers = VoteAndVerify(options, data.matches);
  EXPECT_GT(num_inliers, 0.8 * kNumInliers);
}

FeatureGeometryMatch MakeRetrievalMatch(float x, float y, float dx, float dy) {
  FeatureGeometry f;
  f.x = x;
  f.y = y;
  f.scale = 1;
  FeatureGeometry g = f;
  g.x += dx;
  g.y += dy;
  return {f, g};
}

TEST(VoteAndVerify, EquivalentRelativeOrientationsVoteTogether) {
  std::vector<FeatureGeometryMatch> matches;
  constexpr float angle = 0.75f;
  for (int i = 0; i < 6; ++i) {
    auto m = MakeRetrievalMatch(10 * i, 5 * (i % 3), 0, 0);
    m.geometry1.orientation = i % 2 ? 2.8f : 0.1f;
    m.geometry2.orientation =
        std::remainder(m.geometry1.orientation + angle, 2.0f * float(M_PI));
    m.geometry2.x =
        std::cos(angle) * m.geometry1.x - std::sin(angle) * m.geometry1.y + 27;
    m.geometry2.y =
        std::sin(angle) * m.geometry1.x + std::cos(angle) * m.geometry1.y + 17;
    matches.push_back(m);
  }
  VoteAndVerifyOptions options;
  options.min_num_votes = matches.size();
  options.local_optimization = false;
  options.eff_inlier_count = false;
  EXPECT_EQ(VoteAndVerify(options, matches), matches.size());
}

TEST(VoteAndVerify, EffectiveCountIsIndependentOfCoordinateOrigin) {
  VoteAndVerifyOptions options;
  options.num_eff_inlier_bins = 4;
  options.local_optimization = false;
  for (const float offset : {0.0f, -100.0f}) {
    std::vector<FeatureGeometryMatch> matches;
    for (int i = 0; i < 4; ++i) {
      matches.push_back(MakeRetrievalMatch(offset + i, offset + i, 0, 0));
    }
    EXPECT_EQ(VoteAndVerify(options, matches), 4);
  }
  std::vector<FeatureGeometryMatch> line;
  for (int i = 0; i < 4; ++i) {
    line.push_back(MakeRetrievalMatch(i, 1, 0, 0));
  }
  EXPECT_EQ(VoteAndVerify(options, line), 4);
}

TEST(VoteAndVerify, DistinctSparseBinsWithLargeDimensions) {
  VoteAndVerifyOptions options;
  options.num_trans_bins = 1024;
  options.num_scale_bins = 128;
  options.num_angle_bins = 1024;
  options.min_num_votes = 3;
  options.max_transfer_error = 1;
  options.eff_inlier_count = false;
  options.local_optimization = false;
  std::vector<FeatureGeometryMatch> matches;
  for (float dy : {0.0f, 256.0f}) {
    matches.push_back(MakeRetrievalMatch(10, 10, 0, dy));
    matches.push_back(MakeRetrievalMatch(10, 40, 0, dy));
    matches.push_back(MakeRetrievalMatch(30, 10, 0, dy));
  }
  EXPECT_EQ(VoteAndVerify(options, matches), 3);
}

}  // namespace
}  // namespace retrieval
}  // namespace colmap
