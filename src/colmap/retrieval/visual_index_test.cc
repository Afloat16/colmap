// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/retrieval/visual_index.h"

#include "colmap/feature/types.h"
#include "colmap/math/random_eigen.h"
#include "colmap/retrieval/inverted_index.h"
#include "colmap/util/file.h"
#include "colmap/util/testing.h"

#include <fstream>
#include <sstream>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/impl/io.h>
#include <faiss/index_io.h>
#include <gtest/gtest.h>

namespace colmap {
namespace retrieval {
namespace {

// Helper to create random descriptors with a given type.
FeatureDescriptorsFloat CreateRandomDescriptors(int num_rows,
                                                int desc_dim,
                                                FeatureExtractorType type) {
  return FeatureDescriptorsFloat(
      type, FeatureDescriptorsFloatData::Random(num_rows, desc_dim));
}

VisualIndex::BuildOptions SmallVocabularyOptions() {
  VisualIndex::BuildOptions options;
  options.num_visual_words = 4;
  options.num_iterations = 2;
  options.num_rounds = 1;
  options.num_threads = 1;
  return options;
}

TEST(VisualIndexLifecycle, RebuildDropsPreviouslyIndexedImages) {
  auto index = VisualIndex::Create(32, 16);
  const auto train =
      CreateRandomDescriptors(200, 32, FeatureExtractorType::SIFT);
  index->Build(SmallVocabularyOptions(), train);
  index->Add({}, 7, FeatureKeypoints(train.data.rows()), train);
  index->Prepare();
  index->Build(SmallVocabularyOptions(), train);
  EXPECT_EQ(index->NumImages(), 0);
  EXPECT_FALSE(index->IsImageIndexed(7));
  std::vector<ImageScore> scores;
  EXPECT_ANY_THROW(index->Query({}, train, &scores));
  index->Add({}, 7, FeatureKeypoints(train.data.rows()), train);
  index->Prepare();
  index->Query({}, train, &scores);
  ASSERT_EQ(scores.size(), 1);
  EXPECT_EQ(scores.front().image_id, 7);
}

TEST(VisualIndexLifecycle, FailedWordAssignmentCanBeRetried) {
  auto index = VisualIndex::Create(32, 16);
  const auto train =
      CreateRandomDescriptors(200, 32, FeatureExtractorType::SIFT);
  index->Build(SmallVocabularyOptions(), train);
  index->Prepare();
  VisualIndex::IndexOptions invalid;
  invalid.num_neighbors = 0;
  const FeatureKeypoints keypoints(train.data.rows());
  EXPECT_ANY_THROW(index->Add(invalid, 7, keypoints, train));
  EXPECT_FALSE(index->IsImageIndexed(7));
  std::vector<ImageScore> scores;
  EXPECT_NO_THROW(index->Query({}, train, &scores));
  index->Add({}, 7, keypoints, train);
  index->Prepare();
  index->Query({}, train, &scores);
  ASSERT_EQ(scores.size(), 1);
}

TEST(VisualIndexLifecycle, FeaturelessImageMembershipSurvivesSerialization) {
  const auto directory = CreateTestDir();
  auto index = VisualIndex::Create(32, 16);
  const auto train =
      CreateRandomDescriptors(200, 32, FeatureExtractorType::SIFT);
  index->Build(SmallVocabularyOptions(), train);
  index->Add(
      {}, 5, {}, CreateRandomDescriptors(0, 32, FeatureExtractorType::SIFT));
  index->Add({}, 7, FeatureKeypoints(train.data.rows()), train);
  for (const bool prepared : {false, true}) {
    if (prepared) {
      index->Prepare();
    }
    const auto path = directory / "index.bin";
    index->Write(path);
    auto restored = VisualIndex::Read(path);
    EXPECT_EQ(restored->NumImages(), 2);
    EXPECT_TRUE(restored->IsImageIndexed(5));
    EXPECT_TRUE(restored->IsImageIndexed(7));
  }
}

TEST(VisualIndexLifecycle, RejectIncompleteStoredImageWeight) {
  const auto path = CreateTestDir() / "index.bin";
  auto index = VisualIndex::Create(32, 16);
  const auto train =
      CreateRandomDescriptors(200, 32, FeatureExtractorType::SIFT);
  index->Build(SmallVocabularyOptions(), train);
  index->Add({}, 7, FeatureKeypoints(train.data.rows()), train);
  index->Prepare();
  index->Write(path);
  std::filesystem::resize_file(path, std::filesystem::file_size(path) - 1);
  EXPECT_ANY_THROW(VisualIndex::Read(path));
}

TEST(VisualIndexWeights, SkipUntrainedWordsInAppearanceScores) {
  using Index = InvertedIndex<float, 32, 16>;
  Index index;
  index.Initialize(2);
  const Index::DescType training = Index::DescType::Ones(8, 32);
  index.ComputeHammingEmbedding(training, Index::WordIds::Zero(8, 1));
  const Index::DescType descriptor = training.topRows(1);
  index.AddEntry(1, 0, 0, descriptor, {});
  index.AddEntry(2, 1, 0, descriptor, {});
  index.Finalize();
  std::vector<ImageScore> scores;
  Index::WordIds query_word(1, 1);
  query_word(0) = 1;
  index.Query(descriptor, query_word, &scores);
  EXPECT_TRUE(scores.empty());
  EXPECT_FLOAT_EQ(index.SquaredIDFWeight(1), 0);
  EXPECT_FLOAT_EQ(index.ComputeSelfSimilarity(query_word), 0);
  query_word(0) = 0;
  index.Query(descriptor, query_word, &scores);
  ASSERT_EQ(scores.size(), 1);
  EXPECT_EQ(scores.front().image_id, 1);
}

TEST(VisualIndexWeights, MultipleAssignmentsKeepTheStrongestCorrespondence) {
  // Word 0 occurs in both images and has no discriminative weight. Word 1
  // occurs only in the target image and separates the six descriptor codes.
  using Index = InvertedIndex<float, 32, 16>;
  Index postings;
  postings.Initialize(2);
  Index::DescType training = Index::DescType::Zero(10, 32);
  training.topRows(5).setConstant(10);
  Index::WordIds labels = Index::WordIds::Ones(10, 1);
  labels.topRows(5).setZero();
  postings.ComputeHammingEmbedding(training, labels);

  constexpr int kFeatures = 6;
  FeatureDescriptorsFloat query(FeatureExtractorType::SIFT,
                                Index::DescType::Zero(kFeatures, 32));
  FeatureKeypoints keypoints;
  for (int i = 0; i < kFeatures; ++i) {
    for (int j = 0; j < 16; ++j) {
      query.data(i, j) = std::bitset<4>(i & j).count() % 2 ? 1 : -1;
    }
    FeatureGeometry g;
    g.x = 80 * i;
    g.y = 70 * ((i * i + 1) % 7);
    g.scale = 1;
    keypoints.emplace_back(g.x, g.y, 1, 0);
  }
  for (int i = kFeatures - 1; i >= 0; --i) {
    FeatureGeometry g;
    g.x = keypoints[i].x;
    g.y = keypoints[i].y;
    g.scale = 1;
    for (const int word : {0, 1}) {
      postings.AddEntry(7, word, kFeatures - 1 - i, query.data.row(i), g);
    }
  }
  postings.AddEntry(9, 0, 0, query.data.row(0), {});
  postings.Finalize();

  // The query's nearest word is always 0. Search both words so every true
  // correspondence is encountered first with zero weight.
  faiss::IndexFlatL2 quantizer(32);
  faiss::IndexIVFFlat vocabulary(&quantizer, 32, 1);
  Index::DescType centers = Index::DescType::Zero(2, 32);
  centers.row(1).setConstant(5);
  vocabulary.train(2, centers.data());
  vocabulary.add(2, centers.data());
  const auto path = CreateTestDir() / "words.bin";
  faiss::VectorIOWriter encoded;
  faiss::write_index(&vocabulary, &encoded);
  {
    std::ofstream file(path, std::ios::binary);
    // VisualIndex version 2, descriptor width, embedding width, extractor.
    const int header[] = {
        2, 32, 16, static_cast<int>(FeatureExtractorType::SIFT)};
    file.write(reinterpret_cast<const char*>(header), sizeof(header));
    file.write(reinterpret_cast<const char*>(encoded.data.data()),
               encoded.data.size());
    postings.Write(&file, {7, 9});
  }
  auto index = VisualIndex::Read(path);
  index->Prepare();
  VisualIndex::QueryOptions opts;
  opts.num_neighbors = 2;
  opts.image_id_filter = [](int image_id) { return image_id == 7; };
  std::vector<ImageScore> scores;
  index->Query(opts, query, &scores);
  ASSERT_EQ(scores.size(), 1);
  const float before_verification = scores.front().score;
  opts.num_images_after_verification = 1;
  index->Query(opts, keypoints, query, &scores);
  ASSERT_EQ(scores.size(), 1);
  EXPECT_NEAR(scores.front().score - before_verification, kFeatures, 1e-5);
}

class ParameterizedVisualIndexTests
    : public ::testing::TestWithParam<
          std::pair</*embedding_dim=*/int, /*embedding_dim=*/int>> {};

TEST_P(ParameterizedVisualIndexTests, Nominal) {
  const auto [desc_dim, embedding_dim] = GetParam();

  {
    auto visual_index = VisualIndex::Create(desc_dim, embedding_dim);
    EXPECT_EQ(visual_index->NumVisualWords(), 0);
  }

  VisualIndex::BuildOptions build_options;
  // Keep test runtimes low.
  build_options.num_iterations = 10;
  build_options.num_rounds = 1;

  {
    auto descriptors =
        CreateRandomDescriptors(200, desc_dim, FeatureExtractorType::SIFT);
    auto visual_index = VisualIndex::Create(desc_dim, embedding_dim);
    EXPECT_EQ(visual_index->NumVisualWords(), 0);
    EXPECT_EQ(visual_index->NumImages(), 0);
    EXPECT_EQ(visual_index->DescDim(), desc_dim);
    EXPECT_EQ(visual_index->EmbeddingDim(), embedding_dim);
    build_options.num_visual_words = 5;
    visual_index->Build(build_options, descriptors);
    EXPECT_EQ(visual_index->NumVisualWords(), 5);
    EXPECT_EQ(visual_index->FeatureType(), FeatureExtractorType::SIFT);
  }

  {
    auto descriptors =
        CreateRandomDescriptors(4096, desc_dim, FeatureExtractorType::SIFT);
    auto visual_index = VisualIndex::Create(desc_dim, embedding_dim);
    EXPECT_EQ(visual_index->NumVisualWords(), 0);
    EXPECT_EQ(visual_index->NumImages(), 0);
    build_options.num_visual_words = 512;
    visual_index->Build(build_options, descriptors);
    EXPECT_EQ(visual_index->NumVisualWords(), 512);
  }

  {
    auto descriptors =
        CreateRandomDescriptors(1000, desc_dim, FeatureExtractorType::SIFT);
    auto visual_index = VisualIndex::Create(desc_dim, embedding_dim);
    EXPECT_EQ(visual_index->NumVisualWords(), 0);
    EXPECT_EQ(visual_index->NumImages(), 0);
    build_options.num_visual_words = 100;
    visual_index->Build(build_options, descriptors);
    EXPECT_EQ(visual_index->NumVisualWords(), 100);

    VisualIndex::IndexOptions index_options;
    FeatureKeypoints keypoints1(50);
    auto descriptors1 =
        CreateRandomDescriptors(50, desc_dim, FeatureExtractorType::SIFT);
    visual_index->Add(index_options, 1, keypoints1, descriptors1);
    EXPECT_EQ(visual_index->NumImages(), 1);
    FeatureKeypoints keypoints2(50);
    auto descriptors2 =
        CreateRandomDescriptors(50, desc_dim, FeatureExtractorType::SIFT);
    visual_index->Add(index_options, 2, keypoints2, descriptors2);
    EXPECT_EQ(visual_index->NumImages(), 2);
    visual_index->Prepare();

    VisualIndex::QueryOptions query_options;
    std::vector<ImageScore> image_scores;
    visual_index->Query(query_options, descriptors1, &image_scores);
    EXPECT_EQ(image_scores.size(), 2);
    EXPECT_EQ(image_scores[0].image_id, 1);
    EXPECT_EQ(image_scores[1].image_id, 2);
    EXPECT_GT(image_scores[0].score, image_scores[1].score);

    query_options.max_num_images = 1;
    visual_index->Query(query_options, descriptors1, &image_scores);
    EXPECT_EQ(image_scores.size(), 1);
    EXPECT_EQ(image_scores[0].image_id, 1);

    query_options.max_num_images = 3;
    visual_index->Query(query_options, descriptors1, &image_scores);
    EXPECT_EQ(image_scores.size(), 2);
    EXPECT_EQ(image_scores[0].image_id, 1);
    EXPECT_EQ(image_scores[1].image_id, 2);
    EXPECT_GT(image_scores[0].score, image_scores[1].score);

    query_options.max_num_images = 1;
    query_options.image_id_filter = [](const int image_id) {
      return image_id != 1;
    };
    visual_index->Query(query_options, descriptors1, &image_scores);
    ASSERT_EQ(image_scores.size(), 1);
    EXPECT_EQ(image_scores[0].image_id, 2);
  }
}

TEST_P(ParameterizedVisualIndexTests, ReadWrite) {
  const auto [desc_dim, embedding_dim] = GetParam();
  const auto test_dir = CreateTestDir();
  const auto vocab_tree_path = test_dir / "vocab_tree.bin";

  VisualIndex::BuildOptions build_options;
  // Keep test runtimes low.
  build_options.num_iterations = 10;
  build_options.num_rounds = 1;

  auto descriptors =
      CreateRandomDescriptors(200, desc_dim, FeatureExtractorType::SIFT);
  auto visual_index = VisualIndex::Create(desc_dim, embedding_dim);
  EXPECT_EQ(visual_index->NumVisualWords(), 0);
  EXPECT_EQ(visual_index->DescDim(), desc_dim);
  EXPECT_EQ(visual_index->EmbeddingDim(), embedding_dim);
  build_options.num_visual_words = 5;
  visual_index->Build(build_options, descriptors);
  EXPECT_EQ(visual_index->NumVisualWords(), 5);
  EXPECT_EQ(visual_index->FeatureType(), FeatureExtractorType::SIFT);

  VisualIndex::IndexOptions index_options;
  FeatureKeypoints keypoints1(50);
  auto descriptors1 =
      CreateRandomDescriptors(50, desc_dim, FeatureExtractorType::SIFT);
  visual_index->Add(index_options, 1, keypoints1, descriptors1);
  FeatureKeypoints keypoints2(50);
  auto descriptors2 =
      CreateRandomDescriptors(50, desc_dim, FeatureExtractorType::SIFT);
  visual_index->Add(index_options, 2, keypoints2, descriptors2);

  EXPECT_EQ(visual_index->NumImages(), 2);

  visual_index->Write(vocab_tree_path);
  auto read_visual_index = VisualIndex::Read(vocab_tree_path);
  EXPECT_EQ(visual_index->NumVisualWords(),
            read_visual_index->NumVisualWords());
  EXPECT_EQ(visual_index->NumImages(), read_visual_index->NumImages());
  EXPECT_EQ(visual_index->DescDim(), read_visual_index->DescDim());
  EXPECT_EQ(visual_index->EmbeddingDim(), read_visual_index->EmbeddingDim());
  EXPECT_EQ(visual_index->FeatureType(), read_visual_index->FeatureType());
  EXPECT_TRUE(visual_index->IsImageIndexed(1));
  EXPECT_TRUE(visual_index->IsImageIndexed(2));
  EXPECT_FALSE(visual_index->IsImageIndexed(3));
}

TEST_P(ParameterizedVisualIndexTests, SpatialVerification) {
  const auto [desc_dim, embedding_dim] = GetParam();

  VisualIndex::BuildOptions build_options;
  // Keep test runtimes low.
  build_options.num_iterations = 10;
  build_options.num_rounds = 1;
  build_options.num_visual_words = 200;
  // Use a single thread for deterministic results.
  build_options.num_threads = 1;

  auto train_descriptors =
      CreateRandomDescriptors(1000, desc_dim, FeatureExtractorType::SIFT);
  auto visual_index = VisualIndex::Create(desc_dim, embedding_dim);
  visual_index->Build(build_options, train_descriptors);
  EXPECT_EQ(visual_index->NumVisualWords(), build_options.num_visual_words);

  // Add multiple images with geometries and descriptors
  VisualIndex::IndexOptions index_options;
  constexpr int kNumImages = 5;
  constexpr int kNumFeatures = 50;

  FeatureKeypoints query_keypoints(kNumFeatures);
  for (size_t j = 0; j < query_keypoints.size(); ++j) {
    // Create a grid of keypoints (non-collinear)
    query_keypoints[j].x = static_cast<float>((j % 10) * kNumFeatures);
    query_keypoints[j].y = static_cast<float>((j / 10.f) * kNumFeatures);
  }

  // Use the same descriptors for all images.
  const auto descriptors = CreateRandomDescriptors(
      kNumFeatures, desc_dim, FeatureExtractorType::SIFT);

  for (int i = 0; i < kNumImages; ++i) {
    FeatureKeypoints keypoints(kNumFeatures);
    // For every other image, create keypoints with different but consistent
    // translational transformations to the query keypoints. The other outlier
    // images have inconsistent spatial transformations with random keypoints.
    if (i % 2 == 0) {
      // Create keypoints with different spatial transformations.
      const float x_offset = static_cast<float>(i * 20);
      const float y_offset = static_cast<float>(i * 30);
      for (size_t j = 0; j < keypoints.size(); ++j) {
        keypoints[j].x = query_keypoints[j].x + x_offset;
        keypoints[j].y = query_keypoints[j].y + y_offset;
      }
    } else {
      for (size_t j = 0; j < keypoints.size(); ++j) {
        const Eigen::Vector2f offset = RandomEigenVectorf<2>();
        keypoints[j].x = offset.x();
        keypoints[j].y = offset.y();
      }
    }

    visual_index->Add(index_options, /*image_id=*/i, keypoints, descriptors);
  }

  EXPECT_EQ(visual_index->NumImages(), kNumImages);
  visual_index->Prepare();

  VisualIndex::QueryOptions query_options;

  std::vector<ImageScore> image_scores_without_verification;
  visual_index->Query(query_options,
                      query_keypoints,
                      descriptors,
                      &image_scores_without_verification);
  EXPECT_EQ(image_scores_without_verification.size(), kNumImages);
  for (const auto& score : image_scores_without_verification) {
    EXPECT_LT(score.score, 1.0f);
  }

  query_options.num_images_after_verification = kNumImages;
  std::vector<ImageScore> image_scores_with_verification;
  visual_index->Query(query_options,
                      query_keypoints,
                      descriptors,
                      &image_scores_with_verification);
  EXPECT_EQ(image_scores_with_verification.size(), kNumImages);
  // Make sure inlier images have more keypoints as inliers.
  for (const auto& score : image_scores_with_verification) {
    if (score.image_id % 2 == 0) {
      EXPECT_GT(score.score, kNumFeatures / 2);
    } else {
      EXPECT_LT(score.score, kNumFeatures / 4);
    }
  }

  // Test with max_num_images constraint, should respect max_num_images.
  query_options.max_num_images = 2;
  query_options.num_images_after_verification = 4;
  std::vector<ImageScore> image_scores_limited;
  visual_index->Query(
      query_options, query_keypoints, descriptors, &image_scores_limited);
  EXPECT_LE(image_scores_limited.size(), query_options.max_num_images);
}

TEST_P(ParameterizedVisualIndexTests, TypeMismatch) {
  const auto [desc_dim, embedding_dim] = GetParam();

  VisualIndex::BuildOptions build_options;
  build_options.num_iterations = 10;
  build_options.num_rounds = 1;
  build_options.num_visual_words = 10;

  // Build index with SIFT descriptors.
  auto sift_descriptors =
      CreateRandomDescriptors(100, desc_dim, FeatureExtractorType::SIFT);
  auto visual_index = VisualIndex::Create(desc_dim, embedding_dim);
  visual_index->Build(build_options, sift_descriptors);
  EXPECT_EQ(visual_index->FeatureType(), FeatureExtractorType::SIFT);

  // Adding descriptors with mismatched type should throw.
  VisualIndex::IndexOptions index_options;
  FeatureKeypoints keypoints(10);
  auto undefined_descriptors =
      CreateRandomDescriptors(10, desc_dim, FeatureExtractorType::UNDEFINED);
  EXPECT_THROW(
      visual_index->Add(index_options, 1, keypoints, undefined_descriptors),
      std::invalid_argument);

  // Adding with correct type should succeed.
  auto sift_descriptors_small =
      CreateRandomDescriptors(10, desc_dim, FeatureExtractorType::SIFT);
  EXPECT_NO_THROW(
      visual_index->Add(index_options, 1, keypoints, sift_descriptors_small));
  visual_index->Prepare();

  // Querying with mismatched type should throw.
  VisualIndex::QueryOptions query_options;
  std::vector<ImageScore> image_scores;
  EXPECT_THROW(
      visual_index->Query(query_options, undefined_descriptors, &image_scores),
      std::invalid_argument);

  // Querying with correct type should succeed.
  EXPECT_NO_THROW(visual_index->Query(
      query_options, sift_descriptors_small, &image_scores));
}

TEST(VisualIndex, Print) {
  VisualIndex::BuildOptions build_options;
  build_options.num_iterations = 10;
  build_options.num_rounds = 1;
  build_options.num_visual_words = 10;

  auto descriptors =
      CreateRandomDescriptors(100, 128, FeatureExtractorType::SIFT);
  auto visual_index = VisualIndex::Create(128, 64);
  visual_index->Build(build_options, descriptors);

  VisualIndex::IndexOptions index_options;
  FeatureKeypoints keypoints(10);
  auto descriptors_small =
      CreateRandomDescriptors(10, 128, FeatureExtractorType::SIFT);
  visual_index->Add(index_options, 1, keypoints, descriptors_small);
  visual_index->Add(index_options, 2, keypoints, descriptors_small);

  std::ostringstream stream;
  stream << *visual_index;
  EXPECT_EQ(stream.str(),
            "VisualIndex(num_visual_words=10, num_images=2, desc_dim=128, "
            "embedding_dim=64, feature_type=SIFT)");
}

TEST(VisualIndex, CreateWithDescriptorDimension) {
  for (const auto& [desc_dim, embedding_dim] : {std::make_pair(32, 16),
                                                std::make_pair(128, 64),
                                                std::make_pair(256, 64)}) {
    const auto visual_index = VisualIndex::Create(desc_dim);
    EXPECT_EQ(visual_index->DescDim(), desc_dim);
    EXPECT_EQ(visual_index->EmbeddingDim(), embedding_dim);
  }
}

INSTANTIATE_TEST_SUITE_P(VisualIndexTests,
                         ParameterizedVisualIndexTests,
                         ::testing::Values(std::make_pair(128, 64),
                                           std::make_pair(32, 16),
                                           std::make_pair(256, 64)));

#ifdef COLMAP_DOWNLOAD_ENABLED

TEST(VisualIndex, Download) {
  const auto test_dir = CreateTestDir();
  const auto vocab_tree_path = test_dir / "server_vocab_tree.bin";
  OverwriteDownloadCacheDir(test_dir);

  auto descriptors =
      CreateRandomDescriptors(50, 32, FeatureExtractorType::SIFT);
  auto visual_index = VisualIndex::Create(32, 16);
  VisualIndex::BuildOptions build_options;
  build_options.num_visual_words = 5;
  // NOLINTNEXTLINE(clang-analyzer-optin.cplusplus.VirtualCall)
  visual_index->Build(build_options, descriptors);
  visual_index->Write(vocab_tree_path);

  std::vector<char> vocab_tree_data;
  ReadBinaryBlob(vocab_tree_path, &vocab_tree_data);
  const std::string vocab_tree_uri =
      "file://" + std::filesystem::absolute(vocab_tree_path).string() +
      ";vocab_tree.bin;" +
      ComputeSHA256({vocab_tree_data.data(), vocab_tree_data.size()});
  auto downloaded_visual_index = VisualIndex::Read(vocab_tree_uri);
  EXPECT_EQ(downloaded_visual_index->NumVisualWords(),
            visual_index->NumVisualWords());
  EXPECT_EQ(downloaded_visual_index->FeatureType(),
            visual_index->FeatureType());
}

#endif

}  // namespace
}  // namespace retrieval
}  // namespace colmap
