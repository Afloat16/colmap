// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/sensor/database.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(CameraDatabase, Initialization) {
  CameraDatabase database;
  camera_specs_t specs = InitializeCameraSpecs();
  EXPECT_EQ(database.NumEntries(), specs.size());
}

TEST(CameraDatabase, ExactMatch) {
  CameraDatabase database;
  double sensor_width;
  EXPECT_TRUE(
      database.QuerySensorWidth("canon", "digitalixus100is", &sensor_width));
  EXPECT_EQ(sensor_width, 6.1600f);
}

TEST(CameraDatabase, AmbiguousMatch) {
  CameraDatabase database;
  double sensor_width;
  EXPECT_TRUE(
      !database.QuerySensorWidth("canon", "digitalixus1", &sensor_width));
  EXPECT_EQ(sensor_width, 6.1600f);
}

TEST(CameraDatabase, ExactMatchAfterAmbiguousPrefixes) {
  CameraDatabase database;
  double sensor_width = 0;
  // Earlier entries include both EOS 5D Mark III and EOS 5D Mark II.
  EXPECT_TRUE(
      database.QuerySensorWidth("Canon", "Canon EOS 5D", &sensor_width));
  EXPECT_EQ(sensor_width, 35.8000f);
  // D100 and D1H must not hide the later exact D1 entry.
  EXPECT_TRUE(database.QuerySensorWidth("Nikon", "Nikon D1", &sensor_width));
  EXPECT_EQ(sensor_width, 23.7000f);
  EXPECT_TRUE(database.QuerySensorWidth("Nikon", "Nikon D3", &sensor_width));
  EXPECT_EQ(sensor_width, 36.0000f);
  // "digitalixus" is itself a model in the database, not only a family name.
  EXPECT_TRUE(
      database.QuerySensorWidth("Canon", "Digital IXUS", &sensor_width));
  EXPECT_EQ(sensor_width, 5.3300f);
}

TEST(CameraDatabase, AmbiguousPrefixesWithoutExactMatch) {
  CameraDatabase database;
  double sensor_width = 0;
  EXPECT_FALSE(
      database.QuerySensorWidth("Canon", "EOS 5D Mark", &sensor_width));
  EXPECT_FALSE(database.QuerySensorWidth("Nikon", "D", &sensor_width));
}

}  // namespace
}  // namespace colmap
