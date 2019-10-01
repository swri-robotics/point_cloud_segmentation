#include <tesseract_common/macros.h>
TESSERACT_COMMON_IGNORE_WARNINGS_PUSH
#include <algorithm>
#include <console_bridge/console.h>
#include <gtest/gtest.h>
#include <memory>
#include <tesseract_geometry/mesh_parser.h>
TESSERACT_COMMON_IGNORE_WARNINGS_POP

#include "pcs_scan_integration/octomap_mesh_masking.h"

using namespace pcs_scan_integration;

static const bool SAVE_OUTPUT = false;

class OctomapMeshMaskUnit : public ::testing::Test
{
protected:
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;

  void SetUp() override
  {
    CONSOLE_BRIDGE_logInform("Setting up OctomapMeshMaskingUnit");
    cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
    cloud->width = 255;
    cloud->height = 1;
    cloud->is_dense = false;
    cloud->points.resize(cloud->width * cloud->height);

    // Create a point cloud which consists of points in a row increasing in color value from 0 to 255
    for (size_t i = 0; i < cloud->points.size(); ++i)
    {
      cloud->points[i].x = static_cast<float>(i) / 100.f;
      cloud->points[i].y = 0.0;
      cloud->points[i].z = 0.0;
      cloud->points[i].r = static_cast<uint8_t>(i);
      cloud->points[i].g = static_cast<uint8_t>(i);
      cloud->points[i].b = static_cast<uint8_t>(i);
    }
  }
};

TEST_F(OctomapMeshMaskUnit, Construction) {}

TEST_F(OctomapMeshMaskUnit, getSet)
{
  OctomapMeshMask masker;
  EXPECT_TRUE(masker.getOctree() == nullptr);
  EXPECT_TRUE(masker.getInputMesh() == nullptr);
  EXPECT_TRUE(masker.getMaskedMesh() == nullptr);

  // Get/Set Octomap
  {
    auto octree =
        std::make_shared<tesseract_geometry::Octree>(*cloud, 0.1, tesseract_geometry::Octree::SubType::BOX, true);
    masker.setOctree(octree);
    EXPECT_TRUE(masker.getOctree() != nullptr);
    masker.setOctree(nullptr);
    masker.setOctree(cloud, 0.1);
    EXPECT_TRUE(masker.getOctree() != nullptr);
  }
  // Set Octree from Point Cloud
  {
    masker.setOctree(nullptr);
    masker.setOctree(cloud, 0.1, 0, 255, false);
    tesseract_geometry::Octree::ConstPtr octree = masker.getOctree();
    // Since this function is mostly just calling the Tesseract constructor, we can mostly rely on the Tesseract unit
    // tests
    EXPECT_TRUE(octree != nullptr);
  }
  // Get/Set Mesh
  {
    std::string path = std::string(DATA_DIR) + "/plane_4m.stl";
    masker.setInputMesh(path);
    auto input_mesh = masker.getInputMesh();
    EXPECT_EQ(input_mesh->getVertices()->size(), 6273);
    EXPECT_EQ(input_mesh->getTriangles()->size(), 12288 * 4);
  }
  {
    std::string path = std::string(DATA_DIR) + "/box_2m.ply";
    tesseract_geometry::Mesh::Ptr mesh =
        tesseract_geometry::createMeshFromPath<tesseract_geometry::Mesh>(path, Eigen::Vector3d(1, 1, 1), true)[0];
    masker.setInputMesh(mesh);
    auto input_mesh = masker.getInputMesh();
    EXPECT_EQ(input_mesh->getVertices()->size(), 8);
    EXPECT_EQ(input_mesh->getTriangles()->size(), 12 * 4);
  }
}

TEST_F(OctomapMeshMaskUnit, colorPassthrough)
{
  {
    const bool limit_neg = false;
    // Loop over various options for limits
    for (int lower_lim = 0; lower_lim < 255; lower_lim += 25)
    {
      for (int upper_lim = lower_lim; upper_lim < 255; upper_lim += 25)
      {
        // Apply color filter
        auto result_cloud = pcs_scan_integration::colorPassthrough(cloud, lower_lim, upper_lim, limit_neg);
        // Check that the color filter is obeyed
        EXPECT_EQ(result_cloud->size(), upper_lim - lower_lim - (upper_lim == lower_lim ? 0 : 1));
        for (const pcl::PointXYZRGB& point : result_cloud->points)
        {
          EXPECT_LT(point.r, upper_lim);
          EXPECT_LT(point.g, upper_lim);
          EXPECT_LT(point.b, upper_lim);
          EXPECT_GT(point.r, lower_lim);
          EXPECT_GT(point.g, lower_lim);
          EXPECT_GT(point.b, lower_lim);
        }
      }
    }
  }
  {
    const bool limit_neg = true;
    // Loop over various options for limits
    for (int lower_lim = 0; lower_lim < 255; lower_lim += 25)
    {
      for (int upper_lim = lower_lim; upper_lim < 255; upper_lim += 25)
      {
        // Apply color filter
        auto result_cloud = pcs_scan_integration::colorPassthrough(cloud, lower_lim, upper_lim, limit_neg);
        // Check that the color filter is obeyed
        EXPECT_EQ(result_cloud->size(), cloud->size() - (upper_lim - lower_lim) - 1);
        for (const pcl::PointXYZRGB& point : result_cloud->points)
        {
          EXPECT_FALSE(((point.r > lower_lim) && (point.r < upper_lim)));
          EXPECT_FALSE(((point.g > lower_lim) && (point.g < upper_lim)));
          EXPECT_FALSE(((point.b > lower_lim) && (point.b < upper_lim)));
        }
      }
    }
  }
}

TEST_F(OctomapMeshMaskUnit, maskMesh)
{
  OctomapMeshMask masker;
  std::string path = std::string(DATA_DIR) + "/box_2m.bt";
  std::shared_ptr<octomap::OcTree> ot(new octomap::OcTree(path));
  tesseract_geometry::Octree::Ptr octree(new tesseract_geometry::Octree(ot, tesseract_geometry::Octree::SubType::BOX));
  masker.setOctree(octree);

  tesseract_geometry::Mesh::Ptr mesh = tesseract_geometry::createMeshFromPath<tesseract_geometry::Mesh>(
      std::string(DATA_DIR) + "/plane_4m.stl", Eigen::Vector3d(1, 1, 1), true)[0];
  masker.setInputMesh(mesh);

  {
    EXPECT_TRUE(masker.maskMesh(OctomapMeshMask::MaskType::RETURN_COLORIZED));
    if (SAVE_OUTPUT)
    {
      std::string output_path = std::string(DATA_DIR) + "/results/test_output_RETURN_COLORIZED.ply";
      EXPECT_TRUE(masker.saveMaskedMesh(output_path));
      CONSOLE_BRIDGE_logDebug("Saving file to data directory");
    }
    tesseract_geometry::Mesh::Ptr results = masker.getMaskedMesh();
    EXPECT_EQ(results->getVertices()->size(), 6273);
    EXPECT_EQ(results->getTriangles()->size(), 12288 * 4);
  }
  {
    EXPECT_TRUE(masker.maskMesh(OctomapMeshMask::MaskType::RETURN_INSIDE));
    if (SAVE_OUTPUT)
    {
      std::string output_path = std::string(DATA_DIR) + "/results/test_output_RETURN_INSIDE.ply";
      EXPECT_TRUE(masker.saveMaskedMesh(output_path));
      CONSOLE_BRIDGE_logDebug("Saving file to data directory");
    }
    tesseract_geometry::Mesh::Ptr results = masker.getMaskedMesh();
    EXPECT_EQ(results->getVertices()->size(), 5928);
    EXPECT_EQ(results->getTriangles()->size(), 1976 * 4);
  }
  {
    EXPECT_TRUE(masker.maskMesh(OctomapMeshMask::MaskType::RETURN_OUTSIDE));
    if (SAVE_OUTPUT)
    {
      std::string output_path = std::string(DATA_DIR) + "/results/test_output_RETURN_OUTSIDE.ply";
      EXPECT_TRUE(masker.saveMaskedMesh(output_path));
      CONSOLE_BRIDGE_logDebug("Saving file to data directory");
    }
    tesseract_geometry::Mesh::Ptr results = masker.getMaskedMesh();
    EXPECT_EQ(results->getVertices()->size(), 34200);
    EXPECT_EQ(results->getTriangles()->size(), 11400 * 4);
  }
}

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
