#include <tesseract_common/macros.h>
TESSERACT_COMMON_IGNORE_WARNINGS_PUSH
#include <console_bridge/console.h>
#include <gtest/gtest.h>
TESSERACT_COMMON_IGNORE_WARNINGS_POP

#include <pcs_detection/utils.h>

using namespace pcs_detection;

class UtilsUnit : public ::testing::Test
{
protected:
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;

  void SetUp() override
  {
    CONSOLE_BRIDGE_logInform("Setting up PointCloudAnnotatorUnit");
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

TEST_F(UtilsUnit, CloudImageManipulation)
{
  CONSOLE_BRIDGE_logDebug("UtilsUnit, CloudImageManipulation");

  auto position_image = std::make_shared<cv::Mat>();
  auto image_2d = std::make_shared<cv::Mat>();
  cloudToImage(cloud, *position_image, *image_2d);
  EXPECT_EQ(position_image->rows, image_2d->rows);
  EXPECT_EQ(position_image->cols, image_2d->cols);

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr output_cloud = imageToCloud(*image_2d, *position_image, cloud->header);

  // Chech that the results are the same as the input when the output of cloudToImage is passed into imageToCloud
  EXPECT_EQ(cloud->points.size(), output_cloud->points.size());
  EXPECT_EQ(cloud->width, output_cloud->width);
  EXPECT_EQ(cloud->height, output_cloud->height);
  EXPECT_EQ(cloud->is_dense, output_cloud->is_dense);
  for (std::size_t i = 0; i < cloud->points.size(); i++)
  {
    EXPECT_EQ(cloud->points[i].x, output_cloud->points[i].x);
    EXPECT_EQ(cloud->points[i].y, output_cloud->points[i].y);
    EXPECT_EQ(cloud->points[i].z, output_cloud->points[i].z);
    EXPECT_EQ(cloud->points[i].r, output_cloud->points[i].r);
    EXPECT_EQ(cloud->points[i].g, output_cloud->points[i].g);
    EXPECT_EQ(cloud->points[i].b, output_cloud->points[i].b);
  }
}

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
