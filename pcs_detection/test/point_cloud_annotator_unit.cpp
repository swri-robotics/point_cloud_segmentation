#include <tesseract_common/macros.h>
TESSERACT_COMMON_IGNORE_WARNINGS_PUSH
#include <console_bridge/console.h>
#include <gtest/gtest.h>
TESSERACT_COMMON_IGNORE_WARNINGS_POP

#include <pcs_detection/point_cloud_annotator.h>

using namespace pcs_detection;

class PointCloudAnnotatorUnit : public ::testing::Test
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

std::vector<cv::Mat> ImageAnnotatorCallback_ReturnInputs(const std::vector<cv::Mat> input_images)
{
  CONSOLE_BRIDGE_logDebug("I am tired of annotating. I'm just going to return the inputs..");
  return input_images;
}

std::vector<cv::Mat> ImageAnnotatorCallback_Throw(const std::vector<cv::Mat> input_images)
{
  CONSOLE_BRIDGE_logDebug("Throwing a test exception. This is only a drill. This is only a drill.");
  throw std::exception();
}

void ResultsCallback_ReturnCleanly(std::vector<pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr> results)
{
  CONSOLE_BRIDGE_logDebug("I got your results, and they are appreciated. Returning cleanly.");
}

void ResultsCallback_Throw(std::vector<pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr> results)
{
  CONSOLE_BRIDGE_logDebug("Throwing a test exception. This is only a drill. This is only a drill.");
  throw std::exception();
}

TEST_F(PointCloudAnnotatorUnit, Construction)
{
  CONSOLE_BRIDGE_logDebug("PointCloudAnnotatorUnit, Construction");
  pcs_detection::PointCloudAnnotator pca1(&ImageAnnotatorCallback_ReturnInputs, &ResultsCallback_ReturnCleanly);
  pcs_detection::PointCloudAnnotator pca2(&ImageAnnotatorCallback_Throw, &ResultsCallback_Throw, 123);
}

TEST_F(PointCloudAnnotatorUnit, addPointCloud)
{
  CONSOLE_BRIDGE_logDebug("PointCloudAnnotatorUnit, addPointCloud");
  {
    pcs_detection::PointCloudAnnotator pca1(&ImageAnnotatorCallback_ReturnInputs, &ResultsCallback_ReturnCleanly, 2);
    EXPECT_TRUE(pca1.addPointCloud(cloud));
    pca1.addPointCloud(cloud);
  }
  {
    // Should fail as soon as the buffer gets big enough to trigger annotateImages
    pcs_detection::PointCloudAnnotator pca1(&ImageAnnotatorCallback_Throw, &ResultsCallback_ReturnCleanly, 3);
    EXPECT_TRUE(pca1.addPointCloud(cloud));
    EXPECT_TRUE(pca1.addPointCloud(cloud));
    EXPECT_FALSE(pca1.addPointCloud(cloud));
  }
  {
    // Should fail as soon as the buffer gets big enough to trigger annotateImages
    pcs_detection::PointCloudAnnotator pca1(&ImageAnnotatorCallback_ReturnInputs, &ResultsCallback_Throw, 3);
    EXPECT_TRUE(pca1.addPointCloud(cloud));
    EXPECT_TRUE(pca1.addPointCloud(cloud));
    EXPECT_FALSE(pca1.addPointCloud(cloud));
  }
}

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
