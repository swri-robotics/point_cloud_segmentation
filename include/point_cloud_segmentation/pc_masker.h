#include <ros/ros.h>
#include <opencv2/core/mat.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace pc_segmentation
{
const bool DEBUG = true;

/**
 * @brief The MaskDetectionRepublisher class
 *
 * TODO
 * Header/General Refactor to clean up
 * Add publishers for the images
 * Unit tests
 *  - Refactor thesholding_mask_detector into base class
 *    - Test on base class - correct input/output size, outputs are either 1 or 0
 *  - Apply Mask: input and output are the same size, the masking has been applied
 */
class PCMasker
{
public:
  PCMasker(ros::NodeHandle nh, const std::string& sub_topic, const std::string& pub_topic);

  void newPCCallback(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& scan);

private:
  std::string point_cloud_sub_topic_;
  std::string masked_depth_image_pub_topic_;
  ros::NodeHandle nh_;
  ros::Subscriber point_cloud_sub_;
  ros::Publisher masked_depth_image_pub_;
};

}  // namespace pc_segmentation
