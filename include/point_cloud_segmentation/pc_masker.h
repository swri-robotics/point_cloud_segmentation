#include <ros/ros.h>
#include <opencv2/core/mat.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace pc_segmentation
{
/**
 * @brief The MaskDetectionRepublisher class
 *
 * TODO
 * Refactor to pass in masking function
 * Unit tests
 *  - Refactor thesholding_mask_detector into base class
 *    - Test on base class - correct input/output size, outputs are either 1 or 0
 *  - Apply Mask: input and output are the same size, the masking has been applied
 */
class PCMasker
{
public:
  PCMasker(ros::NodeHandle nh, const std::string& sub_topic, const std::string& pub_topic);

  /** @brief Callback when for input point cloud. This does all of the processing */
  void newPCCallback(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& scan);

  /** @brief If true, the extracted image and masked image are published during processing */
  bool debug_publisher_ = false;
  /** @brief If true, intermediate steps are displayed in an OpenCV window */
  bool debug_viewer_ = false;

private:
  /** @brief Topic used for the input point cloud*/
  std::string point_cloud_sub_topic_;
  /** @brief Topic for the resulting masked point cloud*/
  std::string masked_point_cloud_pub_topic_;
  /** @brief Nodehandle associated with the publishers and subscribers*/
  ros::NodeHandle nh_;
  /** @brief Subscriber to input point cloud*/
  ros::Subscriber point_cloud_sub_;
  /** @brief Publisher for the resulting masked point cloud*/
  ros::Publisher masked_point_cloud_pub_;
  /** @brief Publisher used for extracted image when debug_publisher_=true*/
  ros::Publisher debug_extracted_image_pub_;
  /** @brief Publisher used for masked image when debug_publisher_=true*/
  ros::Publisher debug_masked_image_pub_;
  /** @brief Service client used to process the image */
  ros::ServiceClient image_processing_client_;
};

}  // namespace pc_segmentation
