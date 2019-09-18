#include <ros/ros.h>
#include <pcs_detection/hsv_thresholding.h>
#include <pcs_msgs/ImageProcessing.h>
#include <cv_bridge/cv_bridge.h>

namespace pcs_ros
{
bool process_image(pcs_msgs::ImageProcessing::Request& req, pcs_msgs::ImageProcessing::Response& res)
{
  ROS_DEBUG("Processing image with hsv thresholding");
  // Convert to OpenCV format
  cv_bridge::CvImagePtr cv_ptr;
  cv_ptr = cv_bridge::toCvCopy(req.input_image, req.input_image.encoding);

  // Perform Thresholding
  cv::Mat mask;
  pcs_detection::hsvThresholdingDetector(cv_ptr->image, mask);

  // Convert to the size of the input
  cv::Mat mask_3channel;
  cv::Mat in[] = { mask, mask, mask };
  cv::merge(in, 3, mask_3channel);
  cv_ptr->image = mask_3channel;

  // Convert to ROS msg
  res.returned_image = *cv_ptr->toImageMsg();
  return true;
}
}  // namespace pcs_ros

int main(int argc, char** argv)
{
  ros::init(argc, argv, "hsv_thresholding_server");
  ros::NodeHandle nh;

  ros::ServiceServer service = nh.advertiseService("perform_detection", pcs_ros::process_image);
  ROS_INFO("HSV thresholding service is available");
  ros::spin();

  return 0;
}
