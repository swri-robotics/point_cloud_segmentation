#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <pcs_msgs/ImageProcessing.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>

#include <pcs_detection/hsv_thresholding.h>
#include <pcs_detection/utils.h>
#include <pcs_ros/pc_masker.h>

namespace pc_segmentation
{
static const std::string SERVICE_NAME = "/perform_detection";

PCMasker::PCMasker(ros::NodeHandle nh, const std::string& sub_topic, const std::string& pub_topic)
  : point_cloud_sub_topic_(sub_topic), masked_point_cloud_pub_topic_(pub_topic), nh_(nh)
{
  point_cloud_sub_ = nh_.subscribe(point_cloud_sub_topic_, 1, &PCMasker::newPCCallback, this);
  masked_point_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(masked_point_cloud_pub_topic_, 1);
  debug_extracted_image_pub_ = nh_.advertise<sensor_msgs::Image>("extracted_image_debug", 1);
  debug_masked_image_pub_ = nh_.advertise<sensor_msgs::Image>("masked_image_debug", 1);

  ros::service::waitForService(SERVICE_NAME);
  image_processing_client_ = nh_.serviceClient<pcs_msgs::ImageProcessing>(SERVICE_NAME);

  ROS_INFO("Subscribing to %s", point_cloud_sub_.getTopic().c_str());
  ROS_INFO("Publishing on %s", masked_point_cloud_pub_.getTopic().c_str());
}

void PCMasker::newPCCallback(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& scan)
{
  // -------  Extract RGB Image from cloud ----------
  cv::Mat depth_image;
  cv::Mat image;
  pcs_detection::cloudToImage(scan, depth_image, image);
  if (debug_viewer_)
  {
    cv::imwrite("extracted_image.png", image);
    cv::imshow("window", image);
    cv::waitKey(1000);
  }

  cv_bridge::CvImage debug_msg;
  debug_msg.header.frame_id = scan->header.frame_id;
  debug_msg.header.seq = scan->header.seq;
  pcl_conversions::fromPCL(scan->header.stamp, debug_msg.header.stamp);
  debug_msg.header.stamp = ros::Time::now();
  debug_msg.encoding = sensor_msgs::image_encodings::TYPE_8UC3;
  debug_msg.image = image;
  if (debug_publisher_)
  {
    debug_extracted_image_pub_.publish(debug_msg.toImageMsg());
  }

  // --------- Extract pixel mask from RGB image ---------------
  // Call image processing service
  pcs_msgs::ImageProcessing srv;
  srv.request.input_image = *debug_msg.toImageMsg();
  if (!image_processing_client_.call(srv))
  {
    ROS_ERROR("Image processing service call failed");
    return;
  }

  // Convert response back into, binary CV Mat
  auto mask_ptr = cv_bridge::toCvCopy(srv.response.returned_image, sensor_msgs::image_encodings::TYPE_8UC3);
  cv::Mat mask = mask_ptr->image;

  if (debug_viewer_)
  {
    imshow("window", mask * 255);
    cv::waitKey(1000);

    cv::Mat masked_image;

    pcs_detection::applyMask(image, mask, masked_image);
    imshow("window", masked_image);
    cv::waitKey(1000);
  }
  if (debug_publisher_)
  {
    cv::Mat masked_image;
    pcs_detection::applyMask(image, mask, masked_image);

    cv_bridge::CvImage debug_msg;
    debug_msg.header.frame_id = scan->header.frame_id;
    debug_msg.header.seq = scan->header.seq;
    pcl_conversions::fromPCL(scan->header.stamp, debug_msg.header.stamp);
    debug_msg.encoding = sensor_msgs::image_encodings::TYPE_8UC3;
    debug_msg.image = masked_image;
    debug_masked_image_pub_.publish(debug_msg.toImageMsg());
  }

  // This converts the mask into the 64 bit float type that it needs
  mask.convertTo(mask, CV_64FC3);

  // ---------- Apply mask to the depth image --------------
  cv::Mat masked_depth_image;
  pcs_detection::applyMask(depth_image, mask, masked_depth_image);

  if (debug_viewer_)
  {
    ROS_INFO_STREAM(pcs_detection::type2str(image.type()));
    ROS_INFO_STREAM(pcs_detection::type2str(mask.type()));
    ROS_INFO_STREAM(pcs_detection::type2str(depth_image.type()));
    ROS_INFO_STREAM(pcs_detection::type2str(mask.type()));

    imshow("window", depth_image);
    cv::waitKey(1000);
    imshow("window", masked_depth_image);
    cv::waitKey(1000);
  }

  // Now the masked cv::Mat depth image is converted back into a pcl type to be sent as a msg
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr output = pcs_detection::imageToCloud(image, masked_depth_image);
  output->header = scan->header;

  // Publish masked cloud
  masked_point_cloud_pub_.publish(output);
}
}  // namespace pc_segmentation
