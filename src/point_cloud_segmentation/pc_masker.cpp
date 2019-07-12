#include <sensor_msgs/PointCloud2.h>

#include <pcl_ros/point_cloud.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <point_cloud_segmentation/hsv_thresholding.h>
#include <point_cloud_segmentation/utils.h>
#include <point_cloud_segmentation/pc_masker.h>

namespace pc_segmentation
{
PCMasker::PCMasker(ros::NodeHandle nh, const std::string& sub_topic, const std::string& pub_topic)
  : point_cloud_sub_topic_(sub_topic), masked_depth_image_pub_topic_(pub_topic), nh_(nh)
{
  point_cloud_sub_ = nh.subscribe(point_cloud_sub_topic_, 1, &PCMasker::newPCCallback, this);
  masked_depth_image_pub_ = nh.advertise<sensor_msgs::PointCloud2>(masked_depth_image_pub_topic_, 1);
  ROS_INFO("Subscribing to %s", point_cloud_sub_.getTopic().c_str());
  ROS_INFO("Publishing on %s", masked_depth_image_pub_.getTopic().c_str());
}

void PCMasker::newPCCallback(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& scan)
{
  // Extract RGB Image from cloud
  cv::Mat depth_image;
  cv::Mat image;
  pc_segmentation::cloudToImage(scan, depth_image, image);
  if (DEBUG)
  {
    cv::imwrite("output.png", image);
    cv::imshow("window", image);
    cv::waitKey(1000);
  }

  // Extract pixel mask from RGB image
  cv::Mat mask;
  pc_segmentation::thresholdingMaskDetector(image, mask);

  if (DEBUG)
  {
    imshow("window", mask * 255);
    cv::waitKey(1000);

    cv::Mat mask_3channel;
    cv::Mat in[] = { mask, mask, mask };
    cv::merge(in, 3, mask_3channel);
    cv::Mat masked_image;

    applyMask(image, mask_3channel, masked_image);
    imshow("window", masked_image);
    cv::waitKey(1000);
  }

  // This converts the mask into the 64 bit float type that it needs
  mask.convertTo(mask, CV_64FC1);

  // This duplicates the mask into all 3 channels of the depth image (to a CV_64FC3)
  cv::Mat mask_3channel;
  cv::Mat in[] = { mask, mask, mask };
  cv::merge(in, 3, mask_3channel);

  // Apply mask to the depth image
  cv::Mat masked_depth_image;
  applyMask(depth_image, mask_3channel, masked_depth_image);

  if (DEBUG)
  {
    ROS_INFO_STREAM(type2str(image.type()));
    ROS_INFO_STREAM(type2str(mask.type()));
    ROS_INFO_STREAM(type2str(depth_image.type()));
    ROS_INFO_STREAM(type2str(mask_3channel.type()));

    imshow("window", depth_image);
    cv::waitKey(1000);
    imshow("window", masked_depth_image);
    cv::waitKey(1000);
  }

  // Now the masked cv::Mat depth image is converted back into a pcl type to be sent as a msg
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr output = imageToCloud(image, masked_depth_image);
  output->header = scan->header;

  // Publish masked cloud
  masked_depth_image_pub_.publish(output);
}

}  // namespace pc_segmentation
