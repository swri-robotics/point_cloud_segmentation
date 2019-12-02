/**
 * @file point_cloud_annotator_node.cpp
 * @brief Wraps a pcs_detection::PointCloudAnnotator in ROS 1 interfaces.
 *
 * @author Matthew Powelson
 * @date November 27, 2019
 * @version TODO
 * @bug No known bugs
 *
 * @copyright Copyright (c) 2019, Southwest Research Institute
 *
 * @par License
 * Software License Agreement (Apache License)
 * @par
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * @par
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <sensor_msgs/Image.h>
#include <std_srvs/SetBool.h>
#include <ros/ros.h>

#include <pcs_detection/point_cloud_annotator.h>
#include <pcs_msgs/ImageProcessing.h>

namespace pcs_ros
{
static const std::string PERFORM_DETECTION_SERVICE_NAME = "/perform_detection";
static const std::string TOGGLE_ANNOTATION_SERVICE_NAME = "toggle_annotation";
static const bool PUBLISH_DEBUG_IMAGES = true;

/**
 * @brief This class wraps a pcs_detection::PointCloudAnnotator in ROS 1 interfaces. It subscribes to a pointcloud and
 * publishes a point cloud. The image annotator function that is passed in calls a ROS service to perform the
 * annotation.
 */
class PointCloudAnnotatorNode
{
public:
  /**
   * @brief Subscriber callback for input pointcloud. Expects PointXYZRGB as a **structured pointcloud**. Calls
   * annotator_.addPointCloud
   * @param input_cloud Structured PointXYZRGB pointcloud
   */
  void subscriberCallback(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& input_cloud)
  {
    if (annotation_enabled_)
      annotator_.addPointCloud(input_cloud);
  }

  /**
   * @brief Annotation function that is passed into annotator_. Calls ROS service to perform annotation.
   * @param input_images Vector of 8UC3 cv::Mat images.
   * @return Vector of 8UC3 cv::Mat annotations.
   */
  std::vector<cv::Mat> imageAnnotator(const std::vector<cv::Mat> input_images)
  {
    // Create results vector
    std::vector<cv::Mat> results(input_images.size());

    // Create image processing service
    pcs_msgs::ImageProcessing srv;
    cv_bridge::CvImage cv_image;
    cv_image.header.stamp = ros::Time::now();
    cv_image.encoding = sensor_msgs::image_encodings::TYPE_8UC3;

    // Loop over all images and process them
    for (std::size_t idx = 0; idx < input_images.size(); idx++)
    {
      input_images[idx].copyTo(cv_image.image);
      srv.request.input_image = *cv_image.toImageMsg();

      if (!image_processing_client_.call(srv))
      {
        ROS_ERROR("Image processing service call failed");
      }

      if (PUBLISH_DEBUG_IMAGES)
        debug_image_pub_.publish(srv.response.returned_image);

      cv_bridge::CvImagePtr result =
          cv_bridge::toCvCopy(srv.response.returned_image, sensor_msgs::image_encodings::TYPE_8UC3);
      results[idx] = result->image;
    }

    return results;
  }

  /**
   * @brief Callback passed into annotator_ that is called with results when a new batch of annotated point clouds is
   * ready.
   * @param results These are the resulting annotated point clouds
   */
  void publisherCallback(std::vector<pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr> results)
  {
    ros::Rate r(60);
    for (auto& result : results)
    {
      pointcloud_publisher_.publish(result);
      if (results.size() > 1)
        r.sleep();
    }
  }

  PointCloudAnnotatorNode(ros::NodeHandle& nh)
    : nh_(nh)
    , input_topic_("input")
    , output_topic_("output")
    , annotation_enabled_(true)
    , annotator_([&](std::vector<cv::Mat> input) { return PointCloudAnnotatorNode::imageAnnotator(input); },
                 [&](std::vector<pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr> input) {
                   return PointCloudAnnotatorNode::publisherCallback(input);
                 },
                 1)
  {
    // Bring up image detection service
    ros::service::waitForService(PERFORM_DETECTION_SERVICE_NAME);
    image_processing_client_ = nh_.serviceClient<pcs_msgs::ImageProcessing>(PERFORM_DETECTION_SERVICE_NAME);

    // Advertise service to toggle annotation
    toggle_annotation_server_ =
        nh_.advertiseService(TOGGLE_ANNOTATION_SERVICE_NAME, &PointCloudAnnotatorNode::toggleAnnotationCallback, this);

    // Create publishers and subscribers
    pointcloud_subscriber_ = nh_.subscribe(input_topic_, 1, &PointCloudAnnotatorNode::subscriberCallback, this);
    pointcloud_publisher_ = nh_.advertise<pcl::PointCloud<pcl::PointXYZRGB>>(output_topic_, 5);
    debug_image_pub_ = nh_.advertise<sensor_msgs::Image>("debug_image_topic", 5);

    // Print the topics we are using
    std::string t1 = nh_.resolveName(input_topic_);
    std::string t2 = nh_.resolveName(output_topic_);
    ROS_INFO_STREAM("Subscribing on: " << t1);
    ROS_INFO_STREAM("Publishing  on: " << t2);
  }

  /**
   * @brief Callback for ROS service to enable/disable annotation.
   *
   * This can help to avoid melting your CPU/GPU in case of computationally expensive annotation. Note that this does
   * not unsubscribe. It simply doesn't do anything in the subscriber callback if disabled.
   * @param req Service request containing the boolean value to enable or disable annotation
   * @param res Service response containing message and success
   * @return Returns true
   */
  bool toggleAnnotationCallback(std_srvs::SetBool::Request& req, std_srvs::SetBool::Response& res)
  {
    annotation_enabled_ = req.data;
    res.message = annotation_enabled_ ? "Annotation enabled" : "Annotation disabled";
    res.success = true;
    return true;
  }

private:
  /** @brief Node handle associated with ROS interfaces*/
  ros::NodeHandle nh_;
  /** @brief Input pointcloud topic name*/
  std::string input_topic_;
  /** @brief Output pointcloud topic name*/
  std::string output_topic_;
  /** @brief Subscriber for input pointcloud */
  ros::Subscriber pointcloud_subscriber_;
  /** @brief Publisher for annotated pointcloud*/
  ros::Publisher pointcloud_publisher_;
  /** @brief Publishes the annotated images coming from the image_processing_client for debugging purposes*/
  ros::Publisher debug_image_pub_;
  /** @brief Service client used to process the image */
  ros::ServiceClient image_processing_client_;
  /** @brief std_srvs::SetBool service server used to enable/disable annotation*/
  ros::ServiceServer toggle_annotation_server_;
  /** @brief True if annotation is enabled */
  bool annotation_enabled_;

  /** @brief Annotator that is wrapped in ROS intefaces */
  pcs_detection::PointCloudAnnotator annotator_;
};
}  // namespace pcs_ros

int main(int argc, char** argv)
{
  ros::init(argc, argv, "point_cloud_annotator_node");
  ros::NodeHandle nh;
  pcs_ros::PointCloudAnnotatorNode pcan(nh);
  ros::spin();
  return 0;
}
