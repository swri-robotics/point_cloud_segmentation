/**
 * @file point_cloud_annotator_node.cpp
 * @brief Wraps a pcs_detection::PointCloudAnnotator in ROS 2 interfaces.
 *
 * @author Matthew Powelson
 * @date Feb 27, 2020
 * @version TODO
 * @bug No known bugs
 *
 * @copyright Copyright (c) 2020, Southwest Research Institute
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

#include <pcs_ros2/point_cloud_annotator_node.h>

namespace pcs_ros
{
static const std::string PERFORM_DETECTION_SERVICE_NAME = "/perform_detection";
static const std::string TOGGLE_ANNOTATION_SERVICE_NAME = "toggle_annotation";
static const std::string DEBUG_IMAGE_TOPIC = "debug_image_topic";
static const bool PUBLISH_DEBUG_IMAGES = true;

PointCloudAnnotatorNode::PointCloudAnnotatorNode(std::string name)
  : Node(name)
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
  image_processing_client_ = this->create_client<pcs_msgs::srv::ImageProcessing>(PERFORM_DETECTION_SERVICE_NAME);

  // Advertise service to toggle annotation
  toggle_annotation_server_ = this->create_service<std_srvs::srv::SetBool>(
      TOGGLE_ANNOTATION_SERVICE_NAME,
      std::bind(
          &PointCloudAnnotatorNode::toggleAnnotationCallback, this, std::placeholders::_1, std::placeholders::_2));

    // Create publishers and subscribers
    pointcloud_subscriber_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(input_topic_, 1, std::bind(&PointCloudAnnotatorNode::subscriberCallback, this, std::placeholders::_1));
    pointcloud_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(output_topic_, 5);
    debug_image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(DEBUG_IMAGE_TOPIC, 5);

  //  // Print the topics we are using
  //  std::string t1 = nh_.resolveName(input_topic_);   // I don't think this functionality exists in ROS 2 (2/27/20)
  //  std::string t2 = nh_.resolveName(output_topic_);
  //  ROS_INFO_STREAM("Subscribing on: " << t1);
  //  ROS_INFO_STREAM("Publishing  on: " << t2);

  // Wait for all connections

  while (!image_processing_client_->wait_for_service(std::chrono_literals::operator""s(5)))
  {
    if (!rclcpp::ok())
    {
      RCLCPP_ERROR(this->get_logger(), "Interrupted while waiting for the service. Exiting.");
      break;
    }
    RCLCPP_INFO(this->get_logger(), "Image processing service not available, waiting again...");
  }
}

void PointCloudAnnotatorNode::subscriberCallback(const sensor_msgs::msg::PointCloud2::SharedPtr input_cloud)
{
  if (annotation_enabled_)
  {
    // Convert to Point<type>
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_pnt_t(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::fromROSMsg(*input_cloud, *point_cloud_pnt_t);

    annotator_.addPointCloud(point_cloud_pnt_t);
  }
}

std::vector<cv::Mat> PointCloudAnnotatorNode::imageAnnotator(const std::vector<cv::Mat> input_images)
{
  // Create results vector
  std::vector<cv::Mat> results(input_images.size());

  // Create image processing service
  auto request = std::make_shared<pcs_msgs::srv::ImageProcessing::Request>();
  cv_bridge::CvImage cv_image;
  cv_image.header.stamp = this->now();
  cv_image.encoding = sensor_msgs::image_encodings::TYPE_8UC3;

    // Loop over all images and process them
    for (std::size_t idx = 0; idx < input_images.size(); idx++)
    {
      // Image processing is done through a ROS service to allow Python image processing to be used
      input_images[idx].copyTo(cv_image.image);
      request->input_image = *cv_image.toImageMsg();

      auto response = image_processing_client_->async_send_request(request);
      if (rclcpp::spin_until_future_complete(this->get_node_base_interface(), response) !=
        rclcpp::executor::FutureReturnCode::SUCCESS)
      {
        RCLCPP_ERROR(this->get_logger(), "Failed to call image annotation");
        break;
      }

      // Publish a debugging image of the results
      if (PUBLISH_DEBUG_IMAGES)
        debug_image_pub_->publish(response.get()->returned_image);

      cv_bridge::CvImagePtr result =
          cv_bridge::toCvCopy(response.get()->returned_image, sensor_msgs::image_encodings::TYPE_8UC3);
      results[idx] = result->image;
    }

  return results;
}

void PointCloudAnnotatorNode::publisherCallback(std::vector<pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr> results)
{
  rclcpp::Rate r(60);
  for (auto& result : results)
  {
    sensor_msgs::msg::PointCloud2 pc2_msg;
    pcl::toROSMsg(*result, pc2_msg);

    pointcloud_publisher_->publish(pc2_msg);
    if (results.size() > 1)
      r.sleep();
  }
}

void PointCloudAnnotatorNode::toggleAnnotationCallback(const std_srvs::srv::SetBool::Request::SharedPtr req,
                                                       std_srvs::srv::SetBool::Response::SharedPtr res)
{
  annotation_enabled_ = req->data;
  res->message = annotation_enabled_ ? "Annotation enabled" : "Annotation disabled";
  res->success = true;
}

}  // namespace pcs_ros

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<pcs_ros::PointCloudAnnotatorNode>("point_cloud_annotator_node"));
  rclcpp::shutdown();
  return 0;
}
