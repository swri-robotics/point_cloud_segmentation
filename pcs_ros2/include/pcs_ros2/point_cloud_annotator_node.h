/**
 * @file point_cloud_annotator_node.h
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

#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <rclcpp/rclcpp.hpp>

#include <pcs_detection/point_cloud_annotator.h>
#include <pcs_msgs/srv/image_processing.hpp>

namespace pcs_ros
{

/**
 * @brief This class wraps a pcs_detection::PointCloudAnnotator in ROS 1 interfaces. It subscribes to a pointcloud and
 * publishes a point cloud. The image annotator function that is passed in calls a ROS service to perform the
 * annotation.
 */
class PointCloudAnnotatorNode : public rclcpp::Node
{
public:

    PointCloudAnnotatorNode(std::string name);

  /**
   * @brief Subscriber callback for input pointcloud. Expects PointXYZRGB as a **structured pointcloud**. Calls
   * annotator_.addPointCloud
   * @param input_cloud Structured PointXYZRGB pointcloud
   */
  void subscriberCallback(const sensor_msgs::msg::PointCloud2::SharedPtr input_cloud);
  /**
   * @brief Annotation function that is passed into annotator_. Calls ROS service to perform annotation.
   * @param input_images Vector of 8UC3 cv::Mat images.
   * @return Vector of 8UC3 cv::Mat annotations.
   */
  std::vector<cv::Mat> imageAnnotator(const std::vector<cv::Mat> input_images);

  /**
   * @brief Callback passed into annotator_ that is called with results when a new batch of annotated point clouds is
   * ready.
   * @param results These are the resulting annotated point clouds
   */
  void publisherCallback(std::vector<pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr> results);

  /**
   * @brief Callback for ROS service to enable/disable annotation.
   *
   * This can help to avoid melting your CPU/GPU in case of computationally expensive annotation. Note that this does
   * not unsubscribe. It simply doesn't do anything in the subscriber callback if disabled.
   * @param req Service request containing the boolean value to enable or disable annotation
   * @param res Service response containing message and success
   * @return Returns true
   */
  void toggleAnnotationCallback(const std_srvs::srv::SetBool::Request::SharedPtr req,
                                std_srvs::srv::SetBool::Response::SharedPtr res);

private:
  /** @brief Input pointcloud topic name*/
  std::string input_topic_;
  /** @brief Output pointcloud topic name*/
  std::string output_topic_;
  /** @brief Subscriber for input pointcloud */
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_subscriber_;
  /** @brief Publisher for annotated pointcloud*/
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_publisher_;
  /** @brief Publishes the annotated images coming from the image_processing_client for debugging purposes*/
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_image_pub_;
  /** @brief Service client used to process the image */
  rclcpp::Client<pcs_msgs::srv::ImageProcessing>::SharedPtr image_processing_client_;
  /** @brief std_srvs::SetBool service server used to enable/disable annotation*/
  rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr toggle_annotation_server_;
  /** @brief True if annotation is enabled */
  bool annotation_enabled_;

  /** @brief Annotator that is wrapped in ROS intefaces */
  pcs_detection::PointCloudAnnotator annotator_;
};
}  // namespace pcs_ros
