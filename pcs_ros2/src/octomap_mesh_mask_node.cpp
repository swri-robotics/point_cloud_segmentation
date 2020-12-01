#include <rclcpp/rclcpp.hpp>
//#include <actionlib/server/simple_action_server.h>
//#include <pcl_ros/point_cloud.h>
//#include <pcl_ros/transforms.h>
#include <pcs_msgs/action/apply_octomap_mesh_mask.hpp>
#include <pcs_scan_integration/octomap_mesh_masking.h>
//#include <tf/transform_listener.h>
#include "rclcpp_action/rclcpp_action.hpp"
#include <pcs_ros2/octomap_mesh_mask_node.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2_eigen/tf2_eigen.h>
#include <pcl/common/transforms.h>

namespace pcs_ros
{
OctomapMeshMaskAction::OctomapMeshMaskAction(rclcpp::Node::SharedPtr node, std::string name)
  : node_(node), action_name_(name), tf_buffer_(node_->get_clock()), tf_listener_(tf_buffer_)
{
  action_server_ = rclcpp_action::create_server<ApplyOctomapMeshMask>(
      node_->get_node_base_interface(),
      node_->get_node_clock_interface(),
      node_->get_node_logging_interface(),
      node_->get_node_waitables_interface(),
      name,
      std::bind(&OctomapMeshMaskAction::handleGoal, this, std::placeholders::_1, std::placeholders::_2),
      std::bind(&OctomapMeshMaskAction::handleCancel, this, std::placeholders::_1),
      std::bind(&OctomapMeshMaskAction::handleAccepted, this, std::placeholders::_1));
}
rclcpp_action::GoalResponse OctomapMeshMaskAction::handleGoal(const rclcpp_action::GoalUUID& uuid,
                                                              std::shared_ptr<const ApplyOctomapMeshMask::Goal> goal)
{
  RCLCPP_INFO(node_->get_logger(), "Received OctomapMeshMaskAction Goal");
  (void)uuid;
  // TODO: Reject invalid goals
  //  if (Conditions)
  //  {
  //    return rclcpp_action::GoalResponse::REJECT;
  //  }
  return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}

rclcpp_action::CancelResponse
OctomapMeshMaskAction::handleCancel(const std::shared_ptr<GoalApplyOctomapMeshMask> goal_handle)
{
  RCLCPP_INFO(node_->get_logger(), "Received request to cancel goal");
  (void)goal_handle;
  return rclcpp_action::CancelResponse::ACCEPT;
}

/**
 * @brief Callback for action server
 * @param goal Action goal
 */
void OctomapMeshMaskAction::executeCallback(const std::shared_ptr<GoalApplyOctomapMeshMask> goal_handle)
{
  pcs_scan_integration::OctomapMeshMask masker;

  auto goal = goal_handle->get_goal();
  auto feedback = std::make_shared<ApplyOctomapMeshMask::Feedback>();
  auto result = std::make_shared<ApplyOctomapMeshMask::Result>();
  try
  {
    // Get pointcloud on topic provided
    {
      const double timeout = 5;
      auto subscription = node_->create_subscription<sensor_msgs::msg::PointCloud2>(
          goal->point_cloud_topic,
          1,
          std::bind(&OctomapMeshMaskAction::getPointCloudCallback, this, std::placeholders::_1));
      auto start_time = node_->now().seconds();
      while (point_cloud_ == nullptr && (node_->now().seconds() - start_time) > timeout)
      {
        // Wait for a new topic
      }
    }
    if (point_cloud_ == nullptr)
    {
      RCLCPP_ERROR(node_->get_logger(), "Pointcloud not found within timeout");
      throw std::runtime_error("Pointcloud not found within timeout");
    }

    // Look up transform between octomap frame and mesh frame. Note that we look it up at time now because the octomap
    // message could be pretty old
    geometry_msgs::msg::TransformStamped transform;
    tf2::Duration timeout(1);
    try
    {
      transform = tf_buffer_.lookupTransform(goal->mesh_frame, point_cloud_->header.frame_id, node_->now(), timeout);
    }
    catch (tf2::TransformException& e)
    {
      RCLCPP_ERROR(node_->get_logger(), "Tranform not found within timeout. Error: %s", e.what());
      throw std::runtime_error("Tranform not found within timeout");
    }
    Eigen::Affine3d transform_eigen = tf2::transformToEigen(transform);

    // Convert to PCL
    pcl::PCLPointCloud2 point_cloud_pcl;
    pcl_conversions::toPCL(*point_cloud_, point_cloud_pcl);

    // Convert to Point<type>
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_pnt_t(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::fromPCLPointCloud2(point_cloud_pcl, *point_cloud_pnt_t);

    // Transform point cloud
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_transformed(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::transformPointCloud(*point_cloud_pnt_t, *point_cloud_transformed, transform_eigen, true);

    // Set the octree based on the parameters given
    masker.setOctree(
        point_cloud_transformed, goal->resolution, goal->lower_limit, goal->upper_limit, goal->limit_negative);
  }
  catch (...)
  {
    RCLCPP_ERROR(node_->get_logger(),
                 "Octomap Mesh Mask Action did not receive a pointcloud on %s",
                 goal->point_cloud_topic.c_str());
    result->status_msg = "Failed to get point cloud";
    goal_handle->abort(result);
    return;
  }

  // Set the input mesh
  std::string input_filepath = goal->mesh_path;
  masker.setInputMesh(input_filepath);

  // Perform masking
  switch (goal->mask_type)
  {
    case 0:
      masker.maskMesh(pcs_scan_integration::OctomapMeshMask::MaskType::RETURN_INSIDE);
      break;
    case 1:
      masker.maskMesh(pcs_scan_integration::OctomapMeshMask::MaskType::RETURN_OUTSIDE);
      break;
    case 2:
      masker.maskMesh(pcs_scan_integration::OctomapMeshMask::MaskType::RETURN_COLORIZED);
      break;
    default:
      result->status_msg = "Invalid mask type";
      RCLCPP_ERROR(node_->get_logger(), "Invalid mask type");
      goal_handle->abort(result);
      return;
  }

  // Reset point cloud
  point_cloud_ = nullptr;
  // Save the results
  std::string result_path = goal->results_dir + "/masked_mesh.ply";
  if (!masker.saveMaskedMesh(result_path))
  {
    result->status_msg = "Save mesh failed";
    RCLCPP_ERROR(node_->get_logger(), "Save mesh failed");
    goal_handle->abort(result);
    return;
  }
  result->results_path = result_path;
  goal_handle->succeed(result);
  return;
}

void OctomapMeshMaskAction::handleAccepted(const std::shared_ptr<GoalApplyOctomapMeshMask> goal_handle)
{
  // this needs to return quickly to avoid blocking the executor, so spin up a new thread
  std::thread{ std::bind(&OctomapMeshMaskAction::executeCallback, this, std::placeholders::_1), goal_handle }.detach();
}

void OctomapMeshMaskAction::getPointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
  point_cloud_ = std::make_shared<sensor_msgs::msg::PointCloud2>(*msg);
}

}  // namespace pcs_ros

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("octomap_mesh_mask_node");

  pcs_ros::OctomapMeshMaskAction omma(node, "octomap_mesh_mask_server");

  RCLCPP_INFO(node->get_logger(), "Octomap Mesh Mask Action is available");
  rclcpp::spin(node);

  rclcpp::shutdown();
  return 0;
}
