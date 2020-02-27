#include <rclcpp/rclcpp.hpp>
#include <pcs_msgs/action/apply_octomap_mesh_mask.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>

namespace pcs_ros
{
/**
 * @brief Exposes pcs_scan_integration::OctomapMeshMask as a ROS action server.
 *
 * This implementation takes the octomap as a point cloud that is **currently being published**. If it is not found in 5
 * seconds, it returns failed. It also takes a path to PLY mesh file and returns a path to PLY mesh file. Modifying it
 * to return a Tesseract_geometry::Mesh would not be hard.
 */
class OctomapMeshMaskAction
{
public:
  using ApplyOctomapMeshMask = pcs_msgs::action::ApplyOctomapMeshMask;
  using GoalApplyOctomapMeshMask = rclcpp_action::ServerGoalHandle<ApplyOctomapMeshMask>;

  OctomapMeshMaskAction(rclcpp::Node::SharedPtr node, std::string name);

  rclcpp_action::GoalResponse handleGoal(const rclcpp_action::GoalUUID& uuid,
                                         std::shared_ptr<const ApplyOctomapMeshMask::Goal> goal);

  rclcpp_action::CancelResponse handleCancel(const std::shared_ptr<GoalApplyOctomapMeshMask> goal_handle);

  /**
   * @brief Callback for action server
   * @param goal Action goal
   */
  void executeCallback(const std::shared_ptr<GoalApplyOctomapMeshMask> goal);

  void handleAccepted(const std::shared_ptr<GoalApplyOctomapMeshMask> goal_handle);

  void getPointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);

protected:
  rclcpp::Node::SharedPtr node_;
  rclcpp_action::Server<ApplyOctomapMeshMask>::SharedPtr action_server_;
  std::string action_name_;
  sensor_msgs::msg::PointCloud2::SharedPtr point_cloud_;
  //  pcs_msgs::ApplyOctomapMeshMaskFeedback feedback_;
  //  pcs_msgs::ApplyOctomapMeshMaskResult result_;
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
};
}  // namespace pcs_ros
