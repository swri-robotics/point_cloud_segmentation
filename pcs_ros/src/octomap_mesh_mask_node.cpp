#include <ros/ros.h>
#include <actionlib/server/simple_action_server.h>
#include <pcl_ros/point_cloud.h>
#include <pcs_msgs/ApplyOctomapMeshMaskAction.h>
#include <pcs_scan_integration/octomap_mesh_masking.h>

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
protected:
  ros::NodeHandle nh_;
  actionlib::SimpleActionServer<pcs_msgs::ApplyOctomapMeshMaskAction> as_;
  std::string action_name_;
  pcs_msgs::ApplyOctomapMeshMaskFeedback feedback_;
  pcs_msgs::ApplyOctomapMeshMaskResult result_;

public:
  OctomapMeshMaskAction(std::string name)
    : as_(nh_, name, std::bind(&OctomapMeshMaskAction::execute_callback, this, std::placeholders::_1), false)
    , action_name_(name)
  {
    as_.start();
  }

  /**
   * @brief Callback for action server
   * @param goal Action goal
   */
  void execute_callback(const pcs_msgs::ApplyOctomapMeshMaskGoalConstPtr& goal)
  {
    pcs_scan_integration::OctomapMeshMask masker;

    try
    {
      // Get pointcloud on topic provided
      auto pointcloud =
          ros::topic::waitForMessage<pcl::PointCloud<pcl::PointXYZRGB>>(goal->point_cloud_topic, ros::Duration(5.0));
      // Set the octree based on the parameters given
      masker.setOctree(pointcloud, goal->resolution, goal->lower_limit, goal->upper_limit, goal->limit_negative);
    }
    catch (...)
    {
      ROS_ERROR("Octomap Mesh Mask Action did not receive a pointcloud on %s", goal->point_cloud_topic.c_str());
      result_.status_msg = "Failed to get point cloud";
      as_.setAborted(result_);
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
        result_.status_msg = "Invalid mask type";
        ROS_ERROR("Invalid mask type");
        as_.setAborted(result_);
        return;
    }

    // Save the results
    std::string result_path = goal->results_dir + "/masked_mesh.ply";
    if (!masker.saveMaskedMesh(result_path))
    {
      result_.status_msg = "Save mesh failed";
      ROS_ERROR("Save mesh failed");
      as_.setAborted(result_);
      return;
    }
    result_.results_path = result_path;
    as_.setSucceeded(result_);
    return;
  };
};

}  // namespace pcs_ros

int main(int argc, char** argv)
{
  ros::init(argc, argv, "octomap_mesh_mask_node");
  ros::NodeHandle nh;
  pcs_ros::OctomapMeshMaskAction("octomap_mesh_mask_server");

  ROS_INFO("Octomap Mesh Mask Action is available");
  ros::spin();
  return 0;
}
