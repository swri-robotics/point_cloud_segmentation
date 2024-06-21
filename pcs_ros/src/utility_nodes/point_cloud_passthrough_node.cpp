#include <ros/ros.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <pcs_scan_integration/octomap_mesh_masking.h>

namespace pcs_ros
{
class PointCloudPassthrough
{
public:
  PointCloudPassthrough(ros::NodeHandle& nh) : nh_(nh), input_topic_("input"), output_topic_("output")
  {
    // Create publishers and subscribers
    sub_ = nh_.subscribe(input_topic_, 1, &PointCloudPassthrough::callback, this);
    pub_ = nh_.advertise<pcl::PointCloud<pcl::PointXYZRGB>>(output_topic_, 1);

    // Print the topics we are using
    std::string t1 = nh_.resolveName(input_topic_);
    std::string t2 = nh_.resolveName(output_topic_);
    ROS_INFO_STREAM("Subscribing XYZRGB pointcloud on: " << t1);
    ROS_INFO_STREAM("Publishing filtered XYZRGB pointcloud on: " << t2);

    nh_.param<int>("lower_limit", lower_limit_, 0);
    nh_.param<int>("upper_limit", upper_limit_, 255);
    ROS_INFO_STREAM("Passing Pointclouds with RGB values between " << lower_limit_ << " and " << upper_limit_
                                                                   << std::endl);
  }

  void callback(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr& input_cloud)
  {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr results =
        pcs_scan_integration::colorPassthrough(input_cloud, lower_limit_, upper_limit_, false);

    pub_.publish(results);
  }

private:
  ros::NodeHandle nh_;
  std::string input_topic_;
  std::string output_topic_;
  ros::Subscriber sub_;
  ros::Publisher pub_;

  int upper_limit_;
  int lower_limit_;
};
}  // namespace pcs_ros
int main(int argc, char** argv)
{
  ros::init(argc, argv, "point_cloud_passthrough_node");
  ros::NodeHandle nh("~");
  pcs_ros::PointCloudPassthrough filter(nh);
  ros::spin();
  return 0;
}
