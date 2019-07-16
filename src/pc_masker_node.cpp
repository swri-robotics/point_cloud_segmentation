#include <ros/ros.h>
#include <point_cloud_segmentation/pc_masker.h>

int main(int argc, char** argv)
{
  ros::init(argc, argv, "pc_masker_node");
  ros::NodeHandle pnh("~");

  bool debug_publisher, debug_viewer;
  pnh.param<bool>("debug_publisher", debug_publisher, false);
  pnh.param<bool>("debug_viewer", debug_viewer, false);

  pc_segmentation::PCMasker republisher(pnh, "unmasked_cloud", "masked_cloud");
  republisher.debug_viewer_ = debug_viewer;
  republisher.debug_publisher_ = debug_publisher;

  ros::spin();

  return 0;
}
