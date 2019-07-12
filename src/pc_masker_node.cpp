#include <ros/ros.h>
#include <point_cloud_segmentation/pc_masker.h>

int main(int argc, char** argv)
{
  ros::init(argc, argv, "pc_masker_node");
  ros::NodeHandle pnh("~");

  pc_segmentation::PCMasker republisher(pnh, "unmasked_cloud", "masked_cloud");

  ros::spin();

  return 0;
}
