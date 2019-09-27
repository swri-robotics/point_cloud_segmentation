#include <cv_bridge/cv_bridge.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <pcs_ros/utils.h>

namespace pcs_ros
{
/**
 * @brief This class is used for extracting an image from a point cloud for data collection
 *
 * This is similar in functionality to <node pkg="pcl_ros" type="convert_pointcloud_to_image" name="cloud_to_image" >
 * but this uses the util function in pcs_ros that also extract the xyz position
 */
class ImageExtractor
{
public:
  /**
   * @brief Point cloud callback that extracts image and publishes it
   * @param cloud PointCloud<PointXYZRGB> from which the RGB image is extracted
   */
  void callback(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr& cloud)
  {
    // Extract the image
    cv::Mat position_image;
    pc_segmentation::cloudToImage(cloud, position_image, image_.image);

    // Convert to ROS type and copy header
    image_.encoding = sensor_msgs::image_encodings::TYPE_8UC3;
    image_.header.frame_id = cloud->header.frame_id;
    image_.header.seq = cloud->header.seq;
    pcl_conversions::fromPCL(cloud->header.stamp, image_.header.stamp);
    image_pub_.publish(image_.toImageMsg());
  }

  ImageExtractor() : cloud_topic_("input"), image_topic_("output")
  {
    // Create publishers and subscribers
    sub_ = nh_.subscribe(cloud_topic_, 30, &ImageExtractor::callback, this);
    image_pub_ = nh_.advertise<sensor_msgs::Image>(image_topic_, 30);

    // Print the topics we are using
    std::string r_ct = nh_.resolveName(cloud_topic_);
    std::string r_it = nh_.resolveName(image_topic_);
    ROS_INFO_STREAM("Subscribing to point cloud on: " << r_ct);
    ROS_INFO_STREAM("Publishing image on: " << r_it);
  }

private:
  ros::NodeHandle nh_;
  cv_bridge::CvImage image_;
  std::string cloud_topic_;
  std::string image_topic_;
  ros::Subscriber sub_;
  ros::Publisher image_pub_;
};
}  // namespace pcs_ros
int main(int argc, char** argv)
{
  ros::init(argc, argv, "image_extractor_node");
  pcs_ros::ImageExtractor pci;
  ros::spin();
  return 0;
}
