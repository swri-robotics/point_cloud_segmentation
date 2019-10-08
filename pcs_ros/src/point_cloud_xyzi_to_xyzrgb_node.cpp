#include <ros/ros.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
namespace pcs_ros
{
/**
 * @brief Converts a PointI to a PointRGB setting each color channel equal to the intensity
 * @param in input XYZI cloud
 * @param out output XYZRGB cloud
 */
inline void PointItoRGB(const pcl::PointXYZI& in, pcl::PointXYZRGB& out)
{
  // It appears that intensity is (in the case of the realsense) an 8 bit value padded with 3 extra bytes
  // http://pointclouds.org/documentation/tutorials/adding_custom_ptype.php
  uint8_t intensity = in._PointXYZI::intensity;
  uint8_t r = intensity;
  uint8_t g = intensity;
  uint8_t b = intensity;

  uint32_t rgb = ((uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b);
  out.rgb = *reinterpret_cast<float*>(&rgb);

  out._PointXYZRGB::x = in._PointXYZI::x;
  out._PointXYZRGB::y = in._PointXYZI::y;
  out._PointXYZRGB::z = in._PointXYZI::z;
}

/**
 * @brief This converts a PointCloud2 to a pointcloud XYZI for 8 bit intensity values. This should not be necessary, but
 * the fromROSMsg appears to be broken in this case.
 *
 * Perhaps fromROSMsg does not work because it is expecting a float? When I tried using it I got the error "Failed to
 * find match for field intensity"
 * @param cloud_msg Input cloud to be converted
 * @return pcl::PointCloud<pcl::PointXYZI> with intensity interpretted as a uint8_t
 */
inline pcl::PointCloud<pcl::PointXYZI> fromROSMsgXYZI(const sensor_msgs::PointCloud2& cloud_msg)
{
  pcl::PointCloud<pcl::PointXYZI> cloud;
  pcl_conversions::toPCL(cloud_msg.header, cloud.header);

  // Get the field structure of this point cloud
  int pointBytes = cloud_msg.point_step;
  int offset_x;
  int offset_y;
  int offset_z;
  int offset_int;
  for (int f = 0; f < cloud_msg.fields.size(); ++f)
  {
    if (cloud_msg.fields[f].name == "x")
      offset_x = cloud_msg.fields[f].offset;
    if (cloud_msg.fields[f].name == "y")
      offset_y = cloud_msg.fields[f].offset;
    if (cloud_msg.fields[f].name == "z")
      offset_z = cloud_msg.fields[f].offset;
    if (cloud_msg.fields[f].name == "intensity")
      offset_int = cloud_msg.fields[f].offset;
  }

  // populate point cloud object
  assert(cloud_msg.height == 1);
  cloud.points.resize(cloud_msg.width);
  for (int p = 0; p < cloud_msg.width; ++p)
  {
    pcl::PointXYZI newPoint;

    newPoint.x = *(float*)(&cloud_msg.data[0] + (pointBytes * p) + offset_x);
    newPoint.y = *(float*)(&cloud_msg.data[0] + (pointBytes * p) + offset_y);
    newPoint.z = *(float*)(&cloud_msg.data[0] + (pointBytes * p) + offset_z);
    newPoint.intensity = *(uint8_t*)(&cloud_msg.data[0] + (pointBytes * p) + offset_int);

    cloud.points[p] = newPoint;
  }

  return cloud;
}

/**
 * @brief Subscribes to a XYZI pointcloud and republishes it as an XYZRGB with each color channel equal to the 8 bit
 * intensity
 *
 * This is useful when a pointcloud is colorized with a greyscale image and published as an XYZI
 */
class PointCloudXYZItoXYZRGB
{
public:
  void callback(const sensor_msgs::PointCloud2::ConstPtr& cloud_pc2)
  {
    // Convert to ZYZI from message
    pcl::PointCloud<pcl::PointXYZI> cloud = fromROSMsgXYZI(*cloud_pc2);

    // Convert to XYZRGB where each color channel is the intensity value
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr results(new pcl::PointCloud<pcl::PointXYZRGB>());

    results->width = cloud.width;
    results->height = cloud.height;
    results->header = cloud.header;
    results->is_dense = cloud.is_dense;

    results->points.resize(cloud.points.size());
    for (int idx = 0; idx < cloud.points.size(); idx++)
    {
      PointItoRGB(cloud.points[idx], results->points[idx]);
    }

    pub_.publish(results);
  }

  PointCloudXYZItoXYZRGB() : input_topic_("input"), output_topic_("output")
  {
    // Create publishers and subscribers
    sub_ = nh_.subscribe(input_topic_, 1, &PointCloudXYZItoXYZRGB::callback, this);
    pub_ = nh_.advertise<pcl::PointCloud<pcl::PointXYZRGB>>(output_topic_, 1);

    // Print the topics we are using
    std::string t1 = nh_.resolveName(input_topic_);
    std::string t2 = nh_.resolveName(output_topic_);
    ROS_ERROR_STREAM("Subscribing to XYZI pointcloud on: " << t1);
    ROS_ERROR_STREAM("Publishing XYZRGB pointcloud on: " << t2);
  }

private:
  ros::NodeHandle nh_;
  std::string input_topic_;
  std::string output_topic_;
  ros::Subscriber sub_;
  ros::Publisher pub_;
};
}  // namespace pcs_ros
int main(int argc, char** argv)
{
  ros::init(argc, argv, "point_cloud_xyzi_to_xyzrgb_node");
  pcs_ros::PointCloudXYZItoXYZRGB converter;
  ros::spin();
  return 0;
}
