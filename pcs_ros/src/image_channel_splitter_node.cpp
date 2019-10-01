#include <cv_bridge/cv_bridge.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/Image.h>
#include <opencv2/imgproc/imgproc.hpp>

namespace pcs_ros
{
/**
 * @brief This class/node splits a single channel mono8 image into a 3 channel rgb8 image. Each channel will
 * simply be the same as the input mono8 image
 */
class ImageChannelSplitter
{
public:
  void callback(const sensor_msgs::Image::ConstPtr& monochrome_image)
  {
    // Convert to OpenCV and convert to rgb
    cv_bridge::CvImagePtr monochrome;
    monochrome = cv_bridge::toCvCopy(monochrome_image, sensor_msgs::image_encodings::MONO8);
    std::vector<cv::Mat> vChannels;
    vChannels.push_back(monochrome->image);
    vChannels.push_back(monochrome->image);
    vChannels.push_back(monochrome->image);
    cv::merge(vChannels, image_.image);

    // Convert back to ROS and publish
    image_.encoding = sensor_msgs::image_encodings::RGB8;
    image_.header = monochrome_image->header;
    pub_.publish(image_.toImageMsg());
  }

  ImageChannelSplitter() : monochrome_topic_("input"), image_topic_("output")
  {
    // Create publishers and subscribers
    sub_ = nh_.subscribe(monochrome_topic_, 30, &ImageChannelSplitter::callback, this);
    pub_ = nh_.advertise<sensor_msgs::Image>(image_topic_, 30);

    // Print the topics we are using
    std::string t1 = nh_.resolveName(monochrome_topic_);
    std::string t2 = nh_.resolveName(image_topic_);
    ROS_INFO_STREAM("Subscribing to single channel image on: " << t1);
    ROS_INFO_STREAM("Publishing 3 channel image on: " << t2);
  }

private:
  ros::NodeHandle nh_;
  cv_bridge::CvImage image_;
  std::string monochrome_topic_;
  std::string image_topic_;
  ros::Subscriber sub_;
  ros::Publisher pub_;
};
}  // namespace pcs_ros
int main(int argc, char** argv)
{
  ros::init(argc, argv, "image_channel_splitter_node");
  pcs_ros::ImageChannelSplitter ics;
  ros::spin();
  return 0;
}
