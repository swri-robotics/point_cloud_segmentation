#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <sensor_msgs/Image.h>
#include <opencv2/imgproc/imgproc.hpp>

#include <pcs_detection/point_cloud_annotator.h>
#include <pcs_msgs/ImageProcessing.h>

namespace pcs_ros
{
static const std::string SERVICE_NAME = "/perform_detection";
static const bool PUBLISH_DEBUG_IMAGES = true;

class PointCloudAnnotatorNode
{
public:
  void subscriberCallback(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& input_cloud)
  {
    annotator_.addPointCloud(input_cloud);
  }

  std::vector<cv::Mat> imageAnnotator(const std::vector<cv::Mat> input_images)
  {
    // Create results vector
    std::vector<cv::Mat> results(input_images.size());

    // Create image processing service
    pcs_msgs::ImageProcessing srv;
    cv_bridge::CvImage cv_image;
    cv_image.header.stamp = ros::Time::now();
    cv_image.encoding = sensor_msgs::image_encodings::TYPE_8UC3;

    // Loop over all images and process them
    for (std::size_t idx = 0; idx < input_images.size(); idx++)
    {
      input_images[idx].copyTo(cv_image.image);
      srv.request.input_image = *cv_image.toImageMsg();

      if (!image_processing_client_.call(srv))
      {
        ROS_ERROR("Image processing service call failed");
      }

      if (PUBLISH_DEBUG_IMAGES)
        debug_image_pub_.publish(srv.response.returned_image);

      cv_bridge::CvImagePtr result =
          cv_bridge::toCvCopy(srv.response.returned_image, sensor_msgs::image_encodings::TYPE_8UC3);
      results[idx] = result->image;
    }

    return results;
  }

  void publisherCallback(std::vector<pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr> results)
  {
    ros::Rate r(60);
    for (auto& result : results)
    {
      pub_.publish(result);
      if (results.size() > 1)
        r.sleep();
    }
  }

  PointCloudAnnotatorNode()
    : input_topic_("input")
    , output_topic_("output")
    , annotator_([&](std::vector<cv::Mat> input) { return PointCloudAnnotatorNode::imageAnnotator(input); },
                 [&](std::vector<pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr> input) {
                   return PointCloudAnnotatorNode::publisherCallback(input);
                 },
                 1)
  {
    // Bring up image detection service
    ros::service::waitForService(SERVICE_NAME);
    image_processing_client_ = nh_.serviceClient<pcs_msgs::ImageProcessing>(SERVICE_NAME);

    // Create publishers and subscribers
    sub_ = nh_.subscribe(input_topic_, 1, &PointCloudAnnotatorNode::subscriberCallback, this);
    pub_ = nh_.advertise<pcl::PointCloud<pcl::PointXYZRGB>>(output_topic_, 5);
    debug_image_pub_ = nh_.advertise<sensor_msgs::Image>("debug_image_topic", 5);

    // Print the topics we are using
    std::string t1 = nh_.resolveName(input_topic_);
    std::string t2 = nh_.resolveName(output_topic_);
    ROS_INFO_STREAM("Subscribing on: " << t1);
    ROS_INFO_STREAM("Publishing  on: " << t2);
  }

private:
  ros::NodeHandle nh_;
  cv_bridge::CvImage image_;
  std::string input_topic_;
  std::string output_topic_;
  ros::Subscriber sub_;
  ros::Publisher pub_;
  ros::Publisher debug_image_pub_;
  /** @brief Service client used to process the image */
  ros::ServiceClient image_processing_client_;

  pcs_detection::PointCloudAnnotator annotator_;
};
}  // namespace pcs_ros

int main(int argc, char** argv)
{
  ros::init(argc, argv, "point_cloud_annotator_node");
  pcs_ros::PointCloudAnnotatorNode pcan;
  ros::spin();
  return 0;
}
