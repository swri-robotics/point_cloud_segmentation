#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

namespace pc_segmentation
{
/**
 * @brief Returns a string associated with the openCV typedef (ie CV_8U
 * @param type OpenCV flags specified in typedef in interface.h
 * @return A human readable string coresponding to the typedef
 */
std::string type2str(const int type)
{
  std::string r;

  uchar depth = CV_MAT_DEPTH(type);
  int chans = CV_MAT_CN(type);

  switch (depth)
  {
    case CV_8U:
      r = "8U";
      break;
    case CV_8S:
      r = "8S";
      break;
    case CV_16U:
      r = "16U";
      break;
    case CV_16S:
      r = "16S";
      break;
    case CV_32S:
      r = "32S";
      break;
    case CV_32F:
      r = "32F";
      break;
    case CV_64F:
      r = "64F";
      break;
    default:
      r = "User";
      break;
  }

  r += "C";
  r += std::to_string(chans);

  return r;
}

bool applyMask(const cv::Mat& input_image, const cv::Mat& mask, cv::Mat& masked_image)
{
  masked_image = input_image.mul(mask);
  return true;
}

/**
 * @brief Converts a pointcloud to a color image and a position encoded image
 *
 * Note: There are also the functions in pcl_ros which are similar, but they only extract the rgb
 * http://docs.pointclouds.org/1.1.0/namespacepcl.html#a31460a4b07150db357c690a8ae27d1e6
 * @param cloud Input point cloud. Should be a dense XYZRGB cloud
 * @param position_image cv::Mat with 3 64 bit channels encoding x, y, z position
 * @param color_image cv::Mat encoding extracted rgb image
 */
void cloudToImage(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr& cloud,
                  cv::Mat& position_image,
                  cv::Mat& color_image)
{
  assert(cloud->width != 1 && cloud->height != 1);

  // Resize coordinates to the size of the point cloud (stored in a 64 bit float 3 channel matrix)
  position_image = cv::Mat(480, 640, CV_64FC3);
  // Resize image to the size of the depth image (stored in a 8 bit unsigned 3 channel matrix)
  color_image = cv::Mat(480, 640, CV_8UC3);
  // Iterate over the rows and columns of the structured point cloud
  for (int y = 0; y < color_image.rows; y++)
  {
    for (int x = 0; x < color_image.cols; x++)
    {
      // Pull out the xyz values from the point cloud
      position_image.at<double>(y, x * 3 + 0) = cloud->points.at(y * color_image.cols + x).x;
      position_image.at<double>(y, x * 3 + 1) = cloud->points.at(y * color_image.cols + x).y;
      position_image.at<double>(y, x * 3 + 2) = cloud->points.at(y * color_image.cols + x).z;

      // Pull out the rgb values from the point cloud
      cv::Vec3b color = cv::Vec3b(cloud->points.at(y * color_image.cols + x).b,
                                  cloud->points.at(y * color_image.cols + x).g,
                                  cloud->points.at(y * color_image.cols + x).r);
      // Apply color to that point
      color_image.at<cv::Vec3b>(cv::Point(x, y)) = color;
    }
  }
}

/**
 * @brief Convert a color image and a position encoded image back to a point cloud
 *
 * TODO: Unit tests, handle nans
 * @param color_image CV_8UC3 cv::Mat RGB image
 * @param depth_image CV_64F3 cv::Mat where the channels correspond to x, y, and z position
 * @return Returns an XYZRGB point cloud generated from the inputs
 */
pcl::PointCloud<pcl::PointXYZRGB>::Ptr imageToCloud(const cv::Mat& color_image,
                                                    const cv::Mat& position_image,
                                                    const pcl::PCLHeader& header = pcl::PCLHeader())
{
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
  cloud->header = header;
  for (int y = 0; y < color_image.rows; y++)
  {
    for (int x = 0; x < color_image.cols; x++)
    {
      pcl::PointXYZRGB point;
      point.x = position_image.at<double>(y, x * 3 + 0);
      point.y = position_image.at<double>(y, x * 3 + 1);
      point.z = position_image.at<double>(y, x * 3 + 2);

      cv::Vec3b color = color_image.at<cv::Vec3b>(cv::Point(x, y));
      uint8_t r = (color[2]);
      uint8_t g = (color[1]);
      uint8_t b = (color[0]);

      int32_t rgb = (r << 16) | (g << 8) | b;
      point.rgb = *reinterpret_cast<float*>(&rgb);

      cloud->points.push_back(point);
    }
  }
  return cloud;
}

}  // namespace pc_segmentation
