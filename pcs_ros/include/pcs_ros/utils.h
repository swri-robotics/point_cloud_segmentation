#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

namespace pc_segmentation
{
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

void cloudToImage(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud, cv::Mat& depth_image, cv::Mat& color_image)
{
  // Resize coordinates to the size of the point cloud (stored in a 64 bit float 3 channel matrix)
  depth_image = cv::Mat(480, 640, CV_64FC3);
  // Resize image to the size of the depth image (stored in a 8 bit unsigned 3 channel matrix)
  color_image = cv::Mat(480, 640, CV_8UC3);
  // Iterate over the rows and columns of the structured point cloud
  for (int y = 0; y < color_image.rows; y++)
  {
    for (int x = 0; x < color_image.cols; x++)
    {
      // Pull out the xyz values from the point cloud
      depth_image.at<double>(y, x * 3 + 0) = cloud->points.at(y * color_image.cols + x).x;
      depth_image.at<double>(y, x * 3 + 1) = cloud->points.at(y * color_image.cols + x).y;
      depth_image.at<double>(y, x * 3 + 2) = cloud->points.at(y * color_image.cols + x).z;

      // Pull out the rgb values from the point cloud
      cv::Vec3b color = cv::Vec3b(cloud->points.at(y * color_image.cols + x).b,
                                  cloud->points.at(y * color_image.cols + x).g,
                                  cloud->points.at(y * color_image.cols + x).r);
      // Apply color to that point
      color_image.at<cv::Vec3b>(cv::Point(x, y)) = color;
    }
  }
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr imageToCloud(const cv::Mat& color_image, const cv::Mat& depth_image)
{
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());

  for (int y = 0; y < color_image.rows; y++)
  {
    for (int x = 0; x < color_image.cols; x++)
    {
      pcl::PointXYZRGB point;
      point.x = depth_image.at<double>(y, x * 3 + 0);
      point.y = depth_image.at<double>(y, x * 3 + 1);
      point.z = depth_image.at<double>(y, x * 3 + 2);

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
