#include <pcs_detection/point_cloud_annotator.h>
#include <pcs_detection/utils.h>
#include <console_bridge/console.h>

using namespace pcs_detection;

bool PointCloudAnnotator::addPointCloud(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr input_cloud)
{
  // Preprocess Images
  auto position_image = std::make_shared<cv::Mat>();
  auto image_2d = std::make_shared<cv::Mat>();
  cloudToImage(input_cloud, *position_image, *image_2d);
  PointCloudData data(input_cloud, position_image, image_2d);

  // Add Data to buffer
  buffer_mutex_.lock();
  input_buffer_.push(data);
  buffer_mutex_.unlock();

  // Check buffer size - This could be done in another thread
  if (input_buffer_.size() >= batch_size_)
  {
    if (!annotateImages())
    {
      CONSOLE_BRIDGE_logError("Annotate Images failed");
      return false;
    }
  }
  return true;
}

bool PointCloudAnnotator::annotateImages()
{
  assert(input_buffer_.size() >= batch_size_);

  // Pull data out of queue and place it in a vector
  buffer_mutex_.lock();
  std::vector<cv::Mat> vec(batch_size_);
  for (int idx = 0; idx < batch_size_; idx++)
  {
    vec[idx] = *input_buffer_.front().image_2d_;
  }
  buffer_mutex_.unlock();

  // Send that data to the annotator (blocking and long potentially running)
  std::vector<cv::Mat> image_annotations;
  try
  {
    image_annotations = image_annotator_callback_(vec);
  }
  catch (...)
  {
    CONSOLE_BRIDGE_logError("Image Annotator Callback Exception");
    return false;
  }

  // Apply annotations
  buffer_mutex_.lock();
  pointCloudVec results(batch_size_);
  for (std::size_t idx = 0; idx < batch_size_; idx++)
  {
    results[idx] = pcs_detection::imageToCloud(
        image_annotations[idx], *input_buffer_.front().position_image_, input_buffer_.front().cloud_->header);
    input_buffer_.pop();
  }
  buffer_mutex_.unlock();

  // Send the results to the results callback
  try
  {
    results_callback_(results);
  }
  catch (...)
  {
    CONSOLE_BRIDGE_logError("Results Callback Exception");
    return false;
  }
  return true;
}
