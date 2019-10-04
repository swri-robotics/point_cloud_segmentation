/**
 * @file point_cloud_annotator.h
 * @brief Annotates a colorized pointcloud based on some detection function
 *
 * @author Matthew Powelson
 * @date Oct 2, 2019
 * @version TODO
 * @bug No known bugs
 *
 * @copyright Copyright (c) 2019, Southwest Research Institute
 *
 * @par License
 * Software License Agreement (Apache License)
 * @par
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * @par
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef PCS_DETECTION_POINT_CLOUD_ANNOTATOR_H
#define PCS_DETECTION_POINT_CLOUD_ANNOTATOR_H

#include <queue>
#include <mutex>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <opencv2/core/core.hpp>

namespace pcs_detection
{
typedef std::vector<pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr> pointCloudVec;
/**
 * @brief Contains all data necessary to process a point cloud at a later time
 *
 * The idea is that these could go into a buffer for batch processing. This could be expanded to contain things like the
 * tranform if necessary.
 */
struct PointCloudData
{
  PointCloudData() = default;

  PointCloudData(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr& cloud,
                 std::shared_ptr<cv::Mat>& position_image,
                 std::shared_ptr<cv::Mat>& image_2d)
    : cloud_(cloud), position_image_(position_image), image_2d_(image_2d)
  {
  }
  /** @brief Input point cloud */
  pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud_;
  /** @brief cv::Mat (64FC3) with 3 64 bit channels encoding x, y, z position*/
  std::shared_ptr<cv::Mat> position_image_;
  /** @brief cv::Mat (8UC3) encoding extracted 2D image */
  std::shared_ptr<cv::Mat> image_2d_;
};

/**
 * @brief The PointCloudAnnotator class
 */
class PointCloudAnnotator
{
public:
  PointCloudAnnotator(std::function<std::vector<cv::Mat>(const std::vector<cv::Mat>)> image_annotator_callback,
                      std::function<void(pointCloudVec)> results_callback,
                      long unsigned int batch_size = 1)
    : image_annotator_callback_(std::move(image_annotator_callback))
    , results_callback_(std::move(results_callback))
    , batch_size_(batch_size)
  {
    // This is not a requirement. It can be as big as you want. This is mostly just a sanity check.
    assert(batch_size_ <= 1024);
  }

  /** @brief Adds a pointcloud to the processing queue and does any preprocessing necessary
   * @return false if failed*/
  bool addPointCloud(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr input_cloud);

  /** @brief Remove data from the buffer, calls the annotate callback, and returns results
   * @return false if failed*/
  bool annotateImages();

protected:
  /** @brief Called to annotate a buffer. */
  std::function<std::vector<cv::Mat>(const std::vector<cv::Mat>)> image_annotator_callback_;
  /** @brief Called when results are ready. */
  std::function<void(pointCloudVec)> results_callback_;
  /** @brief Size at which the buffer submits a new batch of images to be annotated */
  long unsigned int batch_size_;

  /** @brief This stores the data until there is enough of it to be batch processed. This will likely need to be a ring
   * buffer or something more intelligent if this becomes threaded */
  std::queue<PointCloudData> input_buffer_;

private:
  std::mutex buffer_mutex_;
};

}  // namespace pcs_detection

#endif
