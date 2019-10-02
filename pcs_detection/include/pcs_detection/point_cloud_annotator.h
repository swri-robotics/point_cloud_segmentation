/**
 * @file point_cloud_annotator.h
 * @brief Annotates a colorized pointcloud based on some detection function
 *
 * @author Matthew Powelson
 * @date OCt 2, 2019
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

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace pcs_detection
{
class PointCloudAnnotator
{
public:
  PointCloudAnnotator() = default;

private:
  std::function<void(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr)> callback_;
};

}  // namespace pcs_detection

#endif
