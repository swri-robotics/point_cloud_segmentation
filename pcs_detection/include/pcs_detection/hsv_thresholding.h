/**
 * @file hsv_thresholding.h
 * @brief Detects features where the hsv values fall within a threshold
 *
 * @author Matthew Powelson
 * @date Sept 18, 2019
 * @version TODO
 * @bug No known bugs
 *
 * @copyright Copyright (c) 2017, Southwest Research Institute
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

#ifndef PCS_DETECTION_HSV_THRESHOLDING_H
#define PCS_DETECTION_HSV_THRESHOLDING_H
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace pcs_detection
{
/**
 * @brief Detects a color using color thresholding and returns an annotation. 255 = detected, 0 = no color by default
 *
 * This is mostly copied from https://docs.opencv.org/3.4/da/d97/tutorial_threshold_inRange.html
 *
 * Use the Python script hsv_threshold_tuning.py to find suitable thresholds
 * @param input_image
 * @param mask Annotation the same size as the input
 * @param inverted Default = false. If true, 0's are returned where the color is detected
 * @return
 */
inline bool hsvThresholdingDetector(const cv::Mat& input_image, cv::Mat& mask, bool inverted = false)
{
  const int hue_lower = 95;
  const int hue_upper = 115;
  const int saturation_lower = 95;
  const int saturation_upper = 220;
  const int value_lower = 40;
  const int value_upper = 150;

  cv::Mat hsv_image;
  cv::cvtColor(input_image, hsv_image, cv::COLOR_BGR2HSV);
  inRange(hsv_image,
          cv::Scalar(hue_lower, saturation_lower, value_lower),
          cv::Scalar(hue_upper, saturation_upper, value_upper),
          mask);
  // Invert if flag is set
  if (inverted)
    mask = (255 - mask);
  else
    mask = mask;

  // Uncomment to return a completely 255 mask.
  //  cv::Mat output(480, 640, CV_8UC1, cv::Scalar(255));
  //  mask = output;
  return true;
}
}  // namespace pcs_detection
#endif
