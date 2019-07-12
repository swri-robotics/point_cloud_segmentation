#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace pc_segmentation
{
/**
 * @brief Detects the masking using color thresholding and returns a mask. 1 = masking, 0 = no masking by default
 *
 * This is mostly copied from https://docs.opencv.org/3.4/da/d97/tutorial_threshold_inRange.html
 *
 * Use the Python script hsv_threshold_tuning.py to find suitable thresholds
 * @param input_image
 * @param mask Binary mask the same size as the input
 * @param inverted Default = false. If true, 0's are returned where masking is detected
 * @return
 */
bool thresholdingMaskDetector(const cv::Mat& input_image, cv::Mat& mask, bool inverted = false)
{
  const int hue_lower = 95;
  const int hue_upper = 115;
  const int saturation_lower = 95;
  const int saturation_upper = 220;
  const int value_lower = 40;
  const int value_upper = 150;

  cv::Mat hsv_image;
  cv::cvtColor(input_image, hsv_image, CV_BGR2HSV);
  inRange(hsv_image,
          cv::Scalar(hue_lower, saturation_lower, value_lower),
          cv::Scalar(hue_upper, saturation_upper, value_upper),
          mask);
  // Set to 0 - 1 rather than 0 - 255
  if (inverted)
    mask = (255 - mask) / 255;
  else
    mask = mask / 255;

  // Uncomment to return a completely 1 mask.
  //  cv::Mat output(480, 640, CV_8UC1, cv::Scalar(1));
  //  mask = output;
  return true;
}
}  // namespace pc_segmentation
