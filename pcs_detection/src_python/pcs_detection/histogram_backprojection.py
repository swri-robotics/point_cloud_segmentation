import numpy as np
import cv2 as cv

class HistogramBackprojection:
    """
    Annotates an image based on a previously provided histogram file.

    For more information see
    https://docs.opencv.org/master/dc/df6/tutorial_py_histogram_backprojection.html
    """
    histogram = None
    threshold_min = 30
    threshold_max = 150

    def __init__(self, hist_filepath):
        """
        Takes an .npy file that contains the trained histogram as an input
        """
        self.histogram = np.load(hist_filepath)
        print("Histogram loaded")

    def annotate_image(self, input_image):
        """
        Returns a binary mask image where 255 corresponds to regions inside the
        histogram and 0 corresponds to regions outside. Mask will be the same
        size as the input image
        """
        # Convert the image to hsv
        hsv_image = cv.cvtColor(input_image, cv.COLOR_BGR2HSV)
        dst = cv.calcBackProject([hsv_image], [0, 1], self.histogram, [0, 180, 0, 255], 1)

        # Now convolute with circular disc
        disc = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        cv.filter2D(dst,-1, disc, dst)

        # Apply treshold based on limits
        _, thresholded = cv.threshold(dst, self.threshold_min, self.threshold_max, 0)

        # Apply erosion and dilation to clean up noise
        kernel = np.ones((5, 5), np.uint8)
        cleaned = cv.morphologyEx(thresholded, cv.MORPH_OPEN, kernel)

        # Normalize to 0 - 1
        cv.normalize(cleaned, cleaned, 0, 1, cv.NORM_MINMAX, cv.CV_8UC3)

        # Convert to 0/255
        cleaned = 255 * cleaned

        # Convert back to 3 Channel Image
        output_image = cv.merge((cleaned, cleaned, cleaned))

        return output_image
