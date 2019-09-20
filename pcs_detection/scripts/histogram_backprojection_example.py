#!/usr/bin/env python

import numpy as np
import cv2 as cv
from pcs_detection.histogram_backprojection import HistogramBackprojection

if __name__ == '__main__':
    # Load the histogram into the annotator
    annotator = HistogramBackprojection("example_data/trained_hist.npy")

    # Load the image
    input_image = cv.imread('example_data/example_image.png')

    # Generate the annotation
    results_image = annotator.annotate_image(input_image)

    # Show the results
    print("Input image of size: " + str(input_image.shape))
    print("Results image of size: " + str(results_image.shape))
    print("Press ESC to exit")
    while True:
        cv.imshow("image", np.hstack((input_image, results_image)))
        k = cv.waitKey(1) & 0xFF
        if k == 27:
            cv.destroyWindow("image")
            break
