#!/usr/bin/env python3
'''
 * @file fcn8_inference.py
 * @brief Used for training neural nets, validating the nets by viewing the predictions, viewing the images passed into the network after preprocessing, and deployment
 *
 * @author Matthew Powelson
 * @date November 6, 2019
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
 '''

# For data loading and visualization
import numpy as np
import cv2 as cv

# For importing the config
import json
import os
import sys

# Inference module
from pcs_detection.inference import Inference

# Hack because code was originally written to have configs as Python modules
class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)

if __name__ == '__main__':
  if len(sys.argv) > 1 and sys.argv[1] == "NO_GPU":
      print("Disabling GPU")
      os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
  elif len(sys.argv):
      print("Invalid argument. Pass argument 'NO_GPU' to run in CPU mode")

  # Import Config json file and convert into format we need
  dir_path = os.path.dirname(os.path.realpath(__file__))
  
  with open(dir_path + '<path to saved off json>') as json_data_file:
      data = json.load(json_data_file)
  config = Config(**data)

  # Construct the annotator
  annotator = Inference(config)

  # Load the image
  input_image = cv.imread(dir_path + '/data/example_dataset_1/validation/0003.png')

  # Generate the annotation and convert to 3 channel image
  res = annotator.make_prediction(input_image[:,:,0])
  results_image = cv.merge((res, res, res)) * 255
  results_image = results_image.astype(np.uint8)

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
