'''
 * @file inference.py 
 * @brief Creates an object that holds the model instance and a method for making inferences
 * @author Jake Janssen
 * @date Oct 28, 2019
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
 '''

import numpy as np
from src_python.pcs_detection.preprocess import preprocessing

class Inference():
    '''
    Edits the config bas ded on the validation weights and builds the model
    '''
    def __init__(self, config):

        self.config=config 

        # evaluate the full image regardless of what is in config 
        self.config.USE_FULL_IMAGE = True

        # display the type of model and image channels
        print('___Config_Options_From_Training___')
        print('Using model:', config.MODEL)
        print('Using channel:',config.CHANNEL)
        print('___________________________________')

        # load in the model
        if config.MODEL == 'fcn8':
            from src_python.pcs_detection.models.fcn8_model import fcn8
        elif config.MODEL == 'fcn_reduced':
            from src_python.pcs_detection.models.fcn8_reduced import fcn8

        # create the model
        weldDetector = fcn8(self.config)
        # load weights into the model file
        weldDetector.build_model(val=True, val_weights = self.config.VAL_WEIGHT_PATH)

        self.model = weldDetector.model
        print("Model loaded and ready")

    def make_prediction(self, img_data_original):
        '''
        Applies preprocessing, makes a prediction, and converts it to a boolean mask 
        Returns np array of size img_height x img_width
        '''

        img_data_original = img_data_original.astype(np.float32)

        if not img_data_original.any():
          print("Input image is invalid")
          return img_data_original

        # do not edit the original image
        img_data = img_data_original.copy()

        # preprocess data and make prediction 
        # apply preprocessing 
        img_data = preprocessing(img_data, self.config)
        # first dimension is used for batch size
        img_data = np.expand_dims(img_data, axis=0)

        # make a prediction and convert it to a boolean mask
        prediction =  self.model.predict(img_data)
        prediction =  prediction[0]

        # chnage the confidence of background predicition
        prediction[:,:,0] += self.config.CONFIDENCE_THRESHOLD

        # practical min max normalization 
        # values will depend on application 
        prediction += 40
        prediction /= 80

        # get the max value of the prediction
        val_prediction = np.max(prediction,axis=-1)

        # probability is based off of the difference between the max class and background
        val_prediction -= prediction[:,:,0]

        # clip any abnormally strong predicitons 
        val_prediction[val_prediction > 1] = 1
        val_prediction[val_prediction < 0] = 0

        return val_prediction