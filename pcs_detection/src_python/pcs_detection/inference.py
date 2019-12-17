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
from pcs_detection.preprocess import preprocessing

import tensorflow as tf
import tensorflow.keras.backend as K


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
        elif config.MODEL == 'fcn_transfer':
            from src_python.pcs_detection.models.fcn8_transfer import fcn8

        # Save the graph and session so it can be set if make_prediction is in another thread
        self.graph = tf.get_default_graph()
        cfg = tf.ConfigProto()
        # This allows GPU memory to dynamically grow. This is a workaround to fix this issue on RTX cards
        # https://github.com/tensorflow/tensorflow/issues/24496
        # However, this can be problematic when sharing memory between applications.
        # TODO: Check and see if issue 24496 has been closed, and change this. Note that since Tensorflow 1.15
        # is the final 1.x release, this might never happen until this code is upgraded to tensorflow 2.x
        cfg.gpu_options.allow_growth = True
        cfg.log_device_placement = False
        self.session = tf.Session(config = cfg)

        # Save the graph and session so it can be set if make_prediction is in another thread
        self.graph = tf.get_default_graph()
        cfg = tf.ConfigProto()
        # This allows GPU memory to dynamically grow. This is a workaround to fix this issue on RTX cards
        # https://github.com/tensorflow/tensorflow/issues/24496
        # However, this can be problematic when sharing memory between applications.
        # TODO: Check and see if issue 24496 has been closed, and change this. Note that since Tensorflow 1.15
        # is the final 1.x release, this might never happen until this code is upgraded to tensorflow 2.x
        cfg.gpu_options.allow_growth = True
        cfg.log_device_placement = False
        self.session = tf.Session(config = cfg)

        # create the model
        K.set_session(self.session)
        weldDetector = fcn8(self.config)
        # load weights into the model file
        weldDetector.build_model()
        self.model = weldDetector.model
        self.model._make_predict_function()
        self.graph.finalize()

        print("Model loaded and ready")

    def make_prediction(self, img_data_original):
        '''
        Applies preprocessing, makes a prediction, and converts it 1d mask of target prediction confidence
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
        with self.session.as_default():
            with self.graph.as_default():
                prediction = self.model.predict(img_data)
        
        prediction = prediction[0]
        prediction[:,:,0] += self.config.CONFIDENCE_THRESHOLD

        # normalize between 0 and 1
        prediction = prediction/self.config.PREDICTION_MAX_RANGE + 0.5

        # get the max value of the prediction
        val_prediction = np.max(prediction,axis=-1)

        # confidence score is based on the difference between the max target prediction and max background prediciton
        val_prediction -= np.max(prediction[:,:,0:(len(config.BACKGROUND_CLASS_NAMES)+1)], axis=-1)

        # clip any abnormally strong predicitons 
        val_prediction[val_prediction > 1] = 1
        val_prediction[val_prediction < 0] = 0

        return val_prediction
