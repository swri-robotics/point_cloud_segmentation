'''
 * @file preprocess.py
 * @brief Applies laplacian preprocessing and mean subtraction prior to input to neural net
 *
 * @author Jake Janssen
 * @date Oct 24, 2019
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

import cv2
import numpy as np

def preprocessing(img_data, config):
    '''
    Apply laplacian preprocessing to image if specified
    and mean subtraction on image 
    '''

    # only use the first channel if grey is being used 
    
    if config.CHANNEL == 'GREY' and len(img_data.shape) != 2 or config.CHANNEL == 'STACKED' and len(img_data.shape) != 2 or config.CHANNEL == 'CLAHE' and len(img_data.shape) != 2:
        img_data = img_data[:,:,0]

    # add third dimension to images with a single channel
    if len(img_data.shape) == 2:
        img_data = np.expand_dims(img_data, axis=-1)

    # convert image to the lab color space 
    if config.CHANNEL == 'LAB':
        img_data = img_data.astype(np.float32)
        img_data /= 255
        img_data = cv2.cvtColor(img_data.astype(np.float32), cv2.COLOR_BGR2LAB)
        channels=cv2.split(img_data)
        img_data = img_data.astype(np.float32)
    # convert image channel to YCR_CB
    elif config.CHANNEL == 'YCR_CB':
        ycrcb=cv2.cvtColor(img_data.astype(np.uint8),cv2.COLOR_BGR2YCR_CB)
        # channels=cv2.split(ycrcb)
        # channels[0] = cv2.equalizeHist(channels[0].astype(np.uint8),channels[0].astype(np.uint8))
        # img_data = cv2.merge(channels)
        # img_data = img_data.astype(np.float32)
        img_data = ycrcb.astype(np.float32)
    elif config.CHANNEL =='HSV':
        img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2HSV)
        img_data = img_data.astype(np.float32)
    
    # add the laplacian as a second channel 
    elif config.CHANNEL == 'COMBINED':
        edge_chnl = cv2.GaussianBlur(img_data, (3, 3), 0).astype(np.uint8)
        ddepth = cv2.CV_16S
        lap_chnl = cv2.Laplacian(edge_chnl, ddepth, ksize=3 )
        lap_chnl = lap_chnl * config.PRE_PROCESS['edge'][1] 
        combined_img = np.zeros((img_data.shape[0], img_data.shape[1], img_data.shape[2]+1))
        combined_img[:,:,0:img_data.shape[2]] = img_data
        combined_img[:,:,img_data.shape[2]] = lap_chnl
        img_data = combined_img
    # add the laplacian onto the grey scale image
    elif config.CHANNEL == 'STACKED':
        # get the laplacian of the image
        edge_chnl = cv2.GaussianBlur(img_data, (3, 3), 0).astype(np.uint8)
        ddepth = cv2.CV_16S
        lap_chnl = cv2.Laplacian(edge_chnl, ddepth, ksize=3 )
        img_data[:,:,0] += 2*lap_chnl

    elif config.CHANNEL == 'CLAHE':
        clahe = cv2.createCLAHE(clipLimit=10.0,tileGridSize=(3,3))
        img_data = clahe.apply(img_data.astype(np.uint16))
        img_data = img_data.astype(np.float32)
        img_data = np.expand_dims(img_data, axis=-1)

    # which mean subtraction values to use 
    if config.CHANNEL == 'RGB':
        dataset_means = config.PRE_PROCESS['rgb']
    elif config.CHANNEL == 'THERMAL':
        dataset_means = config.PRE_PROCESS['thermal'][3]
    elif config.CHANNEL == 'GREY':
        dataset_means = config.PRE_PROCESS['grey']
    elif config.CHANNEL == 'COMBINED':
        dataset_means = config.PRE_PROCESS['grey'][:]
        dataset_means.append(config.PRE_PROCESS['edge'][0])
    elif config.CHANNEL == 'STACKED':
        dataset_means = config.PRE_PROCESS['grey']
    elif config.CHANNEL == 'LAB':
        dataset_means = config.PRE_PROCESS['lab']
    elif config.CHANNEL == 'YCR_CB':
        dataset_means = config.PRE_PROCESS['ycr']
    elif config.CHANNEL == 'HSV':
        dataset_means = config.PRE_PROCESS['hsv']
    elif config.CHANNEL == 'CLAHE':
        dataset_means = config.PRE_PROCESS['clahe']

    else:
        print('Invalid channel')

    # mean subtraction to center image means around 0
    img_data -= np.asarray(dataset_means)

    return img_data
