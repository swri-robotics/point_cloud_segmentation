'''
 * @file utils.py
 * @brief Helper functions for viewing images and reading/writing config and label files 
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

import numpy as np
import pandas as pd
from lxml import etree
import cv2
import os
import json

def get_labels_from_xml(label_path):
    '''
    Reads in labels from an xml file with CVATs format
    and returns a dictionary where the keys are whatever is specified as image name (outer directory and filename)
    and the values are the contours for that image
    '''
    anno = dict()
    root = etree.parse(label_path).getroot()
    
    # name of the outer directory this label corresponds to
    # allows for images in different directories to have the same filename 
    dataset_name = label_path.split('/')[2]

    for image_tag in root.iter('image'):
        image = {}
        # initialize all types of labels
        for label_name in root.iter('label'):
            image[label_name.find('name').text] = list()

        #extract the meta info from the image
        for key, value in image_tag.items():
            image[key] = value

        #keys will be collection folder combined with each images
        #this allows for images in different folders to have the same name
        image['name'] = dataset_name + '/' + image['name']

        image['contour'] = list()

        # loop through the poly elements in the image
        for poly_tag in image_tag:
            # get the type of poly element - eg polyline, polygon
            poly_elem = {}
            poly_type = poly_tag.tag
            poly_elem[poly_type] = []

            for key, value in poly_tag.items():
                poly_elem[key] = value
 
            #Get the contour points
            contour_points = []
            shape = str(poly_elem['points']).split(";") #get the shape from the polyline 
            for pair in shape:
                x, y = pair.split(",")
                contour_points.append([int(float(x)), int(float(y))])

            #create a contour set of points in the proper order from the points
            contour = np.array(pd.DataFrame(contour_points),
                                                        np.int32
                                                        )
            image[poly_elem['label']].append([contour,poly_type])
            
        anno[image['name']] = image
    
    return anno

def dump_validation_config(config):
    '''
    Save the config used to train in the same folder in the model weights.
    This config can later be used to apply the same preprocessing, and model selection
    that was used in training 
    '''

    config_dump = {}
    for key in config.__dict__:
        if (not key.startswith('__')) and (key != 'logger') :
            try:
                config_dump[key] = config.__getattribute__(key)
            except:
                pass
    save_path = os.path.join(os.path.split(config.WEIGHT_SAVE_PATH)[0],'full_config.json')
    with open(save_path, 'w') as outfile:
        json.dump(config_dump, outfile, indent=4) 

def dump_inference_config(config):
    '''
    Save a config used for inference in the same folder as the weights.
    '''
    wanted_keys = ['MODEL', 'VAL_WEIGHT_PATH', 'BATCH_SIZE', 'MODE', 'DISPLAY_SCALE_FACTOR', 'CHANNEL', 'PRE_PROCESS', 'CONFIDENCE_THRESHOLD', 'TARGET_CLASS_NAMES', 'BACKGROUND_CLASS_NAMES', 'PREDICTION_MAX_RANGE' 'ORIG_DIMS', ]
    config_dump = {}
    for  key in config.__dict__:
        if key in wanted_keys:
            try:
                config_dump[key] = config.__getattribute__(key)
            except:
                pass
    save_path = os.path.join(os.path.split(config.WEIGHT_SAVE_PATH)[0],'inference_config.json')
    print('_____CONFIG______')
    print(config_dump)
    print('_________________')
    with open(save_path, 'w') as outfile:
        json.dump(config_dump, outfile, indent=4) 


def resize(image, scale_factor):
    '''
    Used to resize a display image.
    '''
    new_height = int(image.shape[0] * scale_factor)
    new_width = int(image.shape[1]  * scale_factor)
    resized_img = cv2.resize(image, (new_width, new_height))
    return resized_img


def minMaxNormalize(chnls):
    '''
    Normalizes images to have pixel values between 0-255
    This function should only be used for displaying 
    '''
    #loop over all of the channels
    for i in range(chnls.shape[-1]):
        # calculate min and ptp of each channel. 
        min_arr = np.min(chnls[:, :, i][chnls[:, :, i] != 0]) 
        ptp_arr = chnls[:, :, i].ptp()

        chnls[:, :, i] = (((chnls[:, :, i] - min_arr) / ptp_arr) * 255)

    return chnls.astype(np.uint8)

def histogram(original_image):
    '''
    Displays a histogram showing the pixel value distributions per channel of the image.
    This function should be used to check the being fed into the network.
    Each channel should have a mean of zero and all standard deviations should be within the same magnitude. 
    '''
    img_cv = original_image.copy()

    org_means = []
    org_stds = []

    for i in range(img_cv.shape[-1]):
        org_means.append(round(img_cv[:,:,i].mean(),2))
        org_stds.append(round(img_cv[:,:,i].std(),2))

    img_cv = minMaxNormalize(img_cv)

    # Histograms of data distribution 
    # split channels 
    hist_chnls = cv2.split(img_cv)

    # histogram parameters
    histSize = 256
    histRange = (0,256)
    accumulate = False
    hist_w = 512
    hist_h = 400
    bin_w = int(round( hist_w/histSize ))
    histImage = np.zeros((hist_h, hist_w, 4), dtype=np.uint8)

    hists = [] 
    
    # text colors and colors for plot
    t_colors = [(210,0,0), (0,210,0), (0,0,210)]
    colors = [(255,0,0), (0,255,0), (0,0,255) ]

    # starting vertical location of text
    text_h = 30

    # get data into histogram format
    for ii in range(len(hist_chnls)):
        temp_hist = cv2.calcHist(hist_chnls, [ii], None, [histSize], histRange, accumulate=accumulate)
        cv2.normalize(temp_hist, temp_hist, alpha = 0, beta=hist_h, norm_type=cv2.NORM_MINMAX)
        hists.append(temp_hist)

    # add histogram to image 
    for jj in range(1, histSize):
        for curr_hist, color in zip(hists,colors):
            cv2.line(histImage, ( bin_w*(jj-1), hist_h - int((curr_hist[jj-1]).round()) ),
                    ( bin_w*(jj), hist_h - int((curr_hist[jj]).round()) ),
                    color, thickness=1)

    # add text and mean/normal distribution lines 
    for ii, color, t_color in zip(range(len(hist_chnls)), colors, t_colors):
        hist_std = int(round(hist_chnls[ii].std()))
        hist_mean = int(round(hist_chnls[ii].mean()))
        cv2.circle(histImage, (2*hist_mean, 400 ), 2*hist_std, color, thickness=4)
        display_str = 'Mean: ' + str(org_means[ii]) + ', Std: ' + str(org_stds[ii])
        cv2.putText(histImage , display_str, (10,text_h), cv2.FONT_HERSHEY_SIMPLEX, .4, t_color, 1,cv2.LINE_AA)
        text_h += 20

    # display histogram 
    cv2.imshow('RGBT Data Distribution', histImage)

def colorTriLabel(label, colors):
    display_label = np.zeros((label.shape[0], label.shape[1], 3))
    display_label[:,:,0] = label[:,:,0] * 255
    display_label[:,:,-1] = label[:,:,-1] * 255
    for ii, color in enumerate(colors):
        display_label[label[:,:,ii+1] == 1] = color
    return display_label.astype(np.uint8)

def colorPrediction(prediction, orig_img, colors):
    '''
    Creates a three channel bgr image and colors it with the prediciton.
    '''
    prediction_display = orig_img.copy()    
    prediction = np.argmax(prediction, axis=-1)
    for jj, color in enumerate(colors):
        prediction_display[:,:][prediction==jj+1] = color
    prediction_display = prediction_display.astype(np.uint8)
    return prediction_display

def LABtoBGR(image, config):
    '''
    Used to convert the LAB color space back to BGR
    '''
    image += config.PRE_PROCESS['lab']
    image[image==0] = 1e-4
    image = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_LAB2BGR)
    image *= 255
    return image

def get_colors(n):
    '''
    Generates a list of colors
    '''
    colors = [[102,255,153], [255, 102, 204], [102, 204, 255], [51, 102, 153]]
    return colors[0:n]

