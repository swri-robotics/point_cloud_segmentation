'''
 * @file data_loader.py 
 * @brief Provides a generator of preprocessed images and corresponding labels from directories specified in the config 
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

from keras.preprocessing.image import ImageDataGenerator
import os
import random
from itertools import cycle
import cv2
import numpy as np
from utils import get_labels_from_xml
from preprocess import preprocessing

class dataLoader():
    '''
    This class is responsible for generating the image and label that will be fed into training. 
    Preprocessing, augmentations, and class balancing will be done here. 
    The generate funciton should be called in the main program to return new batches of training images
    '''
    def __init__(self, config):
        self.config = config
        self.mode = config.MODE
        self.augmentations = config.AUGMENTATIONS

        # used for augmentations
        self.datagen = ImageDataGenerator(
            rotation_range=config.AUGMENTATIONS["rotation_range"],
            brightness_range = config.AUGMENTATIONS['brightness_range'],
            horizontal_flip = config.AUGMENTATIONS['horizontal_flip'],
            vertical_flip = config.AUGMENTATIONS['vertical_flip'],
            zoom_range = config.AUGMENTATIONS['zoom_range'],
            fill_mode='constant',
            cval = 0 # new area caused by augmentations are black
        )
        
        # init for future use
        self.xml_labels = dict()
        self.num_paths = 0

        # load in the data paths 
        self.loadDataPaths()

    def loadDataPaths(self):
        '''
        Get the filenames from the dataset directories and the countours from labels.xml
        self.datapaths holds these values
        '''
        self.data_paths = []
    
        # use either the training or validation sets 
        if self.mode in ['TRAIN', 'TEST_TRAINING_DATA']:
            data_dirs = self.config.TRAINING_DIRS
        else:
            data_dirs = self.config.VALIDATION_DIRS

        # go through every directory
        for directory in data_dirs:
            if directory['class'] == 'weld':
                label_path = directory['dir_path'].split("/")[0:-1]
                label_path.append("labels.xml")
                label_path = "/".join(label_path)

                # convert labels.xml into contours
                labels = get_labels_from_xml(label_path)
                self.xml_labels = {**self.xml_labels, **labels}

                # get all file names in the directory 
                fnames = list(labels.keys())
                fnames = [fname.split('/')[-1] for fname in fnames]

                if directory['num_imgs'] != 'all':
                    # calculate the number of interval for to get num_imgs of files evenly spaced out
                    skip_interval = round(len(fnames) / directory['num_imgs'])

                    # fnames now holds all the files of interest
                    fnames = fnames[::skip_interval]

                # get the full path for file names
                fnames = ['{0}/{1}'.format(directory['dir_path'], fname) for fname in fnames]

            elif directory['class'] == 'background':
                fnames = os.listdir(directory['dir_path'])
                fnames = [directory['dir_path'] + '/' + fname for fname in fnames]
                
            else:
                print('Invalid class')

            print('[Loading] {} paths from {}.'.format(len(fnames),  directory['dir_path'].split('/')[4]))

            self.data_paths.extend(fnames)
        

        # if training or simulating training shuffle the data paths 
        if self.mode == 'TRAIN' or self.mode == 'TEST_TRAINING_DATA':
            random.shuffle(self.data_paths)

        self.num_paths = len(self.data_paths)
        print('loaded', self.num_paths, 'images')
        return

    def create_multi_label(self, label):
        '''
        Create a label that has a weld mask, background mask, and ignore mask.
        Ignore mask will be created around the weld mask.
        '''
        # multi label will store the annotations, background, and ignore mask
        multi_label = np.zeros((label.shape[0], label.shape[1], 3))

        # create the background mask
        background_mask = np.zeros((label.shape[0], label.shape[1]),  dtype=np.float32)
        kernel = np.ones((5,5),np.uint8)
        bin_label = label.copy()
        # convert the image to 2bit so it plays nice with dilate 
        bin_label[bin_label>0] = 1
        # dialate the label to create the area that will be ignored
        background_mask = cv2.dilate(bin_label,kernel,iterations = 2)
        # flip the dialated label to get background
        background_mask = np.where(background_mask>0, 0, 1)

        #initially create the ignore mask
        ignore_mask = np.zeros((label.shape[0], label.shape[1]), dtype=np.float32)
        # Reduce the max pixel values to be 1
        background_mask[background_mask > 0] = 1
        label[label > 0] = 1

        # background cannot intersect with label
        background_mask[label == 1] = 0
        # ignore mask is everything that is not background or label
        ignore_mask[background_mask == 0] = 1
        ignore_mask[label==1] = 0

        # assign values to the multi label 
        multi_label[:, :, 0] = background_mask
        multi_label[:, :, 1] = label
        multi_label[:, :, 2] = ignore_mask

        return multi_label.astype(np.float32)

    def reduce_background_label(self, label_batch):
        '''
        Reduce the number of background pixels by randomly ignoring them 
        until the ratio of weld to background pixels is satisfied.
        '''
        # get the total amount of labeled pixels in the batch 
        imgs_per_batch = label_batch.shape[0]
        label_px_batch = np.count_nonzero(label_batch[:, :, :, 1])
        background_px_per_img = (label_px_batch/imgs_per_batch) * self.config.BACKGROUND_REDUCTION

        # if the whole batch is background then leave then make it all ignore
        if label_px_batch == 0:
            label_batch[:,:,:,0] = 0
            label_batch[:,:,:,2] = 1
            return label_batch

        for ii in range(imgs_per_batch):
            # get the row and column idxs of the background pixels
            rows, columns = np.nonzero(label_batch[ii,:,:,0])
            background_pxls = len(rows)
            extra_pxls = int(background_pxls - background_px_per_img)

            if extra_pxls > 0:
                # choose the idxs to replace 
                replace_idxs = np.random.choice(
                    background_pxls,
                    size = extra_pxls,
                    replace = False 
                )
                idx = random.randint(0,background_pxls)
                # remove background labels and add to the ignore mask
                label_batch[ii,rows[replace_idxs],columns[replace_idxs],0] = 0
                label_batch[ii,rows[replace_idxs],columns[replace_idxs],2] = 1
      
        return label_batch

    def tile_image(self, image, label):
        '''
        Selects a region which shape is specified in the config that contains 
        a high proportion of the label.
        Only intended to be used for training
        '''
        orig_h = image.shape[0]
        orig_w = image.shape[1]

        label_px = np.count_nonzero(label[:,:,1])

        # if a label exists, create a tile that encompasses the label in the image
        if label_px > 0:
            h,w = np.nonzero(label[:,:,1])

            h_padding = (self.config.IMG_DIMS[0]-h.ptp())
            w_padding = (self.config.IMG_DIMS[1]-w.ptp())

            keep_searching = True
            curr_iter = 0
            max_iter = 50

            # search for a tile with enough labeled pixels 
            while keep_searching:
                if w_padding != 0:
                    w_split = random.randint(0,abs(w_padding))
                    w_split = w_split * (w_padding/abs(w_padding))
                    w_range = [w.min()-w_split, w.max()+(w_padding-w_split)]
                else:
                    w_range = [w.min(), w.max()] 

                if h_padding != 0:
                    h_split = random.randint(0,abs(h_padding))
                    h_split = h_split * (h_padding/abs(h_padding))
                    h_range = [h.min()-h_split, h.max()+(h_padding-h_split)]
                else:
                    h_range = [h.min(), h.max()]

                trimmed_image = image[int(h_range[0]):int(h_range[1]), int(w_range[0]):int(w_range[1]), :]
                trimmed_label = label[int(h_range[0]):int(h_range[1]), int(w_range[0]):int(w_range[1]), :]

                dim_check = trimmed_image.shape[0] == self.config.IMG_DIMS[0] and trimmed_image.shape[1] == self.config.IMG_DIMS[1]
                pixel_check = np.count_nonzero(trimmed_label[:,:,1]) > self.config.MIN_PIXELS_IN_IMG

                if (dim_check and pixel_check) or (dim_check and curr_iter>max_iter):
                    keep_searching = False
                
                curr_iter += 1

            return trimmed_image, trimmed_label

        # Pick a random region for background images 
        else:
            # pick a random h and w point where any point selected can center the new tile
            h_center = random.randint(self.config.IMG_DIMS[0] / 2, orig_h - (self.config.IMG_DIMS[0]) / 2)
            h_range = [h_center - (self.config.IMG_DIMS[0]/2), h_center + (self.config.IMG_DIMS[0]/2)]

            w_center = random.randint(self.config.IMG_DIMS[1] / 2, orig_w - (self.config.IMG_DIMS[1] / 2))
            w_range = [w_center - (self.config.IMG_DIMS[1]/2), w_center + (self.config.IMG_DIMS[1]/2)]

            # create the tile around this center
            trimmed_image = image[int(h_range[0]):int(h_range[1]), int(w_range[0]):int(w_range[1]), :]
            trimmed_label = label[int(h_range[0]):int(h_range[1]), int(w_range[0]):int(w_range[1]), :]

            return trimmed_image, trimmed_label

    def generate(self):
        '''
        Serves up the images and applies preprocessing, augmentation, and background reduction.
        Batches will be returned as a numpy array of shape batch_size x height x width x number of channels.
        '''
        # cycle through all the data paths 
        data_path_cycler = cycle(self.data_paths) 

        # generator goes on forever
        while True:
            img_batch = []
            label_batch = []
            batch_cnt = 0

            # keep going until the batch count is reached
            while batch_cnt < self.config.BATCH_SIZE:
                file_path = next(data_path_cycler)

                # load in image and draw lines on label
                try:
                    img_data = cv2.imread(file_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
                    
                except:
                    continue

                # apply any laplacian preprocessing and mean subtraction
                img_data = preprocessing(img_data, self.config)
                
                # genereate the label
                try:
                    key = file_path.split('/')
                    key = key[4] + '/' + key[-1]
                    label_contours = self.xml_labels[key]
                    label = np.zeros((img_data.shape[0], img_data.shape[1])).astype(np.float32)
                    for contour in label_contours['contour']:
                        #cv2.drawContours(label, [contour], -1, (255, 255, 255), -1)
                        cv2.polylines(label, [contour], False, 1, self.config.LABEL_THICKNESS)

                    multi_label = self.create_multi_label(label)
                # will fail on background
                except:
                    multi_label = np.zeros((img_data.shape[0], img_data.shape[1], 3))
                    multi_label[:,:,0] = 1

                # pick a random tile from the image
                if not self.config.USE_FULL_IMAGE:
                    img_data, multi_label = self.tile_image(img_data, multi_label)

                # add augmented image and label into the batch
                img_batch.append(img_data)
                label_batch.append(multi_label)
                batch_cnt += 1 
                
            img_batch = np.asarray(img_batch)
            label_batch = np.asarray(label_batch)

            # image and label augmentations use the same seed to keep in sync
            seed = random.randint(10, 100000)

            # create an augmented image and label unless we are validating
            if self.mode != 'VALIDATE':
                aug_img = next(self.datagen.flow(img_batch, batch_size=self.config.BATCH_SIZE, seed=seed))
                aug_label = next(self.datagen.flow(label_batch, batch_size=self.config.BATCH_SIZE, seed=seed))
            else:
                aug_img = img_batch
                aug_label = label_batch

            # add any added black pixels into the ignore mask
            empty_pixels = np.sum(aug_label, axis=-1)
            empty_pixels_mask = empty_pixels == 0
            aug_label[empty_pixels_mask, -1] = 1

            # reduce the number of background pixels
            aug_label = self.reduce_background_label(aug_label)

            # send back the batch 
            yield [np.asarray(aug_img), np.asarray(aug_label)]
