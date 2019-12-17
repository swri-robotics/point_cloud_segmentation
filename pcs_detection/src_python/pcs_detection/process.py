'''
 * @file deploy.py
 * @brief Training, validating, verifying training data, and deployment functions 
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
import keras
import sys
import datetime
import os
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau, History
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import time
from pcs_detection.data_loader import dataLoader
from pcs_detection.preprocess import preprocessing
from pcs_detection.utils import histogram, minMaxNormalize, resize, dump_validation_config, dump_inference_config, colorTriLabel, colorPrediction, LABtoBGR, get_colors
from tqdm import tqdm

def _build_model(config):
    '''
    Builds the model for training or validation
    '''

    # merge the two class names so that the background classes are first
    config.CLASS_NAMES = config.BACKGROUND_CLASS_NAMES.copy()
    config.CLASS_NAMES.extend(config.TARGET_CLASS_NAMES)

    # create the model
    if config.MODEL == 'fcn8':
        from src_python.pcs_detection.models.fcn8_model import fcn8
        weldDetector = fcn8(config)
        weldDetector.build_model()
        weldDetector = weldDetector.model
    elif config.MODEL == 'fcn_transfer':
        from src_python.pcs_detection.models.fcn8_transfer import fcn8
        weldDetector = fcn8(config)
        weldDetector.build_model()
        weldDetector = weldDetector.model
    else:
        print('invalid model')

    return weldDetector

def train(config):
    '''
    Trains the model with fit generator and plots the results. 
    '''

    weldDetector = _build_model(config)

    # saving path will also add a time stamp to avoid duplicates and save per epoch
    dir_path = os.path.dirname(os.path.realpath(__file__)).rsplit('/',2)[0] + '/scripts'
    utid = datetime.datetime.now().strftime('%y_%m_%d_%H%M%S')
    model_info = config.MODEL + '_' + config.CHANNEL
    config.WEIGHT_SAVE_PATH = '{0}/{1}/{2}_{3}_{4}/{5}'.format(dir_path, config.WEIGHT_DIR, config.WEIGHT_ID, model_info, utid, '{epoch:02d}.h5')

    # reset umask so new permissions can be written 
    original_umask = os.umask(0)
    #make the model save dir if it does not already exist
    if not os.path.exists(config.WEIGHT_SAVE_PATH):
        print("Making dir: {}".format(config.WEIGHT_SAVE_PATH))
        os.makedirs(os.path.split(config.WEIGHT_SAVE_PATH)[0], mode=0o777)
        os.umask(original_umask)
    else:
        os.chmod(os.path.split(config.WEIGHT_SAVE_PATH)[0], 0o777)
        os.umask(original_umask)

    # create the generator for the data
    DataLoader = dataLoader(config)

    DataLoader_val = dataLoader(config)
    DataLoader_val.mode = "VALIDATION"
    #DataLoader_val.use_full_image = True
    DataLoader_val.loadDataPaths()

    # callback to save the weight files 
    checkpointer = ModelCheckpoint(
                        filepath=config.WEIGHT_SAVE_PATH,
                        verbose=1,               
                        save_best_only=True,
                        mode = 'max',
                        monitor = 'val_labeled_IoU',
                        save_weights_only=False
                        )
    # callback to reduce the learning rate 
    reduce_lr = ReduceLROnPlateau(monitor='loss',
                                            factor=config.LEARNING_RATE['reduce_factor'],
                                            patience=config.LEARNING_RATE['reduce_patience'],
                                            min_lr=config.LEARNING_RATE['end'],
                                            verbose=1)

    # callback to stop early
    early_stop = EarlyStopping(monitor='loss', min_delta=0, patience=30, verbose=1, mode='auto')

    # training takes place here 
    history = weldDetector.fit_generator(
        DataLoader.generate(),
        epochs = config.N_EPOCHS,
        steps_per_epoch = (DataLoader.num_paths//config.BATCH_SIZE) * config.AUGMENTATIONS['number_aug'],
        verbose = 1,
        callbacks = [History(), checkpointer, reduce_lr, early_stop],
        use_multiprocessing=False,
        validation_data = DataLoader_val.generate(),
        validation_steps = DataLoader_val.num_paths // config.BATCH_SIZE
    )

    print("Finished training.")

    # generate a config for validation/inference using the best weights from training
    weights_dir = config.WEIGHT_SAVE_PATH.rsplit('/',1)[0]
    best_weights_file = '01.h5'
    for file in sorted(os.listdir(weights_dir)):
        if file.endswith('.h5'):
            best_weights_file = file 
    config.VAL_WEIGHT_PATH = os.path.join(weights_dir,best_weights_file)
    config.MODE = 'VALIDATE'
    dump_validation_config(config)
    dump_inference_config(config)
    # Display and save training curves 
    fig = plt.figure()
    axis1 = fig.add_subplot(311)
    axis1.plot(history.history['sparse_accuracy_ignoring_last_label_ohot'], 'ro-')
    axis1.plot(history.history['val_sparse_accuracy_ignoring_last_label_ohot'], 'bo-')
    axis1.set_title('Training Metrics')
    axis1.set_ylabel('Accuracy')
    axis1.set_xlabel('Epoch')
    axis1.legend(['Train', 'Test'], loc='upper left')
    
    axis2 = fig.add_subplot(312)
    axis2.plot(history.history['loss'], 'mo-')
    axis2.plot(history.history['val_loss'], 'co-')
    #axis2.set_title('Model loss')
    axis2.set_ylabel('Loss')
    axis2.set_xlabel('Epoch')
    axis2.legend(['Train', 'Test', 'Train'], loc='upper left')

    axis3 = fig.add_subplot(313)
    axis3.plot(history.history['labeled_IoU'], 'mo-')
    axis3.plot(history.history['val_labeled_IoU'], 'co-')
    #axis3.set_title('Model IoU' + config.WEIGHT_SAVE_PATH)
    axis3.set_ylabel('IoU')
    axis3.set_xlabel('Epoch')
    axis3.legend(['Train', 'Test', 'Train'], loc='upper left')

    # saving them off in same folder as weights and config
    metrics_path = os.path.join(os.path.split(config.WEIGHT_SAVE_PATH)[0],'metrics.png')
    print('saving figure to' ,metrics_path)
    fig.savefig(metrics_path)

    return
    #fig.show()

def validate(config):
    '''
    Used for looking at predictions of an already trained model.
    '''

    weldDetector = _build_model(config)
   
    # always validate on full image
    config.USE_FULL_IMAGE = True

    scale_factor = config.DISPLAY_SCALE_FACTOR

    # create the generator for the data
    DataLoader = dataLoader(config)
    gen = DataLoader.generate()

    for img_batch, label_batch in gen:

        start = time.time()
        pred = weldDetector.predict(img_batch)
        pred[:,:,:,0] += config.CONFIDENCE_THRESHOLD
        #pred[:,:,:,-1] += config.CONFIDENCE_THRESHOLD
        #plastic_idx = config.CLASS_NAMES.index('plastic') + 1
        #pred[:,:,:,plastic_idx] += -10
        end = time.time()
        print('Batch Size', config.BATCH_SIZE)
        print('Prediction Time:', end-start)

        for ii in range(img_batch.shape[0]):

            image = img_batch[ii]
            label = label_batch[ii]

            # convert the image back to bgr if needed 
            if config.CHANNEL == 'LAB':
                image = LABtoBGR(image, config)

            elif config.CHANNEL == 'YCR_CB':
                image += np.asarray(config.PRE_PROCESS['ycr'])
                image = cv2.cvtColor(image,cv2.COLOR_YCR_CB2BGR)

            prediction = pred[ii].copy()

            # practical min max normalization 
            # values will depend on application 
            prediction = prediction/config.PREDICTION_MAX_RANGE + 0.5

            # get the max value of the prediction
            val_prediction = np.max(prediction,axis=-1)

            # probability is based off of the difference between the max class and background
            val_prediction -= np.max(prediction[:,:,0:(len(config.BACKGROUND_CLASS_NAMES)+1)], axis=-1)

            # clip any abnormally strong predicitons 
            val_prediction[val_prediction > 1] = 1
            val_prediction[val_prediction < 0] = 0

            results_image = cv2.merge((val_prediction, val_prediction, val_prediction)) * 2*255
            results_image = results_image.astype(np.uint8)

            cv2.imshow('Prob Dist', results_image)

            colors = get_colors(len(config.CLASS_NAMES))

            image = minMaxNormalize(image)
            if image.shape[-1] != 3:
                image = cv2.merge((image, image, image))
            colored_predicition = colorPrediction(pred[ii], image, colors)

            if config.CHANNEL == 'COMBINED':
                edge = minMaxNormalize(image[:,:,-1:image.shape[2]]).astype(np.uint8)
                image = image[:,:,0:image.shape[-1]-1]
                cv2.imshow('edge', edge)

            # normalize image to be between 0 and 255 
            display_label = colorTriLabel(label, colors)

            cv2.imshow('img', resize(image, config.DISPLAY_SCALE_FACTOR))
            cv2.imshow('label', resize(display_label, config.DISPLAY_SCALE_FACTOR))
            cv2.imshow('pred overlay', resize(colored_predicition, config.DISPLAY_SCALE_FACTOR))
            key = cv2.waitKey(0)
            # press q to quit the imshows 
            if 'q' == chr(key & 255):
                sys.exit(0)

def test_dataloader(config):
    '''
    Mode used to check the quality of the images passed in to train.
    Histogram values should have a mean centered at zero (or within a standard deviation).
    Green represents the label, blue the background mask, and red the ignore mask. 
    '''

    config.CLASS_NAMES = config.BACKGROUND_CLASS_NAMES.copy()
    config.CLASS_NAMES.extend(config.TARGET_CLASS_NAMES)

    DataLoader = dataLoader(config)
    gen = DataLoader.generate()

    while True:
        img_batch, label_batch = next(gen)

        for ii in range(len(img_batch)):

            image = img_batch[ii,:,:,:]
            label = label_batch[ii,:,:,:]

            # want a mean within a std of zero
            histogram(image)

            if config.CHANNEL == 'LAB':
                image = LABtoBGR(image, config)

            elif config.CHANNEL == 'YCR_CB':
                image += np.asarray(config.PRE_PROCESS['ycr'])
                image = cv2.cvtColor(image,cv2.COLOR_YCR_CB2BGR)

            if config.CHANNEL == 'COMBINED':
                edge = minMaxNormalize(image[:,:,-1:image.shape[2]]).astype(np.uint8)
                image = image[:,:,0:image.shape[-1]-1]
                cv2.imshow('edge', edge)

            colors = get_colors(len(config.CLASS_NAMES))
            display_label = colorTriLabel(label, colors)

            # normalize image to be between 0 and 255 
            image = minMaxNormalize(image)

            # overylay to view label on image
            if image.shape[-1] != 3:
                image = cv2.merge((image, image, image))
            overlay = image.copy()
            overlay_label = label.copy()
            overlay = colorPrediction(overlay_label, overlay, colors)

            cv2.imshow('img', resize(image, config.DISPLAY_SCALE_FACTOR))
            cv2.imshow('label', resize(display_label, config.DISPLAY_SCALE_FACTOR))
            cv2.imshow('overlay', resize(overlay, config.DISPLAY_SCALE_FACTOR))
            key = cv2.waitKey(0)
            # press q to quit the imshows 
            if 'q' == chr(key & 255):
                sys.exit(0)

def class_analysis(config):
    '''
    Generate a plot showing the distribution of pixels for each class
    '''
    config.CLASS_NAMES = config.BACKGROUND_CLASS_NAMES.copy()
    config.CLASS_NAMES.extend(config.TARGET_CLASS_NAMES)

    DataLoader = dataLoader(config)
    gen = DataLoader.generate()

    class_counts = np.zeros(len(config.CLASS_NAMES)+2)

    for cnt in range(DataLoader.num_paths//config.BATCH_SIZE):
        _, label_batch = next(gen)
        class_counts[0] += np.count_nonzero(label_batch[:,:,:,0])
        class_counts[-1] += np.count_nonzero(label_batch[:,:,:,-1])
        for ii in range(len(config.CLASS_NAMES)):
            class_counts[ii+1] += np.count_nonzero(label_batch[:,:,:,ii+1])
    
    total_count = class_counts.sum()
    class_counts /= total_count

    max_class = class_counts[:-1].max()
    rec_weights = max_class / class_counts[:-1]

    class_names = ['background']
    class_names.extend(config.CLASS_NAMES)
    class_names.append('ignore')
    print('__CLASS_PIXEL_DISTRIBUTION__')
    for jj, class_name in enumerate(class_names):
        print(class_name + ':', class_counts[jj]) 

    print('recommended weights')
    print(rec_weights)
    print(rec_weights.to_list())

def demo_video(config):
    '''
    Creates an MP4 using the images in the validation set
    '''

    weldDetector = _build_model(config)

    # always validate on full image
    config.USE_FULL_IMAGE = True
    config.AUGMENTATIONS = {
        "number_aug": 0,
        "horizontal_flip": False,
        "brightness_range": None,
        "zoom_range": False,
        "rotation_range": None,
        "vertical_flip": False
    }

    scale_factor = config.DISPLAY_SCALE_FACTOR

    # create the generator for the data
    DataLoader = dataLoader(config)
    gen = DataLoader.generate()

    video_name = '<path to save file>'
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    video = cv2.VideoWriter(video_name, 0x00000021, 4, (2*config.ORIG_DIMS[1], config.ORIG_DIMS[0]))

    for _, gen_data in zip(tqdm(range(DataLoader.num_paths//config.BATCH_SIZE)), gen):
        img_batch, label_batch = gen_data
        pred = weldDetector.predict(img_batch)

        for ii in range(img_batch.shape[0]):

            tri_pred = np.zeros((pred[ii].shape[0], pred[ii].shape[1], 3))
            colors = [(102, 204, 255), (255, 153, 102), (0,153,255)]
            prediction = np.argmax(pred[ii],axis=-1)

            for jj, color in enumerate(colors):
                temp_chnl_img = np.zeros((pred[ii].shape[0],pred[ii].shape[1], 2))
                temp_chnl_img[:,:,1][prediction==jj+1] = 1
                kernel = np.ones((10,10),np.uint8)
                tri_pred[temp_chnl_img[:,:,1]==1] = color

            prediction[prediction > 0] = 255
            prediction = prediction.astype(np.uint8)
            image = img_batch[ii]
            if config.CHANNEL == 'LAB':
                image = LABtoBGR(image, config)
            elif config.CHANNEL == 'YCR_CB':
                    image += np.asarray(config.PRE_PROCESS['ycr'])
                    image = cv2.cvtColor(image.astype(np.uint8),cv2.COLOR_YCR_CB2BGR)
            image = minMaxNormalize(image)
            overlay = image.copy()
            _, contours, _ = cv2.findContours(prediction,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (7,24,194), 3)
            combined = np.zeros((overlay.shape[0], overlay.shape[1]*2, 3))
            combined[:, 0:overlay.shape[1],:] = overlay
            combined[:, overlay.shape[1]:2*overlay.shape[1], :] = tri_pred
            video.write(combined.astype(np.uint8))

    video.release()
    return


