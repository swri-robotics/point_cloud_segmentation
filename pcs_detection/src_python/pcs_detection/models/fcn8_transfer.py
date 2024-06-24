'''
 * @file fcn8_model.py 
 * @brief VGG16 as the backbone for fcn8. VGG16 imports the image net weights. All input images must have 3 channels.
 *
 * @author Jake Janssen
 * @date Dec 17, 2019
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

import os, sys
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Lambda, Add, MaxPooling2D, Dropout, Cropping2D, ZeroPadding2D
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling1D, UpSampling1D, Flatten, Reshape, Permute, Conv1D
from tensorflow.keras.layers import Conv2DTranspose, Input, BatchNormalization
from tensorflow.keras import optimizers
from tensorflow.keras.applications.vgg16 import VGG16

class fcn8():
    '''
    fcn8 with VGG weights transfered from the imagenet challenge.
    Inputs must be RGB.
    '''
    def __init__(self, config):
        self.model = None
        self.config = config
        if self.config.MODE in ['VALIDATE', 'VIDEO']:
            self.val = True
        else:
            self.val = False 

    def IoU(self, y_true, y_pred):
        nb_classes = K.int_shape(y_pred)[-1]
        iou = []
        pred_pixels = K.argmax(y_pred, axis=-1)
        for i in range(1, nb_classes):
            true_labels = K.equal(y_true[:,:,:,i], 1)
            pred_labels = K.equal(pred_pixels, i)
            inter = tf.to_int32(true_labels & pred_labels)
            union = tf.to_int32(true_labels | pred_labels)
            union_cnt = tf.math.count_nonzero(union)
            if K.get_value(union_cnt) != 0:
                ious = K.sum(inter) / K.sum(union)
                iou.append(ious)
        if not iou:
            iou = [0]
        iou = tf.stack(iou)
        return K.mean(iou)

    def labeled_IoU(self, y_true, y_pred):
        '''
        Gives the intersection over union for the unioned class predicition without ignore mask
        '''
        # combine the background and ignore label into a general background layer
        bg_label = K.any( K.stack( [y_true[:,:,:,0], y_true[:,:,:,-1]] ), axis=0)
        # everything that is False in the background label will be a labeled class 
        class_label = K.equal(bg_label, False)

        # choose the class that has the highest prediciton per pixel
        pred_pixels = K.argmax(y_pred, axis=-1)
        # get the mask of everything that was not predicited as background
        pred_label = K.not_equal(pred_pixels, 0)

        # find the intersection and union
        inter = tf.to_int32(class_label & pred_label)
        union = tf.to_int32(class_label | pred_label)

        # calculate the IoU
        return K.sum(inter) / K.sum(union)
        
    def sparse_loss(self, y_true, y_pred):
    
        y_pred = K.reshape(y_pred, (-1, K.int_shape(y_pred)[-1]))
        y_true = K.reshape(y_true, (-1, K.int_shape(y_pred)[-1] +1))
        unpacked = tf.unstack(y_true, axis=-1)
        y_true = tf.stack(unpacked[:-1], axis=-1) # Cutoff last label
        y_true = K.argmax(y_true,axis=-1)
        entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true,
                                                        logits=y_pred)
        return tf.reduce_mean(entropy, axis=-1)

    def weighted_sparse_loss(self, y_true, y_pred):
        y_pred = K.reshape(y_pred, (-1, K.int_shape(y_pred)[-1]))
        y_true = K.reshape(y_true, (-1, K.int_shape(y_pred)[-1] +1))
        unpacked = tf.unstack(y_true, axis=-1)
        y_true = tf.stack(unpacked[:-1], axis=-1)
        y_true = K.argmax(y_true,axis=-1)
        class_weights = tf.constant(self.config.LOSS_WEIGHTS)
        weights = tf.gather(class_weights, y_true)
        return tf.losses.sparse_softmax_cross_entropy(y_true, y_pred, weights)

    def sparse_accuracy_ignoring_last_label_ohot(self, y_true, y_pred):

        nb_classes = K.int_shape(y_pred)[-1]
        y_pred = K.reshape(y_pred, (-1, K.int_shape(y_pred)[-1])) # Reshape to h*w, number_of_classes
        y_true = K.reshape(y_true, (-1, K.int_shape(y_pred)[-1] +1))    
        
        valid_labels_mask = tf.not_equal(tf.argmax(y_true, axis=-1), nb_classes)
        indices_to_keep = tf.where(valid_labels_mask)
        unpacked = tf.unstack(y_true, axis=-1)
        unpacked[0] += unpacked[-1] # add ignore label to 
        y_true = tf.stack(unpacked[:-1], axis=-1) # Cutoff last label

        y_true = tf.gather_nd(params = y_true, indices=indices_to_keep)
        y_pred = tf.gather_nd(params = y_pred, indices=indices_to_keep)
        
        return K.mean(K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)))


    def build_model(self):
        # number of classes +1 for background
        num_class = len(self.config.CLASS_NAMES) + 1
        
        if self.config.CHANNEL == 'RGB' or self.config.CHANNEL == 'LAB' or self.config.CHANNEL == 'YCR_CB' or self.config.CHANNEL == 'HSV' or self.config.CHANNEL == 'GREY':
            num_channels = 3
        elif self.config.CHANNEL == 'THERMAL' or self.config.CHANNEL == 'STACKED':
            num_channels = 1
        elif self.config.CHANNEL == 'COMBINED':
            num_channels = 2
        else: 
            print('Invalid channel')

        # relu activiation function 
        act_func='relu'
        kernel_initializer='he_uniform'

        if self.config.USE_FULL_IMAGE:
            h, w = [480, 640]
        else:
            h, w = self.config.IMG_DIMS
        
        # Create input layer to feed into VGG
        # this layers must be >48,>48,3
        input_image = Input(shape=(h,
                                   w,
                                   num_channels), name='input_image')

        # use imagenet weights and create model
        model_vgg16_conv = VGG16(weights='imagenet', include_top=False, input_tensor=input_image, input_shape=(h,w,num_channels))

        # freeze the first five layers of VGG
        #for layer in model_vgg16_conv.layers: #.layers[0:2]:   #model_vgg16_conv.layers[0:5]:
        #    layer.trainable = False

        output_vgg16_conv = model_vgg16_conv(input_image)

        # begin classification
        fc6 = Conv2D(4096, 
                    kernel_size=(7,7), 
                    strides=(1,1), 
                    padding='same', 
                    activation=act_func,
                    kernel_initializer=kernel_initializer, 
                    name='fc6')(output_vgg16_conv)

        drop6 = Dropout(0.5)(fc6)

        fc7 = Conv2D(4096, 
                    kernel_size=(1,1), 
                    strides=(1,1), 
                    padding='same', 
                    activation=act_func, 
                    kernel_initializer=kernel_initializer,
                    name='fc7')(drop6)

        drop7 = Dropout(0.5)(fc7)

        score_frn = Conv2D(num_class, 
                        kernel_size=(1,1), 
                        strides=(1,1), 
                        padding='same', 
                        activation=act_func, 
                        kernel_initializer=kernel_initializer,
                        name='score_frn')(drop7)

        upscore2n = Conv2DTranspose(num_class, 
                            kernel_size=(4,4), 
                            strides=(2, 2), 
                            padding='same', 
                            name='upscore2n')(score_frn)
        
        score_pool4 = Conv2D(num_class, 
                            kernel_size=(1,1), 
                            strides=(1,1), 
                            padding='same', 
                            activation=act_func, 
                            kernel_initializer=kernel_initializer,
                            name='score_pool4n')(model_vgg16_conv.get_layer("block4_pool").output)
                            
        try:
            fuse_pool4 = Add()([upscore2n, score_pool4])
        except:
            upscore2n = Cropping2D(cropping=((1,0),(1,0)))(upscore2n)
            fuse_pool4 = Add()([upscore2n, score_pool4])
            
        upscore_pool4 = Conv2DTranspose(num_class, 
                                    kernel_size=(4,4), 
                                    strides=(2, 2), 
                                    padding='same', 
                                    name='upscore_pool4')(fuse_pool4)

        score_pool3n = Conv2D(num_class, 
                            kernel_size=(1,1), 
                            strides=(1,1), 
                            padding='same', 
                            activation=act_func, 
                            kernel_initializer=kernel_initializer,
                            name='score_pool3n')(model_vgg16_conv.get_layer("block3_pool").output)
        
        fuse_pool3 = Add()([upscore_pool4, score_pool3n])

        upscore8n = Conv2DTranspose(num_class, kernel_size=(16,16), strides=(8, 8), 
                padding='same', name='upscore8n', activation='linear')(fuse_pool3)

        upscore_out = Activation('softmax')(upscore8n)

        model = Model(input_image, upscore8n)
        model.summary()

        if not self.val:
            # use sgd for optimizer (could try adam too)
            sgd = optimizers.SGD(lr=self.config.LEARNING_RATE['start'], momentum=0.9)
            #adam = optimizers.Adam(lr = self.config.LEARNING_RATE['start'])
            model.compile(
                            optimizer=sgd,
                            loss = self.weighted_sparse_loss,
                            metrics=[self.sparse_accuracy_ignoring_last_label_ohot, self.labeled_IoU]
                        )

        self.model = model

        if self.val:
            try:
                model.load_weights(self.config.VAL_WEIGHT_PATH)
                print("[INFO] Loded Model Weights {0}".format(self.config.VAL_WEIGHT_PATH))
            except Exception as e:
                print("[INFO] Model Weight File Not Found")
                print(e)
                sys.exit()
