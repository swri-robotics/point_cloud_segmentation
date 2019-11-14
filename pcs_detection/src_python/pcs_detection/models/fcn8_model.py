'''
 * @file fcn8_model.py 
 * @brief Standard version of fcn8 with added batch normalization
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

import os, sys
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.layers import Activation, Lambda, Add, MaxPooling2D, Dropout, Cropping2D, ZeroPadding2D
from keras.layers import Input, Dense, Conv2D, MaxPooling1D, UpSampling1D, Flatten, Reshape, Permute, Conv1D
from keras.layers import Conv2DTranspose, Input, BatchNormalization
from keras import optimizers


class fcn8():
    def __init__(self, config):
        self.model = None
        self.autoencoder = False
        self.ignore_label = 6
        self.config = config 

    def IoU(self, y_true, y_pred):
        nb_classes = K.int_shape(y_pred)[-1]
        iou = []
        pred_pixels = K.argmax(y_pred, axis=-1)
        for i in range(1, nb_classes):
            true_labels = K.equal(y_true[:,:,:,i], 1)
            pred_labels = K.equal(pred_pixels, i)
            inter = tf.to_int32(true_labels & pred_labels)
            union = tf.to_int32(true_labels | pred_labels)
            legal_batches = K.sum(tf.to_int32(true_labels), axis=1)>0
            ious = K.sum(inter)/K.sum(union)
            iou.append(ious)
        iou = tf.stack(iou)
        return K.mean(iou)
  
    def sparse_loss(self, y_true, y_pred):
    
        y_pred = K.reshape(y_pred, (-1, K.int_shape(y_pred)[-1]))

        y_true = K.reshape(y_true, (-1, K.int_shape(y_pred)[-1] +1))
        unpacked = tf.unstack(y_true, axis=-1)
        y_true = tf.stack(unpacked[:-1], axis=-1) # Cutoff last label

        y_true = K.argmax(y_true,axis=-1)

        entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true,
                                                        logits=y_pred)

        return tf.reduce_mean(entropy, axis=-1)

    def softmax_crossentropy_FCN_OHOT(self, y_true, y_pred):
        # Wrappper function?
        ignore_label = K.int_shape(y_pred)[-1] + 1 # num classes + 1

        # Reshape to h*w, number_of_classes
        y_pred = K.reshape(y_pred, (-1, K.int_shape(y_pred)[-1]))
        y_true = K.reshape(y_true, (-1, K.int_shape(y_pred)[-1] +1))    

        not_ignore_mask = tf.to_float(tf.not_equal(K.argmax(y_true,axis=-1),ignore_label-1)) * 1.0

        # Cut off last label
        #TODO Use this: https://www.tensorflow.org/api_docs/python/tf/gather_nd
        unpacked = tf.unstack(y_true, axis=-1)
        y_true = tf.stack(unpacked[:-1], axis=-1)


        cross_entropy = tf.losses.compute_weighted_loss(
                weights=not_ignore_mask,
                losses = tf.nn.softmax_cross_entropy_with_logits(
                labels = y_true,
                logits = y_pred)
                )
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        return cross_entropy_mean

    def sparse_accuracy_ignoring_last_label_ohot(self, y_true, y_pred):

        nb_classes = K.int_shape(y_pred)[-1]
        y_pred = K.reshape(y_pred, (-1, K.int_shape(y_pred)[-1])) # Reshape to h*w, number_of_classes
        y_true = K.reshape(y_true, (-1, K.int_shape(y_pred)[-1] +1))    
        
        valid_labels_mask = tf.not_equal(tf.argmax(y_true, axis=-1), nb_classes)
        indices_to_keep = tf.where(valid_labels_mask)
        unpacked = tf.unstack(y_true, axis=-1)
        y_true = tf.stack(unpacked[:-1], axis=-1) # Cutoff last label

        y_true = tf.gather_nd(params = y_true, indices=indices_to_keep)
        y_pred = tf.gather_nd(params = y_pred, indices=indices_to_keep)
        
        return K.mean(K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)))

    def pixelwise_crossentropy(self, y_true, y_pred):
        '''
        Expects:
        y_true to be in the form of [h,w,num_class] one-hot encoded labels
        y_pred to be in the form [h,w,num_class]
        if you want an ignore label the class_weights can be applied to zero the label
        '''

        _EPSILON = K.epsilon()
        nb_class=2 # number of objects plus background
        y_pred = K.clip(y_pred, _EPSILON, 1.0-_EPSILON)
            
        loss=0
        for cls in range(0,nb_class):
            tf.print('Hello')
            d = y_true
            d = tf.Print(d, [d], "Inside loss function")
            d = tf.Print(d, [d], str(tf.shape(y_true)))
            d = tf.Print(d, [d], str(tf.shape(y_pred)))

            loss+=y_true[:,:,cls]*K.log(y_pred[:,:,cls])
        
                
        return -loss


    def sparse_crossentropy_ignoring_last_label_ohot_builtin(self, y_true, y_pred):
        '''
        Softmax cross entropy

        Final Activation should be 'linear' this performas softmax inside of the function

        Expects:
            y_pred = [H,W,Num_Classes] Reshpaes to [Samples, Num_Classes]
            y_true = [H,W,Num_classes] reshapes to [Samples, Num_classes +1]
                Removes last label dimension
        '''
        tensor_shape = K.int_shape(y_pred)[-1]
        pixel_shape = K.int_shape(y_pred)[1] * K.int_shape(y_pred)[2]
        y_pred = K.reshape(y_pred, (-1, pixel_shape, tensor_shape))
        
        y_true = K.reshape(y_true, (-1, pixel_shape, tensor_shape+1))
        unpacked = tf.unstack(y_true, axis=-1)
        y_true = tf.stack(unpacked[:-1], axis=-1) # Remove last axis = ignore label
        
        log_softmax = tf.nn.log_softmax(y_pred)
        #cross_entropy = -K.sum(y_true * log_softmax, axis=1)
        #cross_entropy = tf.where(tf.is_nan(cross_entropy), tf.zeros_like(cross_entropy),cross_entropy)
        #cross_entropy_mean = K.mean(cross_entropy)
        
        cross_entropy_mean = K.categorical_crossentropy(y_true, log_softmax)
        return cross_entropy_mean

    def build_model(self, val = False, val_weights = None):
        # welds and background
        num_class = 2
        if self.config.CHANNEL == 'RGB':
            num_channels = 3
        elif self.config.CHANNEL == 'THERMAL' or self.config.CHANNEL == 'GREY' or self.config.CHANNEL == 'STACKED':
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
        
        # layers of neural net 
        input_image = Input(shape=(h,
                                   w,
                                    num_channels))

        conv1_1a = Conv2D(64, 
                        kernel_size=(3,3), 
                        strides=(1,1),
                        padding='same', 
                        activation=act_func, 
                        kernel_initializer=kernel_initializer,
                        name='conv1_1a')(input_image)

        conv1_2 = Conv2D(64, 
                        kernel_size=(3,3), 
                        strides=(1,1),
                        padding='same', 
                        activation=act_func, 
                        kernel_initializer=kernel_initializer,
                        name='conv1_2')(conv1_1a)

        x = BatchNormalization()(conv1_2)
        
        pool1 = MaxPooling2D(pool_size=(2, 2), 
                            strides=(2,2), 
                            padding='same', 
                            name='pool1')(x)
        
        conv2_1 = Conv2D(128, 
                        kernel_size=(3,3), 
                        strides=(1,1),
                        padding='same', 
                        activation=act_func, 
                        kernel_initializer=kernel_initializer,
                        name='conv2_1')(pool1)

        conv2_2 = Conv2D(128, 
                        kernel_size=(3,3), 
                        strides=(1,1),
                        padding='same', 
                        activation=act_func, 
                        kernel_initializer=kernel_initializer,
                        name='conv2_2')(conv2_1)
        
        x = BatchNormalization()(conv2_2)

        pool2 = MaxPooling2D(pool_size=(2, 2), 
                            strides=(2,2), 
                            padding='same', 
                            name='pool2')(x)
        
        
        conv3_1 = Conv2D(256, 
                        kernel_size=(3,3), 
                        strides=(1,1),
                        padding='same', 
                        activation=act_func,
                        kernel_initializer=kernel_initializer, 
                        name='conv3_1')(pool2)

        conv3_2 = Conv2D(256, 
                        kernel_size=(3,3), 
                        strides=(1,1),
                        padding='same', 
                        activation=act_func, 
                        kernel_initializer=kernel_initializer,
                        name='conv3_2')(conv3_1)

        conv3_3 = Conv2D(256, 
                        kernel_size=(3,3), 
                        strides=(1,1),
                        padding='same', 
                        activation=act_func, 
                        kernel_initializer=kernel_initializer,
                        name='conv3_3')(conv3_2)

        x = BatchNormalization()(conv3_3)
        
        pool3 = MaxPooling2D(pool_size=(2, 2), 
                            strides=(2,2), 
                            padding='same', 
                            name='pool3')(x)
    
        conv4_1 = Conv2D(512, 
                        kernel_size=(3,3), 
                        strides=(1,1),
                        padding='same', 
                        activation=act_func, 
                        kernel_initializer=kernel_initializer,
                        name='conv4_1')(pool3)

        conv4_2 = Conv2D(512, 
                        kernel_size=(3,3), 
                        strides=(1,1),
                        padding='same', 
                        activation=act_func, 
                        kernel_initializer=kernel_initializer,
                        name='conv4_2')(conv4_1)

        conv4_3 = Conv2D(512, 
                        kernel_size=(3,3), 
                        strides=(1,1),
                        padding='same', 
                        activation=act_func,
                        kernel_initializer=kernel_initializer, 
                        name='conv4_3')(conv4_2)
        
        x = BatchNormalization()(conv4_3)
        
        pool4 = MaxPooling2D(pool_size=(2, 2), 
                            strides=(2,2), 
                            padding='same', 
                            name='pool4')(x)

        
        conv5_1 = Conv2D(512, 
                        kernel_size=(3,3), 
                        strides=(1,1),
                        padding='same', 
                        activation=act_func,
                        kernel_initializer=kernel_initializer,
                        name='conv5_1')(pool4)

        conv5_2 = Conv2D(512, 
                        kernel_size=(3,3), 
                        strides=(1,1),
                        padding='same', 
                        activation=act_func, 
                        kernel_initializer=kernel_initializer,
                        name='conv5_2')(conv5_1)

        conv5_3 = Conv2D(512, 
                        kernel_size=(3,3), 
                        strides=(1,1),
                        padding='same', 
                        activation=act_func, 
                        kernel_initializer=kernel_initializer,
                        name='conv5_3')(conv5_2)

        x = BatchNormalization()(conv5_3)
        
        pool5 = MaxPooling2D(pool_size=(2, 2), 
                        strides=(2,2), 
                        padding='same', 
                        name='pool5')(x)

        fc6 = Conv2D(4096, 
                    kernel_size=(7,7), 
                    strides=(1,1), 
                    padding='same', 
                    activation=act_func,
                    kernel_initializer=kernel_initializer, 
                    name='fc6')(pool5)

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
                            name='score_pool4n')(pool4)

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
                            name='score_pool3n')(pool3)
        try:
            fuse_pool3 = Add()([upscore_pool4, score_pool3n])
        except:
            fuse_pool3 = Add()([upscore_pool4, score_pool3n])

        
        upscore8n = Conv2DTranspose(num_class, kernel_size=(16,16), strides=(8, 8), 
                padding='same', name='upscore8n', activation='linear')(fuse_pool3)

        upscore_out = Activation('softmax')(upscore8n)

        model = Model(input_image, upscore8n)
        model_prob = Model(input_image, upscore_out)



        if not val:
            # use sgd for optimizer (could try adam too)
            sgd = optimizers.SGD(lr=self.config.LEARNING_RATE['start'], momentum=0.9)
            model.compile(
                            optimizer=sgd,
                            loss = self.sparse_loss,
                            metrics=[self.sparse_accuracy_ignoring_last_label_ohot, self.IoU]
                        )

        self.model = model
        self.model_prob = model_prob

        if val:
            #print("[INFO] Loading {0}".format(val_weights))
            try:
                model.load_weights(val_weights)
                print("[INFO] Loded Model Weights {0}".format(val_weights))
            except Exception as e:
                print("[INFO] Model Weight File Not Found")
                print(e)
                sys.exit()

        #print(model.summary())
