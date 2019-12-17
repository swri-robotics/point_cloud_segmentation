#!/usr/bin/env python3
'''
 * @file fcn8_train.py
 * @brief Used for training neural nets, and view the images passed into the network after preprocessing
 *
 * @author Jake Janssen
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

import json
import os        
from pcs_detection.process import test_dataloader, train, class_analysis

class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)

if __name__ == '__main__':
    # Import Config json file and convert into format we need
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(dir_path + '/training_config.json') as json_data_file:
        data = json.load(json_data_file)
    config = Config(**data)

    # run the training process specified in the config
    if config.MODE == 'DEBUG':
        test_dataloader(config)
    elif config.MODE == 'TRAIN':
        train(config)
    elif config.MODE == 'STATS':
        class_analysis(config)
    else:
        print('Not a valid mode')
