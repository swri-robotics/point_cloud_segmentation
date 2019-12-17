#!/usr/bin/env python3
'''
 * @file fcn8_validate.py
 * @brief Used for viewing the predicitions of trained nets on the validation set
 *
 * @author Jake Janssen
 * @date November 8, 2019
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
from pcs_detection.process import validate, demo_video

class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)

if __name__ == '__main__':
    # Import Config json file and convert into format we need
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(dir_path + '<path to saved off json>') as json_data_file:
        data = json.load(json_data_file)
    config = Config(**data)
 
    if 'VAL_WEIGHT_PATH' in config.__dict__.keys():
        if config.MODE == 'VALIDATE':
            validate(config)
        elif config.MODE == 'VIDEO':
            demo_video(config)
    else:
        print('This config does not have an associated weight file')
