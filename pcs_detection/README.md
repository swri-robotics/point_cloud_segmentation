# Running Instructions 
### Training 
    Training is done by running the fcn8_train.py.
    The training_config.json (see below) file allows for control of the training.
    The model will be trained on the images within the TRAINING_DIRS specified in the config. The classes to be trained on must also be entered into the config under CLASS_NAMES. These names must correspond to the class names used within the label file. 
    Pre-processing done before training is also selected within this config. To select the type of preprocessing, first select the image channels that should be used. The image channels do not have to be the original input to the algoritm (e.g. rgb images can be converted to lab or ycr_cb), though image inputs will not be guarenteed to work with all image channels (e.g. greyscale images can not be converted to the lab color space). 
    After selecting the image channel, the values under PRE_PROCESS may need to be adjusted. These values correspond to the mean of each image channel after preporcessing for the entire dataset. They can be verified by running the fcn8_train.py script in the debug mode (see config explanation below) and looking at the histogram of the image channel values. The goal is to have the mean for the whole dataset be zero. 
    Other preprocessing techniques such as only taking a section with a high intensity of labels may also be configured in this config. 

    Once training begins, the best weights from the training session will be saved off under the script/data/weights directory along with two config files.

### Verification
    Once training is complete, the example_weights.py script will aid in verifying the model. This script will make predictions on the validation sets specified in the training config. To run this script, first change the path inside the script to that of the full_config.json saved off with your weights. This config will load the weights that generated the highest accuracy on the validation set by default (this can be changed by going into the full_config file). Then run the script and press the spacebar to continue onto the next image. Pressing 'q' will cause the script to stop. 

### Inference
    Inference is done in a similar way to verification, but instead of using full_config.json, inference_config.json should be used. Additionally, images will have to be supplied to this script instead of running off the verification directories. 

### Interpreting Images
    The images generated for both the predicitions and labels will follow the same color scheme:
    * Blue represents the background class
    * Red represents regions of the image that are ignored or not used in the loss calculation 
    * Other colors will consistently be a class specified in the config

# Data Requirements 
    Training subsets must contain a folder holding images and a training_labels.xml file. 
    Images will be resized to the ORIG_DIMS specified in the config (aspect ratio may not be preserved). 

# Config Descriptions

### training_config.json:
    This config file is paired with fcn8_train.py. It allows for the selection of a model, types of image and label preprocessing, the image directories used for both training and validation, and various training parameteres. Once the model is done training, a config file used for inference and a config file used for logging the training parameters will be created.

### <model_weights_path>/full_config.json 
    This config is a copy of training_config.json used to train a specific set of weights. It is also used for viewing predicitions of the model in example_weights.py.

### <model_weights_path>/inference_config.json 
    A pruned version of full_config.json used for making inferences in fcn8_inference.py.

### Config Keys 
    Mode - which mode execute when running
        * TRAIN will for training a new model
        * DEBUG will display training images, labels, and statistics on the training data
        * VALIDATE is used for loading in weights and making predicitions
    Save_Model - whether or not to save the training weights
    Class_Names - the names of the classes that are present in the label files
                  Note: background does not have to be put as a class  
    Weight_Id - name of model weights
    Weight_Save_Path - directory to save weight files to
    Model - model (fcn8) used for training 
    Channel - image channels to use
        *  GREY is single channel greyscale
        *  STACKED is a single channel with the Laplacian of the image added to the greyscale 
        *  COMBINED is a two channel image with the first being greyscale and the second the Laplacian
        *  RBG is three channel color
        * LAB will convert a RGB image to the LAB color space
        * YCR_CB will convert a RGB image to the YCR_CB color space
    Augmentations - standard Keras augmentations: https://keras.io/preprocessing/image/
    Learning_Rate - scheduler for the learning rate 
    Batch_Size - how many images to process with a single prediction 
    N_Epochs - maximum number of epochs before training ends
    Background_Reduction - the ratio of (label pixels / background pixels) per batch used in training labels
    Min_Pixels_In_Image - minimum number of labeled pixels per training image in the weld class
    Display_Scale_Factor - amount to scale images viewed in the pipeline
    Use_Full_Image - toggles resizing to IMG_DIMS
    IMG_DIMS - select a region of the image with specified size that has the majority of the label
    Label_Thickness - how thick to draw the weld lines in the label
    Pre_Process - values used for mean subtraction pre-processing (mean should be zero after subtraction)
    Confidence_Threshold - used to increase the threshold needed predicting a weld pixel (post processing)
    Val_Weight_Path - path to the weight file used for predictions 
    Training_Dirs - Directories of images used for training 
        *  dir_path - directory path
        *  num_imgs - number of images to use from that directory ('all' will use all images)
        *  labels - boolean value for if labels are paired with this directory
                    Note: The only directories that should not have labels are the ones that do not contain any instances of the classes.
    Validation_Dirs - same structure as above, these directories will be used for generating validation metrics
        Note: Validation directories with no labels can be run by making the class background