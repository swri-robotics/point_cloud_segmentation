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
        * TEST_TRAINING_DATA will display training images, labels, and statistics on the training data
        * VALIDATE is used for loading in weights and making predicitions
    Save_Model - whether or not to save the training weights 
    Weight_Id - name of model weights
    Weight_Save_Path - directory to save weight files to
    Model - model (fcn8) used for training 
    Channel - image channels to use
        *  GREY is single channel greyscale
        *  RBG is three channel color
        *  STACKED is a single channel with the Laplacian of the image added to the greyscale 
        *  COMBINED is a two channel image with the first being greyscale and the second the Laplacian
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
        *  class - weld must have a label.xml associated with the directory, background will not
    Validation_Dirs - same structure as above, these directories will be used for generating validation metrics
        Note: Validation directories with no labels can be run by making the class background
    