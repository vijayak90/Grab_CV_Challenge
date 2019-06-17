
import tarfile
import scipy.io
import numpy as np
import os
import cv2 as cv
import shutil
import random
from progressbar import ProgressBar


# Module to save the pre-processed training dataset
def save_trainset(g_fname, g_label, g_bbox):
    orig_train_folder = 'cars_train'
    samples = len(g_fname) #Gets the number of samples by counting the length of the filename list

    train_split = 0.8 # Splitting based on Pareto principle, where 80% sample used for training and 20% used for validation
    num_train = int(round(samples * train_split))
    train_indexes = random.sample(range(samples), num_train)
    
    #Creating a progressbar to track the progress of saving the dataset
    pb = ProgressBar()
    
    #Looping through for all the samples
    for i in pb(range(samples)):
        fname = g_fname[i]
        label = g_label[i]
        (x1, y1, x2, y2) = g_bbox[i]

        #sets the path as cars_train/filename of each image to create folder that contains similar images
        orig_path = os.path.join(orig_train_folder, fname)
        orig_image = cv.imread(orig_path) #Read the image
        height, width = orig_image.shape[:2] #And its features

        margin = 16 # Setting the margin as 16 so as to set the bounding box values to resize
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(x2 + margin, width)
        y2 = min(y2 + margin, height)
        
        if i in train_indexes:
            temp_folder = 'data/train'
        else:
            temp_folder = 'data/valid'
        #sets path as data/train or data/valid based on the above condition and stores the resized image in the data folders after finding the similar images corresponding to each image
        temp_path = os.path.join(temp_folder, label)
        if not os.path.exists(temp_path):
            os.makedirs(temp_path)
        temp_path = os.path.join(temp_path, fname)

        resize_image = orig_image[y1:y2, x1:x2]
        temp_img = cv.resize(src=resize_image, dsize=(ih, iw))
        cv.imwrite(temp_path, temp_img)

# Module to save the preprocessed test dataset
# Logic is pretty much the same as the training dataset
def save_testset(g_fname, g_bbox):
    orig_folder = 'cars_test'
    temp_folder = 'data/test'
    samples = len(g_fname)
    pb = ProgressBar()
    for i in pb(range(samples)):
        fname = g_fname[i]
        (x1, y1, x2, y2) = g_bbox[i]
        orig_path = os.path.join(orig_folder, fname)
        
        orig_image = cv.imread(orig_path)
        height, width = orig_image.shape[:2]
        
        margin = 16
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(x2 + margin, width)
        y2 = min(y2 + margin, height)
        
        temp_path = os.path.join(temp_folder, fname)
        resize_image = orig_image[y1:y2, x1:x2]
        temp_img = cv.resize(src=resize_image, dsize=(ih, iw))
        cv.imwrite(temp_path, temp_img)

# Module to preprocess the training data set
def preprocess_trainset():
    print("Preprocessing training dataset")
    # Get the metadata about the dataset annotations and store, transpose it to get the actual features required
    g_cannos = scipy.io.loadmat('devkit/cars_train_annos')
    g_annotations = g_cannos['annotations']
    g_annotations = np.transpose(g_annotations)

    g_fname = [] # List to store the filenames associated with each image
    g_class_id = [] # List to store class ids of the class that the image belongs to
    g_bbox = [] # List to store bounding box values of each image
    g_label = [] # List to store the label associated with each image

    #Looping in the annotations to split the features and store in the corresponding lists as defined above    
    for g_annotation in g_annotations:
        bbox_x1 = g_annotation[0][0][0][0]
        bbox_y1 = g_annotation[0][1][0][0]
        bbox_x2 = g_annotation[0][2][0][0]
        bbox_y2 = g_annotation[0][3][0][0]
        class_id = g_annotation[0][4][0][0]
        g_label.append('%04d' % (class_id,))
        fname = g_annotation[0][5][0]
        g_bbox.append((bbox_x1, bbox_y1, bbox_x2, bbox_y2))
        g_class_id.append(class_id)
        g_fname.append(fname)

    g_lcount = np.unique(g_class_id).shape[0] # Getting the unique class ids to count the number of classifications available
    
    print('There are %d number of car classifications available in this dataset!' % g_lcount)
    
    #Saving the preprocessed training dataset
    save_trainset(g_fname, g_label, g_bbox)

# Module to preprocess the test data set
def preprocess_testset():
    print("Preprocessing testing dataset...")
    #Getting the filename and image information from the metadata extracted folder using scipy
    g_cannos = scipy.io.loadmat('devkit/cars_test_annos')
    g_annotations = g_cannos['annotations']
    g_annotations = np.transpose(g_annotations) # Transposing it to get the actual features
    
    g_fname = [] #List to store the filename associated with each of the image
    g_bbox = [] #List to store the bounding box values for each image - x1, y1, x2, y2 values

    #Loop through to fetch the values into the corresponding lists
    for g_annotation in g_annotations:
        bbox_x1 = g_annotation[0][0][0][0]
        bbox_y1 = g_annotation[0][1][0][0]
        bbox_x2 = g_annotation[0][2][0][0]
        bbox_y2 = g_annotation[0][3][0][0]
        fname = g_annotation[0][4][0]
        g_bbox.append((bbox_x1, bbox_y1, bbox_x2, bbox_y2))
        g_fname.append(fname)
       
    # Saving the preprocessed test dataset
    save_testset(g_fname, g_bbox)

# Module to check the folders for training, validation and test datasets    
def check_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

#As an entry point, the below module covers the process of extracting the datasets and calling the preprocess modules

if __name__ == '__main__':
    # parameters - width and height of the images
    iw, ih = 224, 224
    
    #Extracting the training dataset into cars_train folder    
    print('Extracting cars_train.tgz...')
    if not os.path.exists('cars_train'):
        with tarfile.open('cars_train.tgz', "r:gz") as tar:
            tar.extractall()
    print('Training dataset extracted!')
    
    #Extracting the test dataset into the cars_test folder
    print('Extracting cars_test.tgz...')
    if not os.path.exists('cars_test'):
        with tarfile.open('cars_test.tgz', "r:gz") as tar:
            tar.extractall()
    print('Testing dataset extracted!')
    
    #Extracting the devkit folder that has metadata about the images into the devkit folder
    print('Extracting car_devkit.tgz...')
    if not os.path.exists('devkit'):
        with tarfile.open('car_devkit.tgz', "r:gz") as tar:
            tar.extractall()
    print('Metadata extracted!')
    
    #Extracting the metadata from mat file using the scipy lib and storing in the python variables
    g_cmeta = scipy.io.loadmat('devkit\cars_meta')    
    class_names = g_cmeta['class_names']  
    #Getting the class_names values from the mat file, to store the car classifications and transposing the class_names variable to get the class_names shape
    class_names = np.transpose(class_names)
    print('class_names.shape: ' + str(class_names.shape))
    print('Random class_name: [{}]'.format(class_names[195][0][0]))
    
    #Checks if the folder exists, if not create the folders in order to use it for training and testing
    check_folder('data/train')
    check_folder('data/valid')
    check_folder('data/test')

    preprocess_trainset()
    preprocess_testset()

    # clean up the folders for better memory usage and performance
    # shutil.rmtree('cars_train')
    shutil.rmtree('cars_test') # cleaning just the test dataset folder, as I just started with training
