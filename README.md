# Grab_CV_Challenge
This repository holds the files and the descriptions of the Grab's Computer vision challenge of recognizing the car make, model and colour. I am a beginner in this field and this repository contains the feature extraction and the training of the dataset. I am still in the process of fine-tuning the training and start with the validation and testing.

Extract the ImageNet pre-trained ResNet-152 model into the models folder(https://drive.google.com/file/d/0Byy2AcGyEVxfeXExMzNNOHpEODg/view)

Run the Feature Extraction using the 'feature_extract.py' file

Start the training by running the 'training.py' file
Note - The accuracy is very low, as I just started the training and need some more time to fine-tune the model. When I increase the epochs(1000), with a minimum batchsize (4), the training takes hours to complete, so need to test with higher epochs for a better accuracy.
The validation and testing parts are in progress.
