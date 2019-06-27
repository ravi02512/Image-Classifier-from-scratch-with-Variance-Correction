# Image-Classifier-from-scratch-with-Variance-Correction
 
In this repository we built image classifier from scratch(using only numpy module) which is able to classify cat and dog images and apply L2 regularization and early stopping to correct the variance of the model.

## About the datasest:
'train_catvnoncat.h5' and 'test_catvnoncat.h5' contains matrix in which each column represents the image array of a image.In train dataset we have 209 columns which mean we have 209 images in train dataset and in test dataset we have 50 images.


## About the Python files
  
  ### Image.Py
      Image.py contains the functions needed for creating image classifier Following is the list of fuctions and the working of the          functions:
      1.Initialize Parameters
        Here we are intializing the weight and bias vector for each layers.We can initialize weight and bias with zeros, random and       accoring to 'He' distribution.
       
       2.L_model_forward:
        After initializing the weight and bias vector L_forward function take these parameter and computer the output.
        
       3. Compute cost:
          This function computes the cost (which is 1-accuracy)
          
       4. L_Model_Backward:
          This Fuction updates the weight and bias vector in order to reduce the cost
   
   ### Basic_neural_network.py
       This python script contains the basic structure of neural network with backward propagation(gradient descent)
       
   ### L2_Regularization.py
      In this script we performed the variance reduction technique
       
  ## Variance Reduction
    ### action performed in order to reduce the variance
      1.Early stopping 
      2.L2 Regularization
  
  After tuning the learning rate and running the basic_neural_network.py we get approx 98 percentage train accuracy and 74 percentage test accuracy which show high variance in this model to reduce the variance from the model we used early stopping and tune the lambda parameter and got the train accuracy 80 percentage and 76 percentage test accuaracy  which is promising result for dataset with 250 images.
  
  
  
  ## Results
  
  After reducing the variance from model we got classifier which can classify between the dog and cat picture with 76 percent accuracy.
