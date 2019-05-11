# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image0]: ./report_files/sample_image.png "Test Data Sample"
[image1]: ./report_files/dataset_visualize.png "Visualization"
[image2]: ./report_files/grayscale.png "Grayscaling"
[image3]: ./report_files/image_augment.png "Augmented Image"
[image4]: ./samples/sample_1.jpeg "Traffic Sign 1"
[image5]: ./samples/sample_2.jpeg "Traffic Sign 2"
[image6]: ./samples/sample_3.jpeg "Traffic Sign 3"
[image7]: ./samples/sample_4.jpeg "Traffic Sign 4"
[image8]: ./samples/sample_5.jpeg "Traffic Sign 5"
[image9]: ./samples/top5_sample1.png "Traffic Sign 1"
[image10]: ./samples/top5_sample2.png "Traffic Sign 2"
[image11]: ./samples/top5_sample3.png "Traffic Sign 3"
[image12]: ./samples/top5_sample4.png "Traffic Sign 4"
[image13]: ./samples/top5_sample5.png "Traffic Sign 5"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/roopakingole/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Data Set Summary
I used the numpy library to calculate summary statistics of the traffic
signs data set:

* Number of training examples = 34799
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 43

#### 2. Exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

Along with this, below is how the sample image and its classification is shown.

![alt text][image0]

### Preprocess Data

As a first step, I decided to convert the images to grayscale because for Traffic signs color is not distinguishing feature and no 2 signals of same class will be of different color.

I decided to enhance the exposure little bit since some of the images are dark.
Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data between [0-1] since these values works well with CNN.

I tried to generate additional data because current dataset is largly imbalanced. 

To add more data to the the data set, I used the following tried to augment the image through random rotation, flipping the images and changing perspective randomly. Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 

### Design and Test a Model Architecture

#### 1. Model Architecture

My final model consisted of the following layers based on LeNet architecture:

| Layer         		|     Description	        					| Output  |
|:---------------------:|:---------------------------------------------:|:--------|
| Input         		| 32x32x1 grayscale image => 32x32x1    		| 32x32x1 | 
| Convolution 5x5x1x6  	| 1x1 stride, VALID padding, outputs 28x28x6 	| 28x28x6 |
| RELU					|												|         |
| Max pooling	      	| 2x2x1 stride,  outputs 14x14x6 				| 14x14x6 |
| Convolution 5x5x6x16  | 1x1 stride, VALID padding, outputs 10x10x16   | 10x10x16|
| RELU					|												|         |
| Max pooling	      	| 2x2x1 stride,  outputs 5x5x16 				| 5x5x16  |
| Fully connected		| 400x120     									| 120     |
| RELU					|												|         |
| Fully connected		| 120x84     									| 84      |
| RELU					|												|         |
| Fully connected		| 84x43     									| 43      |
| Softmax				|            									|         |
 

#### 2. Model Training

To train the model, I used Stochastic Gradient Descent optimization by AdamOptimizer with the learning rate of 0.001, batch size of 128 images and in total 14 Epochs. I trained my model on CPU only.


#### 3. Model Accuracy 
Within 15 Epochs, I was able to achieve the validation set accuracy of 94.4%. I used evaluate() to calculate the validation accuracy and test accuracy. My approach to get to final solution is to iterate the training model in batches.


For this problem I chose to keep my architecture as LeNet-5. The original LeNet-5 was designed for character recognition which is very similar to pattern recognition on traffic signs. I thought the exact model is good starting point. When I ran this model with 15 Epochs, model converged to 94.4% accuracy. Hence I decided to stick to this model. 

My final model results were:
* training set accuracy of 0.944
* validation set accuracy of 0.944 
* test set accuracy of 0.925

### Test a Model on New Images

#### 1. Random German Traffic Signs
Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Road work (25)   		|   Road work (25)
| Speed limit (50km/h) (2)   |   Speed limit (50km/h) (2)
| Stop (14)   			|    Roundabout mandatory (40)
| Right-of-way at the next intersection (11)   |    Double curve (21)
| Yield (13)   			|    Yield (13)


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. 

#### 3. Top 5 predictions
![alt text][image9] ![alt text][image10] ![alt text][image11] 
![alt text][image12] ![alt text][image13]



