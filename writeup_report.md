# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/model.png "Model Visualization"
[image2]: ./examples/track1_center.jpg "Center lane driving"
[image3]: ./examples/track1_recovering_1.jpg "Recovery Image"
[image4]: ./examples/track1_recovering_2.jpg "Recovery Image"
[image5]: ./examples/track1_recovering_3.jpg "Recovery Image"
[image6]: ./examples/normal.png "Normal Image"
[image7]: ./examples/shear.png "Sheared Image"
[image8]: ./examples/flip.png "Flipped Image"
[image9]: ./examples/bright.png "Brightness Adjusted Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* track1.mp4 video of track #1 driving
* track2.mp4 video of track #2 (optional) driving
* 'examples' directory with writeup document images

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 4 and 256 (model.py lines 163-182). 3x3 filters are followed by 1x1 convolution layers up to depth 1 - steering angle value. Here 1x1 convolution is an equivalent of fully connected layer (Dense) which could be used instead. 

The model includes ReLU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 173).


#### 2. Attempts to reduce overfitting in the model

The model has ~166K trainable parameters and does not require dropout layers to reduce overfitting. Overfitting is addressed by smaller number of parameters and augmentation of dataset providing unique samples for each training batch. 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 238-243). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track near center.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 253).
 
The batch size is set to 32. Greater batch size up to 128 and above provides smoother steering near center of lane on the first track. But second track has many steep curves and batch size above 32 leads to underperforming steering.
  
As dataset is randomly augmented, greater number of epochs provides better steering model. Even if loss is not decreasing new samples still improves the model by reducing overfitting. Thus the number of epochs is set to 30 to reduce total training time to about 90 minutes (173 seconds per epoch).

#### 4. Appropriate training data

For the first track I used training data provided by Udacity. The data contains a combination of center lane driving, recovering from the left and right sides of the road as well as driving the track in opposite direction. For the second track I recorded three laps of center lane driving in one direction. Accompanied by augmentation the data was sufficient to train successful steering policy. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I took the NVIDIA's model described here https://arxiv.org/abs/1604.07316 as a baseline. The model delivered steering policy to complete track1 and 70% of track2. However, it had to be adjusted by adding dropout layers to address overfitting as it has 2,116,983 trainable parameters.
  
Then I've experimented with the architecture by changing convolution layers kernel sizes, depth, stride, pooling, number of layers and input image size. I found that straight experiments with model training and verification in simulator take too much time as loss value is insufficient to judge the model performance. It is uncertain what loss value is small enough for capable steering policy.

So I changed the model to output steering direction instead of steering angle. I set three categories of direction: left (if steering angle is less than zero), right (if steering angle is above zero) and straight (if steering angle is zero). Then I trained the models with categorical_crossentropy() objective on direction categories. This got me a simple model evaluation criterion based on direction accuracy (not loss value). The direction accuracy is a quality measure of features outputted by the model.  

Using this approach I have selected the model architecture that delivers greatest direction accuracy on training and validation datasets with minimal number of parameters. Then I increased convolution depth at deeper layers assuming steering angle needs more features than steering direction.
 
The architecture experiments unveiled interesting findings. The first finding is that max or average pooling decrease the performance. Indeed, steering requires knowledge of exact position which is degraded by pooling layers. So I removed pooling layers by increasing stride size.
  
The another finding is that more stacked convolutions with 3x3 kernel size delivers greater accuracy than lesser convolution layers with bigger size of kernel, such as 5x5 and above.
   
In addition, resizing initial image to 64x64 has no advantage and even reduces accuracy.

Finally, I noticed that batch normalization decreases the performance. Usually batch normalization improves convergence and reduces training time, but for this task it produces underperforming model. According to my experiments, normalization does not allow train policy with steep turns. Presumably batch normalization is useful on smoother functions which is not the case for steering policy. 

Cropping layer helps to remove irrelevant parts of image, such as vehicle body at the bottom and landscape at top. Removing these parts makes training faster and increase generalization.

Using my findings I came up with VGG-like architecture without pooling layers. Based validation datasets I selected deeper layers depth so overfitting (validation loss is greater than training one) does not occur even without dropout layers and additional regularization. 
  
I was aiming at a model capable to drive both tracks with the same weights. As the second track is more complicated I first verified trained model on second track in the simulator. The typical issue at driving in simulator is underperforming turns. I was able to resolve it by increasing deeper layers depth and decreasing batch size.
 
At the end of the process, the vehicle is able to drive autonomously around the track near the center and without leaving the road.

#### 2. Final Model Architecture


The final model architecture (model.py lines 159-209) consisted of a convolution neural network with the following layers:

Layer|Output Shape|Param #
---|---|---
Cropping2D: ((70, 25), (0, 0))|(None, 65, 320, 3)|0
Lambda: normalization - zero mean and unit variation|(None, 65, 320, 3)|0            
Convolution2D: kernel=3x3, stride=2x2, depth=4|(None, 32, 159, 4)|112
Activation: ReLU |(None, 32, 159, 4)|0
Convolution2D: kernel=3x3, stride=2x2, depth=8|(None, 15, 79, 8)|296
Activation: ReLU|(None, 15, 79, 8)|0
Convolution2D kernel=3x3, stride=2x2, depth=16|(None, 7, 39, 16)|1168
Activation: ReLU|(None, 7, 39, 16)|0
Convolution2D: kernel=3x3, stride=2x2, depth=32|(None, 3, 19, 32)|4640
activation_4 (Activation)|(None, 3, 19, 32)|0
Convolution2D: kernel=3x3, stride=1x2, depth=64|(None, 1, 9, 64)|18496
Activation: ReLU|(None, 1, 9, 64)|0
Convolution2D: kernel=1x3, stride=1x2, depth=128|(None, 1, 4, 128)|24704
Activation: ReLU|(None, 1, 4, 128)|0
Convolution2D: kernel=1x3, stride=1x2, depth=256|(None, 1, 1, 256)|98560
Activation: ReLU|(None, 1, 1, 256)|0
Convolution2D: kernel=1x1, stride=1x1, depth=64|(None, 1, 1, 64)|16448
Activation: ReLU|(None, 1, 1, 64)|0
Convolution2D: kernel=1x1, stride=1x1, depth=32|(None, 1, 1, 32)|2080
Activation: ReLU|(None, 1, 1, 32)|0
Convolution2D: kernel=1x1, stride=1x1, depth=1|(None, 1, 1, 1)|33
Flatten|(None, 1)|0
**Total trainable parameters: 166,537**


Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

For track #1 I used Udacity dataset which contains center lane driving as well as vehicle recovering from the left side and right sides of the road back to center driving. Here is an example image of center lane driving:

![alt text][image2]

These images show what a recovery from right to center looks like:

![alt text][image3]
![alt text][image4]
![alt text][image5]

The dataset also contains driving lap in different directions which helps to improve generalization.

In addition I generated three laps center driving data for track#2.

The key component of dataset is images from left and right camera. Using these images significantly improves steering policy performance and generalization. These steering angle for those images is corrected assuming recovery. So for left camera the correction is to turn slightly right and for left camera the correction is to turn slightly left. The value of correction is adjustable parameter that encodes how quickly recovery should be performed. This normally depends on steepness of the road curves. So I used 0.12 value for track#1 and 0.2 for track#2 images.  

To augment the data set, I used:
* image flipping
* image shearing
* adjusting brightness

Flipping assumes a flipped image with changed steering angle sign. This is equivalent of driving the track in opposite direction.
 
Shearing assumes horizontal shifting of image with corresponding adjusting of steering angle. This is similar to using of images from left and right camera.  

Brightness adjustment helps to generalize model features as simulator's light source produces different brightness of the same objects.  

For example, here is an image that has then been sheared, flipped and brightness adjusted:

![alt text][image6]
![alt text][image7]
![alt text][image8]
![alt text][image9]


Udacity track#1 data plus my three laps track#2 data multiplied by three (number of cameras) produced 56,484 (image, steering angle) tuples. These data were randomly augmented at runtime (see data_generator(), lines 149-151).
 
Data pre-processing is integrated into Keras model by adding cropping and normalization layers. Cropping removes irrelevant parts of image at the top and bottom, and normalization sets color intensities values with zero mean mean unit variance range. 

I finally randomly shuffled the data set and put 0.1 of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The optimization parameters, batch size and number of epochs are set as described in the "3. Model parameter tuning" section of this document.