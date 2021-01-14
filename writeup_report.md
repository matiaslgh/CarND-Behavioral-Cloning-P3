# **Behavioral Cloning Project**

The goals / steps of this project are the following:

- Use the simulator to collect data of good driving behavior
- Build, a convolution neural network in Keras that predicts steering angles from images
- Train and validate the model with a training and validation set
- Test that the model successfully drives around track one without leaving the road
- Summarize the results with a written report

[//]: # "Image References"
[image1]: ./images/nvidia-architecture.png "NVIDIA's architecture"

## Rubric Points

### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

---

### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

- model.py containing the script to create and train the model
- drive.py for driving the car in autonomous mode
- model.h5 containing a trained convolution neural network
- writeup_report.md summarizing the results
- best-run.mp4 to see how the model perform in an entire lap

#### 2. Submission includes functional code

Using the Udacity provided simulator and my drive.py (I only updated the speed to 30) file, the car can be driven autonomously around the track by executing

```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I used an architecture based on the one used by NVIDIA's team ([link](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf))

Theirs looks like this:

![alt text][image1]

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 32 and 64

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

Also, there are some small changes when performing data augmentation that also help to prevent overfitting

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used the three images provided for every line of the csv file. This means I used left, center and right images applying a correction to the steering angle of +/- 0.15 to to the left/right images.

I got training data by driving two complete laps in one direction and two more in the oposite direction. I did that with both tracks.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to create the simplest neural network to ensure I didn't have any errors so far.

Then, created a convolution neural network model similar to the one mentioned above, owned by NVIDIA's team. I thought this model might be appropriate because theirs has provided great results and it was a great starting point to create mine.

First results were pretty bad, but I wasn't using the images of the left and right cameras of the car, so I added logic to start using them with a correction of 0.2 degrees in the steering angle. I kept tweeking this parameter until I got the lowest mean squared error I could with the data I had at that moment. CORRECTION = 0.15 was my final value.

Results were better but still bad, I needed more data. It's here when I decided to drive 2 laps in one direction and 2 laps in the opposite one for both tracks. This improved a lot my results, but still wasn't enough, since I had much better results in the loss for the training than the validation loss. I was overfitting and that's why I started to add dropout layers and spent some time tweaking that until I stopped overfitting.

Also, at that point I had enough data to start using a generator and set and tweak a bit the batch size.

My last step was to perform data augmentation. For doing this, I added a flipped version of the 3 images (left, right and center) and I multiplied their steering angles by -1. For now I had 6 images in my list. I performed a small rotation in all those images by adding/subtracting them 10 degrees and then I had the 6 original images plus their rotated versions. This means I had 18 images now. Then, to all of those images I added an extra augmentation: removing different sections in the image such as the corners or the half of the image.. This caused I have 108 images per csv line -> 3 _ 2 _ 3 \* 6

That last step caused that the training time wasn't really efficient because of the amount of data, but it really helped with removing overfitting and getting a good result represented by a low mean squared error and the car driving perfectly in the simulator.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes:

| Layer           | Description                                          |
| --------------- | ---------------------------------------------------- |
| Input           | 160x320x3 RGB image                                  |
| Crop            | Crop top 70px and bottom 25px                        |
| Lambda          | Normalization (x / 255.0 - 0.5)                      |
| Convolution     | ksize=5x5 / stride=2x2 / depth=24 / output=78x158x24 |
| RELU            |                                                      |
| Dropout         | rate=0.1                                             |
| Convolution     | ksize=5x5 / stride=2x2 / depth=36 / output=37x77x36  |
| RELU            |                                                      |
| Dropout         | rate=0.1                                             |
| Convolution     | ksize=5x5 / stride=2x2 / depth=48 / output=17x37x48  |
| RELU            |                                                      |
| Dropout         | rate=0.1                                             |
| Convolution     | ksize=3x3 / stride=1x1 / depth=64 / output=15x35x64  |
| RELU            |                                                      |
| Dropout         | rate=0.1                                             |
| Convolution     | ksize=3x3 / stride=1x1 / depth=64 / output=13x33x64  |
| RELU            |                                                      |
| Dropout         | rate=0.1                                             |
| Flatten         | outputs 27456                                        |
| Fully connected | outputs 100                                          |
| Dropout         | rate=0.2                                             |
| Fully connected | outputs 50                                           |
| Dropout         | rate=0.2                                             |
| Fully connected | outputs 10                                           |
| Dropout         | rate=0.2                                             |
| Fully connected | outputs 1                                            |
