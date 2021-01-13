import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Conv2D, Dropout
from keras.callbacks import ModelCheckpoint
import sklearn
from sklearn.model_selection import train_test_split
from math import ceil
from scipy import ndimage


CORRECTION = 0.15  # Steering angle to add or subtract to left and right images
IMAGES_PER_CSV_LINE = 108  # Center, left and right images + data augmentation
BATCH_SIZE = IMAGES_PER_CSV_LINE * 4
DATA_DIRECTORY = './simulator-data/IMG/'
IMAGE_WIDTH = 320
IMAGE_HEIGHT = 160
CROP_TOP = 70
CROP_BOTTOM = 25


height, width = IMAGE_HEIGHT, IMAGE_WIDTH
h_half = int(height / 2)
w_half = int(width / 2)

# Areas (rectangles) that are used to perform data augmentation by removing a section of the images
BOTTOM_LEFT = np.array([[0, height], [w_half, height], [w_half, h_half], [0, h_half]])
BOTTOM_RIGHT = np.array([[w_half, h_half], [w_half, height], [width, height], [width, h_half]])
LEFT_HALF = np.array([[0, 0], [0, height], [w_half, height], [w_half, 0]])
BOTTOM_HALF = np.array([[0, h_half], [0, height], [width, height], [width, h_half]])
RIGHT_HALF = np.array([[w_half, 0], [w_half, height], [width, height], [width, 0]])


def remove_section(image, polygon_points):
    result = np.copy(image)
    cv2.fillConvexPoly(result, polygon_points, 0)
    return result


def get_current_path(path):
    '''
    Since the model might be trained in a different computer/path
    than it was recorded, we need this function to correct the path.
    '''
    filename = path.split('/')[-1]
    return DATA_DIRECTORY + filename


def read_img_rgb(img_path):
    '''
    Given an image path, get the updated path, read the image
    and convert BGR to RGB (drive.py reads images in RGB)
    '''
    current_img_path = get_current_path(img_path)
    img = cv2.imread(current_img_path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def get_images_with_measurements(line):
    '''
    Given a tuple with the values of a csv line, load the center image with its steering angle,
    left and right images with their corrected measurement and then perform data augmentation
    by doing rotation and removing sections of the image
    '''
    center_img = read_img_rgb(line[0])
    left_img = read_img_rgb(line[1])
    right_img = read_img_rgb(line[2])
    measurement = float(line[3])

    # 6 images
    initial_data = [
        (center_img, measurement),
        (left_img, measurement + CORRECTION),
        (right_img, measurement - CORRECTION),
        (np.fliplr(center_img), -measurement),
        (np.fliplr(left_img), -measurement - CORRECTION),
        (np.fliplr(right_img), -measurement + CORRECTION),
    ]

    # 6 x 3 = 18 images
    data_with_rotation = []
    for data in initial_data:
        data_with_rotation.append(data)
        data_with_rotation.append((ndimage.rotate(data[0], 10, reshape=False), data[1]))
        data_with_rotation.append((ndimage.rotate(data[0], -10, reshape=False), data[1]))

    # 18 * 6 = 108 images
    result = []
    for data in data_with_rotation:
        result.append(data)
        result.append((remove_section(data[0], BOTTOM_LEFT), data[1]))
        result.append((remove_section(data[0], BOTTOM_RIGHT), data[1]))
        result.append((remove_section(data[0], LEFT_HALF), data[1]))
        result.append((remove_section(data[0], BOTTOM_HALF), data[1]))
        result.append((remove_section(data[0], RIGHT_HALF), data[1]))

    return result


def generator(samples, batch_size=32):
    '''
    Given a list of what it was read from the csv file + the batch size, it creates a generator
    that provides a shuffled batch of images with their measurements. It includes augmented data.
    '''
    num_samples = len(samples)
    batch_size_pre_augmentated = ceil(batch_size / IMAGES_PER_CSV_LINE)
    # Loop forever so the generator never terminates
    while True:
        np.random.shuffle(samples)
        for offset in range(0, num_samples, batch_size_pre_augmentated):
            batch_samples = samples[offset:offset + batch_size_pre_augmentated]

            images = []
            measurements = []

            for batch_sample in batch_samples:
                for img, measurement in get_images_with_measurements(batch_sample):
                    images.append(img)
                    measurements.append(measurement)

            X_train = np.array(images)
            y_train = np.array(measurements)

            yield sklearn.utils.shuffle(X_train, y_train)


samples = []
with open('./simulator-data/driving_log.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

# Actual model
model = Sequential()
model.add(Cropping2D(cropping=((CROP_TOP, CROP_BOTTOM), (0, 0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: x / 255.0 - 0.5))
model.add(Conv2D(filters=24, kernel_size=5, strides=(2, 2), activation='relu'))
model.add(Dropout(0.1))
model.add(Conv2D(filters=36, kernel_size=5, strides=(2, 2), activation='relu'))
model.add(Dropout(0.1))
model.add(Conv2D(filters=48, kernel_size=5, strides=(2, 2), activation='relu'))
model.add(Dropout(0.1))
model.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
model.add(Dropout(0.1))
model.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(50))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Dropout(0.2))
model.add(Dense(1))

# Compile and train the model using the generator function
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
train_generator = generator(train_samples, BATCH_SIZE)
# TODO: Stop performing data augmentation for validation_generator
validation_generator = generator(validation_samples, BATCH_SIZE)

# Checkpoint to save after an epoch got a better val_loss
checkpoint = ModelCheckpoint(
    "model.h5",
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    mode='min',
    period=1,
    save_weights_only=False
)

model.compile(loss='mse', optimizer='adam')
model.fit_generator(
    train_generator,
    steps_per_epoch=ceil(len(train_samples) * IMAGES_PER_CSV_LINE / BATCH_SIZE),
    validation_data=validation_generator,
    validation_steps=ceil(len(validation_samples) * IMAGES_PER_CSV_LINE / BATCH_SIZE),
    epochs=10,
    verbose=1,
    callbacks=[checkpoint]
)
