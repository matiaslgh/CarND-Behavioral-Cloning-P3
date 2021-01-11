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


CORRECTION = 0.15
IMAGES_PER_CSV_LINE = 270
BATCH_SIZE = IMAGES_PER_CSV_LINE * 4
DATA_DIRECTORY = './simulator-data/IMG/'
IMAGE_WIDTH = 320
IMAGE_HEIGHT = 160
CROP_TOP = 70
CROP_BOTTOM = 25


height, width = IMAGE_HEIGHT, IMAGE_WIDTH
h_half = int(height / 2)
w_half = int(width / 2)

TOP_RIGHT = np.array([[w_half, 0], [w_half, h_half], [width, h_half], [width, 0]])
TOP_LEFT = np.array([[0, 0], [0, h_half], [w_half, h_half], [w_half, 0]])
BOTTOM_LEFT = np.array([[0, height], [w_half, height], [w_half, h_half], [0, h_half]])
BOTTOM_RIGHT = np.array([[w_half, h_half], [w_half, height], [width, height], [width, h_half]])

LEFT_HALF = np.array([[0, 0], [0, height], [w_half, height], [w_half, 0]])
BOTTOM_HALF = np.array([[0, h_half], [0, height], [width, height], [width, h_half]])
RIGHT_HALF = np.array([[w_half, 0], [w_half, height], [width, height], [width, 0]])
TOP_HALF = np.array([[0, 0], [0, h_half], [width, h_half], [width, 0]])

TOP_RIGHT_TRIANGLE = np.array([[w_half, 0], [width, h_half], [width, 0]])
TOP_LEFT_TRIANGLE = np.array([[0, 0], [0, h_half], [w_half, 0]])
BOTTOM_LEFT_TRIANGLE = np.array([[0, height], [w_half, height], [0, h_half]])
BOTTOM_RIGHT_TRIANGLE = np.array([[w_half, height], [width, height], [width, h_half]])


def remove_section(image, polygon_points):
    result = np.copy(image)
    cv2.fillConvexPoly(result, polygon_points, 0)
    return result


def remove_four_corners_triangle(image):
    result = remove_section(image, TOP_RIGHT_TRIANGLE)
    result = remove_section(result, TOP_LEFT_TRIANGLE)
    result = remove_section(result, BOTTOM_LEFT_TRIANGLE)
    result = remove_section(result, BOTTOM_RIGHT_TRIANGLE)
    return result


def set_padding_black(image, thickness_percentage=16):
    padding = int(height * thickness_percentage / 100)

    LEFT = np.array([[0, 0], [0, height], [padding, height], [padding, 0]])
    BOTTOM = np.array([[0, height - padding], [0, height], [width, height], [width, height - padding]])
    RIGHT = np.array([[width - padding, 0], [width - padding, height], [width, height], [width, 0]])
    TOP = np.array([[0, 0], [0, padding], [width, padding], [width, 0]])

    result = remove_section(image, LEFT)
    result = remove_section(result, BOTTOM)
    result = remove_section(result, RIGHT)
    result = remove_section(result, TOP)

    return result


def get_current_path(path):
    filename = path.split('/')[-1]
    return DATA_DIRECTORY + filename


def read_img_rgb(img_path):
    current_img_path = get_current_path(img_path)
    img = cv2.imread(current_img_path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def get_images_with_measurements(line):
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

    # 18 * 15 = 270 images
    result = []
    for data in data_with_rotation:
        result.append(data)

        result.append((remove_section(data[0], TOP_RIGHT), data[1]))
        result.append((remove_section(data[0], TOP_LEFT), data[1]))
        result.append((remove_section(data[0], BOTTOM_LEFT), data[1]))
        result.append((remove_section(data[0], BOTTOM_RIGHT), data[1]))

        result.append((remove_section(data[0], LEFT_HALF), data[1]))
        result.append((remove_section(data[0], BOTTOM_HALF), data[1]))
        result.append((remove_section(data[0], RIGHT_HALF), data[1]))
        result.append((remove_section(data[0], TOP_HALF), data[1]))
        result.append((set_padding_black(data[0]), data[1]))

        result.append((remove_section(data[0], BOTTOM_RIGHT_TRIANGLE), data[1]))
        result.append((remove_section(data[0], TOP_RIGHT_TRIANGLE), data[1]))
        result.append((remove_section(data[0], TOP_LEFT_TRIANGLE), data[1]))
        result.append((remove_section(data[0], BOTTOM_LEFT_TRIANGLE), data[1]))
        result.append((remove_four_corners_triangle(data[0]), data[1]))

    return result


def generator(samples, batch_size=32):
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

# compile and train the model using the generator function
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
train_generator = generator(train_samples, BATCH_SIZE)
validation_generator = generator(validation_samples, BATCH_SIZE)

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
