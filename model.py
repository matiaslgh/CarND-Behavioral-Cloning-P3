import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
import sklearn
from sklearn.model_selection import train_test_split
from math import ceil

BATCH_SIZE = 30
CORRECTION = 0.17
IMAGES_PER_CSV_LINE = 6 # Left, Center, Right + their flipped versions
DATA_DIRECTORY = './simulator-data/IMG/'

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

    return [
        (center_img, measurement),
        (left_img, measurement + CORRECTION),
        (right_img, measurement - CORRECTION),
        (np.fliplr(center_img), -measurement),
        (np.fliplr(left_img), -measurement - CORRECTION),
        (np.fliplr(right_img), -measurement + CORRECTION),
    ]

def generator(samples, batch_size=32):
    num_samples = len(samples)
    batch_size_pre_augmentated = ceil(batch_size / IMAGES_PER_CSV_LINE)
    # Loop forever so the generator never terminates
    while True:
        np.random.shuffle(samples)
        for offset in range(0, num_samples, batch_size_pre_augmentated):
            batch_samples = samples[offset:offset + batch_size_pre_augmentated]

            images =  []
            measurements = []

            for batch_sample in batch_samples:
                for img, measurement in get_images_with_measurements(line):
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
model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: x / 255.0 - 0.5))
model.add(Flatten())
model.add(Dense(1))

# compile and train the model using the generator function
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
train_generator = generator(train_samples, BATCH_SIZE)
validation_generator = generator(validation_samples, BATCH_SIZE)

model.compile(loss='mse', optimizer='adam')
model.fit_generator(
    train_generator,
    steps_per_epoch= ceil(len(train_samples) * IMAGES_PER_CSV_LINE / BATCH_SIZE),
    validation_data=validation_generator,
    validation_steps=ceil(len(validation_samples) * IMAGES_PER_CSV_LINE / BATCH_SIZE),
    epochs=1,
    verbose=1
)
model.save('model.h5')