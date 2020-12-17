import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D

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

    CORRECTION = 0.17

    return [
        (center_img, measurement),
        (left_img, measurement + CORRECTION),
        (right_img, measurement - CORRECTION),
        (np.fliplr(center_img), -measurement),
        (np.fliplr(left_img), -measurement - CORRECTION),
        (np.fliplr(right_img), -measurement + CORRECTION),
    ]

images =  []
measurements = []

with open('./simulator-data/driving_log.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        for img, measurement in get_images_with_measurements(line):
            images.append(img)
            measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)

model = Sequential()
model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: x / 255.0 - 0.5))
model.add(Flatten())
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=1)
model.save('model.h5')