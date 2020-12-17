import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda

DATA_DIRECTORY = './simulator-data/IMG/'

def get_current_path(path):
    filename = path.split('/')[-1]
    return DATA_DIRECTORY + filename
    
images =  []
measurements = []

with open('./simulator-data/driving_log.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        img_path = get_current_path(line[0])
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        images.append(img)

        flipped_img = np.fliplr(img)
        images.append(flipped_img)
        
        measurement = float(line[3])
        measurements.append(measurement)

        measurement_for_flipped_img = -measurement
        measurements.append(measurement_for_flipped_img)
        
        
X_train = np.array(images)
y_train = np.array(measurements)

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Flatten())
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=2)
model.save('model.h5')