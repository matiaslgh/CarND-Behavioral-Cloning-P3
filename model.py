import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense

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
        img = cv2.imread(img_path)
        images.append(img)
        
        measurement = float(line[3])
        measurements.append(measurement)
        
        
X_train = np.array(images)
y_train = np.array(measurements)

model = Sequential()
model.add(Flatten(input_shape=(160, 320, 3)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=4)
model.save('model.h5')