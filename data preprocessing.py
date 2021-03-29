import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

DATADIR = "F:\Programming Projects\CNN cats and dogs classification\PetImages"
CATEGORIES = ["Dog", "Cat"]


IMG_SIZE = 96
train_data = []
def create_train_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path): 
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE) # convert the images into arrays
                new_array = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))
                train_data.append([new_array, class_num])
            except Exception as e:
                pass

create_train_data()        

print(len(train_data))

import random

random.shuffle(train_data)
X = []
y= []

for features,label in train_data:
    X.append(features)
    y.append(label)
    
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)  

import pickle

pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()

pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)

