import random
import os
import cv2
import matplotlib.pyplot as plt
#import tensorflow as tf
import pickle
import numpy as np
import pandas as pd

from keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import plot_confusion_matrix

pickle_in = open("y_train.pickle", "rb")
y_train = pickle.load(pickle_in)
pickle_in = open("X_train.pickle", "rb")
X_train = pickle.load(pickle_in)

pickle_in = open("y_val.pickle", "rb")
y_val = pickle.load(pickle_in)
pickle_in = open("X_val.pickle", "rb")
X_val = pickle.load(pickle_in)

pickle_in = open("y_test.pickle", "rb")
y_test = pickle.load(pickle_in)
pickle_in = open("X_test.pickle", "rb")
X_test = pickle.load(pickle_in)

batch_size = 32
categories = ["angry", "disgust", "fear",
              "happy", "neutral", "sad", "surprise"]
model = tf.keras.models.load_model('CNN6.model')


def prepare(filepath):
    pic_size = 48
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (pic_size, pic_size))
    plt.imshow(new_array, cmap='gray')
    plt.show()
    return new_array.reshape(-1, pic_size, pic_size, 1)


# losowanie obrazu do testu
test_path = (
    'C:/Users/Bogunia/Desktop/praca_magisterska/program/database/images/test/')
test_folders = os.listdir(test_path)
print(test_folders)
selected_folder = random.choice(test_folders)
print(selected_folder)
path = ('C:/Users/Bogunia/Desktop/praca_magisterska/program/database/images/test/'+selected_folder)
print(path)

files = os.listdir(path)
print(len(files))

random_filename = random.choice([x for x in os.listdir(path)
                                 if os.path.isfile(os.path.join(path, x))])
print(random_filename)

img = (path+"/"+random_filename)
image = cv2.imread(img, 0)
cv2.imshow("Obraz1", image)

cv2.waitKey(0)
cv2.destroyAllWindows()
train_results = model.evaluate(
    X_train, y_train, batch_size=batch_size)
print(train_results)
prediction = model.predict_classes([prepare(img)])

for i in prediction:
    if i == [0]:
        prediction = 'angry'
        # print('angry')
    elif i == [1]:
        prediction = 'disgust'
        # print('disgust')
    elif i == [2]:
        prediction = 'fear'
        # print('fear')
    elif i == [3]:
        prediction = 'happy'
        # print('happy')
    elif i == [4]:
        prediction = 'neutral'
        # print('neutral')
    elif i == [5]:
        prediction = 'sad'
        # print('sad')
    else:
        prediction = 'surprise'
        # print('surprise')

print("expected emotion: ", selected_folder)
print("predicted emotion: ", prediction)

y_predict = model.predict_classes(X_test)
cm = confusion_matrix(y_test, y_predict)
print(cm)
