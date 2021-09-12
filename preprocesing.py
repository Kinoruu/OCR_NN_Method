import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import pickle
import tensorflow as tf

from keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.ops.gen_batch_ops import batch

data_path = "database/"
train_folder = "database/train"
validation_folder = 'database/validation'
test_folder = 'database/test'

cv2.waitKey(0)
cv2.destroyAllWindows()

batch_size = 1
image_height = 48
image_width = 48
color_channels = 3
target_size = (image_height, image_width)

categories = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
              'v', 'w', 'x', 'y', 'z']

train_data = []


def create_training_data():
    for category in categories:
        path = os.path.join(train_folder, category)
        class_number = categories.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(
                    path, img), cv2.IMREAD_GRAYSCALE)
                #                cv2.imshow(" ",img_array)
                #                cv2.waitKey(0)
                img_array = cv2.resize(img_array, (image_width, image_height))
                train_data.append([img_array, class_number])
            except Exception as e:
                pass


create_training_data()

if len(train_data) == 0:
    print("empty")
print("elements in train folder: ", len(train_data))

print("elements in train subfolders: ")
for i in os.listdir(data_path + "/train"):
    print(i, len(os.listdir(os.path.join(data_path + "/train", i))))

random.shuffle(train_data)
for sample in train_data[:9]:
    print(sample[1])

validation_data = []


def create_validation_data():
    for category in categories:
        path = os.path.join(validation_folder, category)
        class_number = categories.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(
                    path, img), cv2.IMREAD_GRAYSCALE)
                img_array = cv2.resize(img_array, (image_width, image_height))
                validation_data.append([img_array, class_number])
            except Exception as e:
                pass


create_validation_data()
if len(validation_data) == 0:
    print("empty")
print("elements in validation folder: ", len(validation_data))
print("elements in validation subfolders: ")
for i in os.listdir(data_path + "/validation"):
    print(i, len(os.listdir(os.path.join(data_path + "/validation", i))))

test_data = []


def create_test_data():
    for category in categories:
        path = os.path.join(test_folder, category)
        class_number = categories.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(
                    path, img), cv2.IMREAD_GRAYSCALE)
                img_array = cv2.resize(img_array, (image_width, image_height))
                test_data.append([img_array, class_number])
            except Exception as e:
                pass


create_test_data()
if len(test_data) == 0:
    print("empty")
print("elements in test folder: ", len(test_data))
print("elements in test subfolders: ")
for i in os.listdir(data_path + "/test"):
    print(i, len(os.listdir(os.path.join(data_path + "/test", i))))

X_train = []
y_train = []

for features, labels in train_data:
    X_train.append(features)
    y_train.append(labels)

print(len(X_train))

X_train = np.array(X_train).reshape(-1, 48, 48, 1)
# X_train= np.array([X_train], order='C')
# X_train.resize((48,48))
# X_train.shape
# images_nhwc = tf.compat.v1.placeholder(tf.float32, [None, 3, 48, 48])
print(np.array(y_train).shape)
y_train = np.array(y_train).reshape(-1, 1, 1)
app = lambda y: [np.float(i) for i in y]
y_train = np.apply_along_axis(app, 0, y_train)

X_val = []
y_val = []

for features, labels in train_data:
    X_val.append(features)
    y_val.append(labels)

X_val = np.array(X_val).reshape(-1, 48, 48)
# y_val = np.array(y_val).reshape(-1, 1,1)
# y_val = np.apply_along_axis(app, 0, y_val)

print(len(X_val))

X_test = []
y_test = []

for features, labels in test_data:
    X_test.append(features)
    y_test.append(labels)

X_test = np.array(X_test).reshape(-1, 48, 48)

# y_test = np.array(y_test).reshape(-1, 1,1)


def plotImages(X_train, y_train):
    plt.figure(figsize=[8, 8])
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(X_train[i], cmap="gray")
        plt.title(categories[y_train[i]])

    plt.show()


# plotImages(X_train, y_train)

pickle_out = open("X_train.pickle", "wb")  # write-binary
pickle.dump(X_train, pickle_out)
pickle_out.close()

pickle_out = open("y_train.pickle", "wb")  # write-binary
pickle.dump(y_train, pickle_out)
pickle_out.close()

pickle_out = open("X_val.pickle", "wb")  # write-binary
pickle.dump(X_val, pickle_out)
pickle_out.close()

pickle_out = open("y_val.pickle", "wb")  # write-binary
pickle.dump(y_val, pickle_out)
pickle_out.close()

pickle_out = open("X_test.pickle", "wb")  # write-binary
pickle.dump(X_test, pickle_out)
pickle_out.close()

pickle_out = open("y_test.pickle", "wb")  # write-binary
pickle.dump(y_test, pickle_out)
pickle_out.close()
