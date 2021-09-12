import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
import os

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from keras.layers import Dense, Input, Dropout, Flatten, Conv2D, BatchNormalization, Activation, MaxPooling2D
from tensorflow.python.ops.gen_batch_ops import batch

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

pickle_yt = open("y_train.pickle", "rb")
y_train = pickle.load(pickle_yt)
pickle_xt = open("X_train.pickle", "rb")
X_train = pickle.load(pickle_xt)

pickle_yv = open("y_val.pickle", "rb")
y_val = pickle.load(pickle_yv)
pickle_xv = open("X_val.pickle", "rb")
X_val = pickle.load(pickle_xv)

pickle_yt = open("y_test.pickle", "rb")
y_test = pickle.load(pickle_yt)
pickle_xt = open("X_test.pickle", "rb")
X_test = pickle.load(pickle_xt)

image_height = 48
image_width = 48
color_channels = 1
target_size = (image_height, image_width)

batch_size = 32

# input_shape = (X_train.shape[0:])
input_shape = (48, 48, 1)
model = Sequential()

# 1 warstwa, warstwa wejściowa
model.add(Conv2D(64, (3, 3), padding='same',
          input_shape=input_shape))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.8))

# 2 warstwa, warstwa splotowa
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.8))

# 3 warstwa, warstwa splotowa
model.add(Conv2D(256, (3, 3), padding='same'))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.8))

# 4 warstwa, warstwa splotowa
model.add(Conv2D(512, (3, 3), padding='same'))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.8))

model.build(input_shape)
model.summary()

model.add(Flatten())
model.add(Dense(3000))
model.add(Activation("relu"))
image_output = model.add(Dense(26, activation="softmax"))  # 26 means number of categories

model.build(input_shape)
model.summary()

# training
epochs = 10
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

X_train = np.asarray(X_train)
y_train = np.asarray(y_train)
X_val = np.asarray(X_val)
y_val = np.asarray(y_val)

data_generator = ImageDataGenerator()
# try1 = data_generator.flow(X_val, y_val)

# print(len(X_train))
# print(len(X_val))
early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20)

X_train = np.array(X_train).reshape(-1,48, 48, 1)
X_val = np.array(X_val).reshape(-1,48, 48, 1)
# xtrain = tf.expand_dims(X_train, axis=-1)

# rint(np.array(X_train).shape)
# print(np.array(y_train).shape)
# print(np.array(X_val).shape)
# print(np.array(y_val).shape)

y_train = y_train.reshape(-1,)
y_val = y_val.reshape(-1,)

#training_process = model.fit(data_generator.flow(X_train, y_train, batch_size), steps_per_epoch=len(X_train)/batch_size,
                             #epochs=epochs, validation_data=(X_val, y_val))
training_process = model.fit(X_train, y_train, epochs=25, validation_data=(X_val, y_val))

# zapistwanie modelu
model.save('CNNvX.model')

# wyniki uczenia
train_results = model.evaluate(
    X_train, y_train, batch_size=batch_size)
print(train_results)

plt.figure(figsize=(10, 5))
plt.suptitle('CNNvX training ', fontsize=20)

plt.subplot(1, 2, 1)
plt.xlabel('Epochs', fontsize=8)
plt.ylabel('Training Loss', fontsize=8)
plt.plot(training_process.history['loss'], label='Training Loss')
plt.plot(training_process.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')

plt.subplot(1, 2, 2)
plt.ylabel('Training Accuracy', fontsize=8)
plt.xlabel('Epochs', fontsize=8)
plt.plot(training_process.history['accuracy'], label='Training Accuracy')
plt.plot(training_process.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.show()
