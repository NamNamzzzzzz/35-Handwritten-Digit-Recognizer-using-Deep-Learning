from google.colab import drive
drive.mount('/content/drive')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score
tf.__version__
from tensorflow.keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print("Shape of X_train : ", X_train.shape)
print("Shape of X_test : ", X_test.shape)
print("Shape of y_train : ", y_train.shape)
print("Shape of y_test : ", y_test.shape)
from google.colab import files
uploaded = files.upload()
plt.figure(figsize=(20,20))

plt.subplot(1,10,1)
plt.imshow(X_train[0])
plt.subplot(1,10,2)
plt.imshow(X_train[1])
plt.subplot(1,10,3)
plt.imshow(X_train[2])
plt.subplot(1,10,4)
plt.imshow(X_train[3])
plt.subplot(1,10,5)
plt.imshow(X_train[4])
plt.subplot(1,10,6)
plt.imshow(X_train[5])
plt.subplot(1,10,7)
plt.imshow(X_train[6])
plt.subplot(1,10,8)
plt.imshow(X_train[7])
plt.subplot(1,10,9)
plt.imshow(X_train[8])
plt.subplot(1,10,10)
plt.imshow(X_train[9])
plt.figure(figsize=(20,20))

plt.subplot(1,10,1)
plt.imshow(X_train[150])
plt.subplot(1,10,2)
plt.imshow(X_train[162])
plt.subplot(1,10,3)
plt.imshow(X_train[178])
plt.subplot(1,10,4)
plt.imshow(X_train[193])
plt.subplot(1,10,5)
plt.imshow(X_train[205])
plt.subplot(1,10,6)
plt.imshow(X_train[3978])
plt.subplot(1,10,7)
plt.imshow(X_train[456])
plt.subplot(1,10,8)
plt.imshow(X_train[7896])
plt.subplot(1,10,9)
plt.imshow(X_train[57])
plt.subplot(1,10,10)
plt.imshow(X_train[31897])
print("Shape of X_train : ", X_train.shape)
print("Shape of X_test : ", X_test.shape)
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

input_shape = (28,28,1)

y_train[0:11]
y_cat_train = to_categorical(y_train, 10)
y_cat_test = to_categorical(y_test, 10)
y_cat_train[0:11]

X_train[0]
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
X_train[0]

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
model.summary()
early_stop = EarlyStopping(monitor='val_loss', patience=2)


model.fit(X_train, y_cat_train, epochs=50, callbacks = [early_stop], validation_data=(X_test, y_cat_test))

print("The model has successfully trained")
model.save('mnist.h5')
print("Saving the model as mnist.h5")

training_metrics = pd.DataFrame(model.history.history)
training_metrics.columns
training_metrics.head()
training_metrics[['loss','val_loss']].plot()
training_metrics[['accuracy','val_accuracy']].plot()
score = model.evaluate(X_test, y_cat_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
predictions = np.argmax(model.predict(X_test), axis=-1)
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
plt.figure(figsize=(15,7))
sns.heatmap(confusion_matrix(y_test, predictions),annot=True)
new_img = X_test[95]
plt.imshow(new_img)
y_test[95]
np.argmax(model.predict(new_img.reshape(1,28,28,1)), axis=-1)
new_img2 = X_test[0]
plt.imshow(new_img2)
y_test[0]
np.argmax(model.predict(new_img2.reshape(1,28,28,1)), axis=-1)
new_img3 = X_test[397]
plt.imshow(new_img3)
y_test[397]
np.argmax(model.predict(new_img3.reshape(1,28,28,1)), axis=-1)
