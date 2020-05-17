import numpy as np
import pandas as pd
import random
import os
import cv2
import tensorflow as tf
import keras
import h5py
from os import listdir
from keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense, Dropout
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.layers.advanced_activations import LeakyReLU
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

train_data = 'C:\\Users\\Abc\\Desktop\\plant_disease\\color\\'
train_folders = os.listdir(train_data)
print(train_folders)

image_names = []
'''test_image_names = []'''
train_images = []
'''test_images = []'''
train_labels = []
'''test_labels = []'''
size = 64,64

for folder in train_folders:
    for file in os.listdir(os.path.join(train_data,folder)):
        if file.endswith("JPG"):
            image_names.append(os.path.join(train_data,folder,file))
            train_labels.append(folder)
            img = cv2.imread(os.path.join(train_data,folder,file))
            im = cv2.resize(img,size)
            train_images.append(im)
        else:
            continue
            
train = np.array(train_images)
print(train.shape)
'''
for folder in test_folders:
    for file in os.listdir(os.path.join(test_data,folder)):
        if file.endswith("JPG"):
            test_image_names.append(os.path.join(test_data,folder,file))
            test_labels.append(folder)
            img = cv2.imread(os.path.join(test_data,folder,file))
            im = cv2.resize(img,size)
            test_images.append(im)
        else:
            continue
            
test = np.array(test_images)
print(test.shape)

'''


train = train.astype('float32')/255
label_dummies = pd.get_dummies(train_labels)
labels = label_dummies.values.argmax(1)
            
union_list = list(zip(train, labels))
random.shuffle(union_list)
train,labels = zip(*union_list)

train = np.array(train)
labels = np.array(labels)
'''
test = test.astype('float32')/255
label_dummies_test = pd.get_dummies(test_labels)
test_labels = label_dummies_test.values.argmax(1)
            
union_list_2 = list(zip(test,test_labels))
random.shuffle(union_list_2)
test,test_labels = zip(*union_list)

test = np.array(test)
test_labels = np.array(test_labels)



plt.subplot(121)
plt.imshow(train[0, :, :], cmap = 'gray')
plt.title("Ground Truth : {}".format(labels[0]))


test_labels_one_hot = to_categorical(test_labels)
'''
labels_one_hot = to_categorical(labels)
'''Splitting the data into validation and train set'''

train, valid_X, labels, valid_label = train_test_split(train, labels_one_hot, test_size = 0.2, random_state = 13)
print(train.shape, valid_X.shape, labels.shape, valid_label.shape)


batch_size = 64
epochs = 10
num_classes = 38

'''Neural Network Architecture'''

disease_model = Sequential()
disease_model.add(Conv2D(32, kernel_size = (3,3), activation = 'linear', input_shape = (64, 64, 3), padding = 'same'))
disease_model.add(LeakyReLU(alpha = 0.1))
disease_model.add(MaxPooling2D((2,2), padding = 'same'))
disease_model.add(Dropout(0.1))
disease_model.add(Conv2D(64, kernel_size = (3,3), activation = 'linear', padding = 'same'))
disease_model.add(LeakyReLU(alpha = 0.1))
disease_model.add(MaxPooling2D((2,2), padding = 'same'))
disease_model.add(Dropout((0.1)))
disease_model.add(Conv2D(128, kernel_size = (3,3), activation = 'linear', padding = 'same'))
disease_model.add(LeakyReLU(alpha = 0.1))
disease_model.add(MaxPooling2D((2,2), padding = 'same'))
disease_model.add(Dropout(0.3))
disease_model.add(Flatten())
disease_model.add(Dense(128, activation = 'linear'))
disease_model.add(LeakyReLU(alpha = 0.1))
disease_model.add(Dropout(0.2))
disease_model.add(Dense(num_classes, activation = 'softmax'))
    
'''Compiling the Model'''

disease_model.compile(loss = keras.losses.categorical_crossentropy, optimizer = keras.optimizers.Adam(), metrics = ['accuracy'])
disease_model.summary() 

'''Training the Model'''
disease_train = disease_model.fit(train, labels, batch_size = batch_size, epochs = epochs, verbose = 1, validation_data = (valid_X, valid_label))

'''Plots''' 
accuracy = disease_train.history['acc']
val_accuracy = disease_train.history['val_acc']
loss = disease_train.history['loss']
val_loss = disease_train.history
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training Accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.figure()

'''
plt.plot(epochs, loss, 'bo', label = 'Training Loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
'''

disease_model.save("disease_model_2.h5py")
'''


print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])
'''