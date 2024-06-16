# based on code from https://www.tensorflow.org/tutorials

import tensorflow as tf
import numpy as np
import time
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
#from tensorflow.keras import regularizers


start_time = time.time()

# set the random seeds to make sure your results are reproducible
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(1)

# specify path to training data and testing data

folderbig = "big"
foldersmall = "small"

train_x_location = foldersmall + "/" + "x_train.csv"
train_y_location = foldersmall + "/" + "y_train.csv"
test_x_location = folderbig + "/" + "x_test.csv"
test_y_location = folderbig + "/" + "y_test.csv"

print("Reading training data")
x_train_2d = np.loadtxt(train_x_location, dtype="uint8", delimiter=",")
x_train_3d = x_train_2d.reshape(-1,28,28,1)
x_train = x_train_3d
y_train = np.loadtxt(train_y_location, dtype="uint8", delimiter=",")

print("Pre processing x of training data")
x_train = x_train / 255.0


# define the training model
model = tf.keras.models.Sequential([
    tf.keras.layers.MaxPool2D((4, 4), input_shape=(28,28,1)),
    tf.keras.layers.Conv2D(31, (3,3), padding='same', activation=tf.nn.relu),
    tf.keras.layers.MaxPool2D((2,2), padding = 'same'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(950, activation=tf.nn.relu), 
    tf.keras.layers.Dense(10, activation=tf.nn.softmax) 

])
opt = optimizers.Adam(lr=0.002, beta_1=0.9, beta_2=0.9, amsgrad=False) 

model.compile(optimizer=opt,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("train")
model.fit(x_train, y_train, epochs=10, batch_size = 32) 


print("Reading testing data")
x_test_2d = np.loadtxt(test_x_location, dtype="uint8", delimiter=",")
x_test_3d = x_test_2d.reshape(-1,28,28,1)
x_test = x_test_3d
y_test = np.loadtxt(test_y_location, dtype="uint8", delimiter=",")

print("Pre processing testing data")
x_test = x_test / 255.0


print("evaluate")
model.evaluate(x_test, y_test)


time_taken = time.time() - start_time
time_taken = time_taken / 60
print("Time Take = ", time_taken, "mins")