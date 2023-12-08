import tensorflow as tf 
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

(x_train,y_train),(x_test,y_test) = keras.datasets.cifar10.load_data()


# print(y_train[0:4])
# print(y_train.shape)
y_train=y_train.reshape(-1,)
# print(y_train[0:4])
# print(y_train.shape)
def show_image(x,index):
    plt.imshow(x[index])
    plt.show()
print(x_train[0]/255)
x_train=x_train/255
x_test=x_test/255
# print(x_train[0])
# ann=keras.Sequential([
#     keras.layers.Flatten(input_shape=(32,32,3)),
#     keras.layers.Dense(1000,activation="relu"),
#     keras.layers.Dense(500,activation="relu"),
#     keras.layers.Dense(10,activation="sigmoid"),
# ])
# ann.compile(optimizer="SGD",
#             loss="sparse_categorical_crossentropy",
#             metrics=['accuracy'])
# ann.fit(x_train,y_train,epochs=5)

cnn=keras.Sequential([

    #cnn
    keras.layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu',input_shape=(32,32,3)),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu',input_shape=(32,32,3)),
    keras.layers.MaxPooling2D((2,2)),
    # dense 
    keras.layers.Flatten(),
    keras.layers.Dense(64,activation="relu"),
    keras.layers.Dense(10,activation="softmax"),
])
cnn.compile(optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=['accuracy'])
cnn.fit(x_train,y_train,epochs=
