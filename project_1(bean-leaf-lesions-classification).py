import tensorflow as tf

import os
os.environ["KAGGLE_CONFIG_DIR"]="/content"

!chmod 600 /content/kaggle.json

!kaggle datasets download -d marquis03/bean-leaf-lesions-classification

!unzip \*.zip

import matplotlib.pyplot as plt

os.listdir("/content/train")

import cv2

list=["angular_leaf_spot","bean_rust","healthy"]
image=[]
y_train=[]
for x in list:
  images=os.listdir("/content/train/"+x)
  print(len(images))
  path="/content/train/"+x


  for i in images[:100]:
    if x=="angular_leaf_spot":
      y_train.append(0)
    if x=="bean_rust":
      y_train.append(1)
    if x=="healthy":
      y_train.append(2)
    image.append(cv2.imread(path+"/"+i))
print(image[0].dtype)

import numpy as np
image=np.array(image)

list1=[np.array(x) for x in image]
img_data_float32=[]
for i in list1:
  img_data_float32.append(tf.convert_to_tensor(i.astype(np.float32)))

images=np.array(img_data_float32)/225
y_train1=np.array(y_train)

resized_images = tf.image.resize(images, (224, 224))
x_train_resized = resized_images.numpy()

images,y_train
x_train1 = np.array(x_train_resized)
y_train=tf.convert_to_tensor(y_train)

x_train = np.array([image for image in x_train1])

x_train_tensor = tf.convert_to_tensor(x_train, dtype=tf.float32)

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
y_train_encoded = to_categorical(y_train1, num_classes=3)

# from tensorflow.keras.utils import to_categorical

# Assuming you have two separate arrays: x_train (images) and y_train (class labels)

# Convert class labels to one-hot encoding
# y_train_encoded = to_categorical(y_train)

# Define the number of classes based on your dataset
# num_classes = len(set(y_train))

# Create the model
model_1 = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=10, kernel_size=3, activation="relu", input_shape=(224, 224, 3)),
    tf.keras.layers.Conv2D(filters=10, kernel_size=3, activation="relu"),
    tf.keras.layers.MaxPool2D(pool_size=2, padding="valid"),
    tf.keras.layers.Conv2D(filters=10, kernel_size=3, activation="relu"),
    tf.keras.layers.Conv2D(filters=10, kernel_size=3, activation="relu"),
    tf.keras.layers.MaxPool2D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(3, activation="softmax")  # Use 'softmax' for multi-class classification
])

# Compile the model
model_1.compile(
    loss="categorical_crossentropy",  # Use 'categorical_crossentropy' for multi-class classification
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    metrics=["accuracy"]
)

# Fit the model
history_1 = model_1.fit(
    x_train_tensor,  # Input images
    y_train_encoded ,  # One-hot encoded class labels
    epochs=10,
    steps_per_epoch=len(x_train_tensor),
    # validation_steps=len(valid_data)
)

# Display model summary
model_1.summary()
"""Epoch 1/10
300/300 [==============================] - 2s 4ms/step - loss: 1.0327 - accuracy: 0.4567
Epoch 2/10
300/300 [==============================] - 1s 4ms/step - loss: 0.8223 - accuracy: 0.6600
Epoch 3/10
300/300 [==============================] - 1s 4ms/step - loss: 0.6894 - accuracy: 0.7233
Epoch 4/10
300/300 [==============================] - 2s 6ms/step - loss: 0.5576 - accuracy: 0.7767
Epoch 5/10
300/300 [==============================] - 2s 6ms/step - loss: 0.5037 - accuracy: 0.7767
Epoch 6/10
300/300 [==============================] - 1s 4ms/step - loss: 0.4143 - accuracy: 0.8533
Epoch 7/10
300/300 [==============================] - 1s 4ms/step - loss: 0.3302 - accuracy: 0.8967
Epoch 8/10
300/300 [==============================] - 1s 4ms/step - loss: 0.2507 - accuracy: 0.9333
Epoch 9/10
300/300 [==============================] - 1s 4ms/step - loss: 0.1834 - accuracy: 0.9700
Epoch 10/10
300/300 [==============================] - 1s 4ms/step - loss: 0.1392 - accuracy: 0.9600
Model: "sequential_2"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d_8 (Conv2D)           (None, 222, 222, 10)      280       
                                                                 
 conv2d_9 (Conv2D)           (None, 220, 220, 10)      910       
                                                                 
 max_pooling2d_4 (MaxPoolin  (None, 110, 110, 10)      0         
 g2D)                                                            
                                                                 
 conv2d_10 (Conv2D)          (None, 108, 108, 10)      910       
                                                                 
 conv2d_11 (Conv2D)          (None, 106, 106, 10)      910       
                                                                 
 max_pooling2d_5 (MaxPoolin  (None, 53, 53, 10)        0         
 g2D)                                                            
                                                                 
 flatten_2 (Flatten)         (None, 28090)             0         
                                                                 
 dense_2 (Dense)             (None, 3)                 84273     
                                                                 
=================================================================
Total params: 87283 (340.95 KB)
Trainable params: 87283 (340.95 KB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________"""

predictions = model_1.predict(x_train_tensor)

n=280

predictions[n],x_train_tensor[n]

import matplotlib.pyplot as plt
plt.imshow(x_train_tensor[n])
plt.show
