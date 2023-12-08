import tensorflow as tf

import os
os.environ["KAGGLE_CONFIG_DIR"]="/content"

!chmod 600 /content/kaggle.json

!kaggle datasets download -d marquis03/bean-leaf-lesions-classification

!unzip \*.zip

import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy import random
import tensorflow as tf
from tensorflow.keras import layers, models

import os
import cv2
import numpy as np

path = r"/content/train/healthy"
image_list = os.listdir(path)
image_paths = [os.path.join(path, img) for img in image_list]

# Assuming you have exactly 345 images
num_images = len(image_paths)
expected_shape = (num_images, 500, 500, 3)

images = np.zeros(expected_shape, dtype=np.uint8)

for i, img_path in enumerate(image_paths):
    img = cv2.imread(img_path)

    # Check if the image is loaded successfully
    if img is not None:
        # Resize the image to (500, 500)
        img = cv2.resize(img, (500, 500))
        images[i] = img
    else:
        print(f"Error loading image: {img_path}")

print("Final Shape:", images.shape)

x = random.randint(len(images))
plt.imshow(images[x]/225)
plt.show()

images = (images / 127.5) - 1

images.shape,images[0].shape ,(342,500,500,3)

import tensorflow as tf
from tensorflow.keras import layers, models

def create_model(input_shape=(500, 500, 3)):
    # Define the CNN model with Leaky ReLU activation
    model = models.Sequential()

    # Reshape the 1D input to 2D (assuming the input is a 1D array)
    model.add(layers.Reshape((input_shape[0], input_shape[1], input_shape[2]), input_shape=input_shape))

    # Convolutional Layer 1
    model.add(layers.Conv2D(32, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.01), padding='same'))
    model.add(layers.MaxPooling2D(2))

    # Convolutional Layer 2
    model.add(layers.Conv2D(64, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.01), padding='same'))
    model.add(layers.MaxPooling2D(2))

    # Convolutional Layer 3
    model.add(layers.Conv2D(128, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.01), padding='same'))
    model.add(layers.MaxPooling2D(2))

    # Flatten layer to transition from convolutional to dense layers
    model.add(layers.Flatten())

    # Dense (Fully Connected) Layer 1
    model.add(layers.Dense(128, activation=tf.keras.layers.LeakyReLU(alpha=0.01)))

    # Output Layer
    model.add(layers.Dense(500 * 500 * 3, activation='tanh'))  # Output layer with tanh activation

    # Reshape the output to match the desired shape (500, 500, 3)
    model.add(layers.Reshape((500, 500, 3)))

    return model

# Example usage with input_shape = (500, 500, 3)
model= create_model()

# Display the model summary
model.summary()

import tensorflow as tf
from tensorflow.keras import layers, models

def create_binary_cnn_model(input_shape=(500, 500, 3)):
    # Define the CNN model
    model = models.Sequential()

    # Convolutional layers
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    # Dense layers
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))  # Binary classification, so using sigmoid activation

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.002),
                  loss='binary_crossentropy',  # Binary classification loss function
                  metrics=['accuracy'])

    return model
dis=create_binary_cnn_model(input_shape=(500, 500, 3))
dis.summary()

def gan(model1, model2):
    model2.trainable = False
    gan_model = models.Sequential()
    gan_model.add(model1)
    gan_model.add(model2)

    gan_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss='binary_crossentropy',  # Binary classification loss function
                      metrics=['accuracy'])

    return gan_model

# Example usage:
# Example usage:
model1 = create_model()
model2 = create_binary_cnn_model(input_shape=(500, 500, 3))

gan_model = gan(model1, model2)
gan_model.summary()

def prepar_data(real_images=images, batch_size=32):
    if len(real_images) <= batch_size:
        # If not enough images for the specified batch size, set batch_size to the number of available images
        batch_size = len(real_images) - 1  # Adjust batch_size to ensure it's greater than 0

    temp = np.random.randint(len(real_images) - batch_size)
    x = real_images[temp:temp + batch_size]
    y = np.ones(batch_size)  # Assuming all are real images, so labels are set to 1

    return np.array(x), np.array(y)

x = random.rand(len(images),500,500,3)

def noice(x=x,batch_size=32):
  temp=np.random.randint(len(x)-batch_size)
  x=x[temp:temp+batch_size]
  y=[]
  for temp2 in x:
       y.append(0)
  return np.array(x),np.array(y)
x_fake,y_fake=noice(x=x,batch_size=32)
x_fake.shape,y_fake.shape

from numpy import ones

def train(gen=model1, dis=dis, gan_model=gan_model, number_epoch=100, batch_size=32*32):
    for j in range(number_epoch):
        print(f"Epoch: {j+1},")
        for i in range(0, len(images), batch_size):
            x_real, y_real = prepar_data(real_images=images[i:i+batch_size], batch_size=batch_size)
            x_fake, y_fake = noice(x=images, batch_size=batch_size)

            # Ensure that x_fake and y_fake have the same size
            x_fake = x_fake[:len(y_fake)]

            # Create labels for the GAN
            y_gan = ones((len(y_fake), 1))

            # Train the discriminator on real and fake data
            d_loss_real = dis.train_on_batch(x_real, y_real)
            d_loss_fake = dis.train_on_batch(x_fake, y_fake)

            # Train the generator via the GAN model
            gan_loss = gan_model.train_on_batch(x_fake, y_gan)

            print(f"Batch: {i//batch_size + 1}, Discriminator Loss Real: {d_loss_real[0]}, Discriminator Loss Fake: {d_loss_fake[0]}, GAN Loss: {gan_loss}")
    gen.save('cifar_conditional_generator.h5')

train(gen=model1, dis=dis, gan_model=gan_model, number_epoch=20, batch_size=32)

def noice2(x=x, batch_size=32):
    if len(x) <= batch_size:
        # If not enough data for the specified batch size, set batch_size to the number of available data points
        batch_size = len(x) - 1  # Adjust batch_size to ensure it's greater than 0

    temp = np.random.randint(len(x) - batch_size)
    x_batch = x[temp:temp + batch_size]
    y = np.zeros(batch_size)  # Assuming all are fake images, so labels are set to 0

    return x_batch, y

from numpy import asarray
from numpy.random import randn
from numpy.random import randint
from keras.models import load_model
import numpy as np

X,y=noice2(x=x,batch_size=32)

model = load_model('cifar_conditional_generator.h5')



X  = model.predict(X)

X = (X + 1) / 2.0
X = (X*255).astype(np.uint8)


plt.imshow(x[0])
plt.show()
    
