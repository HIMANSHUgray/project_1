from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint
from keras.datasets.cifar10 import load_data
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Concatenate
from matplotlib import pyplot as plt

(trainX, trainy), (testX, testy) = load_data()

# plot 25 images
for i in range(25):
	plt.subplot(5, 5, 1 + i)
	plt.axis('off')
	plt.imshow(trainX[i])
plt.show()

def define_discriminator(in_shape=(32,32,3), n_classes=10):
	
    # label input
	in_label = Input(shape=(1,))  #Shape 1
	
	li = Embedding(n_classes, 50)(in_label) #Shape 1,50
	
	n_nodes = in_shape[0] * in_shape[1]  #32x32 = 1024. 
	li = Dense(n_nodes)(li)  #Shape = 1, 1024
	
	li = Reshape((in_shape[0], in_shape[1], 1))(li)  #32x32x1
    
    
	# image input
	in_image = Input(shape=in_shape) #32x32x3
	
	merge = Concatenate()([in_image, li]) #32x32x4
    
	fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(merge) #16x16x128
	fe = LeakyReLU(alpha=0.2)(fe)
	
	fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(fe) #8x8x128
	fe = LeakyReLU(alpha=0.2)(fe)
	
	fe = Flatten()(fe)  #8192  (8*8*128=8192)
	# dropout
	fe = Dropout(0.4)(fe)
	# output
	out_layer = Dense(1, activation='sigmoid')(fe)  #Shape=1
    

	model = Model([in_image, in_label], out_layer)
	
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model

test_discr = define_discriminator()
print(test_discr.summary())

def define_generator(latent_dim, n_classes=10):
    
	# label input
	in_label = Input(shape=(1,))  
	
	li = Embedding(n_classes, 50)(in_label) 
    
	
	n_nodes = 8 * 8 
	li = Dense(n_nodes)(li) #1,64
	
	li = Reshape((8, 8, 1))(li)
    
    
	
	in_lat = Input(shape=(latent_dim,))  #Input of dimension 100
    
	
	n_nodes = 128 * 8 * 8
	gen = Dense(n_nodes)(in_lat)  #shape=8192
	gen = LeakyReLU(alpha=0.2)(gen)
	gen = Reshape((8, 8, 128))(gen) #Shape=8x8x128
	
	merge = Concatenate()([gen, li])  #Shape=8x8x129 
	# upsample to 16x16
	gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(merge) #16x16x128
	gen = LeakyReLU(alpha=0.2)(gen)
	# upsample to 32x32
	gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(gen) #32x32x128
	gen = LeakyReLU(alpha=0.2)(gen)
	
	out_layer = Conv2D(3, (8,8), activation='tanh', padding='same')(gen) #32x32x3
	
	model = Model([in_lat, in_label], out_layer)
	return model   

test_gen = define_generator(100, n_classes=10)
print(test_gen.summary())

def define_gan(g_model, d_model):
	d_model.trainable = False  
    
    
	gen_noise, gen_label = g_model.input  
	
	gen_output = g_model.output  #32x32x3
    
	
	gan_output = d_model([gen_output, gen_label])
	
	model = Model([gen_noise, gen_label], gan_output)
	
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt)
	return model

def load_real_samples():
	
	(trainX, trainy), (_, _) = load_data()   #cifar
	
	X = trainX.astype('float32')
	
	X = (X - 127.5) / 127.5  
	return [X, trainy]

def generate_real_samples(dataset, n_samples):
	
	images, labels = dataset  
	
	ix = randint(0, images.shape[0], n_samples)
	
	X, labels = images[ix], labels[ix]
	
	y = ones((n_samples, 1))  
	return [X, labels], y

def generate_latent_points(latent_dim, n_samples, n_classes=10):
	
	x_input = randn(latent_dim * n_samples)
	
	z_input = x_input.reshape(n_samples, latent_dim)
	
	labels = randint(0, n_classes, n_samples)
	return [z_input, labels]

def generate_fake_samples(generator, latent_dim, n_samples):
	
	z_input, labels_input = generate_latent_points(latent_dim, n_samples)
	
	images = generator.predict([z_input, labels_input])
	
	y = zeros((n_samples, 1)) 
	return [images, labels_input], y

def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=10000, n_batch=128):
	bat_per_epo = int(dataset[0].shape[0] / n_batch)
	half_batch = int(n_batch / 2) 
                           
	for i in range(n_epochs):
		
		for j in range(bat_per_epo):
			
             
			[X_real, labels_real], y_real = generate_real_samples(dataset, half_batch)

           
			d_loss_real, _ = d_model.train_on_batch([X_real, labels_real], y_real)
            
			
			[X_fake, labels], y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
			
			d_loss_fake, _ = d_model.train_on_batch([X_fake, labels], y_fake)
            
            
			[z_input, labels_input] = generate_latent_points(latent_dim, n_batch)
            
           
			y_gan = ones((n_batch, 1))
             
			g_loss = gan_model.train_on_batch([z_input, labels_input], y_gan)
			
			print('Epoch>%d, Batch%d/%d, d1=%.3f, d2=%.3f g=%.3f' %
				(i+1, j+1, bat_per_epo, d_loss_real, d_loss_fake, g_loss))
	# save the generator model
	g_model.save('cifar_conditional_generator_25epochs.h5')

latent_dim = 100

d_model = define_discriminator()

g_model = define_generator(latent_dim)

gan_model = define_gan(g_model, d_model)

dataset = load_real_samples()

train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=2)

from numpy import asarray
from numpy.random import randn
from numpy.random import randint
from keras.models import load_model
import numpy as np



model = load_model('cifar_conditional_generator_25epochs.h5')



latent_points, labels = generate_latent_points(100, 100)

labels = asarray([x for _ in range(10) for x in range(10)])

X  = model.predict([latent_points, labels])

X = (X + 1) / 2.0
X = (X*255).astype(np.uint8)


def show_plot(examples, n):
	for i in range(n * n):
		plt.subplot(n, n, 1 + i)
		plt.axis('off')
		plt.imshow(examples[i, :, :, :])
	plt.show()
    
show_plot(X, 1)
