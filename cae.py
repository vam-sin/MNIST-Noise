# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 11:43:26 2018

@author: Vamsi
"""
# Importing the libraries
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.models import Model,Sequential
from keras.layers import Conv2D,MaxPooling2D,UpSampling2D
from keras.models import load_model

# Loading the data
(x_train,_),(x_test,_)=mnist.load_data()

# Scaling the values 
max_value=float(x_train.max())
x_train=x_train.astype('float32')/max_value
x_test=x_test.astype('float32')/max_value

# Properties of the dataset
x_train.shape,x_test.shape

# Reshaping the dataset
x_train = x_train.reshape((len(x_train), 28, 28, 1))
x_test = x_test.reshape((len(x_test), 28, 28, 1))
(x_train.shape,x_test.shape)

# Autoencoder

autoencoder=Sequential()

# Encoder
autoencoder.add(Conv2D(32,(3,3),activation='relu',padding='same',input_shape=x_train.shape[1:]))
autoencoder.add(MaxPooling2D((2,2),padding='same'))
autoencoder.add(Conv2D(32,(3,3),activation='relu',padding='same'))
autoencoder.add(MaxPooling2D((2,2),padding='same'))

# Decoder
autoencoder.add(Conv2D(32,(3,3),activation='relu',padding='same'))
autoencoder.add(UpSampling2D((2,2)))
autoencoder.add(Conv2D(32,(3,3),activation='relu',padding='same'))
autoencoder.add(UpSampling2D((2,2)))
autoencoder.add(Conv2D(1,(3,3),activation='sigmoid',padding='same'))

autoencoder.summary()

iteration=100

# Adding noise
x_train_noisy=x_train+np.random.normal(loc=0.0,scale=0.5,size=x_train.shape)
x_train_noisy=np.clip(x_train_noisy,0.,1.)

x_test_noisy=x_test+np.random.normal(loc=0.0,scale=0.5,size=x_test.shape)
x_tesy_noisy=np.clip(x_test_noisy,0.,1.)

# Training
autoencoder.compile(optimizer='adam',loss='binary_crossentropy')
for i in range(iteration):
    autoencoder.fit(x_train_noisy,x_train,epochs=1,batch_size=128,validation_data=(x_test_noisy,x_test))
    autoencoder.save('model_' + str(i) + '.h5')
    
    
# Reconstructing and checking
autoencoder=load_model('model_99.h5')
num_images=10
np.random.seed(42)
random_test_images=np.random.randint(x_test.shape[0],size=num_images)

x_test_denoised=autoencoder.predict(x_test_noisy)

plt.figure(figsize=(18,4))

for i,image_idx in enumerate(random_test_images):
    
    # Noisy image
    ax=plt.subplot(2,num_images,i+1)
    plt.imshow(x_test_noisy[image_idx].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Reconstructed Image
    ax=plt.subplot(2,num_images,num_images+1+i)
    plt.imshow(x_test_denoised[image_idx].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
plt.show()
