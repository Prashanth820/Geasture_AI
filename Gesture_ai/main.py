import matplotlib
import numpy as np

import matplotlib.pyplot as plt

import utils

import os

#%matplotlib inline

from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Dense, Input, Dropout,Flatten, Conv2D

from keras.layers import BatchNormalization, Activation, MaxPooling2D

from keras.models import Model, Sequential

from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from tensorflow.keras.utils import plot_model

from IPython.display import SVG, Image

import tensorflow as tf

for expression in os.listdir("/content/drive/MyDrive/MyDocs/train/"):

    print(str(len(os.listdir("/content/drive/MyDrive/MyDocs/train/"+expression)))+" "+expression+' images')


#===========================
#Validation
#===========================
img_size=64

batch_size=64

datagen_train=ImageDataGenerator(horizontal_flip=True)

train_generator=datagen_train.flow_from_directory("/content/drive/MyDrive/MyDocs/train/",

                                                 target_size=(img_size,img_size),

                                                 color_mode='grayscale',

                                                 batch_size=batch_size,

                                                 class_mode='categorical',

                                                 shuffle=True)

datagen_validation=ImageDataGenerator(horizontal_flip=True)

validation_generator=datagen_train.flow_from_directory("/content/drive/MyDrive/MyDocs/test/",

                                                 target_size=(img_size,img_size),

                                                 color_mode='grayscale',

                                                 batch_size=batch_size,

                                                 class_mode='categorical',

                                                 shuffle=True)


#===========================
#Building layers
#===========================

model=Sequential()

#conv-1

model.add(Conv2D(64,(3,3),padding='same',input_shape=(64,64,1)))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))

#2 -conv layer

model.add(Conv2D(128,(5,5),padding='same'))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))

#3 -conv layer

model.add(Conv2D(512,(3,3),padding='same'))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))

#4 -conv layer

model.add(Conv2D(512,(3,3),padding='same'))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(256))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(Dropout(0.25))

model.add(Dense(512))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(Dropout(0.25))

model.add(Dense(37,activation='softmax'))

opt=Adam(lr=0.0005)

#lr-learning rate

model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])

model.summary()

#===========================
#Training
#===========================

ephocs=20

steps_per_epoch=train_generator.n//train_generator.batch_size

steps_per_epoch

validation_steps=validation_generator.n//validation_generator.batch_size

validation_steps

history=model.fit(

    x=train_generator,

    steps_per_epoch=18,

    epochs=ephocs,

    validation_data=validation_generator,

    validation_steps=validation_steps,

 #   callbacks=callbacks

)

model.save('hand_gesture1.h5')


#===========================
#Testing
#===========================
import numpy as np

from keras.preprocessing import image
from tensorflow.keras.models import load_model
model=load_model("/content/drive/MyDrive/hand_gesture1.h5")

test_image = image.load_img('/content/drive/MyDrive/MyDocs/test/Y/1.jpg', target_size = (64,64),color_mode = "grayscale")

plt.imshow(test_image)

test_image = image.img_to_array(test_image)

test_image = np.expand_dims(test_image, axis = 0)

result = model.predict(test_image)

a=result.argmax()

s=train_generator.class_indices

        #print(s)

name=[]

for i in s:

    name.append(i)

for i in range(len(s)):

    if(i==a):

        q=name[i]

print(q)