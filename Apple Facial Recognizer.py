#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import tensorflow as tf
import tarfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import shutil
from PIL import Image
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from shutil import copyfile
from os import getcwd


# In[2]:


os.mkdir(f"{getcwd()}/data")
base_dir = f"{getcwd()}/data"

os.mkdir(os.path.join(base_dir,"testing"))
os.mkdir(os.path.join(base_dir,"training"))

base_training = os.path.join(base_dir,"training")
base_testing = os.path.join(base_dir,"testing")

os.mkdir(os.path.join(base_training,"angry"))
os.mkdir(os.path.join(base_training,"disgust"))
os.mkdir(os.path.join(base_training,"fear"))
os.mkdir(os.path.join(base_training,"happy"))
os.mkdir(os.path.join(base_training,"neutral"))
os.mkdir(os.path.join(base_training,"sad"))
os.mkdir(os.path.join(base_training,"surprise"))

os.mkdir(os.path.join(base_testing,"angry"))
os.mkdir(os.path.join(base_testing,"disgust"))
os.mkdir(os.path.join(base_testing,"fear"))
os.mkdir(os.path.join(base_testing,"happy"))
os.mkdir(os.path.join(base_testing,"neutral"))
os.mkdir(os.path.join(base_testing,"sad"))
os.mkdir(os.path.join(base_testing,"surprise"))


# In[3]:


csv_file = f"{getcwd()}/fer2013.csv"   
df = pd.read_csv(csv_file)
print(df.head())


# In[4]:


#parsing CSV Files, and transforming pixels column into 2 by 2 int np array
train = df[["emotion", "pixels"]][df["Usage"] == "Training"]
train['pixels'] = train['pixels'].apply(lambda im: np.fromstring(im, sep=' '))
x_train = np.vstack(train['pixels'].values)
y_train = np.array(train["emotion"])

test = df[["emotion", "pixels"]][df["Usage"]=="PublicTest"]
test["pixels"] = test["pixels"].apply(lambda im: np.fromstring(im, sep=' '))
x_preval = np.vstack(test["pixels"].values)
y_preval = np.array(test["emotion"])

val = df[["emotion", "pixels"]][df["Usage"]=="PrivateTest"]
val["pixels"] = val["pixels"].apply(lambda im: np.fromstring(im, sep=' '))
x_val = np.vstack(val["pixels"].values)
y_val = np.array(val["emotion"])

x_test = np.concatenate((x_val, x_preval))
y_test = np.concatenate((y_val, y_preval))

x_train = x_train.reshape(-1, 48, 48)
x_test  = x_test.reshape(-1, 48, 48)

print(x_train.shape)
print(y_train.shape)

print(x_test.shape)
print(y_test.shape) 


# In[5]:


print(y_test[3589])
print(x_test[3589])


# In[6]:


# Converting 2D np array into image, and storing them in a separate folder for each classes
x_train_size = len(x_train)
try:
    # Change the current working Directory    
    os.chdir(f"{getcwd()}/data/training/")
except OSError:
     print("Can't change the Current Working Directory") 
        
# Looping through the entire traindataset
for x in range(0,x_train_size):
        if (y_train[x]==0).all() == True:
            try:
            # Change the current working Directory    
                os.chdir(f"{getcwd()}/angry")
                im = Image.fromarray(x_train[x,:,:])
                im = im.convert("L")
                im.save(str(x)+"your_file.jpg")
            except OSError:
                print("Can't change the Current Working Directory")   
        elif (y_train[x]==1).all() == True:
            try:
            # Change the current working Directory    
                os.chdir(f"{getcwd()}/disgust")
                im = Image.fromarray(x_train[x,:,:])
                im = im.convert("L")
                im.save(str(x)+"your_file.jpg")
            except OSError:
                print("Can't change the Current Working Directory")   
        elif (y_train[x]==2).all() == True:
            try:
            # Change the current working Directory    
                os.chdir(f"{getcwd()}/fear")
                im = Image.fromarray(x_train[x,:,:])
                im = im.convert("L")
                im.save(str(x)+"your_file.jpg")
            except OSError:
                print("Can't change the Current Working Directory")     
        elif (y_train[x]==3).all() == True:
            try:
            # Change the current working Directory    
                os.chdir(f"{getcwd()}/happy")
                im = Image.fromarray(x_train[x,:,:])
                im = im.convert("L")
                im.save(str(x)+"your_file.jpg")
            except OSError:
                print("Can't change the Current Working Directory")    
        elif (y_train[x]==4).all() == True:
            try:
            # Change the current working Directory    
                os.chdir(f"{getcwd()}/neutral")
                im = Image.fromarray(x_train[x,:,:])
                im = im.convert("L")
                im.save(str(x)+"your_file.jpg")
            except OSError:
                print("Can't change the Current Working Directory")   
        elif (y_train[x]==5).all() == True:
            try:
            # Change the current working Directory    
                os.chdir(f"{getcwd()}/sad")
                im = Image.fromarray(x_train[x,:,:])
                im = im.convert("L")
                im.save(str(x)+"your_file.jpg")
            except OSError:
                print("Can't change the Current Working Directory")    
        elif (y_train[x]==6).all() == True:
            try:
            # Change the current working Directory    
                os.chdir(f"{getcwd()}/surprise")
                im = Image.fromarray(x_train[x,:,:])
                im = im.convert("L")
                im.save(str(x)+"your_file.jpg")
            except OSError:
                print("Can't change the Current Working Directory")     
        try:
            # Change the current working Directory    
            os.chdir(f"{getcwd()}/..")
        except OSError:
            print("Can't change the Current Working Directory")  
try:
    # Change the current working Directory    
    os.chdir(f"{getcwd()}/..")
except OSError:
    print("Can't change the Current Working Directory")  


# In[7]:


# Converting 2D np array into image, and storing them in a separate folder for each classes
x_test_size = len(x_test)
try:
    # Change the current working Directory    
    os.chdir(f"{getcwd()}/testing/")
except OSError:
     print("Can't change the Current Working DirectoryMaju")   
# Looping for the entire test data set
for x in range(0,x_test_size):
        if (y_test[x]==0).all() == True:
            try:
            # Change the current working Directory    
                os.chdir(f"{getcwd()}/angry")
                im = Image.fromarray(x_test[x,:,:])
                im = im.convert("L")
                im.save(str(x)+"your_file.jpg")
            except OSError:
                print("Can't change the Current Working Directory")   
        elif (y_test[x]==1).all() == True:
            try:
            # Change the current working Directory    
                os.chdir(f"{getcwd()}/disgust")
                im = Image.fromarray(x_test[x,:,:])
                im = im.convert("L")
                im.save(str(x)+"your_file.jpg")
            except OSError:
                print("Can't change the Current Working Directory")   
        elif (y_test[x]==2).all() == True:
            try:
            # Change the current working Directory    
                os.chdir(f"{getcwd()}/fear")
                im = Image.fromarray(x_test[x,:,:])
                im = im.convert("L")
                im.save(str(x)+"your_file.jpg")
            except OSError:
                print("Can't change the Current Working Directory")     
        elif (y_test[x]==3).all() == True:
            try:
            # Change the current working Directory    
                os.chdir(f"{getcwd()}/happy")
                im = Image.fromarray(x_test[x,:,:])
                im = im.convert("L")
                im.save(str(x)+"your_file.jpg")
            except OSError:
                print("Can't change the Current Working Directory")    
        elif (y_test[x]==4).all() == True:
            try:
            # Change the current working Directory    
                os.chdir(f"{getcwd()}/neutral")
                im = Image.fromarray(x_test[x,:,:])
                im = im.convert("L")
                im.save(str(x)+"your_file.jpg")
            except OSError:
                print("Can't change the Current Working Directory")   
        elif (y_test[x]==5).all() == True:
            try:
            # Change the current working Directory    
                os.chdir(f"{getcwd()}/sad")
                im = Image.fromarray(x_test[x,:,:])
                im = im.convert("L")
                im.save(str(x)+"your_file.jpg")
            except OSError:
                print("Can't change the Current Working Directory")    
        elif (y_test[x]==6).all() == True:
            try:
            # Change the current working Directory    
                os.chdir(f"{getcwd()}/surprise")
                im = Image.fromarray(x_test[x,:,:])
                im = im.convert("L")
                im.save(str(x)+"your_file.jpg")
            except OSError:
                print("Can't change the Current Working Directory")     
        try:
            # Change the current working Directory    
            os.chdir(f"{getcwd()}/..")
        except OSError:
            print("Can't change the Current Working DirectoryMundur")   
try:
    # Change the current working Directory    
    os.chdir(f"{getcwd()}/..")
except OSError:
    print("Can't change the Current Working Directory")  


# In[8]:


# My Computer doesnt have a dedicated gpu, hence im not able to use cnn for this part.
model = tf.keras.models.Sequential([
#     tf.keras.layers.Conv2D(16,(3,3), activation='relu',input_shape=(48,48,1)),
#     tf.keras.layers.MaxPooling2D(2,2),
#     tf.keras.layers.Conv2D(32,(3,3), activation = 'relu'),
#     tf.keras.layers.MaxPooling2D(2,2),
#     tf.keras.layers.Conv2D(64,(3,3), activation = 'relu'),
#     tf.keras.layers.MaxPooling2D(2,2),
#     tf.keras.layers.Conv2D(64,(3,3), activation = 'relu'),
#     tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dense(512, activation = 'relu', input_shape=(48,48,3)),
    tf.keras.layers.Dense(6, activation = 'softmax')
])
model.compile(loss = 'sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


# In[17]:


TRAINING_DIR = f"{getcwd()}/training/"
VALIDATION_DIR = f"{getcwd()}/testing/"

train_datagen = ImageDataGenerator(
                  rescale=1./255,
                  rotation_range=40,
                  width_shift_range=0.2,
                  height_shift_range=0.2,
                  shear_range=0.2,
                  zoom_range=0.2,
                  horizontal_flip=True,
                  fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(
                  TRAINING_DIR,  
                  target_size=(48, 48),  
                  batch_size=10000,
                  class_mode='categorical'
                 )

validation_datagen = ImageDataGenerator(rescale=1./255)


validation_generator = train_datagen.flow_from_directory(
                      VALIDATION_DIR,  
                      target_size=(48, 48), 
                      batch_size=10000,
                      class_mode='categorical'
                     )


# In[ ]:


#Even without CNN im not able to train the model, insufficient amount of ram.
history = model.fit(train_generator,
                              epochs=1000,
                              verbose=1,
                              validation_data=validation_generator)


# In[ ]:


# PLOT LOSS AND ACCURACY
get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.image  as mpimg
import matplotlib.pyplot as plt

acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc)) # Get number of epochs

plt.plot(epochs, acc, 'r', "Training Accuracy")
plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
plt.title('Training and validation accuracy')
plt.figure()

plt.plot(epochs, loss, 'r', "Training Loss")
plt.plot(epochs, val_loss, 'b', "Validation Loss")


plt.title('Training and validation loss')

# Desired output. Charts with training and validation metrics. No crash :)


# In[ ]:





# In[ ]:




