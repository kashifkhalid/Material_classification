
# dependencies 
import os, cv2, random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns
import tensorflow as tf

from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Conv2D, MaxPooling2D, ZeroPadding2D, Dense, Activation, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.metrics import categorical_accuracy
from keras.optimizers import RMSprop, Adam
from keras.callbacks import EarlyStopping
from keras.utils import np_utils

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras import backend as K



# functions to read datasets 
path="Material_classification/newfmd"

def get_images(mat):
        """Load files from train folder"""
        mat_dir = path+'{}'.format(mat)
        images = [mat+'/'+im for im in os.listdir(mat_dir)]

        return images

def read_image(src):
     """Read and resize individual images"""
     im = cv2.imread(src,1)
     print(src)
     im = cv2.resize(im, (COLS, ROWS), interpolation=cv2.INTER_LINEAR)

     return im

#reading images:
#*************************************
mat_classes = os.listdir(path)   
ROWS = 220
COLS = 220
CHANNELS = 3

files = []
y_all = []
  
for mat in mat_classes:

    mat_files = get_images(mat)
    files.extend(mat_files)
    y_mat = np.tile(mat, len(mat_files))
    y_all.extend(y_mat)
 
print(y_all[0]) 
print(y_all[200])
print(y_all[400])
print(y_all[600])
print(y_all[800])
print(y_all[1000])
print(y_all[1200])
print(y_all[1400])
      
# reshape dataset to meet the images propreties   
X_all = np.ndarray((len(files), ROWS, COLS, CHANNELS), dtype=np.uint8)

for i, im in enumerate(files):

    X_all[i] = read_image(path+im)

# One Hot Encoding Labels:
#***********************************  
y_all = np.array(y_all)
y_all = LabelEncoder().fit_transform(y_all)
y_all= np_utils.to_categorical(y_all)
print(y_all.shape)

# spliting the data for training and testing sets:
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, 
                                                    test_size=0.15, random_state=23, 
                                                    stratify=y_all)

# optimization algorithm 
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

# loss of the first compilation of the model 
objective = 'mean_squared_error'

# normalization functinn to normalize input
def center_normalize(x):
    return (x - K.mean(x)) / K.std(x)

# Download pre-trained network and build the proposed model  
base_model = ResNet50(weights='imagenet', include_top=False)
x = base_model.output

# build a classifier model to put on top of the convolutional model
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(8, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# fine-tuning the network 
for layer in base_model.layers:
    layer.trainable = False

model.compile(loss=objective, optimizer=optimizer, metrics=['accuracy'])
#print(model.summary())

model.fit(X_train, y_train, batch_size=32, epochs=5,
              validation_split=0.15, verbose=1, shuffle=True)
              
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)
   
for layer in model.layers[:141]:
   layer.trainable = False
for layer in model.layers[141:]:
   layer.trainable = True

from keras.optimizers import SGD
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])



early_stopping = EarlyStopping(monitor='val_loss', patience=6, verbose=1, mode='auto')        
        
history = model.fit(X_train, y_train, batch_size=32, epochs=10,
              validation_split=0.15, verbose=1, shuffle=True, callbacks=[early_stopping])

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

# save the model :
#*************************************************
model.save('resnet50.h5')         
  
#save weights:
model.save_weights('resnet50_weights.h5')

# plot
plt.figure(1)  
plt.subplot(211)  
plt.plot(history.history['acc'])  
plt.plot(history.history['val_acc'])  
plt.title('model accuracy')  
plt.ylabel('accuracy')  
plt.xlabel('epoch')  
plt.legend(['train', 'test'], loc='upper left') 

plt.subplot(212)  
plt.plot(history.history['loss'])  
plt.plot(history.history['val_loss'])  
plt.title('model loss')  
plt.ylabel('loss')  
plt.xlabel('epoch')  
plt.legend(['train', 'test'], loc='upper left')  
plt.show()  


