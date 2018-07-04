# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 11:43:42 2018

@author: hugo
"""

import matplotlib.pyplot as plt 


import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import backend as K

import data

import datetime
import glob
import numpy as np
import os
import platform
import SimpleITK as sitk

import pandas as pd

if platform.node() == 'bighugo':
    rootPath = "D:/data/wml-distance/training"
    testPath = "D:/data/wml-distance/testing"
elif platform.node() == 'hugo':
    rootPath = "X:/wml-distance/training"
    testPath = "X:/wml-distance/testing"
else:
    print("New computer, please provide root path to data.")
    raise 

batchSize = 60
batchShape = (batchSize, 51, 51, 2)
rs = np.random.RandomState(354351)
num_classes = 14


#%%

def getRandomPatchCoordinates(imageshape, patchshape):
    zs = rs.randint(0, imageshape[0] - 1)
    ys = rs.randint(0, imageshape[1] - patchshape[0])
    xs = rs.randint(0, imageshape[2] - patchshape[1])
    ze = zs + 1
    ye = ys + patchshape[0]
    xe = xs + patchshape[1]
    return zs, ze, ys, ye, xs, xe



def batchGenerator(images, labels, shape, rs=np.random):    
    while True:
        batchImages = np.zeros(shape, dtype=np.float32)
        batchLabels = np.zeros((shape[0], 1), dtype=np.float32)
        
        sampleIdx = rs.randint(0, len(images), shape[0])
        
        for idx in range(shape[0]):
            image = images[sampleIdx[idx]]
            label = labels[sampleIdx[idx]]
            
            zs, ze, ys, ye, xs, xe = getRandomPatchCoordinates(image.shape, shape[1:3])
            batchImages[idx,:,:,:] = image[zs:ze, ys:ye, xs:xe]
            batchLabels[idx] = label[int(zs+0.5*(ze-1-zs)), int(ys+0.5*(ye-1-ys)), int(xs+0.5*(xe-1-xs))]
            
        yield batchImages, batchLabels



def batchGeneratorMasked(images, labels, nonZeroIdx, shape, rs=np.random):    
    while True:
        batchImages = np.zeros(shape, dtype=np.float32)
        batchLabels = np.zeros((shape[0], 1), dtype=np.float32)
        
        sampleIdx = rs.randint(0, len(nonZeroIdx[0]), shape[0])
        
        for idx in range(shape[0]):
            
            zs = nonZeroIdx[1][sampleIdx[idx]]
            ze = zs + 1
            
            ys = int(nonZeroIdx[2][sampleIdx[idx]] - 0.5*(shape[1]-1))
            ye = int(nonZeroIdx[2][sampleIdx[idx]] + 0.5*(shape[1]-1) + 1)
            
            xs = int(nonZeroIdx[3][sampleIdx[idx]] - 0.5*(shape[2]-1))
            xe = int(nonZeroIdx[3][sampleIdx[idx]] + 0.5*(shape[2]-1) + 1)            
            
            try:
                batchImages[idx,:,:,:] = images[nonZeroIdx[0][sampleIdx[idx]], zs:ze, ys:ye, xs:xe]
                batchLabels[idx] = labels[nonZeroIdx[0][sampleIdx[idx]], nonZeroIdx[1][sampleIdx[idx]], nonZeroIdx[2][sampleIdx[idx]], nonZeroIdx[3][sampleIdx[idx]] ]
            except:
                print(xs,xe,ys,ye,zs,ze)    
                raise
                                    
        yield batchImages, batchLabels



#%%

images, labels, masks = data.loadData(rootPath, padding=25)
nonZeroIdx = np.nonzero(masks)

testImages, testLabels, testMasks = data.loadData(testPath, padding=25)
testNonZeroIdx = np.nonzero(testMasks)


#%%
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=batchShape[1:]))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
#model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())

#model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))


model.compile(loss=keras.losses.categorical_crossentropy,
              #optimizer=keras.optimizers.Adadelta(),
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

model.summary()

#%%

#iterator = batchGenerator(images, labels, batchShape, rs)
iterator = batchGeneratorMasked(images, labels, nonZeroIdx, batchShape, rs)
tbCallBack = keras.callbacks.TensorBoard(log_dir='D:/tensorflow_log/train', histogram_freq=0, write_graph=True, write_images=True)

df = None
max_iters = 50000
for idx, (x_train, y_train) in enumerate(iterator):
    y_train = keras.utils.to_categorical(y_train, num_classes)
    
    if idx % 500 == 0:
        x_test, y_test = next(batchGeneratorMasked(testImages, testLabels, testNonZeroIdx, batchShape))
        y_test = keras.utils.to_categorical(y_test, num_classes)        
        
        score = model.fit(x_train, y_train,
                  batch_size=batchSize,
                  verbose=0,
                  validation_data=(x_test, y_test),
                  )
        
        score.history['idx']= idx
        
        if df is None:
            df = pd.DataFrame.from_dict(score.history)
        else:
            df = df.append(pd.DataFrame.from_dict(score.history))
            
        # Store the params
        model.save(rootPath.replace('training', 'model/cnn_brain_segmentation_model_') + datetime.datetime.today().strftime('%Y%m%d_%H%M%S') + '_idx.h5')
                
        plt.plot(df.idx, df.acc, df.idx, df.val_acc, df.idx, df.acc.rolling(10).mean(), df.idx, df.val_acc.rolling(10).mean())
        try:
            plt.savefig('D:/tensorflow_log/train.png', dpi=200)
            plt.close()
        except:
            pass
    else:
        model.fit(x_train, y_train,
                  batch_size=batchSize,
                  verbose=0,
                  )
    
    if idx == max_iters:
        break

#11.42 gestart
model.save(rootPath.replace('training', 'cnn_brain_segmentation_model_') + datetime.datetime.today().strftime('%Y%m%d_%H%M%S') + '.h5')