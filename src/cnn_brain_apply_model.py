# -*- coding: utf-8 -*-
"""
Created on Mon May 14 12:26:13 2018

@author: hugo
"""



import keras
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import data

import datetime
import glob
import numpy as np
import os
import platform
import SimpleITK as sitk

if platform.node() == 'bighugo':
    rootPath = "D:/data/wml-distance/training"
    testPath = "D:/data/wml-distance/testing"
elif platform.node() == 'hugo':
    rootPath = "X:/wml-distance/training"
    testPath = "X:/wml-distance/testing"
else:
    print("New computer, please provide root path to data.")
    raise 
    
batchSize = 150
batchShape = (batchSize, 25, 25, 2)



#%%        
        
testImages, testLabels, testMasks = data.loadData(testPath)
testNonZeroIdx = np.nonzero(testMasks)

#%%

model = keras.models.load_model("X:/wml-distance/cnn_brain_segmentation_model_20180529_125910.h5")
novelty_model = Model(inputs=model.input, outputs=model.get_layer("dense_1").output)

#%%
iterator = data.batchTestGeneratorMasked(testImages, testLabels, testNonZeroIdx, batchShape)

i = 0
result = None
novelty = None
for predictImages, predictLabels in iterator:
    resultModel = model.predict_on_batch(predictImages)
    resultNovelty = novelty_model.predict_on_batch(predictImages)
    
    if result is None:
        result = resultModel
        novelty = resultNovelty
    else:
        result = np.concatenate([result, resultModel])
        novelty = np.concatenate([novelty, resultNovelty])

#result = model.predict_generator(iterator, len(testNonZeroIdx[0]))


#%%
dst = np.zeros((1, 48, 240, 240, 1), dtype=np.int8)
dst[testMasks] = np.argmax(result, axis=1)
for k,i in enumerate(dst):
    a = sitk.GetImageFromArray(i[:,:,:,0])
    sitk.WriteImage(a,'X:/wml-distance/testing/'+str(k)+'.nii') 
    print(i[:,:,:,0].shape)