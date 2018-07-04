# -*- coding: utf-8 -*-
"""
Created on Tue May 29 15:19:03 2018

@author: hugo
"""

import glob
import numpy as np
import os
import SimpleITK as sitk


def loadData( rootPath, padding=0 ):
    imageFilenames = glob.glob(os.path.join(rootPath, "*", "mri", "miT1W_3D_TFE.nii"))
    if len(imageFilenames) == 0: # Load a single subject
        imageFilenames = glob.glob(os.path.join(rootPath, "mri", "miT1W_3D_TFE.nii"))        
    
    flairFilenames = [x.replace('mri\\miT1W_3D_TFE', 'features\\T2_FLAIR') for x in imageFilenames]
    labelFilenames = [x.replace('mri\\miT1W_3D_TFE', 'features\\labels') for x in imageFilenames]
    
    images = None # shape: (numImages, z, y, x, channels=1)
    labels = None
    masks  = None
    
    for imageFilename, flairFilename, labelFilename in zip(imageFilenames, flairFilenames, labelFilenames):
        # Load the images
        imageImage = sitk.ReadImage(imageFilename)
        flairImage = sitk.ReadImage(flairFilename)
        labelImage = sitk.ReadImage(labelFilename)
        
        # Convert to arrays
        imageArray = np.pad(sitk.GetArrayFromImage(imageImage), [(0,0),(padding,padding),(padding,padding)], 'constant')
        flairArray = np.pad(sitk.GetArrayFromImage(flairImage), [(0,0),(padding,padding),(padding,padding)], 'constant')
        labelArray = np.pad(sitk.GetArrayFromImage(labelImage), [(0,0),(padding,padding),(padding,padding)], 'constant')
        maskArray = labelArray > 0
        
        # Add to the images/labels array
        if images is None:
            images = imageArray.reshape([1] + list(imageArray.shape) + [1])
            images = np.concatenate([images, flairArray.reshape([1] + list(flairArray.shape) + [1])], axis=4)
            labels = labelArray.reshape([1] + list(labelArray.shape) + [1])
            masks  = maskArray.reshape([1] + list(maskArray.shape) + [1])
        else:
            tempArray = np.concatenate([imageArray.reshape([1] + list(imageArray.shape) + [1]), flairArray.reshape([1] + list(flairArray.shape) + [1])], axis=4)
            images = np.concatenate([images, tempArray])
            
            labels = np.concatenate([labels, labelArray.reshape([1] + list(labelArray.shape) + [1])])
            masks  = np.concatenate([masks, maskArray.reshape([1] + list(maskArray.shape) + [1])])
                    
    return images, labels, masks





def batchTestGeneratorMasked(images, labels, nonZeroIdx, shape):  
    idx = 0
    while idx < len(nonZeroIdx[0]):        
        batchImages = np.zeros(shape, dtype=np.float32)
        batchLabels = np.zeros((shape[0], 1), dtype=np.float32)
        
        for i in range(shape[0]):              
            zs = nonZeroIdx[1][idx]
            ze = zs + 1
            
            ys = int(nonZeroIdx[2][idx] - 0.5*(shape[1]-1))
            ye = int(nonZeroIdx[2][idx] + 0.5*(shape[1]-1) + 1)
            
            xs = int(nonZeroIdx[3][idx] - 0.5*(shape[2]-1))
            xe = int(nonZeroIdx[3][idx] + 0.5*(shape[2]-1) + 1)            
            
            batchImages[i,:,:,:] = images[nonZeroIdx[0][idx], zs:ze, ys:ye, xs:xe]
            batchLabels[i] = labels[nonZeroIdx[0][idx], nonZeroIdx[1][idx], nonZeroIdx[2][idx], nonZeroIdx[3][idx] ]
            
            idx += 1      
            
            # Check if we go beyond the image(s)
            if idx >= len(nonZeroIdx[0]):
                break
        
        # In case we have reached the last batch, crop to i+1                      
        yield batchImages[:i+1], batchLabels[:i+1]
    
    return