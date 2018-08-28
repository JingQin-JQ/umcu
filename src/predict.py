import os
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
import SimpleITK as sitk
import scipy.spatial
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose,ZeroPadding2D, Dropout,UpSampling2D,Activation, Cropping2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K
import difflib

def dice_coef_for_training(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1.-dice_coef_for_training(y_true, y_pred)

def conv_bn_relu(nd, k=3, inputs=None):
    conv = Conv2D(nd, k, padding='same')(inputs) #, kernel_initializer='he_normal'
    #bn = BatchNormalization()(conv)
    relu = Activation('relu')(conv)
    return relu

def get_crop_shape(target, refer):
        # width, the 3rd dimension
        cw = (target.get_shape()[2] - refer.get_shape()[2]).value
        assert (cw >= 0)
        if cw % 2 != 0:
            cw1, cw2 = int(cw/2), int(cw/2) + 1
        else:
            cw1, cw2 = int(cw/2), int(cw/2)
        # height, the 2nd dimension
        ch = (target.get_shape()[1] - refer.get_shape()[1]).value
        assert (ch >= 0)
        if ch % 2 != 0:
            ch1, ch2 = int(ch/2), int(ch/2) + 1
        else:
            ch1, ch2 = int(ch/2), int(ch/2)

        return (ch1, ch2), (cw1, cw2)

def get_unet():
    concat_axis = -1
    filters = 3
    inputs = Input(batchShape[1:])    
    conv1 = conv_bn_relu(64, filters, inputs)
    conv1 = conv_bn_relu(64, filters, conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = conv_bn_relu(96, 3, pool1)
    conv2 = conv_bn_relu(96, 3, conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = conv_bn_relu(128, 3, pool2)
    conv3 = conv_bn_relu(128, 3, conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = conv_bn_relu(256, 3, pool3)
    conv4 = conv_bn_relu(256, 4, conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = conv_bn_relu(512, 3, pool4)
    conv5 = conv_bn_relu(512, 3, conv5)

    up_conv5 = UpSampling2D(size=(2, 2))(conv5)
    ch, cw = get_crop_shape(conv4, up_conv5)
    crop_conv4 = Cropping2D(cropping=(ch,cw))(conv4)
    up6 = concatenate([up_conv5, crop_conv4], axis=concat_axis)
    conv6 = conv_bn_relu(256, 3, up6)
    conv6 = conv_bn_relu(256, 3, conv6)

    up_conv6 = UpSampling2D(size=(2, 2))(conv6)
    ch, cw = get_crop_shape(conv3, up_conv6)
    crop_conv3 = Cropping2D(cropping=(ch,cw))(conv3)
    up7 = concatenate([up_conv6, crop_conv3], axis=concat_axis)
    conv7 = conv_bn_relu(128, 3, up7)
    conv7 = conv_bn_relu(128, 3, conv7)

    up_conv7 = UpSampling2D(size=(2, 2))(conv7)
    ch, cw = get_crop_shape(conv2, up_conv7)
    crop_conv2 = Cropping2D(cropping=(ch,cw))(conv2)
    up8 = concatenate([up_conv7, crop_conv2], axis=concat_axis)
    conv8 = conv_bn_relu(96, 3, up8)
    conv8 = conv_bn_relu(96, 3, conv8)

    up_conv8 = UpSampling2D(size=(2, 2))(conv8)
    ch, cw = get_crop_shape(conv1, up_conv8)
    crop_conv1 = Cropping2D(cropping=(ch,cw))(conv1)
    up9 = concatenate([up_conv8, crop_conv1], axis=concat_axis)
    conv9 = conv_bn_relu(64, 3, up9)
    conv9 = conv_bn_relu(64, 3, conv9)

    ch, cw = get_crop_shape(inputs, conv9)
    conv9 = ZeroPadding2D(padding=(ch, cw))(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid', padding='same')(conv9) #, kernel_initializer='he_normal'
    model = Model(inputs=inputs, outputs=conv10)
    model.compile(optimizer=Adam(lr=(2e-4)), loss=dice_coef_loss)
    model.summary()
    return model

def do():
    """Main function"""
#     resultFilename = getResultFilename(participantDir)
    dsc_l = []
    avd_l = []
    recall_l = []
    f1_l = []
    for i in range(50):
        testImage, resultImage = getImages(testFilename, resultFilename, i)
        dsc = getDSC(testImage, resultImage)
        avd = getAVD(testImage, resultImage)
        recall, f1 = getLesionDetection(testImage, resultImage)
        dsc_l.append(dsc)
        avd_l.append(avd)
        recall_l.append(recall)
        f1_l.append(f1)
        
        
        
    print ('Dice', np.mean(dsc_l) , '(higher is better, max=1)')
    print ('AVD', np.mean(avd_l), '%', '(lower is better, min=0)')
    print ('Lesion detection', np.mean(recall_l), '(higher is better, max=1)')
    print ('Lesion F1', np.mean(f1_l), '(higher is better, max=1)')

def getImages(testImage, resultImage, i):
    """Return the test and result images, thresholded and non-WMH masked."""
#     testImage = sitk.ReadImage(testFilename)
#     resultImage = sitk.ReadImage(resultFilename)
#     # Check for equality
#     print(type(testImage) , testImage.shape,testImage.max())
    testImage= sitk.GetImageFromArray(testImage[i,:,:,0])
    resultImage= sitk.GetImageFromArray(resultImage[i,:,:,0])
#     print(type(testImage), sitk.GetArrayFromImage(testImage).shape,sitk.GetArrayFromImage(testImage).max())
#     assert testImage.GetSize() == resultImage.GetSize()
    # Get meta data from the test-image, needed for some sitk methods that check this
    
    resultImage.CopyInformation(testImage)
    # Remove non-WMH from the test and result images, since we don't evaluate on that
#     print(type(testImage), sitk.GetArrayFromImage(testImage).shape,sitk.GetArrayFromImage(testImage).max())
    maskedTestImage = sitk.BinaryThreshold(testImage, 0.5, 1.5, 1, 0) # WMH == 1
    nonWMHImage = sitk.BinaryThreshold(testImage, 1.5, 2.5, 0, 1) # non-WMH == 2
    maskedResultImage = sitk.Mask(resultImage, nonWMHImage)
    # Convert to binary mask
    if 'integer' in maskedResultImage.GetPixelIDTypeAsString():
        bResultImage = sitk.BinaryThreshold(maskedResultImage, 1, 1000, 1, 0)
    else:
        bResultImage = sitk.BinaryThreshold(maskedResultImage, 0.5, 1000, 1, 0)
    return maskedTestImage, bResultImage

def getDSC(testImage, resultImage):
    """Compute the Dice Similarity Coefficient."""
    testArray = sitk.GetArrayFromImage(testImage).flatten()
    resultArray = sitk.GetArrayFromImage(resultImage).flatten()
    # similarity = 1.0 - dissimilarity
    return 1.0 - scipy.spatial.distance.dice(testArray, resultArray)

def getLesionDetection(testImage, resultImage):
    """Lesion detection metrics, both recall and F1."""
    # Connected components will give the background label 0, so subtract 1 from all results
    ccFilter = sitk.ConnectedComponentImageFilter()
    ccFilter.SetFullyConnected(True)
    # Connected components on the test image, to determine the number of true WMH.
    # And to get the overlap between detected voxels and true WMH
    ccTest = ccFilter.Execute(testImage)
    lResult = sitk.Multiply(ccTest, sitk.Cast(resultImage, sitk.sitkUInt32))
    ccTestArray = sitk.GetArrayFromImage(ccTest)
    lResultArray = sitk.GetArrayFromImage(lResult)
    # recall = (number of detected WMH) / (number of true WMH)
    nWMH = len(np.unique(ccTestArray)) - 1
    if nWMH == 0:
        recall = 1.0
    else:
        recall = float(len(np.unique(lResultArray)) - 1) / nWMH
    # Connected components of results, to determine number of detected lesions
    ccResult = ccFilter.Execute(resultImage)
    lTest = sitk.Multiply(ccResult, sitk.Cast(testImage, sitk.sitkUInt32))
    ccResultArray = sitk.GetArrayFromImage(ccResult)
    lTestArray = sitk.GetArrayFromImage(lTest)
    # precision = (number of detections that intersect with WMH) / (number of all detections)
    nDetections = len(np.unique(ccResultArray)) - 1
    if nDetections == 0:
        precision = 1.0
    else:
        precision = float(len(np.unique(lTestArray)) - 1) / nDetections
    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2.0 * (precision * recall) / (precision + recall)
    return recall, f1

def getAVD(testImage, resultImage):
    """Volume statistics."""
    # Compute statistics of both images
    testStatistics = sitk.StatisticsImageFilter()
    resultStatistics = sitk.StatisticsImageFilter()
    testStatistics.Execute(testImage)
    resultStatistics.Execute(resultImage)
    return float(abs(testStatistics.GetSum() - resultStatistics.GetSum())) / float(testStatistics.GetSum()) * 100

def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0],img_rows, img_cols,imgs.shape[-1]), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i],(img_cols, img_rows,imgs.shape[-1]), preserve_range=True)
    return imgs_p

def predict(model, nr):    
    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)
    testrange = range(len(ytest))
    imgs_test, imgs_id_test = Xtest, ytest
#     print("before test pre",imgs_test.shape,imgs_id_test)
    imgs_test = preprocess(imgs_test)
#     print("after test pre",imgs_test.shape,imgs_id_test)
    imgs_test = imgs_test.astype('float32')
    mean = np.mean(imgs_test)  # mean for data centering
    std = np.std(imgs_test)  # std for data normalization
    imgs_test -= mean
    imgs_test /= std

    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    model.load_weights('weights.h5')
    
    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
#     print("test model",imgs_test.shape)
    imgs_mask_test = model.predict(imgs_test, verbose=1)
    print("test model finished",imgs_mask_test.shape)
    np.save('imgs_mask_test.npy', imgs_mask_test)
    
    imgs_mask_test[imgs_mask_test[:,:,:,0] > 0.5] = 1      #thresholding 
    imgs_mask_test[imgs_mask_test[:,:,:,0] <= 0.5] = 0
    print('-' * 30)
    print('Saving predicted masks to files...')
    print('-' * 30)
    pred_dir = 'preds'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    for img,targ, image, image_id in zip( Xtest, imgs_id_test, imgs_mask_test, testrange):
#         print(image_id)
        nn = 0
#         print(image.shape)
        img = (img[:, :, 0]* 255.).astype(np.uint8)
        targ = (targ[:, :, 0] * 255.).astype(np.uint8) 
        image = (image[:, :, 0] * 255.).astype(np.uint8)

        imsave(os.path.join(pred_dir,str(image_id) + "_" + str(nr) + '_ori.png'), img)
        imsave(os.path.join(pred_dir,str(image_id) + "_" + str(nr) + '_trg.png'), targ)
        imsave(os.path.join(pred_dir,str(image_id) + "_" + str(nr) + '_pred.png'), image)
        nn+=1
    print(imgs_id_test.shape,imgs_mask_test.shape)
    return imgs_id_test,imgs_mask_test


img_rows =200
img_cols =200

smooth = 1.

batchSize = 40
batchShape = (batchSize, 200,200, 2)

Xtest = np.load('validationimages.npy')
ytest = np.load('validationlables.npy')
ytest[ytest >1] = 0
print(Xtest.shape, Xtest.min(), Xtest.max()) # (240, 240, 4) -0.380588 2.62761
print(ytest.shape, ytest.min(), ytest.max())
model = get_unet()

for i in [9]:
    
    testFilename, resultFilename = predict(model,i)
    
    do()