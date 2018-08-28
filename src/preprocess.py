from __future__ import print_function
import os
import numpy as np
from random import shuffle
import SimpleITK as sitk
import scipy.spatial
from scipy import ndimage


### ----define loss function for U-net ------------
smooth = 1.

def Utrecht_preprocessing(FLAIR_image, T1_image,labelArray):

    channel_num = 2
    num_selected_slice = np.shape(FLAIR_image)[0]
    image_rows_Dataset = np.shape(FLAIR_image)[1]
    image_cols_Dataset = np.shape(FLAIR_image)[2]
    T1_image = np.float32(T1_image)

    brain_mask_FLAIR = np.ndarray((np.shape(FLAIR_image)[0],image_rows_Dataset, image_cols_Dataset), dtype=np.float32)
    brain_mask_T1 = np.ndarray((np.shape(FLAIR_image)[0],image_rows_Dataset, image_cols_Dataset), dtype=np.float32)
    imgs_two_channels = np.ndarray((num_selected_slice, rows_standard, cols_standard, channel_num), dtype=np.float32)
    brain_label = np.ndarray((np.shape(FLAIR_image)[0],image_rows_Dataset, image_cols_Dataset), dtype=np.float32)
    imgs_mask_two_channels = np.ndarray((num_selected_slice, rows_standard, cols_standard,1), dtype=np.float32)

    # FLAIR --------------------------------------------
    brain_mask_FLAIR[FLAIR_image >=thresh_FLAIR] = 1
    brain_mask_FLAIR[FLAIR_image < thresh_FLAIR] = 0
    for iii in range(np.shape(FLAIR_image)[0]):
        brain_mask_FLAIR[iii,:,:] = scipy.ndimage.morphology.binary_fill_holes(brain_mask_FLAIR[iii,:,:])  #fill the holes inside brain
    
    FLAIR_image = FLAIR_image[:, int(image_rows_Dataset/2-rows_standard/2):int(image_rows_Dataset/2+rows_standard/2), int(image_cols_Dataset/2-cols_standard/2):int(image_cols_Dataset/2+cols_standard/2)]
    brain_mask_FLAIR = brain_mask_FLAIR[:, int(image_rows_Dataset/2-rows_standard/2):int(image_rows_Dataset/2+rows_standard/2), int(image_cols_Dataset/2-cols_standard/2):int(image_cols_Dataset/2+cols_standard/2)]
    ###------Gaussion Normalization here
    FLAIR_image -=np.mean(FLAIR_image[brain_mask_FLAIR == 1])      #Gaussion Normalization
    FLAIR_image /=np.std(FLAIR_image[brain_mask_FLAIR == 1])
    # T1 -----------------------------------------------
    brain_mask_T1[T1_image >=thresh_T1] = 1
    brain_mask_T1[T1_image < thresh_T1] = 0
    for iii in range(np.shape(T1_image)[0]):
        brain_mask_T1[iii,:,:] = scipy.ndimage.morphology.binary_fill_holes(brain_mask_T1[iii,:,:])  #fill the holes inside brain
    T1_image = T1_image[:, int(image_rows_Dataset/2-rows_standard/2):int(image_rows_Dataset/2+rows_standard/2), int(image_cols_Dataset/2-cols_standard/2):int(image_cols_Dataset/2+cols_standard/2)]
    brain_mask_T1 = brain_mask_T1[:, int(image_rows_Dataset/2-rows_standard/2):int(image_rows_Dataset/2+rows_standard/2), int(image_cols_Dataset/2-cols_standard/2):int(image_cols_Dataset/2+cols_standard/2)]
    #------Gaussion Normalization
    T1_image -=np.mean(T1_image[brain_mask_T1 == 1])      
    T1_image /=np.std(T1_image[brain_mask_T1 == 1])
    # lable----------------
    brain_label[labelArray == 1] = 1
    brain_label[labelArray != 1] = 0
    imgs_mask_two_channels = brain_label[:, int(image_rows_Dataset/2-rows_standard/2):int(image_rows_Dataset/2+rows_standard/2), int(image_cols_Dataset/2-cols_standard/2):int(image_cols_Dataset/2+cols_standard/2)]
    #---------------------------------------------------
    FLAIR_image  = FLAIR_image[..., np.newaxis]
    T1_image  = T1_image[..., np.newaxis]
    imgs_two_channels = np.concatenate((FLAIR_image, T1_image), axis = 3)
    maskArray = imgs_mask_two_channels > 0

    return imgs_two_channels,imgs_mask_two_channels, maskArray


def GE3T_preprocessing(FLAIR_image, T1_image,labelArray):

  #  start_slice = 10
    channel_num = 2
    start_cut = 46
    num_selected_slice = np.shape(FLAIR_image)[0]
    image_rows_Dataset = np.shape(FLAIR_image)[1]
    image_cols_Dataset = np.shape(FLAIR_image)[2]
    FLAIR_image = np.float32(FLAIR_image)
    T1_image = np.float32(T1_image)

    brain_mask_FLAIR = np.ndarray((np.shape(FLAIR_image)[0],image_rows_Dataset, image_cols_Dataset), dtype=np.float32)
    brain_mask_T1 = np.ndarray((np.shape(FLAIR_image)[0],image_rows_Dataset, image_cols_Dataset), dtype=np.float32)
    imgs_two_channels = np.ndarray((num_selected_slice, rows_standard, cols_standard, channel_num), dtype=np.float32)
    
    FLAIR_image_suitable = np.ndarray((num_selected_slice, rows_standard, cols_standard), dtype=np.float32)
    T1_image_suitable = np.ndarray((num_selected_slice, rows_standard, cols_standard), dtype=np.float32)
    brain_label = np.ndarray((np.shape(FLAIR_image)[0],image_rows_Dataset, image_cols_Dataset), dtype=np.float32)
    imgs_mask_two_channels = np.ndarray((num_selected_slice, rows_standard, cols_standard), dtype=np.float32)

    # FLAIR --------------------------------------------
    brain_mask_FLAIR[FLAIR_image >=thresh_FLAIR] = 1
    brain_mask_FLAIR[FLAIR_image < thresh_FLAIR] = 0
    for iii in range(np.shape(FLAIR_image)[0]):
  
        brain_mask_FLAIR[iii,:,:] = scipy.ndimage.morphology.binary_fill_holes(brain_mask_FLAIR[iii,:,:])  #fill the holes inside brain
        #------Gaussion Normalization
    FLAIR_image -=np.mean(FLAIR_image[brain_mask_FLAIR == 1])      #Gaussion Normalization
    FLAIR_image /=np.std(FLAIR_image[brain_mask_FLAIR == 1])
    FLAIR_image_suitable[...] = np.min(FLAIR_image)
    FLAIR_image_suitable[:, :, int(cols_standard/2-image_cols_Dataset/2):int(cols_standard/2+image_cols_Dataset/2)] = FLAIR_image[:, start_cut:start_cut+rows_standard, :]
   
    # T1 -----------------------------------------------
    brain_mask_T1[T1_image >=thresh_T1] = 1
    brain_mask_T1[T1_image < thresh_T1] = 0
    for iii in range(np.shape(T1_image)[0]):
 
        brain_mask_T1[iii,:,:] = scipy.ndimage.morphology.binary_fill_holes(brain_mask_T1[iii,:,:])  #fill the holes inside brain
        #------Gaussion Normalization
    T1_image -=np.mean(T1_image[brain_mask_T1 == 1])      #Gaussion Normalization
    T1_image /=np.std(T1_image[brain_mask_T1 == 1])
    T1_image_suitable[...] = np.min(T1_image)
    T1_image_suitable[:, :, int((cols_standard-image_cols_Dataset)/2):int((cols_standard+image_cols_Dataset)/2)] = T1_image[:, start_cut:start_cut+rows_standard, :]
    # lable----------------
    brain_label[labelArray == 1] = 1
    brain_label[labelArray != 1] = 0
    imgs_mask_two_channels[:, :, int((cols_standard-image_cols_Dataset)/2):int((cols_standard+image_cols_Dataset)/2)] = brain_label[:, start_cut:start_cut+rows_standard, :]
    
    #---------------------------------------------------
    FLAIR_image_suitable  = FLAIR_image_suitable[..., np.newaxis]
    T1_image_suitable  = T1_image_suitable[..., np.newaxis]
    
    imgs_two_channels = np.concatenate((FLAIR_image_suitable, T1_image_suitable), axis = 3)
    maskArray = imgs_mask_two_channels > 0
    return imgs_two_channels,imgs_mask_two_channels,maskArray


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def apply_transform(x, transform_matrix, channel_index=2, fill_mode='nearest', cval=0.):
    x = np.rollaxis(x, channel_index, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [scipy.ndimage.interpolation.affine_transform(x_channel, final_affine_matrix,
                      final_offset, order=0, mode=fill_mode, cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_index+1)
    return x


def augmentation(x_0, x_1, y):
    theta = (np.random.uniform(-15, 15) * np.pi) / 180.
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])
    shear = np.random.uniform(-.1, .1)
    shear_matrix = np.array([[1, -np.sin(shear), 0],
                             [0, np.cos(shear), 0],
                             [0, 0, 1]])
    zx, zy = np.random.uniform(.9, 1.1, 2)
    zoom_matrix = np.array([[zx, 0, 0],
                            [0, zy, 0],
                            [0, 0, 1]])
    augmentation_matrix = np.dot(np.dot(rotation_matrix, shear_matrix), zoom_matrix)
    transform_matrix = transform_matrix_offset_center(augmentation_matrix, x_0.shape[0], x_0.shape[1])
    x_0 = apply_transform(x_0[..., np.newaxis], transform_matrix)
    x_1 = apply_transform(x_1[..., np.newaxis], transform_matrix)
    y = apply_transform(y[..., np.newaxis], transform_matrix)
    return x_0[..., 0], x_1[..., 0], y[..., 0]

patient_num =45
patient_count = 0
rows_standard = 200
cols_standard = 200
thresh_FLAIR = 70      #to mask the brain
thresh_T1 = 30
para_array = [[0.958, 0.958, 3], [1.00, 1.00, 3], [1.20, 0.977, 3]]    # parameters of the scanner
para_array = np.array(para_array, dtype=np.float32)

images = None # shape: (numImages, z, y, x, channels=1)
labels = None
masks  = None

type_data = str(input("validation or training dataset: "))
images_l = []
lables_l = []

#read the dirs of test data 
input_dir_1 = '../data/' + type_data + '/Utrecht'
input_dir_2 = '../data/' + type_data + '/Singapore'
input_dir_3 = '../data/' + type_data + '/Amsterdam'
  
#-------------------------------------------   
dirs = os.listdir(input_dir_1) + os.listdir(input_dir_2) + os.listdir(input_dir_3)
# #All the slices and the corresponding patients id
# imgs_three_datasets_two_channels = np.load('imgs_three_datasets_two_channels.npy')
# imgs_mask_three_datasets_two_channels = np.load('imgs_mask_three_datasets_two_channels.npy')
# slices_patient_id_label = np.load('slices_patient_id_label.npy')
dirs = [f for f in dirs if not f.startswith('.')] # macbook
for dir_name in dirs:
    
    if patient_count < len(os.listdir(input_dir_1)):
        inputDir = input_dir_1
    elif patient_count < len(os.listdir(input_dir_1))+len(os.listdir(input_dir_2)):
        inputDir = input_dir_2
    elif patient_count >= len(os.listdir(input_dir_1))+len(os.listdir(input_dir_2)):
        inputDir = input_dir_3
    FLAIR_image = sitk.ReadImage(os.path.join(inputDir, dir_name, 'pre', 'FLAIR.nii.gz'))
    T1_image = sitk.ReadImage(os.path.join(inputDir, dir_name, 'pre', 'T1.nii.gz'))
    label_image= sitk.ReadImage(os.path.join(inputDir, dir_name, "wmh.nii.gz"))
    FLAIR_array = sitk.GetArrayFromImage(FLAIR_image)
    T1_array = sitk.GetArrayFromImage(T1_image)
    labelArray = sitk.GetArrayFromImage(label_image)
    
    #Proccess testing data-----
    para_FLAIR = np.ndarray((1,3), dtype=np.float32)
    para_FLAIR_ = FLAIR_image.GetSpacing()
    para_FLAIR[0,0] = round(para_FLAIR_[0],3)   # get spacing parameters of the data
    para_FLAIR[0,1] = round(para_FLAIR_[1],3)  
    para_FLAIR[0,2] = round(para_FLAIR_[2],3) 
    if np.array_equal(para_FLAIR[0], para_array[0]) :
        imgs_test,label,maskArray = Utrecht_preprocessing(FLAIR_array, T1_array, labelArray)
    elif np.array_equal(para_FLAIR[0], para_array[1]):
        imgs_test,label,maskArray  = Utrecht_preprocessing(FLAIR_array, T1_array, labelArray)
    elif np.array_equal(para_FLAIR[0], para_array[2]):
        imgs_test,label,maskArray  = GE3T_preprocessing(FLAIR_array, T1_array, labelArray)
    patient_count+=1
    
           
    # Add to the images/labels array
    images = imgs_test.reshape([1] + list(imgs_test.shape) )
    labels = label.reshape([1] + list(label.shape) + [1])
    masks  = maskArray.reshape([1] + list(maskArray.shape) + [1])
    for i in range(masks.shape[0]):
        for j in range(masks.shape[1]):
            if not np.all(masks[i,j,:,:,0]== False):
                lables_l.append(labels[i,j,:,:,:])
#                 print("max:",images[i,j,:,:,:].max())
                images_l.append(images[i,j,:,:,:])



aug_nr = int(input("augmentation times(0-10)?"))
if aug_nr > 0:
    images = np.load('trainimages.npy')
    masks = np.load('trainlables.npy')
    masks[masks >1] = 0

    for ii in range(aug_nr):
        images_aug = np.zeros(images.shape, dtype=np.float32)
        masks_aug = np.zeros(masks.shape, dtype=np.float32)
        for i in range(images.shape[0]):
            images_aug[i, ..., 0], images_aug[i, ..., 1], masks_aug[i, ..., 0] = augmentation(images[i, ..., 0], images[i, ..., 1], masks[i, ..., 0])
        image = np.concatenate((images, images_aug), axis=0)
        mask = np.concatenate((masks, masks_aug), axis=0)
    print(type_data,"_ augmentation_",aug_nr,":", image.shape,mask.shape)
    np.save(type_data +'_aug_' + str(aug_nr) + 'lables.npy', mask)    
    np.save(type_data +'_aug_' + str(aug_nr) + 'images.npy', image) 

else:
    print(type_data,":", np.asarray(lables_l).shape,np.asarray(images_l).shape)
    np.save(type_data +'lables.npy', np.asarray(lables_l))    
    np.save(type_data +'images.npy', np.asarray(images_l)) 