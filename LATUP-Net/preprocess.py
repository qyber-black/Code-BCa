#get data ready
#1-Combine 
#2-Changing mask pixel values (labels) from 4 to 3 (as the original labels are 0, 1, 2, 4)
#3-Visualize

import numpy as np
import nibabel as nib
import os
import glob
from sklearn.preprocessing import MinMaxScaler

# Set the base directory (you can modify this to point to your dataset's location)
base_directory = os.getcwd()

# Define the dataset path dynamically
TRAIN_DATASET_PATH = os.path.join(base_directory, 'Results', 'data2020')

# Initialize a MinMaxScaler for normalization
scaler = MinMaxScaler()

# Use glob to find files using relative paths
t2_list = sorted(glob.glob(os.path.join(TRAIN_DATASET_PATH, '*/*t2.nii')))
t1ce_list = sorted(glob.glob(os.path.join(TRAIN_DATASET_PATH, '*/*t1ce.nii')))
flair_list = sorted(glob.glob(os.path.join(TRAIN_DATASET_PATH, '*/*flair.nii')))
mask_list = sorted(glob.glob(os.path.join(TRAIN_DATASET_PATH, '*/*seg.nii')))

# Process each image in the dataset
for img in range(len(t2_list)):
    print("Now preparing image and masks number: ", img)

    # Load and scale T2, T1ce, Flair images
    temp_image_t2 = nib.load(t2_list[img]).get_fdata()
    temp_image_t2 = scaler.fit_transform(temp_image_t2.reshape(-1, temp_image_t2.shape[-1])).reshape(temp_image_t2.shape)

    temp_image_t1ce = nib.load(t1ce_list[img]).get_fdata()
    temp_image_t1ce = scaler.fit_transform(temp_image_t1ce.reshape(-1, temp_image_t1ce.shape[-1])).reshape(temp_image_t1ce.shape)

    temp_image_flair = nib.load(flair_list[img]).get_fdata()
    temp_image_flair = scaler.fit_transform(temp_image_flair.reshape(-1, temp_image_flair.shape[-1])).reshape(temp_image_flair.shape)

    # Load the mask and change pixel values from 4 to 3
    temp_mask = nib.load(mask_list[img]).get_fdata()
    temp_mask[temp_mask == 4] = 3

    # Combine the processed images into a single array
    temp_combined_images = np.stack([temp_image_flair, temp_image_t1ce, temp_image_t2], axis=3)

    # Crop the images and masks to (128, 128, 128)
    temp_combined_images = temp_combined_images[56:184, 56:184, 13:141]
    temp_mask = temp_mask[56:184, 56:184, 13:141]

    # Check if the mask contains enough non-background information
    val, counts = np.unique(temp_mask, return_counts=True)
    if (1 - (counts[0] / counts.sum())) > 0.01:  # Check if non-background is more than 1%
        print("Save Me")

        # Create directories and save the processed images and masks
        train_dir = os.path.join(TRAIN_DATASET_PATH, f'train{img}')
        os.makedirs(train_dir, exist_ok=True)
        np.save(os.path.join(train_dir, f'image_{img}.npy'), temp_combined_images)
        np.save(os.path.join(train_dir, f'mask_{img}.npy'), temp_mask)
    else:
        print("I am useless")

#  KFold split as saved pickle file for data split 

import os
import pickle
from sklearn.model_selection import KFold

# Set the base directory (you can modify this to point to your dataset's location)
base_directory = os.getcwd()

# Define the dataset path dynamically
DATA_PATH = os.path.join(base_directory,'Results','data2020')  # Modify  as needed

# Initialize KFold
kf = KFold(n_splits=5, random_state=1, shuffle=True)

# Get the list of images from the dynamic path
train_img_list = os.listdir(DATA_PATH)

# Perform KFold splitting
Kfolds = kf.split(train_img_list, train_img_list)
train_5fold = []
valid_5fold = []
nb_fold = 0

for train_idx, val_idx in Kfolds:
    print('Training for fold ' + str(nb_fold) + ' started...')

    train_img_fold = [train_img_list[k] for k in list(train_idx)]
    valid_img_fold = [train_img_list[k] for k in list(val_idx)]

    # Save the lists to files
    with open('valid_list' + str(nb_fold) + '.pkl', "wb") as list_valid:
        pickle.dump(valid_img_fold, list_valid)

    with open('train_list' + str(nb_fold) + '.pkl', "wb") as list_train:
        pickle.dump(train_img_fold, list_train)

    print(len(train_img_fold), len(valid_img_fold))
    train_5fold.append(train_img_fold)
    valid_5fold.append(valid_img_fold)
    nb_fold += 1

# Save the 5-fold data dictionary
five_fold_dic = dict(train=train_5fold, validation=valid_5fold)
with open('folds_dic.pkl', "wb") as list_train:
    pickle.dump(five_fold_dic, list_train)

# Load the split
with open('folds_dic.pkl', 'rb') as m:
    folds_dict = pickle.load(m)

valid_img_fold = folds_dict['train'][0]
train_img_fold = folds_dict['validation'][0]

print(valid_img_fold)
print(train_img_fold)
