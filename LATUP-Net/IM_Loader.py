import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import os

# Set the base directory dynamically
base_directory = os.getcwd()

# Define the dataset path dynamically
DATA_PATH = os.path.join(base_directory, 'Results', 'data2020')  # Modify 'data2020' as needed

def load_img_mask(img_mask_list):
    images = []
    masks = []

    for i, image_name in enumerate(img_mask_list):
        image_path = os.path.join(DATA_PATH, image_name, 'image_*.npy')
        mask_path = os.path.join(DATA_PATH, image_name, 'mask_*.npy')

        try:
            image = np.load(glob.glob(image_path)[0])
            mask = np.load(glob.glob(mask_path)[0])

            mask = to_categorical(mask, num_classes=4)
            mask = mask.astype(np.float64)

            masks.append(mask)
            images.append(image)

        except Exception as e:
            print('Error: ', e)
            print('List: ', image_name, os.listdir(os.path.join(DATA_PATH, image_name)))
            print('Shapes: ', mask.shape, image.shape)

    images = tf.convert_to_tensor(np.array(images))
    masks = tf.convert_to_tensor(np.array(masks))

    return images, masks

def imageLoader(images_names, batch_size):
    L = len(images_names)
    while True:
        batch_start = 0
        batch_end = batch_size

        while batch_start < L:
            limit = min(batch_end, L)
            X, Y = load_img_mask(images_names[batch_start:limit])
            yield (X, Y)  # a tuple with two numpy arrays with batch_size samples     

            batch_start += batch_size   
            batch_end += batch_size