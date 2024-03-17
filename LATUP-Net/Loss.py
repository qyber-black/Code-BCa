import os
import numpy as np
import glob
import keras
import tensorflow as tf
import keras.backend as K

# compute the class weights according to the ENet paper:
print("computing class weights")

base_directory = os.getcwd()
images_path = os.path.join(base_directory, 'Results', 'data2020')
image_names = os.listdir(images_path)

for i, dir_name in enumerate(image_names):
    print('*********', i, dir_name)

def get_class_weights(images_dir_path, class_names=['backg', '1', '2', '3']):
    num_classes = len(class_names)
    trainId_to_count = [0] * num_classes
    image_names_list = os.listdir(images_dir_path)
    for i, dir_name in enumerate(image_names_list):
        print('*********', i, dir_name)
        mask_path = glob.glob(os.path.join(images_dir_path, dir_name, 'mask_*.npy'))[0]
        label_img = np.load(mask_path)
        for trainId in range(num_classes):
            trainId_mask = np.equal(label_img, trainId)
            trainId_to_count[trainId] += np.sum(trainId_mask)
        if i % 10 == 0:
            print(i + 1, '- patients so far', trainId_to_count)
# compute the class weights according to the ENet paper:
    trainId_to_count[1]=trainId_to_count0[1]+trainId_to_count0[2]+trainId_to_count0[3]
    trainId_to_count[2]=trainId_to_count0[1]+trainId_to_count0[3]
    trainId_to_count[3]=trainId_to_count0[3]
    class_weights = []
    total_count = sum(trainId_to_count[1:])
    for trainId, count in enumerate(trainId_to_count[1:]):
        trainId_prob = float(count)/float(total_count)
        trainId_weight = 1/np.log(1.22 + trainId_prob)
        class_weights.append(trainId_weight)
    s=sum(class_weights)
    for idx, w in enumerate(class_weights):
        class_weights[idx]=class_weights[idx]/s
    data_info=dict(class_seg=['WT','CT','ET'],pixel_count=trainId_to_count[1:],class_weights=class_weights)
    return data_info
images_dir_path = os.path.join(base_directory,'Results', 'data2020')
data_pixels_weights_info = get_class_weights(images_dir_path, class_names=['backg', '1', '2', '3'])
print(data_pixels_weights_info)

#Define the weighted dice loss 

def dice_coef_class(y_true, y_pred,i, epsilon=0.00001):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf

    """
    axis = (0,1,2,3)
    dice_numerator = 2. * K.sum(y_true[:,:,:,:,i:i+1] * y_pred[:,:,:,:,i:i+1], axis=axis) + epsilon
    dice_denominator = K.sum(y_true[:,:,:,:,i:i+1]*y_true[:,:,:,:,i:i+1], axis=axis) + K.sum(y_pred[:,:,:,:,i:i+1]*y_pred[:,:,:,:,i:i+1], axis=axis) + epsilon
    return K.mean((dice_numerator)/(dice_denominator))


def dice_coef_loss_3classes(y_true, y_pred):
    core=2-dice_coef_class(y_true, y_pred,1) -dice_coef_class(y_true, y_pred,3)
    whole=3-dice_coef_class(y_true, y_pred,1)-dice_coef_class(y_true, y_pred,2) -dice_coef_class(y_true, y_pred,3)
    enhance=1-dice_coef_class(y_true, y_pred,3)
    return 0.6*core + 0.5*whole +0.7*enhance





