
import numpy as np
import pickle
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import copy
import cv2
import numpy as np
from IM_Loader import *
from LATUP_Net import *

# GradCAM 
def make_gradcam_heatmap(img_array, model, layer_name, class_idx):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[0, :, :, :, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()
def make_gradcam_heatmapmine(img_array, model, layer_name, class_idx):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[0, :, :, :, class_idx]

    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]
    gate_f = tf.cast(output > 0, 'float32')
    gate_r = tf.cast(grads > 0, 'float32')
    guided_grads = tf.cast(output > 0, 'float32') * tf.cast(grads > 0, 'float32') * grads

    weights = tf.reduce_mean(guided_grads, axis=(0, 1,2))
   
    cam = np.ones(output.shape[0: 3], dtype = np.float32)

    for i, w in enumerate(weights):
        
        cam += w * output[:, :, :, i]
    heatmap=cam
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap

def visualise_attention_per_model(image,mask,prediction,attention_heatmaps,layer_index):
    fig, axs = plt.subplots(3, len(attention_heatmaps[0])+3, figsize=(14,10))
    model_names=['Attention', 'No Attention']
    class_names=['Background','(NCR/NET)','(ED)','(ET)']
    for k in range(0,2):
      heatmap_3clsses=np.zeros(attention_heatmaps[k][0][layer_index].shape[0:2])
      for class_idx in range(len(attention_heatmaps[0])):
         # selected_slice = int(attention_heatmaps[k][class_idx][layer_index].shape[2]/2)
          selected_slice = 80
          # First display the image slice
          axs[k,class_idx].imshow(image[0, :, :, selected_slice, 0], cmap='gray')
          # Then overlay the heatmap, using a suitable alpha
          axs[k,class_idx].imshow(attention_heatmaps[k][class_idx][layer_index][:, :, selected_slice], cmap='hot', alpha=0.5)
          axs[k,class_idx].set_title(f'{class_names[class_idx]}')
          heatmap_3clsses=heatmap_3clsses+attention_heatmaps[k][class_idx][layer_index][:, :, selected_slice]
      axs[k,class_idx+1].imshow(mask[:, :, selected_slice])
      axs[k,class_idx+1].set_title('Ground Truth')
      axs[k,class_idx+2].imshow(prediction[k][:, :, selected_slice])
      axs[k,class_idx+2].set_title('Prediction')
      axs[k,class_idx+3].imshow(heatmap_3clsses)
      axs[k,class_idx+3].set_title(model_names[k])
    plt.show()


def get_heatmap_layers(image,my_model,list_layers,class_idx):
  features_layer=[]
  for i, layer_name in enumerate(list_layers):
      heatmap = make_gradcam_heatmapmine(image, my_model, layer_name, class_idx)
      features_layer.append(heatmap)
  return features_layer
from matplotlib import colors
import numpy as np
import matplotlib.pyplot as plt

def visualise_attention_per_model(image, mask, prediction, attention_heatmaps, layer_index):
    # Change here: 2 rows instead of 3
    fig, axs = plt.subplots(2, len(attention_heatmaps[0])+3, figsize=(14, 10))

    model_names = ['Attention', 'No Attention']
    class_names = ['Background', '(NCR/NET)', '(ED)', '(ET)']

    # Custom colormap
    cmap = colors.ListedColormap(['blue', 'green', 'red'])
    norm = colors.BoundaryNorm([0.5, 1.5, 2.5, 4.5], cmap.N)

    for k in range(2):
        heatmap_3classes = np.zeros(attention_heatmaps[k][0][layer_index].shape[:2])
        for class_idx in range(len(attention_heatmaps[0])):
            selected_slice = 80
            # First display the image slice
            axs[k, class_idx].imshow(image[0, :, :, selected_slice, 0], cmap='gray')
            # Then overlay the heatmap
            axs[k, class_idx].imshow(attention_heatmaps[k][class_idx][layer_index][:, :, selected_slice], cmap='hot', alpha=0.5)
            axs[k, class_idx].set_title(f'{class_names[class_idx]}')

            heatmap_3classes += attention_heatmaps[k][class_idx][layer_index][:, :, selected_slice]

        # Adjustments for the indexing of Ground Truth and Prediction plots
        axs[k, len(attention_heatmaps[0])].imshow(image[0, :, :, selected_slice, 0], cmap='gray')
        masked_mask = np.ma.masked_where(mask[:, :, selected_slice] == 0, mask[:, :, selected_slice])
        axs[k, len(attention_heatmaps[0])].imshow(masked_mask, cmap='jet', alpha=0.5)
        axs[k, len(attention_heatmaps[0])].set_title('Ground Truth')

        axs[k, len(attention_heatmaps[0])+1].imshow(image[0, :, :, selected_slice, 0], cmap='gray')
        masked_prediction = np.ma.masked_where(prediction[k][:, :, selected_slice] == 0, prediction[k][:, :, selected_slice])
        axs[k, len(attention_heatmaps[0])+1].imshow(masked_prediction, cmap='jet', alpha=0.5)
        axs[k, len(attention_heatmaps[0])+1].set_title('Prediction')

        # Plotting the combined heatmap
        axs[k, len(attention_heatmaps[0])+2].imshow(heatmap_3classes, cmap='hot', alpha=0.5)
        axs[k, len(attention_heatmaps[0])+2].set_title(model_names[k])

    plt.tight_layout()
    plt.show()


#How to use and plot GradCam heatmaps 
# load my_model and my_modelno  
# my_model=CNN_Model(.....)
# load my_mdoel weights
# my_modelno=CNN_Model_no_attention(....) 
#load my_mdoelno weights
# images_path = "path/to/image/directory"    
#visualise_heatmaps(image_names, index_of_image_in_image_names, images_path, my_model, my_modelno)



 # Confusion Matrics
import numpy as np
import os
import glob
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import load_model

def plot_confusion_matrix(y_true, y_pred):
    y_true = np.reshape(y_true, -1)
    y_pred = np.reshape(y_pred, -1)
    cm = confusion_matrix(y_true, y_pred)
    return cm
    
def get_con_matrix_all_sklrn(best_model, images_dir_path, class_names=['Background', '(NCR/NET)', '(ED)', '(ET)']):
    et_class_index = class_names.index('(ET)')
    aggregated_cm = np.zeros([len(class_names), len(class_names)])
    conf_matrices_all = []
    image_names_list = os.listdir(images_dir_path)

    for dir_name in image_names_list:
        image_files = glob.glob(os.path.join(images_dir_path, dir_name, 'image_*.npy'))
        mask_files = glob.glob(os.path.join(images_dir_path, dir_name, 'mask_*.npy'))

        if not image_files or not mask_files:
            continue

        image = np.load(image_files[0])
        mask = np.load(mask_files[0])
        image = np.expand_dims(image, 0)
        prediction = best_model.predict(image)
        prediction_mask = np.argmax(prediction, axis=-1)[0, :, :, :]

        cm = plot_confusion_matrix(mask, prediction_mask)

        # Include this matrix only if 'ET' class is present in ground truth or predictions
        if et_class_index in np.unique(mask):
            conf_matrices_all.append(cm)
            aggregated_cm += cm

    return aggregated_cm, conf_matrices_all

def calculate_mean_std(matrices):
    normalized_matrices = []

    for cm in matrices:
        row_sums = cm.sum(axis=1)[:, np.newaxis]
        normalized_cm = cm.astype('float') / (row_sums + 1e-10)
        normalized_matrices.append(normalized_cm)

    normalized_matrices_array = np.array(normalized_matrices)
    mean_matrix = np.mean(normalized_matrices_array, axis=0)
    std_matrix = np.std(normalized_matrices_array, axis=0)

    return mean_matrix, std_matrix


# How to use and plot  confusion Matrics 
# my_model = ... # Initialize your model here
# images_path = "path/to/image/directory"
# cf_matrix_sklrn, all_conf_mat = get_con_matrix_all_sklrn(my_model, images_path)

#How to Plot
# Compute the mean and std from individual normalized matrices
mn, std = calculate_mean_std(all_conf_mat)

# Normalize the mean and std matrices
row_sums_mean = mn.sum(axis=1)[:, np.newaxis]
mn_normalized = mn.astype('float') / (row_sums_mean + 1e-10)

row_sums_std = std.sum(axis=1)[:, np.newaxis]
std_normalized = std.astype('float') / (row_sums_std + 1e-10)

# Plotting
class_names = ['Background', '(NCR/NET)', '(ED)', '(ET)']
fig, ax = plt.subplots(figsize=(len(class_names) * 1.3, len(class_names) * 1.3))
sns.heatmap(mn_normalized, annot=True, fmt=".2f", cmap="RdBu_r", xticklabels=class_names, yticklabels=class_names)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

fig, ax = plt.subplots(figsize=(len(class_names) * 1.3, len(class_names) * 1.3))
sns.heatmap(std_normalized, annot=True, fmt=".2f", cmap="RdBu_r", xticklabels=class_names, yticklabels=class_names)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()