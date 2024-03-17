#training
import os
import keras.backend as K
import tensorflow as tf
import glob
import pickle
import pandas as pd 
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import KFold # import KFold
import time
from IM_Loader import *
from Loss import dice_coef_loss_3classes
from LATUP_Net import CNN_Model

# import evaluation matrics from segmentation_models_3D and set the optimizer, and the learning rate for training 

from segmentation_models_3D.metrics import IOUScore, FScore
#from segmentation_models_3D.losses import DiceLoss
#dice_loss = DiceLoss()
FScores = FScore()
IOUScores = IOUScore(threshold=0.5)
LR=0.0001
metrics = ['accuracy',FScores,IOUScores]
optim = tf.keras.optimizers.Adam(LR)

# Set the base directory dynamically
base_directory = os.getcwd()

# Define the dataset path dynamically
DATA_PATH = os.path.join(base_directory, 'Results', 'data2020')  # Modify  as needed

# Define the checkpoint directory
CHECKPOINT_DIR = os.path.join(base_directory, 'Results','2020')  # Modify as needed

# Load histories
with open('folds_dic.pkl', 'rb') as m:
      folds_dict = pickle.load(m)

kf = KFold(n_splits=5, random_state=1, shuffle=True)
train_img_list = os.listdir(DATA_PATH)
Kfolds = kf.split(train_img_list, train_img_list)
trainlist = []
validationlist = []
Histories = []
batch_size = 1

for nb_fold in range(0, 5):
    print('Training for the fold ' + str(nb_fold) + ' started ...')

    train_img_fold = folds_dict['train'][nb_fold]
    valid_img_fold = folds_dict['validation'][nb_fold]
    steps_per_epoch = len(train_img_fold) // batch_size
    val_steps_per_epoch = len(valid_img_fold) // batch_size
    train_img_datagen1 = imageLoader(train_img_fold, batch_size)
    val_img_datagen1 = imageLoader(valid_img_fold, batch_size)
    LR = 0.0001
    metrics = ['accuracy', FScores, IOUScores]

    optim = tf.keras.optimizers.Adam(LR)
    model = CNN_Model(IMG_HEIGHT=128, IMG_WIDTH=128, IMG_DEPTH=128, IMG_CHANNELS=3, num_classes=4)

    print(model.summary())
    checkpoint_filepath = os.path.join(CHECKPOINT_DIR, '2020fold_' + str(nb_fold) + '-{epoch:03d}-{val_f1-score:.04f}.hdf5')

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                                   monitor='val_f1-score', verbose=1,
                                                                   save_best_only=True, mode='max')

    model.compile(optimizer=optim, loss=dice_coef_loss_3classes, metrics=metrics)

    start_time = time.time()
    model_history = model.fit(train_img_datagen1,
                              steps_per_epoch=steps_per_epoch,
                              epochs=200,
                              verbose=1,
                              validation_data=val_img_datagen1,
                              validation_steps=val_steps_per_epoch,
                              callbacks=[model_checkpoint_callback])

    end_time = time.time()
    training_time = end_time - start_time
    print("Training time: ", training_time, "seconds")
    model.save(os.path.join(DATA_PATH, 'last_model_fold' + str(nb_fold) + '.hdf5'))

    history_df = pd.DataFrame(model_history.history)
    with open(os.path.join(DATA_PATH, 'fold_' + str(nb_fold) + '_history.csv'), mode='w') as f:
        history_df.to_csv(f)

    with open(os.path.join(DATA_PATH, 'model_history_fold_' + str(nb_fold) + '.pkl'), 'wb') as f:
        pickle.dump(model_history.history, f)

    Histories.append(model_history.history)

# Save all histories
with open('all_models_history.pkl', 'wb') as f:
    pickle.dump(Histories, f)