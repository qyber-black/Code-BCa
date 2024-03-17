#Build the model
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Reshape, Dense, Conv3D, BatchNormalization, UpSampling3D,Concatenate,Activation,Multiply, AveragePooling3D, MaxPooling3D, concatenate, GlobalAveragePooling3D, GlobalMaxPooling3D, Conv3DTranspose, BatchNormalization, Dropout, Lambda
from keras import regularizers
from tensorflow.keras.optimizers import Adam
from keras.metrics import MeanIoU
Lrelu = tf.keras.layers.LeakyReLU(alpha=0.1)
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.python.keras.layers import Dropout, SpatialDropout3D
from Att_Module import SqueezeAndExcitation  
# Attention Model


def CNN_Model(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS, num_classes):
    kernel_initializer = 'he_uniform'
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS), name='input_layer')
    s = inputs

    # Initial layers
    conv = Conv3D(32, (3, 3, 3), activation=Lrelu, kernel_initializer=kernel_initializer, padding='same', name='layer_1')(s)
    conv1 = Conv3D(32, (1, 1, 1), activation=Lrelu, kernel_initializer=kernel_initializer, padding='same', name='layer_2')(conv)
    pool1 = MaxPooling3D((2, 2, 2), name='layer_3_maxpool')(conv1)
    conv2 = Conv3D(32, (3, 3, 3), activation=Lrelu, kernel_initializer=kernel_initializer, padding='same', name='layer_4')(conv)
    pool2 = MaxPooling3D((2, 2, 2), name='layer_5_maxpool')(conv2)
    conv3 = Conv3D(32, (5, 5, 5), activation=Lrelu, kernel_initializer=kernel_initializer, padding='same', name='layer_6')(conv)
    pool3 = MaxPooling3D((2, 2, 2), name='layer_7_maxpool')(conv3)
    layer_out = concatenate([pool1, pool2, pool3], axis=-1, name='layer_8_concatenate')
    attention_layer1 = SqueezeAndExcitation(layer_out, name='attention_layer1')

    conv4 = Conv3D(64, (3, 3, 3), activation=Lrelu, kernel_initializer=kernel_initializer, kernel_regularizer=regularizers.l2(0.02), padding='same', name='layer_9')(attention_layer1)
    B4 = InstanceNormalization(axis=-1, name='layer_10_instance_norm')(conv4)
    conv4 = Conv3D(64, (3, 3, 3), activation=Lrelu, kernel_initializer=kernel_initializer, kernel_regularizer=regularizers.l2(0.02), padding='same', name='layer_11')(B4)
    drop2 = Dropout(0.2, name='layer_12_dropout')(conv4)
    pool4 = MaxPooling3D((2, 2, 2), name='layer_13_maxpool')(drop2)

    # Following layers
    conv5 = Conv3D(128, (3, 3, 3), activation=Lrelu, kernel_initializer=kernel_initializer, kernel_regularizer=regularizers.l2(0.02), padding='same', name='layer_14_conv')(pool4)
    B5 = InstanceNormalization(axis=-1, name='layer_15_instance_norm')(conv5)
    conv5 = Conv3D(128, (3, 3, 3), activation=Lrelu, kernel_initializer=kernel_initializer, kernel_regularizer=regularizers.l2(0.02), padding='same', name='layer_16_conv')(B5)
    drop5 = Dropout(0.2, name='layer_17_dropout')(conv5)
    pool5 = MaxPooling3D((2, 2, 2), name='layer_18_maxpool')(drop5)
    attention_layer2 = SqueezeAndExcitation(pool5, name='attention_layer2')

    # Upsampling layers
    u9 = Conv3D(128, (2, 2, 2), activation=Lrelu, kernel_initializer=kernel_initializer, padding='same', name='upsample_layer_1_conv')(UpSampling3D(size=(2,2,2))(attention_layer2))
    u9 = InstanceNormalization(axis=-1, name='upsample_layer_1_instance_norm')(u9)
    u9 = concatenate([u9, conv5], name='upsample_layer_1_concatenate')
    c9 = Conv3D(128, (3, 3, 3), activation=Lrelu, kernel_initializer=kernel_initializer,
                kernel_regularizer=regularizers.l2(0.02), padding='same', name='upsample_layer_1_conv1')(u9)
    c9 = SqueezeAndExcitation(c9, name='attention_layer3')
    c9 = Dropout(0.2, name='upsample_layer_1_dropout')(c9)
    c9 = Conv3D(128, (3, 3, 3), activation=Lrelu, kernel_initializer=kernel_initializer,
               kernel_regularizer=regularizers.l2(0.02), padding='same', name='upsample_layer_1_conv2')(c9)


    u10= Conv3D(64, (2, 2, 2), activation=Lrelu, kernel_initializer=kernel_initializer, padding='same', name='upsample_layer_2_conv')(UpSampling3D(size =(2,2,2))(c9))
    u10 = InstanceNormalization(axis=-1, name='upsample_layer_2_instance_norm')(u10)
    u10 = concatenate([u10, conv4], name='upsample_layer_2_concatenate')
    c10= Conv3D(64, (3, 3, 3), activation=Lrelu, kernel_initializer=kernel_initializer,
              kernel_regularizer=regularizers.l2(0.02), padding='same', name='upsample_layer_2_conv1')(u10)
    c10 = SqueezeAndExcitation(c10, name='attention_layer4')
    c10 = Dropout(0.2, name='upsample_layer_2_dropout')(c10)
    c10= Conv3D(64, (3, 3, 3), activation=Lrelu, kernel_initializer=kernel_initializer,
                 kernel_regularizer=regularizers.l2(0.02), padding='same', name='upsample_layer_2_conv2')(c10)

    u11= Conv3D(32, (2, 2, 2), activation=Lrelu, kernel_initializer=kernel_initializer, padding='same', name='upsample_layer_3_conv')(UpSampling3D(size =(2,2,2))(c10))
    u11 = InstanceNormalization(axis=-1, name='upsample_layer_3_instance_norm')(u11)
    u11 = concatenate([u11,  conv], name='upsample_layer_3_concatenate')
    c11= Conv3D(32, (3, 3, 3), activation=Lrelu, kernel_initializer=kernel_initializer,
             kernel_regularizer=regularizers.l2(0.02), padding='same', name='upsample_layer_3_conv1')(u11)
    c11 = SqueezeAndExcitation(c11, name='attention_layer5')
    c11 = Dropout(0.2, name='upsample_layer_3_dropout')(c11)
    c11= Conv3D(32, (3, 3, 3), activation=Lrelu, kernel_initializer=kernel_initializer,
                 kernel_regularizer=regularizers.l2(0.02), padding='same', name='upsample_layer_3_conv2')(c11)

    outputs = Conv3D(num_classes, (1, 1, 1), kernel_regularizer=regularizers.l2(0.02), activation='softmax', name='final_output')(c11)

    model = Model(inputs=[inputs], outputs=[outputs])

    return model

#Test if everything is working ok.
model = CNN_Model(128, 128, 128, 3,4)

model.summary()
print(model.input_shape)
print(model.output_shape)



# No Attention Model
Lrelu = tf.keras.layers.LeakyReLU(alpha=0.1)
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.python.keras.layers import Dropout, SpatialDropout3D


def CNN_ModelNo(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS, num_classes):
    kernel_initializer = 'he_uniform'
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS), name='input_layer')
    s = inputs

    # Initial layers
    conv = Conv3D(32, (3, 3, 3), activation=Lrelu, kernel_initializer=kernel_initializer, padding='same', name='layer_1')(s)
    conv1 = Conv3D(32, (1, 1, 1), activation=Lrelu, kernel_initializer=kernel_initializer, padding='same', name='layer_2')(conv)
    pool1 = MaxPooling3D((2, 2, 2), name='layer_3_maxpool')(conv1)
    conv2 = Conv3D(32, (3, 3, 3), activation=Lrelu, kernel_initializer=kernel_initializer, padding='same', name='layer_4')(conv)
    pool2 = MaxPooling3D((2, 2, 2), name='layer_5_maxpool')(conv2)
    conv3 = Conv3D(32, (5, 5, 5), activation=Lrelu, kernel_initializer=kernel_initializer, padding='same', name='layer_6')(conv)
    pool3 = MaxPooling3D((2, 2, 2), name='layer_7_maxpool')(conv3)
    layer_out = concatenate([pool1, pool2, pool3], axis=-1, name='layer_8_concatenate')

    conv4 = Conv3D(64, (3, 3, 3), activation=Lrelu, kernel_initializer=kernel_initializer, kernel_regularizer=regularizers.l2(0.02), padding='same', name='layer_9')(layer_out)
    B4 = InstanceNormalization(axis=-1, name='layer_10_instance_norm')(conv4)
    conv4 = Conv3D(64, (3, 3, 3), activation=Lrelu, kernel_initializer=kernel_initializer, kernel_regularizer=regularizers.l2(0.02), padding='same', name='layer_11')(B4)
    drop2 = Dropout(0.2, name='layer_12_dropout')(conv4)
    pool4 = MaxPooling3D((2, 2, 2), name='layer_13_maxpool')(drop2)

    # Following layers
    conv5 = Conv3D(128, (3, 3, 3), activation=Lrelu, kernel_initializer=kernel_initializer, kernel_regularizer=regularizers.l2(0.02), padding='same', name='layer_14_conv')(pool4)
    B5 = InstanceNormalization(axis=-1, name='layer_15_instance_norm')(conv5)
    conv5 = Conv3D(128, (3, 3, 3), activation=Lrelu, kernel_initializer=kernel_initializer, kernel_regularizer=regularizers.l2(0.02), padding='same', name='layer_16_conv')(B5)
    drop5 = Dropout(0.2, name='layer_17_dropout')(conv5)
    pool5 = MaxPooling3D((2, 2, 2), name='layer_18_maxpool')(drop5)

    # Upsampling layers
    u9 = Conv3D(128, (2, 2, 2), activation=Lrelu, kernel_initializer=kernel_initializer, padding='same', name='upsample_layer_1_conv')(UpSampling3D(size=(2,2,2))(pool5))
    u9 = InstanceNormalization(axis=-1, name='upsample_layer_1_instance_norm')(u9)
    u9 = concatenate([u9, conv5], name='upsample_layer_1_concatenate')
    c9 = Conv3D(128, (3, 3, 3), activation=Lrelu, kernel_initializer=kernel_initializer,
                kernel_regularizer=regularizers.l2(0.02), padding='same', name='upsample_layer_1_conv1')(u9)
    c9 = Dropout(0.2, name='upsample_layer_1_dropout')(c9)
    c9 = Conv3D(128, (3, 3, 3), activation=Lrelu, kernel_initializer=kernel_initializer,
               kernel_regularizer=regularizers.l2(0.02), padding='same', name='upsample_layer_1_conv2')(c9)


    u10= Conv3D(64, (2, 2, 2), activation=Lrelu, kernel_initializer=kernel_initializer, padding='same', name='upsample_layer_2_conv')(UpSampling3D(size =(2,2,2))(c9))
    u10 = InstanceNormalization(axis=-1, name='upsample_layer_2_instance_norm')(u10)
    u10 = concatenate([u10, conv4], name='upsample_layer_2_concatenate')
    c10= Conv3D(64, (3, 3, 3), activation=Lrelu, kernel_initializer=kernel_initializer,
              kernel_regularizer=regularizers.l2(0.02), padding='same', name='upsample_layer_2_conv1')(u10)
    c10 = Dropout(0.2, name='upsample_layer_2_dropout')(c10)
    c10= Conv3D(64, (3, 3, 3), activation=Lrelu, kernel_initializer=kernel_initializer,
                 kernel_regularizer=regularizers.l2(0.02), padding='same', name='upsample_layer_2_conv2')(c10)

    u11= Conv3D(32, (2, 2, 2), activation=Lrelu, kernel_initializer=kernel_initializer, padding='same', name='upsample_layer_3_conv')(UpSampling3D(size =(2,2,2))(c10))
    u11 = InstanceNormalization(axis=-1, name='upsample_layer_3_instance_norm')(u11)
    u11 = concatenate([u11,  conv], name='upsample_layer_3_concatenate')
    c11= Conv3D(32, (3, 3, 3), activation=Lrelu, kernel_initializer=kernel_initializer,
             kernel_regularizer=regularizers.l2(0.02), padding='same', name='upsample_layer_3_conv1')(u11)
    c11 = Dropout(0.2, name='upsample_layer_3_dropout')(c11)
    c11= Conv3D(32, (3, 3, 3), activation=Lrelu, kernel_initializer=kernel_initializer,
                 kernel_regularizer=regularizers.l2(0.02), padding='same', name='upsample_layer_3_conv2')(c11)

    outputs = Conv3D(num_classes, (1, 1, 1), kernel_regularizer=regularizers.l2(0.02), activation='softmax', name='final_output')(c11)

    model = Model(inputs=[inputs], outputs=[outputs])

    return model

#Test if everything is working ok.
modelno = CNN_ModelNo(128, 128, 128, 3,4)


modelno.summary()
print(modelno.input_shape)
print(modelno.output_shape)
layer_namesno = [layer.name for layer in model.layers]

print(layer_namesno)