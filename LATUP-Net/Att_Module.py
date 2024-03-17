import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Reshape, Dense,Add, Conv3D, BatchNormalization, UpSampling3D,Concatenate,Activation,Multiply, AveragePooling3D, MaxPooling3D, concatenate, GlobalAveragePooling3D, GlobalMaxPooling3D, Conv3DTranspose, BatchNormalization, Dropout, Lambda

# Inception Module
def inception_module_3d(x, base_channels=32):
    # 1x1x1 convolution
    a = Conv3D(base_channels*2, (1, 1, 1), activation='relu')(x)

    # 1x1x1 followed by 3x3x3 convolution
    b_1 = Conv3D(base_channels*4, (1, 1, 1), activation='relu')(x)
    b_2 = Conv3D(base_channels*4, (3, 3, 3), padding='same', activation='relu')(b_1)

    # 1x1x1 followed by 5x5x5 convolution
    c_1 = Conv3D(base_channels, (1, 1, 1), activation='relu')(x)
    c_2 = Conv3D(base_channels, (5, 5, 5), padding='same', activation='relu')(c_1)

    # 3x3x3 max-pooling followed by 1x1x1 convolution
    d_1 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    d_2 = Conv3D(base_channels, (1, 1, 1), activation='relu')(d_1)

    return Concatenate(axis=-1)([a, b_2, c_2, d_2])

# Squeeze And Excitation attention Module 

def SqueezeAndExcitation(inputs, ratio=8 ,name="attention"):
    b,_, _, _,c= inputs.shape
    x = GlobalAveragePooling3D()(inputs)
    x = Dense(c//ratio, activation="relu", use_bias=False)(x)
    x = Dense(c, activation="sigmoid", use_bias=False)(x)
    x = inputs * x
    return x

# ECA attention Module 
def ECALayer(inputs):
    b,_, _, _,c= inputs.shape
    x = GlobalAveragePooling3D()(inputs)
    x = Dense(c, activation="softmax", use_bias=False)(x)
    x = tf.expand_dims(tf.expand_dims(tf.expand_dims(x, 1), 1), 1)
    x = inputs * x
    return x

# Squeeze And Excitation with 3d Conv instead of dense layer Module
def SqueezeAndExcitation3dConv(inputs, ratio=8):
    b,_, _, _,c= inputs.shape
    x = GlobalAveragePooling3D()(inputs)
    x = Reshape((1, 1, 1, c))(x)
    x = Conv3D(c//ratio, kernel_size=(1, 1, 1), strides=(1, 1, 1), activation="relu", use_bias=False)(x)
    x = Conv3D(c, kernel_size=(1, 1, 1), strides=(1, 1, 1), activation="sigmoid", use_bias=False)(x)
    x = Multiply()([inputs, x])
    return x

# CBAM Module
def channel_attention(inputs, ratio=8):
    b,_, _, _,c= inputs.shape
    l1 = Dense(c, activation="sigmoid", use_bias=False)
    l2 = Dense(c, use_bias=False)

    #average pooling 
    x1 = GlobalAveragePooling3D()(inputs)
    x1= l1(x1)
    x1= l2(x1)

    #max pooling 
    x2 = GlobalMaxPooling3D()(inputs)
    x2= l1(x2)
    x2= l2(x2)

    #add both and apply sigmoid

    feats = x1 + x2 
    feats = Activation("sigmoid")(feats)
    feats = Multiply()([inputs, feats]) 
    return feats 


def spatial_attention(inputs):

    b,_, _, _,c= inputs.shape
    #average pooling 
    x1 = tf.reduce_mean(inputs, axis=-1)
    x1 = tf.expand_dims(x1, axis=-1)

    #max pooling 
    x2 = tf.reduce_max(inputs, axis=-1)
    x2 = tf.expand_dims(x2, axis=-1)

    #contatenate 

    feats = Concatenate()([x1 , x2])

    #conv layer  
    feats = Conv3D(c, kernel_size=7 ,padding='same', activation="sigmoid")(feats)
    feats = Multiply()([inputs, feats]) 
    return feats 


def CBAM(x):
        x = channel_attention(x)
        x = spatial_attention(x)
        return x

# Proposed MultiModal Attention

def MultiModalAttention(inputs, ratio=16):
    # compute attention maps for each channel
    _, h, w, d, c = inputs.shape
    x = Reshape((h*w*d, c))(inputs)
    x = Dense(c//ratio, activation=relu, use_bias=False)(x)
    x = Dense(c, activation=relu, use_bias=False)(x)
    x = Reshape((h, w, d, c))(x)
    # sum the attention maps across channels
    x = Lambda(lambda x: tf.reduce_sum(x, axis=-1, keepdims=True))(x)
    x = Softmax()(x)
    # multiply attention maps with original feature maps
    x = Multiply()([inputs, x])
    return x

