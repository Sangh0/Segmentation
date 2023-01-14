import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Conv2DTranspose,
    concatenate,
    BatchNormalization,
    UpSampling2D,
)
from tensorflow.keras.applications.vgg16 import VGG16

# build encoder
def encoder(img_size):
    width, height = img_size
    
    vgg = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(width, height, 3),
    )

    pool3 = vgg.get_layer('block3_pool').output
    pool4 = vgg.get_layer('block4_pool').output
    pool5 = vgg.get_layer('block5_pool').output

    conv6 = Conv2D(512, kernel_size=3, strides=1, padding='same', activation='relu')(pool5)
    conv6 = BatchNormalization()(conv6)
    conv7 = Conv2D(512, kernel_size=3, strides=1, padding='same', activation='relu')(conv6)
    conv7 = BatchNormalization()(conv7)

    encoder_model = Model(inputs=[vgg.input], outputs=[conv7])

    return pool3, pool4, encoder_model

def decoder(
    encoder_network, 
    num_classes, 
    model_type='fcn_8s'
):
    assert model_type in ('fcn_32s', 'fcn_16s', 'fcn_8s'), \
        'you should be select in fcn_32s or fcn_16s or fcn_8s'

    pool3, pool4, encoder = encoder_network

    if model_type=='fcn_32s':
        upsample_x32 = UpSampling2D(size=(32,32), data_format='channels_last', interpolation='bilinear')(encoder.output)
        conv_out = Conv2D(num_classes, kernel_size=3, strides=1, padding='same', activation='softmax')(upsample_x32)
        model = Model(inputs=[encoder.input], outputs=[conv_out])

    elif model_type=='fcn_16s':
        conv7_x2 = UpSampling2D(size=(2,2), data_format='channels_last', interpolation='bilinear')(encoder.output)
        pool4 = pool4
        concat = concatenate([pool4, conv7_x2])
        upsample_x16 = UpSampling2D(size=(16,16), data_format='channels_last', interpolation='bilinear')(concat)
        conv_out = Conv2D(num_classes, kernel_size=3, strides=1, padding='same', activation='softmax')(upsample_x16)
        model = Model(inputs=[encoder.input], outputs=[conv_out])

    else:
        conv7_x4 = UpSampling2D(size=(4,4), data_format='channels_last', interpolation='bilinear')(encoder.output)
        pool4_x2 = UpSampling2D(size=(2,2), data_format='channels_last', interpolation='bilinear')(pool4)
        pool3 = pool3
        concat = concatenate([conv7_x4, pool4_x2, pool3])
        upsample_x8 = UpSampling2D(size=(8,8), data_format='channels_last', interpolation='bilinear')(concat)
        conv_out = Conv2D(num_classes, kernel_size=3, strides=1, padding='same', activation='softmax')(upsample_x8)
        model = Model(inputs=[encoder.input], outputs=[conv_out])
    
    return model

def get_fcn_32s(img_size, num_classes):
    return decoder(encoder(img_size), num_classes, model_type='fcn_32s')

def get_fcn_16s(img_size, num_classes):
    return decoder(encoder(img_size), num_classes, model_type='fcn_16s')

def get_fcn_8s(img_size, num_classes):
    return decoder(encoder(img_size), num_classes, model_type='fcn_8s')