import tensorflow as tf
import numpy as np
from custom_op import *
from tensorflow.keras.layers import Conv2D, Dense, ReLU
from tensorflow_addons.layers import InstanceNormalization


class ResnetGenerator(tf.keras.layers.Layer):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=6, img_size=256, light=False):
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.n_blocks = n_blocks
        self.img_size = img_size
        self.light = light
        
        DownBlock = [
            ReflectionPad2D(3),
            Conv2D(filters=ngf, kernel_size=7, strides=1, padding='valid', use_bias=False),
            InstanceNormalization(axis=3),
            ReLU()
        ]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            DownBlock += [
                ReflectionPad2D(1),
                Conv2D(filters=ngf*mult*2, kernel_size=3, strides=2, padding='valid', use_bias=False),
                InstanceNormalization(axis=3),
                ReLU()
            ]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            DownBlock += []

        self.gap_fc = Dense(1, use_bias=False)
        self.gmp_fc = Dense(1, use_bias=False)
        self.conv1x1 = Conv2D(filters=ngf*mult, kernel_size=1, strides=1, use_bias=False)
        self.relu = ReLU()

        FC = [
            Dense(ngf*mult, use_bias=False), 
            ReLU(),
            Dense(ngf*mult, use_bias=False),
            ReLU()
        ]

        self.gamma = Dense(ngf*mult, use_bias=False)
        self.beta = Dense(ngf*mult, use_bias=False)

        for i in range(n_blocks):
            pass

        UpBlock2 = []
        for i in range(n_downsampling):
            mult = 2**(n_downsampling-i)
            UpBlock2 += [

            ]


    def call(self, inputs):
        pass


class ResnetBlock(tf.keras.layers.Layer):
    def __init__(self, dim, use_bias):
        super(ResnetBlock, self).__init__()
        conv_block = []
        conv_block += [
            ReflectionPad2D(1),
            Conv2D(filters=dim, kernel_size=3, strides=1, padding='valid', use_bias=use_bias),
            InstanceNormalization(axis=3),
            ReLU()
        ]

        conv_block += [
            ReflectionPad2D(1),
            Conv2D(filters=dim, kernel_size=3, strides=1, padding='valid', use_bias=use_bias),
            InstanceNormalization(axis=3)
        ]
        self.conv_block = tf.keras.Sequential(conv_block)

    def call(self, x):
        out = x + self.conv_block(x)
        return out


if __name__ == '__main__':
    x = tf.ones((1,256,256, 3))
    model = ResnetBlock(3, False)

    print(model(x).shape)