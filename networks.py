import tensorflow as tf
import numpy as np
from custom_op import *
from tensorflow.keras.layers import Conv2D, Dense, ReLU, UpSampling2D, LeakyReLU
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
            DownBlock += [ResnetBlock(ngf*mult, use_bias=False)]

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
            setattr(self, 'UpBlock1_'+str(i+1), ResnetAdaILNBlock(ngf*mult, use_bias=False))

        UpBlock2 = []
        for i in range(n_downsampling):
            mult = 2**(n_downsampling-i)
            UpBlock2 += [
                UpSampling2D(size=(2,2), interpolation='nearest'),
                ReflectionPad2D(1),
                Conv2D(filters=int(ngf*mult/2), kernel_size=3, strides=1, padding='valid', use_bias=False),
                ILN(int(ngf*mult/2)),
                ReLU()
            ]
        UpBlock2 += [
            ReflectionPad2D(3),
            Conv2D(filters=output_nc, kernel_size=7, strides=1, padding='valid', use_bias=False),
            Tanh()
        ]
        self.DownBlock = tf.keras.Sequential(DownBlock)
        self.FC = tf.keras.Sequential(FC)
        self.UpBlock2 = tf.keras.Sequential(UpBlock2)

    def call(self, inputs):
        x = self.DownBlock(inputs)

        gap = tf.reduce_mean(x, axis=[1,2], keepdims=True)
        gap_logit = self.gap_fc(tf.reshape(gap, shape=[x.shape[0], -1]))
        gap_weight = tf.transpose(self.gap_fc.get_weights()[0], perm=[1,0])
        gap = x * tf.expand_dims(tf.expand_dims(gap_weight, axis=1), axis=2)

        gmp = tf.reduce_max(x, axis=[1,2], keepdims=True)
        gmp_logit = self.gmp_fc(tf.reshape(gmp, shape=[x.shape[0], -1]))
        gmp_weight = tf.transpose(self.gmp_fc.get_weights()[0], perm=[1,0])
        gmp = x * tf.expand_dims(tf.expand_dims(gmp_weight, axis=1), axis=2)

        cam_logit = tf.concat([gap_logit, gmp_logit], 1)
        x = tf.concat([gap, gmp], 3)
        x = self.relu(self.conv1x1(x))

        heatmap = tf.reduce_sum(x, axis=[3], keepdims=True)

        if self.light:
            x_ = tf.reduce_mean(x, axis=[1,2], keepdims=True)
            x_ = self.FC(tf.reshape(x_, shape=[x_.shape[0], -1]))
        else:
            x_ = self.FC(tf.reshape(x_, shape=[x.shape[0], -1]))
        gamma, beta = self.gamma(x_), self.beta(x_)

        for i in range(self.n_blocks):
            x = getattr(self, 'UpBlock1_'+str(i+1))(x, gamma, beta)

        out = self.UpBlock2(x)
        return out, cam_logit, heatmap


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


class ResnetAdaILNBlock(tf.keras.layers.Layer):
    def __init__(self, dim, use_bias):
        super(ResnetAdaILNBlock, self).__init__()
        self.pad1 = ReflectionPad2D(1)
        self.conv1 = Conv2D(filters=dim, kernel_size=3, strides=1, padding='valid', use_bias=use_bias)
        self.norm1 = adaILN(dim)
        self.relu1 = ReLU()

        self.pad2 = ReflectionPad2D(1)
        self.conv2 = Conv2D(filters=dim, kernel_size=3, strides=1, padding='valid', use_bias=use_bias)
        self.norm2 = adaILN(dim)

    def call(self, x, gamma, beta):
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.norm1(out, gamma, beta)
        out = self.relu1(out)
        out = self.pad2(out)
        out = self.conv2(out)
        out = self.norm2(out, gamma, beta)
        return out + x


class adaILN(tf.keras.layers.Layer):
    def __init__(self, num_features, eps=1e-5):
        super(adaILN, self).__init__()
        self.eps = eps
        self.rho = self.add_weight("rho", shape=(1, 1, 1, num_features), dtype='float32', trainable=True,
            initializer=tf.constant_initializer(0.9))

    def call(self, inputs, gamma, beta):
        in_mean = tf.math.reduce_mean(inputs, axis=[1,2], keepdims=True)
        in_var = tf.keras.backend.var(inputs, axis=[1,2], keepdims=True)
        out_in = (inputs - in_mean) / tf.sqrt(in_var + self.eps)
        ln_mean = tf.math.reduce_mean(inputs, axis=[1,2,3], keepdims=True)
        ln_var = tf.keras.backend.var(inputs, axis=[1,2,3], keepdims=True)
        out_ln = (inputs - ln_mean) / tf.sqrt(ln_var, self.eps)
        out = self.rho*out_in + (1-self.rho)*out_ln
        out = out * gamma + beta
        return out

class ILN(tf.keras.layers.Layer):
    def __init__(self, num_features, eps=1e-5):
        super(ILN, self).__init__()
        self.eps = eps
        self.rho = self.add_weight("rho", shape=(1, 1, 1, num_features), dtype='float32', trainable=True,
            initializer=tf.constant_initializer(0.0))
        self.gamma = self.add_weight("gamma", shape=(1, 1, 1, num_features), dtype='float32', trainable=True,
            initializer=tf.constant_initializer(1.0))
        self.beta = self.add_weight("beta", shape=(1, 1, 1, num_features), dtype='float32', trainable=True,
            initializer=tf.constant_initializer(0.0))

    def call(self, inputs):
        in_mean = tf.math.reduce_mean(inputs, axis=[1,2], keepdims=True)
        in_var = tf.keras.backend.var(inputs, axis=[1,2], keepdims=True)
        out_in = (inputs-in_mean) / tf.sqrt(in_var+self.eps)
        ln_mean = tf.math.reduce_mean(inputs, axis=[1,2,3], keepdims=True)
        ln_var = tf.keras.backend.var(inputs, axis=[1,2,3], keepdims=True)
        out_ln = (inputs-ln_mean) / tf.sqrt(ln_var+self.eps)
        out = self.rho*out_in + (1-self.rho)*out_ln
        out = out * self.gamma + self.beta
        return out

class Discriminator(tf.keras.layers.Layer):
    def __init__(self, input_nc, ndf=64, n_layers=5):
        super(Discriminator, self).__init__()
        model = [
            ReflectionPad2D(1),
            SpectralNormalization(Conv2D(filters=ndf, kernel_size=4, strides=2, padding='valid', use_bias=True)),
            LeakyReLU(0.2)
        ]

        for i in range(1, n_layers-2):
            mult = 2 ** (i-1)
            model += [
                ReflectionPad2D(1),
                SpectralNormalization(Conv2D(filters=ndf*mult*2, kernel_size=4, strides=2, padding='valid', use_bias=True)),
                LeakyReLU(0.2)
            ]

        mult = 2 ** (n_layers-2-1)
        model += [
            ReflectionPad2D(1),
            SpectralNormalization(Conv2D(filters=ndf*mult*2, kernel_size=4, strides=1, padding='valid', use_bias=True)),
            LeakyReLU(0.2)
        ]

        mult = 2 ** (n_layers-2)
        self.gap_fc = SpectralNormalization(Dense(1, use_bias=False))
        self.gmp_fc = SpectralNormalization(Dense(1, use_bias=False))
        self.conv1x1 = Conv2D(filters=ndf*mult, kernel_size=1, strides=1, use_bias=True)
        self.leaky_relu = LeakyReLU(0.2)

        self.pad = ReflectionPad2D(1)
        self.conv = SpectralNormalization(Conv2D(filters=1, kernel_size=4, strides=1, padding='valid', use_bias=False))
        self.model = tf.keras.Sequential(model)

    def call(self, inputs):
        x = self.model(inputs)
        gap = tf.math.reduce_mean(x, axis=[1,2], keepdims=True)
        gap_logit = self.gap_fc(tf.reshape(gap, shape=[x.shape[0], -1]))
        gap_weight = tf.transpose(self.gap_fc.get_weights()[0], perm=[1,0])
        gap = x * tf.expand_dims(tf.expand_dims(gap_weight, axis=1), axis=2)

        gmp = tf.math.reduce_max(x, axis=[1,2], keepdims=True)
        gmp_logit = self.gmp_fc(tf.reshape(gmp, shape=[x.shape[0], -1]))
        gmp_weight = tf.transpose(self.gmp_fc.get_weights()[0], perm=[1,0])
        gmp = x * tf.expand_dims(tf.expand_dims(gmp_weight, axis=1), axis=2)

        cam_logit = tf.concat([gap_logit, gmp_logit], 1)
        x = tf.concat([gap, gmp], axis=3)
        x = self.leaky_relu(self.conv1x1(x))

        heatmap = tf.reduce_sum(x, dim=[3], keepdims=True)

        x = self.pad(x)
        out = self.conv(x)
        return out, cam_logit, heatmap


if __name__ == '__main__':
    x = tf.ones((1,256,256, 3))
    model = Discriminator(3, 1, 5)

    print(model(x)[0].shape)