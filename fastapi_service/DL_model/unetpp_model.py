import tensorflow as tf
from tensorflow.keras import layers




def double_conv_block(x, n_filters):
   # Conv2D then ReLU activation
   x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
   # Conv2D then ReLU activation
   x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
   return x

def downsample_block(x, n_filters):
   f = double_conv_block(x, n_filters)
   p = layers.MaxPool2D(2)(f)
   p = layers.Dropout(0.5)(p)
   return f, p


def upsample_block_part1(x, n_filters):
   # upsample
   x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
   return x

def upsample_block_part2(x, n_filters):
   # dropout
   x = layers.Dropout(0.5)(x)
   # Conv2D twice with ReLU activation
   x = double_conv_block(x, n_filters)
   return x


def build_unetPP_model():
    # inputs
    inputs = layers.Input(shape=(128,128,3))

    # encoder: contracting path - downsample
    f00, p00 = downsample_block(inputs, 64)
    f10, p10 = downsample_block(p00, 128)
    f20, p20 = downsample_block(p10, 256)
    f30, p30 = downsample_block(p20, 512)
    bottleneck = double_conv_block(p30, 1024)

    # decoders: expanding path - upsample
    u01 = upsample_block_part1 (p00, 64)
    u01 = layers.concatenate ([u01, f00])
    u01 = upsample_block_part2 (u01, 64)

    u11 = upsample_block_part1 (p10, 128)
    u11 = layers.concatenate ([u11, f10])
    u11 = upsample_block_part2 (u11, 128)

    u21 = upsample_block_part1 (p20, 256)
    u21 = layers.concatenate ([u21, f20])
    u21 = upsample_block_part2 (u21, 256)

    u31 = upsample_block_part1 (bottleneck, 512)
    u31 = layers.concatenate ([u31, f30])
    u31 = upsample_block_part2 (u31, 512)

    u02 = upsample_block_part1 (u11, 64)
    u02 = layers.concatenate ([u02, f00, u01])
    u02 = upsample_block_part2 (u02, 64)

    u12 = upsample_block_part1 (u21, 128)
    u12 = layers.concatenate ([u12, f10, u11])
    u12 = upsample_block_part2 (u12, 128)

    u22 = upsample_block_part1 (u31, 256)
    u22 = layers.concatenate([u22, f20, u21])
    u22 = upsample_block_part2(u22, 256)

    u03 = upsample_block_part1 (u12, 64)
    u03 = layers.concatenate([u03, f00, u01, u02])
    u03 = upsample_block_part2(u03, 64)

    u13 = upsample_block_part1 (u22, 128)
    u13 = layers.concatenate([u13, f10, u11, u12])
    u13 = upsample_block_part2(u13, 128)

    u04 = upsample_block_part1 (u13, 64)
    u04 = layers.concatenate([u04, f00, u01, u02, u03])
    u04 = upsample_block_part2(u04, 64)

    # outputs
    outputs = layers.Conv2D(1, 1, padding="same", activation = "sigmoid")(u04)
    # unet model with Keras Functional API
    unetPP_model = tf.keras.Model(inputs, outputs, name="UNetPP")
    return unetPP_model



