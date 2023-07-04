from time import sleep
from tkinter import E
import cv2
import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import layers
from keras.models import load_model
import sys
from PIL import Image

AUTOTUNE = tf.data.AUTOTUNE
div2k_data = tfds.image.Div2k(config="bicubic_x4")
div2k_data.download_and_prepare()
train = div2k_data.as_dataset(split="train", as_supervised=True)
train_cache = train.cache()
val = div2k_data.as_dataset(split="validation", as_supervised=True)
val_cache = val.cache()

def flip_left_right(lowres_img, highres_img):
    rn = tf.random.uniform(shape=(), maxval=1)
    return tf.cond(rn < 0.5,lambda: (lowres_img, highres_img),lambda: (tf.image.flip_left_right(lowres_img),tf.image.flip_left_right(highres_img),),)


def random_rotate(lowres_img, highres_img):
    rn = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)
    return tf.image.rot90(lowres_img, rn), tf.image.rot90(highres_img, rn)


def random_crop(lowres_img, highres_img, hr_crop_size=96, scale=4):
    lowres_crop_size = hr_crop_size // scale
    lowres_img_shape = tf.shape(lowres_img)[:2]
    lowres_width = tf.random.uniform(shape=(), maxval=lowres_img_shape[1] - lowres_crop_size + 1, dtype=tf.int32)
    lowres_height = tf.random.uniform(shape=(), maxval=lowres_img_shape[0] - lowres_crop_size + 1, dtype=tf.int32)
    highres_width = lowres_width * scale
    highres_height = lowres_height * scale
    lowres_img_cropped = lowres_img[lowres_height : lowres_height + lowres_crop_size,lowres_width : lowres_width + lowres_crop_size,]
    highres_img_cropped = highres_img[highres_height : highres_height + hr_crop_size,highres_width : highres_width + hr_crop_size,]
    return lowres_img_cropped, highres_img_cropped

def dataset_object(dataset_cache, training=True):
    ds = dataset_cache
    ds = ds.map(lambda lowres, highres: random_crop(lowres, highres, scale=4),num_parallel_calls=AUTOTUNE,)
    if training:
        ds = ds.map(random_rotate, num_parallel_calls=AUTOTUNE)
        ds = ds.map(flip_left_right, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(16)
    if training:
        ds = ds.repeat()
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


train_ds = dataset_object(train_cache, training=True)
val_ds = dataset_object(val_cache, training=False)
lowres, highres = next(iter(train_ds))


def PSNR(super_resolution, high_resolution):
    psnr_value = tf.image.psnr(high_resolution, super_resolution, max_val=255)[0]
    return psnr_value

class EDSRModel(tf.keras.Model):
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}
    def predict_step(self, x):
        x = tf.cast(tf.expand_dims(x, axis=0), tf.float32)
        super_resolution_img = self(x, training=False)
        super_resolution_img = tf.clip_by_value(super_resolution_img, 0, 255)
        super_resolution_img = tf.round(super_resolution_img)
        super_resolution_img = tf.squeeze(tf.cast(super_resolution_img, tf.uint8), axis=0)
        return super_resolution_img
    def get_config(self):
        config = super().get_config()
        return config

def ResBlock(inputs):
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(inputs)
    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.Add()([inputs, x])
    return x

def Upsampling(inputs, factor=2, **kwargs):
    x = layers.Conv2D(64 * (factor**2), 3, padding="same", **kwargs)(inputs)
    x = tf.nn.depth_to_space(x, block_size=factor)
    x = layers.Conv2D(64 * (factor**2), 3, padding="same", **kwargs)(x)
    x = tf.nn.depth_to_space(x, block_size=factor)
    return x


def make_model(num_filters, num_of_residual_blocks):
    input_layer = layers.Input(shape=(None, None, 3))
    x = layers.Rescaling(scale=1.0 / 255)(input_layer)
    x = x_new = layers.Conv2D(num_filters, 3, padding="same")(x)
    for _ in range(num_of_residual_blocks):
        x_new = ResBlock(x_new)
    x_new = layers.Conv2D(num_filters, 3, padding="same")(x_new)
    x = layers.Add()([x, x_new])
    x = Upsampling(x)
    x = layers.Conv2D(3, 3, padding="same")(x)
    output_layer = layers.Rescaling(scale=255)(x)
    return EDSRModel(input_layer, output_layer)

model = make_model(num_filters=64, num_of_residual_blocks=16)
model = load_model("edsr_model.h5", custom_objects={"EDSRModel": EDSRModel, "PSNR": PSNR})

if __name__ == '__main__':
    im = Image.open(sys.argv[1])
    dta = np.asarray(im)
    print(dta.shape)
    res = model.predict_step(dta)
    print(res.shape)
    print(type(res))
    plt.imshow(res)
    temp = Image.fromarray(res.numpy())
    temp.save(sys.argv[2])