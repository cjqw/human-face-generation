#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from fuel.datasets.hdf5 import H5PYDataset
from config import get_config
from model import build_net
from keras.models import Model
from keras.layers import Input

import keras
import numpy as np

from utils import *

if get_config("env") == "GPU":
    # tensorflow setting
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = get_config("GPU-number")
    os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    set_session(sess)


def convert_rgb_img(img):
    shape = img[0].shape
    x = np.zeros((shape[0],shape[1],3))
    x[:,:,0] = img[2,:,:]
    x[:,:,1] = img[1,:,:]
    x[:,:,2] = img[0,:,:]
    return x

def gen_noise(n,shape):
    return np.random.randint(2,size=(n,shape))

def get_batch(train_set,handle,l):
    r = l + get_config("batchs")
    imgs,features = train_set.get_data(handle,slice(l,r))
    imgs = imgs / 255
    imgs = np.moveaxis(imgs,1,3)
    return imgs,features

def convert_to_img(img):
    shape = img.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            img[i][j] = img[i][j][::-1]
    return img

def save_result(epoch,generator):
    feature_dim = get_config("feature-dim") or 40
    noise_dim = get_config("noise-dim") or 10
    input_dim = feature_dim + noise_dim
    shape = get_config("img-shape")

    r,c = 5,5
    feature = np.random.normal(0,1,(1, feature_dim))
    feature = np.array(feature,r*c,axis=0)
    noise = np.random.normal(0,1,(r*c, noise_dim))
    real_features = np.concatenate((feature,noise),axis=1)

    gen_imgs = generator.predict(real_features)
    gen_imgs = [convert_to_img(img) for img in gen_imgs]

    figure = np.zeros(shape * np.array([r,c,1]))
    for i in range(r):
        for j in range(c):
            figure[i*shape[0]:i*shape[0]+shape[0],j*shape[1]:j*shape[1]+shape[1],:] = gen_imgs[i*r+j]

    img_path = "img/epoch_%d" % epoch
    model_path = "model/epoch_%d" % epoch
    save_img(img_path,figure)
    save_model(model_path,generator)

def train_model():
    data_path = get_config("data-path")
    batchs = get_config("batchs")
    half_batch = batchs // 2
    n_batchs = (get_config("train-datasets") or train_set.num_examples) // batchs
    feature_dim = get_config("feature-dim") or 40
    noise_dim = get_config("noise-dim") or 10
    input_dim = feature_dim + noise_dim

    train_set = H5PYDataset(data_path,which_sets=('train',))
    handle = train_set.open()

    generator,discriminator,gan = build_net(input_dim)
    save_result(0,generator)

    for i in range(get_config("epochs")):
        for j in range(n_batchs):
            imgs,features = get_batch(train_set,handle,j * batchs)

            noise = np.random.normal(0,1,(half_batch,input_dim))
            gen_imgs = generator.predict(noise)

            idx = np.random.randint(0,imgs.shape[0],half_batch)
            real_imgs = imgs[idx]
            real_noise = np.random.normal(0,1,(half_batch,noise_dim))
            real_features = np.concatenate((features[idx],real_noise),axis=1)

            # train Discriminator
            # d_loss_real = discriminator.train_on_batch([real_features,real_imgs],np.ones((half_batch,1)))
            # d_loss_fake = discriminator.train_on_batch([noise,gen_imgs],np.zeros((half_batch,1)))
            d_loss_real = discriminator.train_on_batch(real_imgs,np.ones((half_batch,1)))
            d_loss_fake = discriminator.train_on_batch(gen_imgs,np.zeros((half_batch,1)))

            d_loss = np.add(d_loss_real,d_loss_fake) * 0.5


            # train Generator
            noise = np.random.normal(0,1,(batchs,input_dim))
            g_loss = gan.train_on_batch(noise,np.ones((batchs,1)))

            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (i, d_loss[0], 100*d_loss[1], g_loss))

            if get_config("env") == "CPU":
                return

            if i % 10 == 0:
                save_result(i,generator)

    # save models

def generate():
    pass

if (get_config("train")):
    train_model()
else:
    generate()
