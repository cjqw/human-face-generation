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
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    set_session(sess)

def save_result(epoch,generator,feature):
    feature_dim = get_config("feature-dim") or 40
    noise_dim = get_config("noise-dim") or 10
    input_dim = feature_dim + noise_dim
    shape = get_config("img-shape")

    r,c = 5,5
    feature = np.repeat(feature,r*c,axis=0)
    noise =  np.random.normal(0,1,(r*c, noise_dim))

    gen_imgs = generator.predict([noise,feature])
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
    data_path = get_config("data-file")
    batchs = get_config("batchs")
    half_batch = batchs // 2
    quarter_batch = half_batch // 2
    n_batchs = (get_config("train-datasets") or train_set.num_examples) // batchs
    feature_dim = get_config("feature-dim") or 40
    noise_dim = get_config("noise-dim") or 10
    input_dim = feature_dim + noise_dim

    train_set = H5PYDataset(data_path,which_sets=('train',))
    handle = train_set.open()

    generator,discriminator,gan = build_net(input_dim)
    save_result(0,generator,np.zeros(shape=(1,feature_dim)))

    for i in range(get_config("epochs")):
        for j in range(n_batchs):
            imgs,features = get_batch(train_set,handle,j * batchs)

            idx = np.random.randint(0,imgs.shape[0],half_batch)
            real_imgs = imgs[idx]
            real_features = features[idx]

            gen_features = features[idx[:quarter_batch]]# np.random.normal(0,1,(batchs,feature_dim))
            noise = np.random.normal(0,1,(quarter_batch,noise_dim))
            gen_imgs = generator.predict([noise,gen_features])

            # real feature and real img
            d_loss_real = discriminator.train_on_batch([real_features,real_imgs],[np.ones((half_batch,1)),real_features])
            # fake feature and fake img
            d_loss_fake = discriminator.train_on_batch([gen_features,gen_imgs],[np.zeros((quarter_batch,1)),gen_features])
            # fake feature and real img
            d_loss_half = discriminator.train_on_batch([gen_features,real_imgs[:quarter_batch]],
                                                       [np.ones((quarter_batch,1)),real_features[:quarter_batch]])

            d_loss = np.add(d_loss_real,np.add(d_loss_half,d_loss_fake) * 0.5) * 0.5


            # train Generator
            noise = np.random.normal(0,1,(batchs,noise_dim))
            gen_features = features[np.random.randint(0,imgs.shape[0],batchs)]# np.random.normal(0,1,(batchs,feature_dim))
            g_loss = gan.train_on_batch([noise,gen_features],[np.ones((batchs,1)),gen_features])

            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (i, d_loss[0], 100*d_loss[1], g_loss[0]))

            if i % 5 == 0 and get_config("env") == "GPU":
                save_result(i,generator,gen_features[0:1,:feature_dim])

    # save models

def generate():
    generator = get_model(get_config("model-file"))
    attribute = read_feature(get_config("feature-file"))
    shape = get_config("img-shape")
    feature_dim = get_config("feature-dim") or 40
    noise_dim = get_config("noise-dim") or 10
    input_dim = feature_dim + noise_dim
    r,c = 5,5

    noise = np.random.normal(0,1,(r*c,noise_dim))
    # feature = np.array([set_feature(feature,np.random.choice(2,size=feature_dim,p=[0.9,0.1]))])
    # show_feature(feature)
    # feature = np.repeat(feature,r*c,axis=0)
    feature = np.random.choice(2,size=(r*c,feature_dim),p=[0.9,0.1])
    feature = np.array([set_feature(attribute,x) for x in feature])
    imgs = generator.predict([noise,feature])
    imgs = [convert_to_img(img) for img in imgs]

    figure = np.zeros(shape * np.array([5,5,1]))
    for i in range(r):
        for j in range(c):
            figure[i*shape[0]:i*shape[0]+shape[0],j*shape[1]:j*shape[1]+shape[1],:] = imgs[i*r+j]

    save_img("result",figure)

if (get_config("train")):
    train_model()
else:
    generate()
