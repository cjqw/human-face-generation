#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from fuel.datasets.hdf5 import H5PYDataset
from config import get_config,feature_map
from model import build_net
from keras.models import Model
from keras.layers import Input

import keras
import numpy as np

from utils import *
# from score import get_inception_score

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

            gen_features = get_features(features)[np.random.randint(0,imgs.shape[0],half_batch)]
            noise = np.random.normal(0,1,(half_batch,noise_dim))
            gen_imgs = generator.predict([noise,gen_features])

            # real feature and real img
            d_loss_real = discriminator.train_on_batch([real_imgs],[np.ones((half_batch,1)),real_features])
            # fake feature and fake img
            d_loss_fake = discriminator.train_on_batch([gen_imgs],[np.zeros((half_batch,1)),gen_features])

            d_loss = np.add(d_loss_real,d_loss_fake) * 0.5

            # train Generator
            noise = np.random.normal(0,1,(batchs,noise_dim))
            gen_features = get_features(features)
            g_loss = gan.train_on_batch([noise,gen_features],[np.ones((batchs,1)),gen_features])

            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f] %f%%" % (i, d_loss[0], 100*d_loss[1], g_loss[0], j*100/n_batchs))

            if i % 10 == 0 and get_config("env") == "GPU":
                save_result(i,generator,gen_features[0:1,:feature_dim])

    # save models

def generate():
    generator = get_model(get_config("model-file"))
    data_path = get_config("data-file")
    shape = get_config("img-shape")
    feature_dim = get_config("feature-dim") or 40
    attribute = read_feature(get_config("feature-file"))
    r,c = 5,5

    train_set = H5PYDataset(data_path,which_sets=('train',))
    handle = train_set.open()
    _ ,real_features = train_set.get_data(handle,slice(0,50000))

    with open("feature.txt","r") as fin:
        feature = fin.readline().strip().split(",")
    feature = choose_feature(real_features,feature)
    feature = np.array(np.repeat([feature],r*c,axis=0),dtype='float64')
    # key = feature_map["Young"]
    # for i in range(10):
    #     feature[i][key-1] = 0.1 * i

    show_feature(feature[0])
    noise = get_noise(r*c)
    imgs = generator.predict([noise,feature])
    imgs = [convert_to_img(img) for img in imgs]

    figure = fill_figure(r,c,shape,imgs)
    save_img("result",figure)

    # print(get_inception_score(imgs))

if (get_config("train")):
    train_model()
else:
    generate()
