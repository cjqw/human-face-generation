#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from fuel.datasets import CelebA
from config import get_config
from model import build_generator,build_discriminator
from keras.models import Model
from keras.layers import Input
import keras
import cv2

import numpy as np

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

def get_dis_train_pair(imgs,features):
    """Get train pair for discriminator."""
    batch = get_config("batchs")
    x,y = imgs,features
    return x,y

def get_gen_train_pair():
    x = gen_noise(get_config("batchs"),get_config("feature-number"))
    return x,x

def train_model():
    train_set = CelebA('64',which_sets=('train',))
    print(train_set.num_examples)
    handle = train_set.open()

    n_batchs = (get_config("train-datasets") or train_set.num_examples) // get_config("batchs")
    generator = build_generator()
    discriminator = build_discriminator()
    discriminator.compile(loss='categorical_crossentropy', optimizer='RMSprop')
    generator.compile(loss='binary_crossentropy', optimizer='RMSprop')


    gan_input = Input(shape=get_config("feature-shape"))
    H = generator(gan_input)
    gan_output = discriminator(H)
    gan = Model(gan_input,gan_output)
    gan.compile(loss='categorical_crossentropy', optimizer='RMSprop')

    for i in range(get_config("epochs")):
        for j in range(n_batchs):
            imgs,features = get_batch(train_set,handle,j*get_config("batchs"))
            # x = imgs[0]
            # for ii in range(64):
            #     for jj in range(64):
            #         x[ii][jj] = x[ii][jj][::-1]
            # cv2.imshow("img",x)
            # cv2.waitKey(0)

            # train Discriminator
            discriminator.trainable = True
            X,Y = get_dis_train_pair(imgs,features)
            d_loss = discriminator.train_on_batch(X,Y)

            # train Generator
            discriminator.trainable = False
            X,Y = get_gen_train_pair()
            g_loss = gan.train_on_batch(X,Y)

    # save models

def generate():
    pass

if (get_config("train")):
    train_model()
else:
    generate()
