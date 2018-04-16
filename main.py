#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from fuel.datasets import CelebA
from config import get_config
from model import build_generator,build_discriminator
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

def train_model():
    train_set = CelebA('64',which_sets=('train',))
    print(train_set.num_examples)
    handle = train_set.open()

    n_batchs = (get_config("train-datasets") or train_set.num_examples) // get_config("batchs")
    generator = build_generator()
    discriminator = build_discriminator()

    for i in range(get_config("epochs")):
        for j in range(n_batchs):
            l = j * get_config("batchs")
            r = (j + 1) * get_config("batchs")
            imgs,features = train_set.get_data(handle,slice(l,r))
            imgs = imgs / 255
            imgs = np.moveaxis(imgs,1,3)
            # x = imgs[0]
            # for ii in range(64):
            #     for jj in range(64):
            #         x[ii][jj] = x[ii][jj][::-1]
            # cv2.imshow("img",x)
            # cv2.waitKey(0)
            print(l,r)
            # train models

    # save models

def generate():
    pass

if (get_config("train")):
    train_model()
else:
    generate()
