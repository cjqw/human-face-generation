import cv2
import numpy as np
from config import get_config

def save_img(title,img):
    cv2.imwrite(title+".jpg",img * 255)

def show_img(img):
    cv2.imshow(title,img)
    cv2.waitKey(0)

def save_np_array(title,arr):
    np.savetxt(title + ".txt",arr.flatten())

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
