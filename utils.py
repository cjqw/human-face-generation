import cv2
import numpy as np
from keras.models import load_model

def save_img(path,img):
    cv2.imwrite(path+".jpg",img * 255)

def save_model(path,model):
    model.save(path+".h5")

def get_model(path):
    return load_model(path)

def show_img(img):
    cv2.imshow(title,img)
    cv2.waitKey(0)

def save_np_array(title,arr):
    np.savetxt(title + ".txt",arr.flatten())
