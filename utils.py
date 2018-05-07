import cv2
import numpy as np

def save_img(title,img):
    cv2.imwrite(title+".jpg",img * 255)

def show_img(img):
    cv2.imshow(title,img)
    cv2.waitKey(0)

def save_np_array(title,arr):
    np.savetxt(title + ".txt",arr.flatten())
