import cv2
import numpy as np
from config import get_config,feature_map
from keras.models import load_model

def save_img(path,img):
    cv2.imwrite(path+".jpg",img)

def save_model(path,model):
    model.save(path+".h5")

def get_model(path):
    return load_model(path)

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
    return np.maximum(img,0) * 255

def read_feature(file_path):
    with open(file_path,"r") as fin:
        feature = fin.read()
    return list(map(lambda x: x.strip(),feature.split(" ")))

def set_feature(desc,feature):
    for word in desc:
        feature[feature_map[word] - 1] = 1
    return feature

def show_feature(feature):
    print("Features:")
    for key in feature_map:
        if feature[feature_map[key]-1] == 1:
            print(key)

def get_features(f):
    l = f.shape[0]
    result =np.zeros(f.shape)
    result[:l//2] = f[:l//2]
    for i in range(l//2):
        result[i+l//2] = interpolate(f[i],f[l-i-1])
    return result

def interpolate(x,y):
    l = np.random.rand()
    return x*l + y*(1-l)

def fill_figure(r,c,shape,imgs):
    figure = np.zeros(shape * np.array([r,c,1]))
    for i in range(r):
        for j in range(c):
            figure[i*shape[0]:i*shape[0]+shape[0],j*shape[1]:j*shape[1]+shape[1],:] = imgs[i*r+j]
    return figure

def get_noise(l):
    return np.random.normal(0,1,(l,get_config("noise-dim")))

def get_same_noise(l):
    return np.repeat(np.random.normal(0,1,(1,get_config("noise-dim"))),l,axis=0)

def satisfied(f,desc):
    for d in desc:
        if d[0] == '-':
            if f[feature_map[d[1:]] - 1] > 0.5: return False
        else:
            if f[feature_map[d] - 1] < 0.5: return False

    return True

def choose_feature(fs,desc):
    f = []
    for i in fs:
        if satisfied(i,desc):
            f.append(i)
    if len(f) == 0:
        return fs[0]
    else:
        return f[np.random.randint(len(f))-1]
