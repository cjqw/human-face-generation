config = {"data-path":"./data/lfw/",
          "epochs":1,
          "batchs":100,
          "train-datasets":300,
          "model-path":"./model/",
          "img-shape":(64,64,3),
          "feature-shape":(40,),
          "feature-number":40,
          "train":True}

def get_config(para):
    try:
        value = config[para]
    except:
        value = None
    return value

# tensorflow configuration
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)
