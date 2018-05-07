config = {"data-path":"./data/celeba_64.hdf5",
          "epochs":1001,
          "batchs":200,
          "train-datasets":5000,
          "model-path":"./model/",
          "img-shape":(64,64,3),
          "feature-dim": 40,
          "noise-dim": 20,
          "train":True,
          "GPU-number":"3",
          "env":"CPU"}

def get_config(para,default = None):
    try:
        value = config[para]
    except:
        value = default
    return value
