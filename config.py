config = {"data-path":"./data/celeba_64.hdf5",
          "epochs":101,
          "batchs":64,
          "train-datasets":5000,
          "model-path":"./model/",
          "img-shape":(64,64,3),
          "feature-dim": 40,
          "noise-dim": 20,
          "train":True,
          "GPU-number":"1",
          "env":"GPU"}

def get_config(para,default = None):
    try:
        value = config[para]
    except:
        value = default
    return value
