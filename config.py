config = {"data-path":"./data/lfw/",
          "epochs":1,
          "batchs":100,
          "train-datasets":300,
          "model-path":"./model/",
          "img-shape":(3,64,64),
          "feature-shape":(40,),
          "train":True}

def get_config(para):
    try:
        value = config[para]
    except:
        value = None
    return value
