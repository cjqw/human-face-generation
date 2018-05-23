from config import get_config
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import Dot,Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, MaxPooling2D
from keras.models import Sequential, Model
from keras.initializers import RandomNormal
from keras.optimizers import Adam
import keras.backend as K

def build_generator():
    feature_dim = get_config("feature-dim") or 40
    noise_dim = get_config("noise-dim") or 10
    input_dim = feature_dim + noise_dim

    model = Sequential()
    model.add(Dense(128*16*16,activation="relu",input_shape=(input_dim,)))
    model.add(Reshape((16,16,128)))

    model.add(BatchNormalization(momentum=0.8))
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(Activation("relu"))

    model.add(BatchNormalization(momentum=0.8))
    model.add(UpSampling2D())
    model.add(Conv2D(64, kernel_size=3, padding="same"))
    model.add(Activation("relu"))

    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(3, kernel_size=3, padding="same"))
    model.add(Activation("tanh"))
    model.summary()

    noise = Input(shape=(noise_dim,))
    feature = Input(shape=(feature_dim,))
    input_v = Concatenate()([noise,feature])
    img = model(input_v)

    return Model([noise,feature], img)


def build_discriminator():
    img_shape = get_config("img-shape") or (64,64,3)
    feature_dim = get_config("feature-dim") or 40
    noise_dim = get_config("noise-dim") or 10
    input_dim = feature_dim + noise_dim

    dropout_prob = 0.4
    kernel_init = "glorot_uniform"

    model = Sequential()

    model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=img_shape, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(ZeroPadding2D(padding=((0,1),(0,1))))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.summary()

    img = Input(shape=img_shape)
    model = model(img)

    validity = Dense(1,activation='sigmoid')(model)

    feature = Dense(feature_dim,activation='sigmoid')(model)

    return Model([img],[validity,feature])

def build_net(input_dim):
    feature_dim = get_config("feature-dim") or 40
    noise_dim = get_config("noise-dim") or 10
    input_dim = feature_dim + noise_dim

    optimizer = Adam(0.0002, 0.5)
    generator = build_generator()
    discriminator = build_discriminator()
    discriminator.compile(loss=['binary_crossentropy',
                                'binary_crossentropy'],
                          optimizer = optimizer,
                          metrics = ['accuracy'])

    noise = Input(shape=(noise_dim,))
    feature = Input(shape=(feature_dim,))

    img = generator([noise,feature])
    discriminator.trainable = False
    gan_output = discriminator(img)
    gan = Model([noise,feature],gan_output)
    gan.compile(loss=['binary_crossentropy',
                      'binary_crossentropy'],
                optimizer=optimizer)
    return generator,discriminator,gan
