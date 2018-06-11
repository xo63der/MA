# convolutional autoencoder in keras

import os
#os.environ["KERAS_BACKEND"] = "tensorflow"

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K

input_img = Input(shape=(28, 28,1)) # 1ch=black&white, 28 x 28

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img) #nb_filter, nb_row, nb_col
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D(pool_size=(2, 2))(x)

print("shape of encoded", K.int_shape(encoded))

x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D(size=(2, 2))(x)

x = Conv2D(16, (3, 3), activation='relu', padding='same')(x) 
x = UpSampling2D(size=(2, 2))(x)

decoded = Conv2D(1, (5, 5), activation='sigmoid', padding='same')(x)
print("shape of decoded", K.int_shape(decoded))

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

from keras.datasets import mnist 
import numpy as np

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32')/255. # 0-1.に変換
x_test = x_test.astype('float32')/255. 

x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

print(x_train.shape)

from keras.callbacks import TensorBoard

autoencoder.fit(x_train, x_train, epochs=5, batch_size=200,
               shuffle=True, validation_data=(x_test, x_test), verbose=1)
#               callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
### if you use TensorFlow backend, you can set TensorBoard callback

import matplotlib.pyplot as plt
# %matplotlib inline

# utility function for showing images
def show_imgs(x_test, decoded_imgs=None, n=10):
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(2, n, i+1)
        plt.imshow(x_test[i].reshape(28,28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        if decoded_imgs is not None:
            ax = plt.subplot(2, n, i+ 1 +n)
            plt.imshow(decoded_imgs[i].reshape(28,28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    plt.show()

    
decoded_imgs = autoencoder.predict(x_test)
# print "input (upper row)\ndecoded (bottom row)"
show_imgs(x_test, decoded_imgs)