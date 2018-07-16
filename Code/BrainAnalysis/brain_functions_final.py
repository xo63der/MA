###### %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

# convolutional autoencoder in keras

from __future__ import print_function

import tensorflow as tf

import os
os.environ["KERAS_BACKEND"] = "tensorflow"

from keras.models import Model
from keras.layers import *
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras import regularizers
from keras import backend as K

import matplotlib.pyplot as plt
import matplotlib.colors as mcol
# %matplotlib inline

from scipy.ndimage.filters import gaussian_laplace
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

import gc

import warnings
warnings.filterwarnings('ignore')

import nilearn
from nilearn import plotting

### show

def show_skull(path, n=0):
    listing = os.listdir(path)
    img = nilearn.image.load_img(path + listing[n])
    plotting.plot_anat(img)

def show_whole(x_test, decoded_imgs=None, n=3):
    plt.figure(figsize=(20, 30))
    for i in range(n):
        ax = plt.subplot(6, n, i+1)
        plt.imshow(x_test[i][int(np.shape(x_test[i])[0]/2)+1])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(6, n, i+1+n)
        plt.imshow(x_test[i][:,int(np.shape(x_test[i])[1]/2)+1,:])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(6, n, i+1+2*n)
        plt.imshow(x_test[i][:,:,int(np.shape(x_test[i])[2]/2)+1])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        if decoded_imgs is not None:
            ax = plt.subplot(6, n, i+1+3*n)
            plt.imshow(decoded_imgs[i][int(np.shape(x_test[i])[0]/2)+1])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            ax = plt.subplot(6, n, i+1+4*n)
            plt.imshow(decoded_imgs[i][:,int(np.shape(x_test[i])[1]/2)+1,:])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            ax = plt.subplot(6, n, i+1+5*n)
            plt.imshow(decoded_imgs[i][:,:,int(np.shape(x_test[i])[2]/2)+1])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    plt.show()

def show_batch(x_test, size, decoded_imgs=None, n=7):
    plt.figure(figsize=(20, 10))
    for i in range(n):
        ax = plt.subplot(6, n, i+1)
        plt.imshow(x_test[i][int(size/2)+1].reshape(size,size))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(6, n, i+1+n)
        plt.imshow(x_test[i][:,int(size/2)+1,:].reshape(size,size))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(6, n, i+1+2*n)
        plt.imshow(x_test[i][:,:,int(size/2)+1].reshape(size,size))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        if decoded_imgs is not None:
            ax = plt.subplot(6, n, i+1+3*n)
            plt.imshow(decoded_imgs[i][int(size/2)+1].reshape(size,size))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            ax = plt.subplot(6, n, i+1+4*n)
            plt.imshow(decoded_imgs[i][:,int(size/2)+1,:].reshape(size,size))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            ax = plt.subplot(6, n, i+1+5*n)
            plt.imshow(decoded_imgs[i][:,:,int(size/2)+1].reshape(size,size))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    plt.show()

def show_encoded(x_test, size, encoded_imgs, n1, n=10):

    plt.figure(figsize=(20, 10))
    for i in range(n):
        ax = plt.subplot(3*(n1+1), n, i+1)
        plt.imshow(x_test[i][int(size/2)+1].reshape(size,size))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(3*(n1+1), n, i+1+n)
        plt.imshow(x_test[i][:,int(size/2)+1,:].reshape(size,size))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(3*(n1+1), n, i+1+2*n)
        plt.imshow(x_test[i][:,:,int(size/2)+1].reshape(size,size))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        if encoded_imgs is not None:
            for j in range(n1):
                ax = plt.subplot(3*(n1+1), n, i+1+(3+3*j)*n)
                plt.imshow(encoded_imgs[i][int(size/6)+1,:,:,j].reshape(int(size/3),int(size/3)))
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

                ax = plt.subplot(3*(n1+1), n, i+1+(4+3*j)*n)
                plt.imshow(encoded_imgs[i][:,int(size/6)+1,:,j].reshape(int(size/3),int(size/3)))
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

                ax = plt.subplot(3*(n1+1), n, i+1+(5+3*j)*n)
                plt.imshow(encoded_imgs[i][:,:,int(size/6)+1,j].reshape(int(size/3),int(size/3)))
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

    plt.show()

### heat maps

def show_heat_grads(x_test, size, out, n=7):
    a=0.3
    cm1 = mcol.LinearSegmentedColormap.from_list("cmapBR",["b","r"])
    plt.figure(figsize=(20, 10))
    for i in range(n):
        ax = plt.subplot(3, n, i+1)
        plt.imshow(x_test[i][int(size/2)+1].reshape(size,size),cmap='gray')
        plt.imshow(out[i][int(size/2)+1].reshape(size,size),cmap=cm1, alpha=a)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(3, n, i+1+n)
        plt.imshow(x_test[i][:,int(size/2)+1,:].reshape(size,size),cmap='gray')
        plt.imshow(out[i][:,int(size/2)+1,:].reshape(size,size),cmap=cm1, alpha=a)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(3, n, i+1+2*n)
        plt.imshow(x_test[i][:,:,int(size/2)+1].reshape(size,size),cmap='gray')
        plt.imshow(out[i][:,:,int(size/2)+1].reshape(size,size),cmap=cm1, alpha=a)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()

def show_heat_encoded(x_test, size, encoded_imgs, n=7):
    a=0.3
    cm1 = mcol.LinearSegmentedColormap.from_list("cmapBR",["b","r"])
    plt.figure(figsize=(20, 10))
    for i in range(n):
        ax = plt.subplot(3, n, i+1)
        plt.imshow(x_test[i][int(size/2)+1].reshape(size,size),cmap='gray')
        plt.imshow(encoded_imgs[i][int(size/6)+1,:,:,j].reshape(size,size),cmap=cm1, alpha=a)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(3, n, i+1+n)
        plt.imshow(x_test[i][:,int(size/2)+1,:].reshape(size,size),cmap='gray')
        plt.imshow(encoded_imgs[i][:,int(size/6)+1,:,j].reshape(size,size),cmap=cm1, alpha=a)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(3, n, i+1+2*n)
        plt.imshow(x_test[i][:,:,int(size/2)+1].reshape(size,size),cmap='gray')
        plt.imshow(encoded_imgs[i][:,:,int(size/6)+1,j].reshape(size,size),cmap=cm1, alpha=a)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()

### load/preprocess data

def loadbatches(path, size, overlap=False, select=True):

    print(overlap)

    batches = []

    allpositions=[]

    listing = os.listdir(path)

    if select:

        for file in listing:

            img = nilearn.image.load_img(path + file).get_fdata()
            log = np.absolute(gaussian_laplace(img,1))

            maxval=max(img.flatten())
            minval=min(img.flatten())
            img=(img-minval)/(maxval-minval)

            maxval=max(log.flatten())
            minval=min(log.flatten())
            log=(log-minval)/(maxval-minval)

            if overlap:
                x_cuts=2*int(len(img)/size)-1
                y_cuts=2*int(len(img[0])/size)-1
                z_cuts=2*int(len(img[0][0])/size)-1
            else:
                x_cuts=int(len(img)/size)
                y_cuts=int(len(img[0])/size)
                z_cuts=int(len(img[0][0])/size)
            x_rest=len(img)%size
            y_rest=len(img[0])%size
            z_rest=len(img[0][0])%size

            pos=np.zeros((x_cuts,x_cuts,z_cuts,3))
            mean_log=np.zeros((x_cuts,x_cuts,z_cuts))

            for i in range(x_cuts):
                if overlap:
                    x1=int(size*i/2)+int(x_rest/2)
                else:
                    x1=size*i+int(x_rest/2)
                x2=x1+size
                for j in range(y_cuts):
                    if overlap:
                        y1=int(size*j/2)+int(y_rest/2)
                    else:
                        y1=size*j+int(y_rest/2)
                    y2=y1+size
                    for k in range(z_cuts):
                        if overlap:
                            z1=int(size*k/2)+int(z_rest/2)
                        else:
                            z1=size*k+int(z_rest/2)
                        z2=z1+size
                        mean_log[i,j,k]=np.mean(log[x1:x2,y1:y2,z1:z2].flatten())
                        pos[i,j,k]=np.array([x1,y1,z1])

            allpositions.append(pos)

            med_log=np.percentile(mean_log.flatten(),95)

            good_pos=[]

            for i in range(x_cuts):
                for j in range(y_cuts):
                    for k in range(z_cuts):
                        if med_log < mean_log[i,j,k]:
                            good_pos.append(pos[i,j,k])

            for i in range(len(good_pos)):
                x1=int(good_pos[i][0])
                x2=x1+size
                y1=int(good_pos[i][1])
                y2=y1+size
                z1=int(good_pos[i][2])
                z2=z1+size

                batches.append(img[x1:x2,y1:y2,z1:z2])

    else:
        for file in listing:

            img = nilearn.image.load_img(path + file).get_fdata()

            maxval=max(img.flatten())
            minval=min(img.flatten())
            img=(img-minval)/(maxval-minval)

            x_cuts=int(len(img)/size)
            y_cuts=int(len(img[0])/size)
            z_cuts=int(len(img[0][0])/size)
            x_rest=len(img)%size
            y_rest=len(img[0])%size
            z_rest=len(img[0][0])%size

            pos=np.zeros((x_cuts,x_cuts,z_cuts,3))

            for i in range(int(x_cuts)):
                x1=size*i+int(x_rest/2)
                x2=x1+size
                for j in range(int(y_cuts)):
                    y1=size*j+int(y_rest/2)
                    y2=y1+size
                    for k in range(int(z_cuts)):
                        z1=size*k+int(z_rest/2)
                        z2=z1+size
                        im_cut=img[x1:x2,y1:y2,z1:z2]
                        batches.append(im_cut)
                        pos[i,j,k]=np.array([x1,y1,z1])

            allpositions.append(pos)

    batches=np.asarray(batches)
    shape=np.shape(batches)
    print(shape)
    return batches.reshape(shape[0],shape[1],shape[2],shape[3],1),allpositions

def load_whole(path, val=10):

    listing = os.listdir(path)

    pics=[]

    i=0

    for file in listing:
        if i<val:
            i+=1

            img=nilearn.image.load_img(path + file).get_fdata()

            maxval=max(img.flatten())
            minval=min(img.flatten())
            img=(img-minval)/(maxval-minval)

            pics.append(img)

    return np.asarray(pics)

### recombine from batches to pictures

def recombine(control_batches, control_allpos):
    n=0
    cpics=[]
    for i in range(len(control_allpos)):
        x_cuts=np.shape(control_allpos[i])[0]
        y_cuts=np.shape(control_allpos[i])[1]
        z_cuts=np.shape(control_allpos[i])[2]

        for j in range(x_cuts):
            for k in range(y_cuts):
                for l in range(z_cuts):
                    if l==0:
                        a=control_batches[n]
                    else:
                        a=np.concatenate((a,control_batches[n+j*y_cuts*z_cuts+k*z_cuts+l]),axis=2)
                if k==0:
                    b=a
                else:
                    b=np.concatenate((b,a),axis=1)
            if j==0:
                c=b
            else:
                c=np.concatenate((c,b),axis=0)
        n=n+x_cuts*y_cuts*z_cuts
        cpics.append(c)

    cpics=np.asarray(cpics)
    shape=np.shape(cpics)
    return cpics.reshape(shape[0],shape[1],shape[2],shape[3])

### KL-Div Regularizer

def kl_reg(x):
    beta=0.001
    p=0.01
    not_nan=0.000001

    p_hat=K.mean(x, axis=0)

    kl_div=(p * K.log(p/p_hat+not_nan)) + ((1-p) * K.log((1-p)/(1-p_hat-not_nan)))
    result=K.sum(kl_div)

    return beta * result

### pretrain

def pretrain(train_batches,name,size,maxfil,bs_z,eps):

    ### pretrain1

    print('pretrain #1')

    input_img = Input(shape=(size, size,size,1), dtype='float32')

    train1_c1=Conv3D(maxfil, (3, 3, 3), activation='sigmoid', padding='same', name='t1c1', activity_regularizer=kl_reg)(input_img)
    train1_decoded=Conv3D(1, (3, 3, 3), activation='sigmoid', padding='same', name='t1tc1')(train1_c1)

    train1_autoencoder = Model(input_img, train1_decoded)
    train1_autoencoder.compile(optimizer='adadelta', loss='mean_squared_error', metrics=['accuracy'])

    #with tf.device('/gpu:0'):
    train1_autoencoder.fit(train_batches, train_batches,
                               epochs=eps, batch_size=bs_z, verbose=1)

    train1_autoencoder.save_weights('weights/'+name+'_train1.h5')

    ### pretrain2

    print('pretrain #2')

    creator_train2 = Model(input_img, train1_c1)
    creator_train2.get_layer("t1c1").set_weights(train1_autoencoder.get_layer('t1c1').get_weights())

    #input_train2 = creator_train2.predict(train_batches)
    #print(input_train2.shape)

    #---

    input_img_train2=Input(shape=(size, size, size, maxfil))
    train2_c1=Conv3D(maxfil, (3, 3, 3), activation='sigmoid', padding='same', name='t2c1', activity_regularizer=kl_reg)(input_img_train2)
    train2_decoded=Conv3D(maxfil, (3, 3, 3), activation='sigmoid', padding='same', name='t2tc1')(train2_c1)

    train2_autoencoder = Model(input_img_train2, train2_decoded)
    train2_autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

    #with tf.device('/gpu:0'):
    train2_autoencoder.fit(creator_train2.predict(train_batches), creator_train2.predict(train_batches), epochs=eps, batch_size=bs_z, verbose=1)

    train2_autoencoder.save_weights('weights/'+name+'_train2.h5')

    ### pretrain3

    print('pretrain #3')

    creator_train3 = Model(input_img_train2, train2_c1)
    creator_train3.get_layer("t2c1").set_weights(train2_autoencoder.get_layer('t2c1').get_weights())

    #input_train3 = creator_train3.predict(input_train2)
    #print(input_train3.shape)

    #---

    input_img_train3=Input(shape=(size, size, size, maxfil))
    train3_c1=Conv3D(int(maxfil/2), (3, 3, 3), activation='sigmoid', padding='same', name='t3c1', activity_regularizer=kl_reg)(input_img_train3)
    train3_decoded=Conv3D(maxfil, (3, 3, 3), activation='sigmoid', padding='same', name='t3tc1')(train3_c1)

    train3_autoencoder = Model(input_img_train3, train3_decoded)
    train3_autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

    #with tf.device('/gpu:0'):
    train3_autoencoder.fit(creator_train3.predict(creator_train2.predict(train_batches)), creator_train3.predict(creator_train2.predict(train_batches)), epochs=eps, batch_size=bs_z, verbose=1)

    train3_autoencoder.save_weights('weights/'+name+'_train3.h5')

    ### pretrain4

    print('pretrain #4')

    creator_train4 = Model(input_img_train3, train3_c1)
    creator_train4.get_layer("t3c1").set_weights(train3_autoencoder.get_layer('t3c1').get_weights())

    #input_train4 = creator_train4.predict(creator_train3.predict(creator_train2.predict(train_batches)))
    #print(input_train4.shape)

    #---

    input_img_train4=Input(shape=(size, size, size, int(maxfil/2)))
    train4_p1=MaxPooling3D(pool_size=(3, 3, 3), name='t4p1')(input_img_train4)
    train4_c1=Conv3D(int(maxfil/4), (3, 3, 3), activation='sigmoid', padding='same', name='t4c1', activity_regularizer=kl_reg)(train4_p1)
    train4_tc1=Conv3D(int(maxfil/2), (3, 3, 3), activation='sigmoid', padding='same', name='t4tc1')(train4_c1)
    train4_decoded=UpSampling3D(size=(3, 3, 3), name='t4u1')(train4_tc1)

    train4_autoencoder = Model(input_img_train4, train4_decoded)
    train4_autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

    #with tf.device('/gpu:0'):
    train4_autoencoder.fit(creator_train4.predict(creator_train3.predict(creator_train2.predict(train_batches))), creator_train4.predict(creator_train3.predict(creator_train2.predict(train_batches))), epochs=eps, batch_size=bs_z, verbose=1)

    train4_autoencoder.save_weights('weights/'+name+'_train4.h5')

### unroll/train

def unrollAndTrain(train_batches,name,size,maxfil,bs_z,eps):

    ### unroll

    input_img = Input(shape=(size, size, size,1),dtype='float32')

    c1 = Conv3D(maxfil, (3, 3, 3), activation='sigmoid', padding='same', name='t1c1', activity_regularizer=kl_reg)(input_img)
    c2 = Conv3D(maxfil, (3, 3, 3), activation='sigmoid', padding='same', name='t2c1', activity_regularizer=kl_reg)(c1)
    c3 = Conv3D(int(maxfil/2), (3, 3, 3), activation='sigmoid', padding='same', name='t3c1', activity_regularizer=kl_reg)(c2)
    p3 = MaxPooling3D(pool_size=(3, 3, 3), name='t4p1')(c3)
    encoded = Conv3D(int(maxfil/4), (3, 3, 3), activation='sigmoid', padding='same', name='t4c1', activity_regularizer=kl_reg)(p3)
    tc4 = Conv3D(int(maxfil/2), (3, 3, 3), activation='sigmoid', padding='same', name='t4tc1')(encoded)
    u3 = UpSampling3D(size=(3, 3, 3), name='t4u1')(tc4)
    tc3 = Conv3D(maxfil, (3, 3, 3), activation='sigmoid', padding='same', name='t3tc1')(u3)
    tc2 = Conv3D(maxfil, (3, 3, 3), activation='sigmoid', padding='same', name='t2tc1')(tc3)
    decoded = Conv3D(1, (3, 3, 3), activation='sigmoid', padding='same', name='t1tc1')(tc2)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='mean_squared_error', metrics=['accuracy'])

    autoencoder.load_weights('weights/'+name+'_train1.h5', by_name=True)
    autoencoder.load_weights('weights/'+name+'_train2.h5', by_name=True)
    autoencoder.load_weights('weights/'+name+'_train3.h5', by_name=True)
    autoencoder.load_weights('weights/'+name+'_train4.h5', by_name=True)

    ### finetune
    #with tf.device('/gpu:0'):
    autoencoder.fit(train_batches, train_batches, epochs=eps, batch_size=bs_z, verbose=1)

    autoencoder.save_weights('weights/'+name+'_autoencoder.h5')

    encoder = Model(input_img, encoded)
    encoder.compile(optimizer='adadelta', loss='mean_squared_error')

    return autoencoder, encoder

### load autoencoder

def loadAutoencoder(name,size,maxfil):

    input_img = Input(shape=(size, size, size,1))

    c1 = Conv3D(maxfil, (3, 3, 3), activation='sigmoid', padding='same', name='t1c1', activity_regularizer=kl_reg)(input_img)
    c2 = Conv3D(maxfil, (3, 3, 3), activation='sigmoid', padding='same', name='t2c1', activity_regularizer=kl_reg)(c1)
    c3 = Conv3D(int(maxfil/2), (3, 3, 3), activation='sigmoid', padding='same', name='t3c1', activity_regularizer=kl_reg)(c2)
    p3 = MaxPooling3D(pool_size=(3, 3, 3), name='t4p1')(c3)
    encoded = Conv3D(int(maxfil/4), (3, 3, 3), activation='sigmoid', padding='same', name='t4c1', activity_regularizer=kl_reg)(p3)
    tc4 = Conv3D(int(maxfil/2), (3, 3, 3), activation='sigmoid', padding='same', name='t4tc1')(encoded)
    u3 = UpSampling3D(size=(3, 3, 3), name='t4u1')(tc4)
    tc3 = Conv3D(maxfil, (3, 3, 3), activation='sigmoid', padding='same', name='t3tc1')(u3)
    tc2 = Conv3D(maxfil, (3, 3, 3), activation='sigmoid', padding='same', name='t2tc1')(tc3)
    decoded = Conv3D(1, (3, 3, 3), activation='sigmoid', padding='same', name='t1tc1')(tc2)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

    autoencoder.load_weights('weights/'+name+'_autoencoder.h5')

    encoder = Model(input_img, encoded)
    encoder.compile(optimizer='adadelta', loss='mean_squared_error')

    return autoencoder, encoder

### per pretrain per pic

#load batches of one pic
def loadOnly(path, file, size, overlap=False):

    batches = []

    img = nilearn.image.load_img(path + file).get_fdata()
    log = np.absolute(gaussian_laplace(img,1))

    maxval=max(img.flatten())
    minval=min(img.flatten())
    img=(img-minval)/(maxval-minval)

    maxval=max(log.flatten())
    minval=min(log.flatten())
    log=(log-minval)/(maxval-minval)

    if overlap:
        x_cuts=2*int(len(img)/size)-1
        y_cuts=2*int(len(img[0])/size)-1
        z_cuts=2*int(len(img[0][0])/size)-1
    else:
        x_cuts=int(len(img)/size)
        y_cuts=int(len(img[0])/size)
        z_cuts=int(len(img[0][0])/size)
    x_rest=len(img)%size
    y_rest=len(img[0])%size
    z_rest=len(img[0][0])%size

    pos=np.zeros((x_cuts,x_cuts,z_cuts,3))
    mean_log=np.zeros((x_cuts,x_cuts,z_cuts))

    for i in range(x_cuts):
        if overlap:
            x1=int(size*i/2)+int(x_rest/2)
        else:
            x1=size*i+int(x_rest/2)
        x2=x1+size
        for j in range(y_cuts):
            if overlap:
                y1=int(size*j/2)+int(y_rest/2)
            else:
                y1=size*j+int(y_rest/2)
            y2=y1+size
            for k in range(z_cuts):
                if overlap:
                    z1=int(size*k/2)+int(z_rest/2)
                else:
                    z1=size*k+int(z_rest/2)
                z2=z1+size
                mean_log[i,j,k]=np.mean(log[x1:x2,y1:y2,z1:z2].flatten())
                pos[i,j,k]=np.array([x1,y1,z1])

    med_log=np.percentile(mean_log.flatten(),95)

    good_pos=[]

    for i in range(x_cuts):
        for j in range(y_cuts):
            for k in range(z_cuts):
                if med_log < mean_log[i,j,k]:
                    good_pos.append(pos[i,j,k])

    for i in range(len(good_pos)):
        x1=int(good_pos[i][0])
        x2=x1+size
        y1=int(good_pos[i][1])
        y2=y1+size
        z1=int(good_pos[i][2])
        z2=z1+size

        batches.append(img[x1:x2,y1:y2,z1:z2])

    batches=np.asarray(batches)
    shape=np.shape(batches)
    return batches.reshape(shape[0],shape[1],shape[2],shape[3],1)

def pretrain1(train_batches,name,size,maxfil,bs_z,eps,p,load_name=None):

    K.set_learning_phase(1)

    input_img = Input(shape=(size, size,size,1), dtype='float32')

    train1_c1=Conv3D(maxfil, (3, 3, 3), activation='sigmoid', padding='same', name='t1c1', activity_regularizer=kl_reg)(input_img)
    train1_decoded=Conv3D(1, (3, 3, 3), activation='sigmoid', padding='same', name='t1tc1')(train1_c1)

    train1_autoencoder = Model(input_img, train1_decoded)
    train1_autoencoder.compile(optimizer='adadelta', loss='mean_squared_error', metrics=['accuracy'])

    if p==0:
        if load_name != None:
            train1_autoencoder.load_weights('weights/'+load_name+'_train1.h5')
        train1_autoencoder.fit(train_batches, train_batches, epochs=eps, batch_size=bs_z, verbose=1)
    else:
        train1_autoencoder.load_weights('weights/'+name+'_train1.h5', by_name=True)
        train1_autoencoder.fit(train_batches, train_batches, epochs=eps, batch_size=bs_z, verbose=1)

    train1_autoencoder.save_weights('weights/'+name+'_train1.h5')

    del train_batches
    del train1_autoencoder
    del input_img
    K.clear_session
    K.set_learning_phase(0)
    gc.collect()

def pretrain2(train_batches,name,size,maxfil,bs_z,eps,p,load_name=None):

    K.set_learning_phase(1)

    input_img = Input(shape=(size, size,size,1), dtype='float32')
    train1_c1=Conv3D(maxfil, (3, 3, 3), activation='sigmoid', padding='same', name='t1c1', activity_regularizer=kl_reg)(input_img)

    creator_train2 = Model(input_img, train1_c1)
    creator_train2.load_weights('weights/'+name+'_train1.h5', by_name=True)

    input_train2 = creator_train2.predict(train_batches)

    #---

    input_img_train2=Input(shape=(size, size, size, maxfil))
    train2_c1=Conv3D(maxfil, (3, 3, 3), activation='sigmoid', padding='same', name='t2c1', activity_regularizer=kl_reg)(input_img_train2)
    train2_decoded=Conv3D(maxfil, (3, 3, 3), activation='sigmoid', padding='same', name='t2tc1')(train2_c1)

    train2_autoencoder = Model(input_img_train2, train2_decoded)
    train2_autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

    if p==0:
        if load_name != None:
            train2_autoencoder.load_weights('weights/'+load_name+'_train2.h5')
        train2_autoencoder.fit(input_train2, input_train2, epochs=eps, batch_size=bs_z, verbose=1)
    else:
        train2_autoencoder.load_weights('weights/'+name+'_train2.h5', by_name=True)
        train2_autoencoder.fit(input_train2, input_train2, epochs=eps, batch_size=bs_z, verbose=1)

    train2_autoencoder.save_weights('weights/'+name+'_train2.h5')

    del creator_train2
    del train2_autoencoder
    del train_batches
    del input_train2
    del input_img
    K.clear_session
    K.set_learning_phase(0)
    gc.collect()

def pretrain3(train_batches,name,size,maxfil,bs_z,eps,p,load_name=None):

    K.set_learning_phase(1)

    input_img = Input(shape=(size, size,size,1), dtype='float32')
    train1_c1=Conv3D(maxfil, (3, 3, 3), activation='sigmoid', padding='same', name='t1c1', activity_regularizer=kl_reg)(input_img)

    creator_train2 = Model(input_img, train1_c1)
    creator_train2.load_weights('weights/'+name+'_train1.h5', by_name=True)

    input_img_train2=Input(shape=(size, size, size, maxfil))
    train2_c1=Conv3D(maxfil, (3, 3, 3), activation='sigmoid', padding='same', name='t2c1', activity_regularizer=kl_reg)(input_img_train2)

    creator_train3 = Model(input_img, train2_c1)
    creator_train3.load_weights('weights/'+name+'_train2.h5', by_name=True)

    input_train3 = creator_train3.predict(creator_train2.predict(train_batches))

    #---

    input_img_train3=Input(shape=(size, size, size, maxfil))
    train3_c1=Conv3D(maxfil, (3, 3, 3), activation='sigmoid', padding='same', name='t3c1', activity_regularizer=kl_reg)(input_img_train3)
    train3_decoded=Conv3D(maxfil, (3, 3, 3), activation='sigmoid', padding='same', name='t3tc1')(train3_c1)

    train3_autoencoder = Model(input_img_train3, train3_decoded)
    train3_autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

    if p==0:
        if load_name != None:
            train3_autoencoder.load_weights('weights/'+load_name+'_train3.h5')
        train3_autoencoder.fit(input_train3, input_train3, epochs=eps, batch_size=bs_z, verbose=1)
    else:
        train3_autoencoder.load_weights('weights/'+name+'_train3.h5', by_name=True)
        train3_autoencoder.fit(input_train3, input_train3, epochs=eps, batch_size=bs_z, verbose=1)

    train3_autoencoder.save_weights('weights/'+name+'_train3.h5')

    del creator_train2
    del creator_train3
    del train3_autoencoder
    del train_batches
    del input_train3
    del input_img
    K.clear_session
    K.set_learning_phase(0)
    gc.collect()

def pretrain4(train_batches,name,size,maxfil,bs_z,eps,p,load_name=None):

    K.set_learning_phase(1)

    input_img = Input(shape=(size, size,size,1), dtype='float32')
    train1_c1=Conv3D(maxfil, (3, 3, 3), activation='sigmoid', padding='same', name='t1c1', activity_regularizer=kl_reg)(input_img)

    creator_train2 = Model(input_img, train1_c1)
    creator_train2.load_weights('weights/'+name+'_train1.h5', by_name=True)

    input_img_train2=Input(shape=(size, size, size, maxfil))
    train2_c1=Conv3D(maxfil, (3, 3, 3), activation='sigmoid', padding='same', name='t2c1', activity_regularizer=kl_reg)(input_img_train2)

    creator_train3 = Model(input_img_train2, train2_c1)
    creator_train3.load_weights('weights/'+name+'_train2.h5', by_name=True)

    input_img_train3=Input(shape=(size, size, size, maxfil))
    train3_c1=Conv3D(maxfil, (3, 3, 3), activation='sigmoid', padding='same', name='t3c1', activity_regularizer=kl_reg)(input_img_train3)

    creator_train4 = Model(input_img_train3, train3_c1)
    creator_train4.load_weights('weights/'+name+'_train3.h5', by_name=True)

    input_train4 = creator_train4.predict(creator_train3.predict(creator_train2.predict(train_batches)))

    #---

    input_img_train4=Input(shape=(size, size, size, int(maxfil/2)))
    train4_p1=MaxPooling3D(pool_size=(3, 3, 3), name='t4p1')(input_img_train4)
    train4_c1=Conv3D(int(maxfil/4), (3, 3, 3), activation='sigmoid', padding='same', name='t4c1', activity_regularizer=kl_reg)(train4_p1)
    train4_tc1=Conv3D(int(maxfil/2), (3, 3, 3), activation='sigmoid', padding='same', name='t4tc1')(train4_c1)
    train4_decoded=UpSampling3D(size=(3, 3, 3), name='t4u1')(train4_tc1)

    train4_autoencoder = Model(input_img_train4, train4_decoded)
    train4_autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

    if p==0:
        if load_name != None:
            train4_autoencoder.load_weights('weights/'+load_name+'_train4.h5')
        train4_autoencoder.fit(input_train4, input_train4, epochs=eps, batch_size=bs_z, verbose=1)
    else:
        train4_autoencoder.load_weights('weights/'+name+'_train4.h5', by_name=True)
        train4_autoencoder.fit(input_train4, input_train4, epochs=eps, batch_size=bs_z, verbose=1)

    train4_autoencoder.save_weights('weights/'+name+'_train4.h5')

    del creator_train2
    del creator_train3
    del creator_train4
    del train3_autoencoder
    del train_batches
    del input_train4
    del input_img
    K.clear_session
    K.set_learning_phase(0)
    gc.collect()

def pretrainPerPic(path, size, name, maxfil, eps, overlap=False):
    listing = os.listdir(path)

    p=0
    for file in listing:
        batches = loadOnly(path, file, size, overlap)
        if overlap:
            bs_z=int(np.shape(batches)[0]/16)
        else:
            bs_z=int(np.shape(batches)[0]/2)
        pretrain1(batches,name,size,maxfil,bs_z,eps,p)
        p+=1

    p=0
    for file in listing:
        if overlap:
            bs_z=int(np.shape(batches)[0]/16)
        else:
            bs_z=int(np.shape(batches)[0]/2)
        batches = loadOnly(path, file, size, overlap)
        pretrain2(batches,name,size,maxfil,bs_z,eps,p)
        p+=1

    p=0
    for file in listing:
        if overlap:
            bs_z=int(np.shape(batches)[0]/16)
        else:
            bs_z=int(np.shape(batches)[0]/2)
        batches = loadOnly(path, file, size, overlap)
        pretrain3(batches,name,size,maxfil,bs_z,eps,p)
        p+=1

    p=0
    for file in listing:
        if overlap:
            bs_z=int(np.shape(batches)[0]/16)
        else:
            bs_z=int(np.shape(batches)[0]/2)
        batches = loadOnly(path, file, size, overlap)
        pretrain4(batches,name,size,maxfil,bs_z,eps,p)
        p+=1

#single pretrain
def pretrain1_perPic(path, size, name, maxfil, eps, overlap=False,load_name=None):
    listing = os.listdir(path)

    p=0
    for file in listing:
        batches = loadOnly(path, file, size, overlap)
        if overlap:
            bs_z=int(np.shape(batches)[0]/32)
        else:
            bs_z=int(np.shape(batches)[0]/4)
        pretrain1(batches,name,size,maxfil,bs_z,eps,p,load_name)
        p+=1
        print('pretrain 1: MRI ', p)

        del batches
        gc.collect()

def pretrain2_perPic(path, size, name, maxfil, eps, overlap=False,load_name=None):
    listing = os.listdir(path)

    p=0
    for file in listing:
        print(file)
        batches = loadOnly(path, file, size, overlap)
        if overlap:
            bs_z=int(np.shape(batches)[0]/32)
        else:
            bs_z=int(np.shape(batches)[0]/4)
        pretrain2(batches,name,size,maxfil,bs_z,eps,p,load_name)
        p+=1
        print('pretrain 2: MRI ', p)

        del batches
        gc.collect()

def pretrain3_perPic(path, size, name, maxfil, eps, overlap=False,load_name=None):
    listing = os.listdir(path)

    p=0
    for file in listing:
        batches = loadOnly(path, file, size, overlap)
        if overlap:
            bs_z=int(np.shape(batches)[0]/32)
        else:
            bs_z=int(np.shape(batches)[0]/4)
        pretrain3(batches,name,size,maxfil,bs_z,eps,p,load_name)
        p+=1
        print('pretrain 3: MRI ', p)

        del batches
        gc.collect()

def pretrain4_perPic(path, size, name, maxfil, eps, overlap=False,load_name=None):
    listing = os.listdir(path)

    p=0
    for file in listing:
        batches = loadOnly(path, file, size, overlap)
        if overlap:
            bs_z=int(np.shape(batches)[0]/32)
        else:
            bs_z=int(np.shape(batches)[0]/4)
        pretrain4(batches,name,size,maxfil,bs_z,eps,p,load_name)
        p+=1
        print('pretrain 4: MRI ', p)

        del batches
        gc.collect()

### get grads

def getMaxGrads(input_set,name,size,maxfil,layer_name, n=None, nof=None):

    input_img = Input(shape=(size, size, size,1))

    c1 = Conv3D(maxfil, (3, 3, 3), activation='sigmoid', padding='same', name='t1c1', activity_regularizer=kl_reg)(input_img)
    c2 = Conv3D(maxfil, (3, 3, 3), activation='sigmoid', padding='same', name='t2c1', activity_regularizer=kl_reg)(c1)
    c3 = Conv3D(int(maxfil/2), (3, 3, 3), activation='sigmoid', padding='same', name='t3c1', activity_regularizer=kl_reg)(c2)
    p3 = MaxPooling3D(pool_size=(3, 3, 3), name='t4p1')(c3)
    encoded = Conv3D(int(maxfil/4), (3, 3, 3), activation='sigmoid', padding='same', name='t4c1', activity_regularizer=kl_reg)(p3)
    tc4 = Conv3D(int(maxfil/2), (3, 3, 3), activation='sigmoid', padding='same', name='t4tc1')(encoded)
    u3 = UpSampling3D(size=(3, 3, 3), name='t4u1')(tc4)
    tc3 = Conv3D(maxfil, (3, 3, 3), activation='sigmoid', padding='same', name='t3tc1')(u3)
    tc2 = Conv3D(maxfil, (3, 3, 3), activation='sigmoid', padding='same', name='t2tc1')(tc3)
    decoded = Conv3D(1, (3, 3, 3), activation='sigmoid', padding='same', name='t1tc1')(tc2)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

    autoencoder.load_weights('weights/'+name+'_autoencoder.h5')

    layer_dict = dict([(layer.name, layer) for layer in autoencoder.layers[1:]])

    if n == None:
        len1=len(input_set)
    else:
        len1=n
    len2=len(input_set[0])
    len3=len(input_set[0][0])
    len4=len(input_set[0][0][0])

    out = np.zeros((len1,len2,len3,len4))
    out_fil = np.zeros((len1,len2,len3,len4))

    if nof == None:
        nof=int(maxfil/4)

    for i in range(nof):
        filter_index = i

        layer_output = layer_dict[layer_name].output
        loss = K.mean(layer_output[:, :, :, :, filter_index])

        # compute the gradient of the input picture wrt this loss
        grads = K.gradients(loss, input_img)[0]

        #normalization trick: we normalize the gradient
        grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)  ### normalize? ... later

        # this function returns the loss and grads given the input picture
        iterate = K.function([input_img], [loss, grads])

        it = iterate([input_set])

        for j in range(len1):
            for k in range(len2):
                for l in range(len3):
                    for m in range(len4):
                        if np.mean(abs(it[1][j][k][l][m])) > out[j][k][l][m]:   #abs()?
                            out[j][k][l]=abs(np.mean(it[1][j][k][l]))

        for j in range(len1):
            q = np.percentile(out[i],99)
            for k in range(len2):
                for l in range(len3):
                    for m in range(len4):
                        if(out[j][k][l][m]>q):
                            out_fil[j][k][l][m]=out[j][k][l][m]

        print('patch done')

    return out, out_fil
