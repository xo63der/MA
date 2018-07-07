###### %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

# convolutional autoencoder in keras

from __future__ import print_function
from keras.models import Model
from keras.layers import *
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras import backend as K

import os
#os.environ["KERAS_BACKEND"] = "tensorflow"

import matplotlib.pyplot as plt
# %matplotlib inline

from keras.datasets import mnist 

from scipy.ndimage.filters import gaussian_laplace
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
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

### load/preprocess data

def loadbatches(path, size, select=True):

    batches_temp = []
    log_batches_temp = []
    
    allpositions=[]
    
    listing = os.listdir(path)
    
    for file in listing:
        
        img = nilearn.image.load_img(path + file).get_fdata()
        log = np.absolute(gaussian_laplace(img,1))
    
        img=np.asarray(img)                
        maxval=max(img.flatten())
        minval=min(img.flatten())
        img=(img-minval)/(maxval-minval)
        
        log=np.asarray(log)                
        maxval=max(log.flatten())
        minval=min(log.flatten())
        log=(log-minval)/(maxval-minval)
        
        x_cuts=int(len(img)/size)
        y_cuts=int(len(img[0])/size)
        z_cuts=int(len(img[0][0])/size)
        x_rest=len(img)%size
        y_rest=len(img[0])%size
        z_rest=len(img[0][0])%size
        
        pos=[]
        
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
                    log_cut=log[x1:x2,y1:y2,z1:z2]
                    batches_temp.append(im_cut)
                    log_batches_temp.append(log_cut)
                    pos.append(np.array([x1,y1,z1]))
        
        allpositions.append(pos)
    
    batches_temp=np.asarray(batches_temp)                
    batches=[]
    
    vip=[]
                    
    if select :                
        log_mean=np.zeros((len(batches_temp)))
        for i in range(len(batches_temp)):
            log_mean[i]=np.mean(log_batches_temp[i].flatten())

        k=0
        med_log_mean=np.percentile(log_mean,99)
        for i in range(len(allpositions)):
            for j in range(len(allpositions[i])):
                if log_mean[k]>med_log_mean:
                    batches.append(batches_temp[k])
                    vip.append(np.array([i,j]))
                k=k+1
    
    else : batches = batches_temp
    
    batches=np.asarray(batches)
    
    shape=np.shape(batches)
    return batches.reshape(shape[0],shape[1],shape[2],shape[3],1),allpositions,vip

### pretrain

def pretrain(train_batches,test_batches,name,size,maxfil,bs_z,eps):   
    
    ### pretrain1

    print('pretrain #1')
    
    input_img = Input(shape=(size, size,size,1), dtype='float64')

    train1_c1=Conv3D(maxfil, (3, 3, 3), activation='sigmoid', padding='same', name='t1c1')(input_img)
    train1_decoded=Conv3D(1, (3, 3, 3), activation='sigmoid', padding='same', name='t1tc1')(train1_c1)

    train1_autoencoder = Model(input_img, train1_decoded)
    train1_autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

    train1_autoencoder.fit(train_batches, train_batches, 
                           validation_data=(test_batches, test_batches),
                           epochs=eps, batch_size=bs_z, verbose=1)

    train1_autoencoder.save_weights('weights/'+name+'_train1.h5')

    #show_batch(test_batches, size, train1_autoencoder.predict(test_batches))

    ### pretrain2

    print('pretrain #2')

    creator_train2 = Model(input_img, train1_c1)
    creator_train2.get_layer("t1c1").set_weights(train1_autoencoder.get_layer('t1c1').get_weights())

    input_train2 = creator_train2.predict(train_batches)

    print(input_train2.shape)

    #---

    input_img_train2=Input(shape=(size, size, size, maxfil))
    train2_c1=Conv3D(maxfil, (3, 3, 3), activation='sigmoid', padding='same', name='t2c1')(input_img_train2)
    train2_decoded=Conv3D(maxfil, (3, 3, 3), activation='sigmoid', padding='same', name='t2tc1')(train2_c1)

    train2_autoencoder = Model(input_img_train2, train2_decoded)
    train2_autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

    train2_autoencoder.fit(input_train2, input_train2, epochs=eps, batch_size=bs_z, verbose=1)
    
    train2_autoencoder.save_weights('weights/'+name+'_train2.h5')

    ### pretrain3

    print('pretrain #3')

    creator_train3 = Model(input_img_train2, train2_c1)
    creator_train3.get_layer("t2c1").set_weights(train2_autoencoder.get_layer('t2c1').get_weights())

    input_train3 = creator_train3.predict(input_train2)

    print(input_train3.shape)

    #---

    input_img_train3=Input(shape=(size, size, size, maxfil))
    train3_c1=Conv3D(int(maxfil/2), (3, 3, 3), activation='sigmoid', padding='same', name='t3c1')(input_img_train3)
    train3_decoded=Conv3D(maxfil, (3, 3, 3), activation='sigmoid', padding='same', name='t3tc1')(train3_c1)

    train3_autoencoder = Model(input_img_train3, train3_decoded)
    train3_autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

    train3_autoencoder.fit(input_train3, input_train3, epochs=eps, batch_size=bs_z, verbose=1)
    
    train3_autoencoder.save_weights('weights/'+name+'_train3.h5')

    ### pretrain4

    print('pretrain #4')

    creator_train4 = Model(input_img_train3, train3_c1)
    creator_train4.get_layer("t3c1").set_weights(train3_autoencoder.get_layer('t3c1').get_weights())

    input_train4 = creator_train4.predict(input_train3)

    print(input_train4.shape)

    #---

    input_img_train4=Input(shape=(size, size, size, int(maxfil/2)))
    train4_p1=MaxPooling3D(pool_size=(3, 3, 3), name='t4p1')(input_img_train4)
    train4_c1=Conv3D(int(maxfil/4), (3, 3, 3), activation='sigmoid', padding='same', name='t4c1')(train4_p1)
    train4_tc1=Conv3D(int(maxfil/2), (3, 3, 3), activation='sigmoid', padding='same', name='t4tc1')(train4_c1)
    train4_decoded=UpSampling3D(size=(3, 3, 3), name='t4u1')(train4_tc1)

    train4_autoencoder = Model(input_img_train4, train4_decoded)
    train4_autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

    train4_autoencoder.fit(input_train4, input_train4, epochs=eps, batch_size=bs_z, verbose=1)
    
    train4_autoencoder.save_weights('weights/'+name+'_train4.h5')

### unroll/train

def unrollAndTrain(train_batches,test_batches,name,size,maxfil,bs_z,eps):

    ### unroll

    input_img = Input(shape=(size, size, size,1))

    c1 = Conv3D(maxfil, (3, 3, 3), activation='sigmoid', padding='same', name='t1c1')(input_img)
    c2 = Conv3D(maxfil, (3, 3, 3), activation='sigmoid', padding='same', name='t2c1')(c1)
    c3 = Conv3D(int(maxfil/2), (3, 3, 3), activation='sigmoid', padding='same', name='t3c1')(c2)
    p3 = MaxPooling3D(pool_size=(3, 3, 3), name='t4p1')(c3)
    encoded = Conv3D(int(maxfil/4), (3, 3, 3), activation='sigmoid', padding='same', name='t4c1')(p3)
    tc4 = Conv3D(int(maxfil/2), (3, 3, 3), activation='sigmoid', padding='same', name='t4tc1')(encoded)
    u3 = UpSampling3D(size=(3, 3, 3), name='t4u1')(tc4)
    tc3 = Conv3D(maxfil, (3, 3, 3), activation='sigmoid', padding='same', name='t3tc1')(u3)
    tc2 = Conv3D(maxfil, (3, 3, 3), activation='sigmoid', padding='same', name='t2tc1')(tc3)
    decoded = Conv3D(1, (3, 3, 3), activation='sigmoid', padding='same', name='t1tc1')(tc2)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

    autoencoder.load_weights('weights/'+name+'_train1.h5', by_name=True)
    autoencoder.load_weights('weights/'+name+'_train2.h5', by_name=True)
    autoencoder.load_weights('weights/'+name+'_train3.h5', by_name=True)
    autoencoder.load_weights('weights/'+name+'_train4.h5', by_name=True)

    ### finetune

    autoencoder.fit(train_batches, train_batches, epochs=eps, validation_data=(test_batches, test_batches), batch_size=bs_z, verbose=1)
    
    autoencoder.save_weights('weights/'+name+'_autoencoder.h5')

    encoder = Model(input_img, encoded)
    encoder.compile(optimizer='adadelta', loss='mean_squared_error')

    return autoencoder, encoder

### load autoencoder

def loadAutoencoder(name,size,maxfil):

    input_img = Input(shape=(size, size, size,1))

    c1 = Conv3D(maxfil, (3, 3, 3), activation='sigmoid', padding='same', name='t1c1')(input_img)
    c2 = Conv3D(maxfil, (3, 3, 3), activation='sigmoid', padding='same', name='t2c1')(c1)
    c3 = Conv3D(int(maxfil/2), (3, 3, 3), activation='sigmoid', padding='same', name='t3c1')(c2)
    p3 = MaxPooling3D(pool_size=(3, 3, 3), name='t4p1')(c3)
    encoded = Conv3D(int(maxfil/4), (3, 3, 3), activation='sigmoid', padding='same', name='t4c1')(p3)
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

### get grads

def getMaxGrads(input_set,name,size,maxfil,layer_name):

    input_img = Input(shape=(size, size, size,1))

    c1 = Conv3D(maxfil, (3, 3, 3), activation='sigmoid', padding='same', name='t1c1')(input_img)
    c2 = Conv3D(maxfil, (3, 3, 3), activation='sigmoid', padding='same', name='t2c1')(c1)
    c3 = Conv3D(int(maxfil/2), (3, 3, 3), activation='sigmoid', padding='same', name='t3c1')(c2)
    p3 = MaxPooling3D(pool_size=(3, 3, 3), name='t4p1')(c3)
    encoded = Conv3D(int(maxfil/4), (3, 3, 3), activation='sigmoid', padding='same', name='t4c1')(p3)
    tc4 = Conv3D(int(maxfil/2), (3, 3, 3), activation='sigmoid', padding='same', name='t4tc1')(encoded)
    u3 = UpSampling3D(size=(3, 3, 3), name='t4u1')(tc4)
    tc3 = Conv3D(maxfil, (3, 3, 3), activation='sigmoid', padding='same', name='t3tc1')(u3)
    tc2 = Conv3D(maxfil, (3, 3, 3), activation='sigmoid', padding='same', name='t2tc1')(tc3)
    decoded = Conv3D(1, (3, 3, 3), activation='sigmoid', padding='same', name='t1tc1')(tc2)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

    autoencoder.load_weights('weights/'+name+'_autoencoder.h5')

    layer_dict = dict([(layer.name, layer) for layer in autoencoder.layers[1:]])
    nof = int(maxfil/4)

    len1=len(input_set)
    len2=len(input_set[0])
    len3=len(input_set[0][0])
    len4=len(input_set[0][0][0])

    out = np.zeros((len1,len1,len1,len1))
    out_fil = np.zeros((len1,len1,len1,len1))

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
                        #if np.mean(it[1][j][k][l]) > out[j][k][l]:   #abs()?
                            out[j][k][l]=np.mean(it[1][j][k][l])

        for j in range(len1):
            q = np.percentile(out[i],99)
            for k in range(len2):
                for l in range(len3):
                    for m in range(len4):
                        if(out[j][k][l][m]>q):
                            out_fil[j][k][l][m]=out[j][k][l][m]
                        
    return out, out_fil