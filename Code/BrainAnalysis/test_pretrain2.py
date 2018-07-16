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
from keras import backend as K

import matplotlib.pyplot as plt
# %matplotlib inline

from scipy.ndimage.filters import gaussian_laplace
import matplotlib.pyplot as plt
import matplotlib.colors as mcol
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import gc

import nilearn
from nilearn import plotting

import brain_functions_final as bf

### Create input

size=21

#train_path = "C:/Users/A2/Documents/CDS/MA/MA/test_pics/"
#test_path = "C:/Users/A2/Documents/CDS/MA/MA/test_pics/"

train_path = "/home/hoffmann/MRI/preprocessed/pre_train_data/"
test_path = "/home/hoffmann/MRI/preprocessed/pre_test_data/"

#bf.show_skull(test_path)

### Create Encoder

maxfil=512
bs_z=16
eps=5

name='512_maxfil_bsz_16_eps_5'

print('loading pretraining')

#bf.pretrainPerPic(train_path, size, name, maxfil, eps, True)

bf.pretrain1_perPic(train_path, size, name, maxfil, eps, True)
gc.collect()

bf.pretrain2_perPic(train_path, size, name, maxfil, eps, True)
gc.collect()

bf.pretrain3_perPic(train_path, size, name, maxfil, eps, True)
gc.collect()

bf.pretrain4_perPic(train_path, size, name, maxfil, eps, True)
gc.collect()


print('training')

train_batches, train_allpos=bf.loadbatches(train_path, size, True)
print(np.shape(train_batches))
autoencoder, encoder = bf.unrollAndTrain(train_batches,name,size,maxfil,bs_z,eps*10)

print('done')
