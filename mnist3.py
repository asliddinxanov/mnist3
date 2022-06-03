#!/usr/bin/env python3

'''MNIST Sample Ver.3'''

### Library Declaration ###
import numpy  as np
from   pandas import DataFrame
import matplotlib.pyplot as plt
import tensorflow        as tf
from   PIL               import Image, ImageOps
from   tensorflow.keras.callbacks           import CSVLogger, ModelCheckpoint
from   tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img
from   tensorflow.keras.models import *
from   tensorflow.keras.layers import *
import sys, os
import time
import csv
import argparse
import glob

### MNIST dataset download ###
# Download to ~/.keras/datasets/mnist.npz
from tensorflow.keras.datasets import mnist

### Global Variables ###
dataset = None
result  = None

# User Defined Model
def udf_model():
    model = Sequential()
    model.add(Reshape((28, 28, 1), input_shape=(28*28,), name='reshape'))

    ### Please modify python code FROM HERE (Please start from column 4) ###



    ### Please modify python code TO HERE  ###
    model.add(Flatten(name='flatten_final'))
    model.add(Dense(10, activation='softmax', name='softmax'))
    return model

# Check Arguments
def checkargs():
    # Gloval vaiables (too many return values...)
    global INETWORK, IEPOCH, ISEED, RATIO, IGRAPHIC, OUTDIR, INDIR, ILOAD

    # Parse Arguments
    parser = argparse.ArgumentParser(description='MNIST Sample Program')
    parser.add_argument('-n','--INETWORK', type=int,   help='NEURAL NETWORK (0:Single Dense Layer / 1:CNN / 2:User Defined)', choices=[0,1,2])
    parser.add_argument('-e','--IEPOCH',   type=int,   help='Number of epochs (Training)')
    parser.add_argument('-r','--RATIO',    type=float, help='Ratio of validation/(training+validation) [0.0-1.0]')
    parser.add_argument('-s','--ISEED',    type=int,   help='Seed of pseudo random number')
    parser.add_argument('-g','--IGRAPHIC', type=int,   help='Graphic Display (0:Only File-output / 1:Display Window & File-output)', choices=[0,1])
    parser.add_argument('-o','--OUTDIR',               help='Result Directory')
    parser.add_argument('-i','--INDIR',                help='Input Directory')
    parser.add_argument('-l','--ILOAD',                help='Use PreTrained Weights/bias')
    args = parser.parse_args()

    if args.INETWORK != None:
        INETWORK = args.INETWORK
    if args.IEPOCH   != None:
        IEPOCH   = args.IEPOCH
    if args.RATIO    != None:
        if args.RATIO >= 0.0  and args.RATIO <=1.0:
            RATIO    = args.RATIO
        else:
            print("[ERROR] RATIO must be range from 0.0 to 1.0.")
            exit
    if args.ISEED    != None:
        ISEED    = args.ISEED
    if args.IGRAPHIC != None:
        IGRAPHIC = args.IGRAPHIC
    if args.OUTDIR   != None:
        OUTDIR   = args.OUTDIR
    if args.INDIR   != None:
        INDIR    = args.INDIR
    if args.ILOAD   != None:
        ILOAD    = args.ILOAD

    ### Print Settings ###
    print("INETWORK =",INETWORK)
    print("IEPOCH   =",IEPOCH)
    print("ISEED    =",ISEED)
    print("RATIO    =",RATIO)
    print("IGRAPHIC =",IGRAPHIC)
    print("OUTDIR   =",OUTDIR)
    print("INDIR    =",INDIR)
    print("ILOAD    =",ILOAD)

    return

# Initialize
def init(ISEED, INETWORK, IEPOCH, RATIO, INDIR):
    ### Disable pycache for safety ###
    sys.dont_write_bytecode = True
    os.environ['PYTHONDONTWRITEBYTECODE']  = '1'

    ### Thread ###
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    ### Set Seed of Pseudo Random Number ###
    np.random.seed(ISEED)
    tf.random.set_seed(ISEED)

    ### Define Dataset (images and labels) of Training, Inference ###
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    ### User Defined Test Data (28px X 28px)###

# For exception handling for variable type mismatch
    if INDIR != None:
        print("[INFO] Using user images for test.")
        i = 0
        # Check PNG files
        FILES = glob.glob(INDIR + "/*")
        nf    = len(FILES)
        # Create Numpy Array
        ntmp  = nf*28*28
        tmp_images  = np.empty(ntmp, np.int)
        test_images = np.reshape(tmp_images, (nf,28,28))
        test_images[:,:,:] = 0.0
        test_labels = np.empty(nf, np.int)
        test_labels[:]     = 0
        for file in FILES:
            # Image
            image_org = load_img(file, color_mode="grayscale")
            # Resize Image
            image     = image_org.resize(size=(28,28), resample=Image.BICUBIC)
            # Convert to negative
            imagen    = ImageOps.invert(image)
            # Convert Image to Numpy Array
            npimg     = np.array(img_to_array(imagen))
            test_images[i,:,:] = npimg[:,:,0]
            # Label
            tmp = os.path.splitext(os.path.basename(file))[0]
            # Only use the 1st character
            test_labels[i] = tmp[0]
            i += 1

        ### Resize and Normalizations (0-255 -> 0-1) of images ###
    train_images = train_images.reshape((len(train_images), 784)).astype('float32') / 255
    test_images = test_images.reshape((len(test_images), 784)).astype('float32') / 255

    ### Convert Labels to One Hot Encoding ###
    train_labels = tf.keras.utils.to_categorical(train_labels, 10)
    test_labels = tf.keras.utils.to_categorical(test_labels, 10)

    return train_images, train_labels, test_images, test_labels


# Define model of neural network architecture
def conv_model(id):
    if (int(id) == 0):
        # Single Dense Layer
        model = Sequential()
        model.add(Dense(10, activation='softmax', input_shape=(28 * 28,), name='softmax'))
    elif (int(id) == 1):
        # CNN
        model = Sequential()
        model.add(Reshape((28, 28, 1), input_shape=(28 * 28,), name='reshape'))
        model.add(Conv2D(32, (3, 3), padding='same', use_bias=True, activation='relu', name='conv_filter1'))
        model.add(Conv2D(64, (3, 3), padding='same', use_bias=True, activation='relu', name='conv_filter2'))
        model.add(Dropout(rate=0.25, name='dropout1'))
        model.add(Flatten(name='flatten'))
        model.add(Dense(128, activation='relu', name='hiden'))
        model.add(Dropout(rate=0.5, name='dropout2'))
        model.add(Dense(10, activation='softmax', name='softmax'))
    else:
        # User Defined Neural Network (Please write or modify python code in def udf_model(model)
        model = udf_model()

    model.summary()
    return model
