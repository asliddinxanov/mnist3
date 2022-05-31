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

