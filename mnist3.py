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