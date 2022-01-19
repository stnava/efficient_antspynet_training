################################
import os
from os.path import exists
import glob
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import sys
import re
t1fns = glob.glob( "images/*_QCsim*.nii.gz" )
# now we can do the processing we need ....
import ants
import antspynet
import antspymm
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
K.set_floatx("float32")
import antspyt1w
import numpy as np
import random
nChannels=1
#
# create_resnet_model_3d(input_image_size, input_scalars_size=0,
# number_of_classification_labels=1000,
# layers=(1, 2, 3, 4), residual_block_schedule=(3, 4, 6, 3),
# lowest_resolution=64, cardinality=1, squeeze_and_excite=False, mode='classification')

from scipy.stats import spearmanr
def compute_spearmanr(y, y_pred):
    mycorr = spearmanr(y_pred, y).correlation * (-1.0)
    return mycorr

mdl = antspynet.create_resnet_model_3d( [None,None,None,nChannels],
    lowest_resolution = 32,
    number_of_classification_labels = 4,
    cardinality = 32,
    squeeze_and_excite = True )
# mdl.summary()

refimg = ants.image_read( t1fns[0] )

def batch_generator( n = 8 ):
    s = random.sample( range( 0, len(t1fns)), n )
    xshape = list( refimg.shape )
    xshape.append( 1 )
    xshape = [n] + xshape
    x = np.zeros( xshape )
    y = np.zeros( [n,4] )
    yNum = np.zeros( n )
    for k in range( n ):
        bfn = t1fns[ s[k] ]
        temp = os.path.basename( bfn ).split( "_" )
        grade = temp[3]
        if grade == 'a':
            y[k,0]=1
            yNum[k]=3
        if grade == 'b':
            y[k,1]=1
            yNum[k]=2
        if grade == 'c':
            y[k,2]=1
            yNum[k]=1
        if grade == 'f':
            y[k,3]=1
            yNum[k]=0
        # print( str(s[k]) + " : " + os.path.basename( bfn ) + " : " + grade )
        img=ants.image_read( bfn )
        x[k,:,:,:,0]=img.numpy()
    return [x, y, yNum ]

# temp=batch_generator()



weights_filename='resnet_grader.h5'
csv_filename=re.sub("h5","csv",weights_filename)
optimizerE = tf.keras.optimizers.Adam(1.e-3)
batchsize = 32
epoch=0
num_epochs = np.round( len( t1fns ) * 5 / batchsize  ).astype(int)
mydf=None

# load the testing data
with tf.device('/CPU:0'):
    temp=batch_generator( 32 )
    testX = temp[0]
    testY = temp[1]
    testYnum = temp[2]

scoreNums = np.zeros( 4 )
scoreNums[3]=3
scoreNums[2]=2
scoreNums[1]=1
scoreNums=scoreNums.reshape( [4,1] )

corrwt = 1.0

for epoch in range(epoch, num_epochs):
    xy = batch_generator( batchsize )
    with tf.GradientTape(persistent = False) as tape:
          preds = mdl( xy[0] )
          predNum = tf.linalg.matmul( preds, scoreNums ).reshape(batchsize)
          traincorr = compute_spearmanr( xy[2], predNum.numpy() )
          cceloss = tf.reduce_mean(  tf.losses.categorical_crossentropy( xy[1], preds ) )
          loss = cceloss + traincorr * corrwt
          my_gradients = tape.gradient( loss, mdl.trainable_variables)
    optimizerE.apply_gradients(  zip( my_gradients, mdl.trainable_variables ) )
    testloss=tf.cast( np.math.inf, 'float32')
    testcorr=0
    if epoch == 1 or epoch % int(20) == 0:
        preds = mdl.predict( testX, batch_size=batchsize )
        with tf.device('/CPU:0'):
            print("Testing")
            testloss = tf.cast( 0.0, 'float32' )
            predNum = tf.linalg.matmul( preds, scoreNums ).reshape(testX.shape[0])
            testcorr = compute_spearmanr( testYnum, predNum )
            testcce = tf.reduce_mean( tf.losses.categorical_crossentropy( testY, preds ) )
            testloss = testcorr * corrwt + testcce
    ismin = False
    if epoch > 1:
        if testloss.numpy() <= mydf['test_loss'].min():
            mdl.save_weights(weights_filename)
            ismin=True
    temp = pd.DataFrame({
        'train_loss': [loss.numpy()],
        'train_corr': [traincorr],
        'train_cce': [cceloss.numpy()],
        'test_corr': [testcorr],
        'test_loss': [testloss.numpy()],
        "epoch": [epoch], "ismin":ismin } )
    if mydf is None:
        mydf = temp
    else:
        mydf = mydf.append(temp, ignore_index = True)
    mydf.to_csv( csv_filename )
    print( temp )
