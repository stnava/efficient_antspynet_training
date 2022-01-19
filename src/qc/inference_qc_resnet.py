################################
import os
from os.path import exists
import glob
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import sys
import re
# now we can do the processing we need ....
import ants
import antspynet
import antspymm
import tensorflow as tf
import tensorflow.keras.backend as K
K.set_floatx("float32")
import antspyt1w
import numpy as np
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

weights_filename='resnet_grader.h5'
mdl = antspynet.create_resnet_model_3d( [None,None,None,nChannels],
    lowest_resolution = 32,
    number_of_classification_labels = 4,
    cardinality = 1,
    squeeze_and_excite = False )
mdl.load_weights( weights_filename )
# mdl.summary()


scoreNums = np.zeros( 4 )
scoreNums[3]=0
scoreNums[2]=1
scoreNums[1]=2
scoreNums[0]=3
scoreNums=scoreNums.reshape( [4,1] )

t1fn='t1weighted_strides14_SR.nii.gz'
t1fn='/Users/stnava/data/ADNI_studies/basalforebrainlibrary/images_train/002_S_0729-20060802-T1w-000-SRnocsf.nii.gz'
dobxt=False
x=ants.image_read( t1fn )
if dobxt:
    bxt=antspyt1w.brain_extraction( x )
    t1 = ants.iMath( x * bxt,  "Normalize" )
else:
    t1 = ants.iMath( x,  "Normalize" )
t1 = ants.rank_intensity( t1, get_mask=True )
templateb = ants.image_read( antspyt1w.get_data( "S_template3_brain", target_extension='.nii.gz' ) )
templateb = ants.crop_image( templateb ).resample_image( [1,1,1] )
templateb = antspynet.pad_image_by_factor( templateb, 8 )
templatebsmall = ants.resample_image( templateb, [2,2,2] )
reg = ants.registration( templatebsmall, t1, 'Similarity', verbose=False )
#############################
ilist = list()
refimg=templateb
ilist.append( [refimg] )
print("simulate")
nsim = 6
uu = antspynet.randomly_transform_image_data( refimg, ilist,
    number_of_simulations = nsim,
    transform_type='scaleShear', sd_affine=0.02 )
print("simulate done")

fwdaffgd = ants.read_transform( reg['fwdtransforms'][0])
invaffgd = ants.invert_ants_transform( fwdaffgd )

for k in range( nsim ):
    print( "k: " + str(k) )
    simtx = uu['simulated_transforms'][k]
    cmptx = ants.compose_ants_transforms( [simtx,fwdaffgd] ) # good
    subjectsim = ants.apply_ants_transform_to_image( cmptx, t1, refimg, interpolation='linear' )
    xarr = subjectsim.numpy()
    newshape = list( xarr.shape )
    newshape = [1] + newshape + [1]
    xarr = np.reshape(  xarr, newshape  )
    preds = mdl.predict( xarr )
    predsnum = tf.matmul(  preds, scoreNums )
    print( preds )
    print( predsnum )
