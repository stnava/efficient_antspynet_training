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
t1fn='/Users/stnava/data/ADNI_studies/basalforebrainlibrary/images_train/002_S_0729-20060802-T1w-000-SRnocsf.nii.gz'; dobxt=False
t1fn='/Users/stnava/.antspyt1w/28523-00000000-T1w-05.nii.gz'; dobxt=True
# t1fn='/Users/stnava/.antspyt1w/ADNI-024_S_1393-20080407-T1w-000.nii.gz'
t1fn='/Users/stnava/.antspyt1w/ADNI-073_S_4300-20140107-T1w-000.nii.gz'
t1fn='/Users/stnava/.antspyt1w/28364-00000000-T1w-00.nii.gz'
t1fn='sub-056_T1wH_v1SR.nii.gz'; dobxt=False
t1fn='Mindboggle_MMRR-21-5_T1wSRHierarchical_SR.nii.gz'
t1fn='sub-094-SRHIERbrain_n4_dnz.nii.gz'
t1fn='/Users/stnava/.antspyt1w/28405-00000000-T1w-02.nii.gz'; dobxt=True
t1fn='/Users/stnava/.antspyt1w/28497-00000000-T1w-04.nii.gz'
t1fn='/Users/stnava/.antspyt1w/sub-094_T1w_n3.nii.gz'
t1fn='/Users/stnava/.antspyt1w/28364-00000000-T1w-00nnlow.nii.gz'
t1fn='/Users/stnava/.antspyt1w/28364-00000000-T1w-00srup.nii.gz'
t1fn='/Users/stnava/.antspyt1w/28386-00000000-T1w-01.nii.gz'
t1fn='/Users/stnava/.antspyt1w/28575-00000000-T1w-07.nii.gz'
t1fn='Landman_1399_20110819_366886505_301_WIP_T1_3D_TFE_iso0_70_SENSE_T1_3D_TFE_iso0_70.nii.gz'; dobxt=True
t1fn='/Users/stnava/Downloads/eximg.nii.gz' ; dobxt=False
t1fn='PPMI-107099-20210914-T1wHierarchical-I1498901-SR.nii.gz'; dobxt=False
x=ants.image_read( t1fn )
print( t1fn )
if dobxt:
    # otsumask=ants.threshold_image( x, "Otsu", 1).iMath("MD",2)
    bxt=antspyt1w.brain_extraction( x ) # * ants.iMath( otsumask, "FillHoles" ) )
    # bxt=antspynet.brain_extraction( ants.iMath( x, "TruncateIntensity", 0.01, 0.95) , "t1" ).threshold_image(0.5, 1)
    t1 = ants.iMath( x * bxt,  "Normalize" )
else:
    t1 = ants.iMath( x,  "Normalize" )
    bxt = ants.threshold_image( t1, 0.01, 1 )
t1 = ants.rank_intensity( t1, mask=bxt, get_mask=True )
# t1 = ants.add_noise_to_image( t1, 'additivegaussian', (0,0.5) ) * bxt # should fail
# ants.plot( t1, nslices=21, ncol=7, crop=True )
templateb = ants.image_read( antspyt1w.get_data( "S_template3_brain", target_extension='.nii.gz' ) )
templateb = ants.crop_image( templateb ).resample_image( [1,1,1] )
templateb = antspynet.pad_image_by_factor( templateb, 8 )
templatebsmall = ants.resample_image( templateb, [2,2,2] )
reg = ants.registration( templatebsmall, t1, 'Similarity', verbose=False )
#############################
ilist = list()
# refimg=ants.resample_image( templateb, ants.get_spacing( t1 ) )
refimg=templateb
ilist.append( [refimg] )
print("simulate")
nsim = 16
uu = antspynet.randomly_transform_image_data( refimg, ilist,
    number_of_simulations = nsim,
    transform_type='scaleShear', sd_affine=0.075 )
print("simulate done")

fwdaffgd = ants.read_transform( reg['fwdtransforms'][0])
invaffgd = ants.invert_ants_transform( fwdaffgd )
meanpred = 0.0
minpred = np.math.inf
for k in range( nsim ):
    print( "k: " + str(k) )
    simtx = uu['simulated_transforms'][k]
    cmptx = ants.compose_ants_transforms( [simtx,fwdaffgd] ) # good
    subjectsim = ants.apply_ants_transform_to_image( cmptx, t1, refimg, interpolation='linear' )
    subjectsim = ants.add_noise_to_image( subjectsim, 'additivegaussian', (0,0.01) )
    xarr = subjectsim.numpy()
    newshape = list( xarr.shape )
    newshape = [1] + newshape + [1]
    xarr = np.reshape(  xarr, newshape  )
    preds = mdl.predict( xarr )
    predsnum = tf.matmul(  preds, scoreNums )
    if float(predsnum.numpy()) < minpred:
        minpred = float(predsnum.numpy())
    meanpred = meanpred + predsnum/nsim
    print( preds )
    print( predsnum )

grade='f'
meanpred=float(meanpred )
if meanpred >= 2.25:
    grade='a'
elif meanpred >= 1.5:
    grade='b'
elif meanpred >= 0.75:
    grade='c'

print( grade )


# quantile-based grading
grademin='f'
if minpred >= 2.25:
    grademin='a'
elif minpred >= 1.5:
    grademin='b'
elif minpred >= 0.75:
    grademin='c'

print( grademin + " " + str( float(meanpred ) ) )
