################################
import os
from os.path import exists
import glob
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import sys
import re
import statistics
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

def most_frequent(List):
    return max(set(List), key = List.count)

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
t1fn='/Users/stnava/.antspyt1w/28497-00000000-T1w-04.nii.gz'
t1fn='/Users/stnava/.antspyt1w/sub-094_T1w_n3.nii.gz'
t1fn='/Users/stnava/.antspyt1w/28364-00000000-T1w-00nnlow.nii.gz'
t1fn='/Users/stnava/.antspyt1w/28364-00000000-T1w-00srup.nii.gz'
t1fn='/Users/stnava/.antspyt1w/28386-00000000-T1w-01.nii.gz' # goood
t1fn='/Users/stnava/.antspyt1w/28575-00000000-T1w-07.nii.gz'
t1fn='Landman_1399_20110819_366886505_301_WIP_T1_3D_TFE_iso0_70_SENSE_T1_3D_TFE_iso0_70.nii.gz'; dobxt=True
t1fn='PPMI-107099-20210914-T1wHierarchical-I1498901-SR.nii.gz'; dobxt=False
t1fn='sub-094-SRHIERbrain_n4_dnz.nii.gz'
t1fn='PPMI-3810-20110913-MRI_T1-I269587-antspyt1w-V0-brain_n4_dnz-SR.nii.gz'
t1fn='/Users/stnava/code/code_old/adu/ADNI/reviewData/SRPro/20070820/T1w/000/deep_dkt/ADNI-016_S_0769-20070820-T1w-000-deep_dkt-deep_dkt_Labels-1006-1007-1015-1016_SR.nii.gz'
t1fn='/Users/stnava/Downloads/temp/basal_forebrain_trt/evaldf/005_S_0929/20061002/T1w/001/antspyt1wsr/V0/ADNI-005_S_0929-20061002-T1w-001-antspyt1wsr-V0-brain_n4_dnz_SR.nii.gz'

dobxt=True
t1fn='/Users/stnava/.antspyt1w/28575-00000000-T1w-07.nii.gz'
t1fn='/Users/stnava/Downloads/eximg.nii.gz' ; dobxt=False
# head=ants.image_read( t1fn )
t1fn='/Users/stnava/.antspyt1w/28405-00000000-T1w-02.nii.gz'; dobxt=True # bad

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

insp=antspyt1w.inspect_raw_t1( t1*bxt, option='brain', output_prefix='ASS' )

# mygrade = antspyt1w.resnet_grader( x )
