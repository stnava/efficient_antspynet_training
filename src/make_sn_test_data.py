import os
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "96"
os.environ["TF_NUM_INTEROP_THREADS"] = "96"
os.environ["TF_NUM_INTRAOP_THREADS"] = "96"
import glob
import numpy as np
import sys
import random
rseed = 88
random.seed( rseed )
import ants
import antspynet
import antspyt1w
import re
import pandas as pd
import tensorflow as tf
tf.random.set_seed( rseed )
import tensorflow.keras as keras
import tensorflow.keras.backend as K
K.set_floatx("float32")
import numpy as np

def special_crop( x, pt, domainer ):
        pti = np.round( ants.transform_physical_point_to_index( x, pt ) )
        xdim = x.shape
        for k in range(len(xdim)):
            if pti[k] < 0:
                pti[k]=0
            if pti[k] > (xdim[k]-1):
                pti[k]=(xdim[k]-1)
        mim = ants.make_image( domainer )
        ptioff = pti.copy()
        for k in range(len(xdim)):
            ptioff[k] = ptioff[k] - np.round( domainer[k] / 2 )
        domainerlo = []
        domainerhi = []
        for k in range(len(xdim)):
            domainerlo.append( int(ptioff[k] - 1) )
            domainerhi.append( int(ptioff[k] + 1) )
        loi = ants.crop_indices( x, tuple(domainerlo), tuple(domainerhi) )
        mim = ants.copy_image_info(loi,mim)
        return ants.resample_image_to_target( x, mim )

refimg = ants.image_read( antspyt1w.get_data( "CIT168_T1w_700um_pad", target_extension='.nii.gz' ))
refimg = ants.rank_intensity( refimg )
refimg = ants.resample_image( refimg, [0.5,0.5,0.5] )
refimgseg = ants.image_read( antspyt1w.get_data( "det_atlas_25_pad_LR", target_extension='.nii.gz' ))
refimgsmall = ants.resample_image( refimg, [2.0,2.0,2.0] )

# generate the data

def preprocess( imgfn ):
    img = ants.image_read( imgfn )
    imgbxt = antspyt1w.brain_extraction( img, method='v1' )
    img = antspyt1w.preprocess_intensity( img, imgbxt, intensity_truncation_quantiles=[0.000001, 0.999999 ] )
    imgr = ants.rank_intensity( img )
    reg = ants.registration( refimg, imgr, 'SyN',
        reg_iterations = [200,200,100,20,5],
        verbose=False )
    imgraff = ants.apply_transforms( refimg, imgr, reg['fwdtransforms'][1], interpolator='linear' )
    imgseg = ants.apply_transforms( refimg, refimgseg, reg['invtransforms'][1], interpolator='nearestNeighbor' )
    binseg = ants.mask_image( imgseg, imgseg, pt_labels, binarize=True )
    imgseg = ants.mask_image( imgseg, imgseg, group_labels_target )
    com = ants.get_center_of_mass( binseg )
    return {
        "img": imgraff,
        "seg": imgseg,
        "imgc": special_crop( imgraff, com, crop_size ),
        "segc": special_crop( imgseg, com, crop_size )
        }

data_directory = "/mnt/cluster/data/anatomicalLabels/Mindboggle101_volumes/simulated_whole_brain/"
data_directory = "/Users/stnava/Downloads/temp/traveling_subjects/SRPBTravel/"
data_directory = "/mnt/cluster/data/SRPBS_multidisorder_MRI/traveling_subjects_repro_study/"
srchstring = "sub-*/T1wH/sub-*v1SR.nii.gz"
exfn = glob.glob( data_directory + srchstring)[0]
eximg = ants.image_read( exfn )
group_labels_target = [0,7,8,9,23,24,25,33,34]
pt_labels = [7,9,23,25]

crop_size = [96,96,64]
image_size = list(eximg.shape)
print( eximg )

# temp=preprocess(exfn)

print("Loading brain data.")

t1_fns = glob.glob( data_directory + srchstring )
print("Total training image files: ", len(t1_fns))

# convert to numpy files
def batch_generator(
    image_filenames,
    image_size,
    batch_size=64,
    ):
    X = np.zeros( (batch_size, *(image_size), 1) )
    Y = np.zeros( (batch_size, *(image_size) ) )
    batch_count = 0
    lo=0
    if len(image_filenames) > 20:
        lo=20
    print("BeginBatch " + str(lo) )
    while batch_count < batch_size:
        i = random.sample(list(range(lo,len(image_filenames))), 1)[0]
        print( str(i) + " " + image_filenames[i] )
        locdata = preprocess( image_filenames[i] )
        X[batch_count,:,:,:,0] = locdata['imgc'].numpy()
        Y[batch_count,:,:,:] = locdata['segc'].numpy()
        batch_count = batch_count + 1
        if batch_count >= batch_size:
                break

    return X, Y

import random, string
def randword(length):
   letters = string.ascii_lowercase
   return ''.join(random.choice(letters) for i in range(length))
randstring = randword( 8 )
outpre = "/mnt/cluster/data/anatomicalLabels/Mindboggle101_volumes/numpySNSegRankTest/TRT_" + randstring
print( outpre )

batch_size = 32
generator = batch_generator( t1_fns,
        image_size=crop_size,
        batch_size = batch_size )

if False:
    for k in range(batch_size):
        t0= ants.from_numpy( generator[0][k,:,:,:,0] )
        t1= ants.from_numpy( generator[1][k,:,:,:] )
        t0=ants.copy_image_info( refimg,t0)
        t1=ants.copy_image_info( refimg,t1)
        ants.plot(t0,t1,axis=2)

np.save( outpre + "_Ximages.npy", generator[0] )
np.save( outpre + "_Y.npy", generator[1] )
