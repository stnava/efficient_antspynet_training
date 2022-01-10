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
refimgseg = ants.image_read( antspyt1w.get_data( "CIT168_T1w_700um_pad_bf", target_extension='.nii.gz' ))
refimgsmall = ants.resample_image( refimg, [2.0,2.0,2.0] )

# generate the data

def preprocess( imgfn, segfn, bxt=False ):
    img = ants.image_read( imgfn )
    seg = ants.image_read( segfn )
    if bxt:
        imgbxt = antspyt1w.brain_extraction( img, method='v1' )
        img = antspyt1w.preprocess_intensity( img, imgbxt, intensity_truncation_quantiles=[0.000001, 0.999999 ] )
    imgr = ants.rank_intensity( img )
    print("BeginReg")
    reg = ants.registration( refimgsmall, imgr, 'SyN', verbose=False )
    print("EndReg")
    imgraff = ants.apply_transforms( refimg, imgr, reg['fwdtransforms'][1], interpolator='linear' )
    imgseg = ants.apply_transforms( refimg, refimgseg, reg['invtransforms'][1], interpolator='nearestNeighbor' )
    binseg = ants.mask_image( imgseg, imgseg, pt_labels, binarize=True )
    imgseg = ants.mask_image( imgseg, imgseg, group_labels_target, binarize=False  )
    com = ants.get_center_of_mass( binseg )
    return {
        "img": imgraff,
        "seg": imgseg,
        "imgc": special_crop( imgraff, com, crop_size ),
        "segc": special_crop( seg, com, crop_size )
        }


libdir = "/mnt/cluster/data/anatomicalLabels/basalforebrainlibrary/"
t1_fns = glob.glob( libdir + "images_test/*SRnocsf.nii.gz" )
t1_fns.sort()
segfns = t1_fns.copy()
for x in range( len( t1_fns )):
    segfns[x] = re.sub( "SRnocsf.nii.gz", "SRnbm3CH13.nii.gz" , t1_fns[x] )
    segfns[x] = re.sub( "images_test", "segmentations" , segfns[x] )

exfn = t1_fns[0]
eximg = ants.image_read( exfn )

group_labels_target = [0,1,2,3,4,5,6,7,8]
reflection_labels =   [0,2,1,6,7,8,3,4,5]
pt_labels = [1,2,3,4,5,6,7,8]

crop_size = [144,96,64]

print("Loading brain data.")
print("Total training image files: ", len(t1_fns))

# temp=preprocess(t1_fns[0],segfns[0])

# convert to numpy files
def batch_generator(
    image_filenames,
    seg_filenames,
    image_size,
    batch_size=48,
    ):
    X = np.zeros( (batch_size, *(image_size), 1) )
    Y = np.zeros( (batch_size, *(image_size) ) )
    batch_count = 0
    lo=0
    print("BeginBatch " + str(lo) )
    while batch_count < batch_size:
        i = random.sample(list(range(lo,len(image_filenames))), 1)[0]
        print( str(i) + " " + image_filenames[i] )
        locdata = preprocess( image_filenames[i], seg_filenames[i] )
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
outpre = libdir + "numpystufftest/ADNITEST_" + randstring
print( outpre )

batch_size = 48
generator = batch_generator(
    t1_fns,
    segfns,
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
