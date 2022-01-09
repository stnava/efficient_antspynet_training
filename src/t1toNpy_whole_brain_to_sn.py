import os
import glob
import numpy as np
import sys
import random
if len( sys.argv ) == 1:
    rseed=0
else:
    rseed = int(sys.argv[1])
random.seed( rseed )
import ants
import antspynet
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

def coordinate_images( mask ):
  idim = mask.dimension
  myr = list()
  for k in range(idim):
      myr.append(0)
  temp = ants.get_neighborhood_in_mask( mask, mask, myr,
    boundary_condition = "image", spatial_info = True,
    physical_coordinates = True, get_gradient = False)
  ilist = []
  for i in range(idim):
      ilist.append( ants.make_image( mask, temp['indices'][:,i] ) )
  return ilist


def batch_generator(
    image_filenames,
    segmentation_filenames,
    image_size,
    group_labels_in,
    batch_size=64,
    ):
    X = np.zeros( (batch_size, *(image_size), 1) )
    Xcc = np.zeros( (batch_size, *(image_size), 3) )
    Y = np.zeros( (batch_size, *(image_size) ) )
    Ypr = np.zeros( (batch_size, *(image_size) ) )
    zz=pd.read_csv( pt_fns[0] )
    npts = zz.shape[0]
    Ypts = np.zeros( ( batch_size,  npts, 3 ) )
    batch_count = 0
    print("BeginBatch")
    while batch_count < batch_size:
        i = random.sample(list(range(len(image_filenames))), 1)[0]
        t1 = ants.image_read(image_filenames[i])
        zz=pd.read_csv( pt_fns[i] )
        mypr = ants.image_read( pr_fns[i] )
        seg = ants.image_read( segmentation_filenames[i] )
        seg = ants.mask_image( seg, seg, group_labels_in, binarize=False)
        comMask = ants.mask_image( mypr, mypr, pt_labels, binarize=True )
        com = ants.get_center_of_mass( comMask )
        t1=special_crop( t1, com, image_size )
        seg=special_crop( seg, com, image_size )
        mypr=special_crop( mypr, com, image_size )
        mycc = coordinate_images( t1 * 0 + 1 )
        X[batch_count,:,:,:,0] = t1.numpy()
        Xcc[batch_count,:,:,:,0] = mycc[0].numpy()
        Xcc[batch_count,:,:,:,1] = mycc[1].numpy()
        Xcc[batch_count,:,:,:,2] = mycc[2].numpy()
        Y[batch_count,:,:,:] = seg.numpy()
        Ypr[batch_count,:,:,:] = mypr.numpy()
        Ypts[batch_count,:,0]=zz['x']
        Ypts[batch_count,:,1]=zz['y']
        Ypts[batch_count,:,2]=zz['z']
        batch_count = batch_count + 1
        if batch_count >= batch_size:
                break

    return X, Xcc, Y, Ypts, Ypr

data_directory = "/mnt/cluster/data/anatomicalLabels/Mindboggle101_volumes/simulated_whole_brain/"
# data_directory = "/tmp/simulated_whole_brain/"
exfn = glob.glob( data_directory + "*img*sim_0.nii.gz" )[0]
eximg = ants.image_read( exfn )
group_labels_target = [0,7,8,9,23,24,25,33,34]
pt_labels = [7,9,23,25]

crop_size = [96,96,64]
image_size = list(eximg.shape)

print("Loading brain data.")

t1_fns = glob.glob( data_directory + "*img*_sim_*.nii.gz" )
seg_fns = t1_fns.copy()
for k in range(len(t1_fns)):
    seg_fns[k] = re.sub( "img", "seg", seg_fns[k] )
pt_fns = t1_fns.copy()
pr_fns = t1_fns.copy()
for k in range(len(t1_fns)):
    pr_fns[k] = re.sub( "img", "pri", pr_fns[k] )
    pt_fns[k] = re.sub( ".nii.gz", "points.csv", seg_fns[k] )

print("Total training image files: ", len(t1_fns))

import random, string
def randword(length):
   letters = string.ascii_lowercase
   return ''.join(random.choice(letters) for i in range(length))
randstring = randword( 8 )
outpre = "/mnt/cluster/data/anatomicalLabels/Mindboggle101_volumes/numpySNSegRank/MBSN_" + randstring
print( outpre )

seg = ants.image_read( seg_fns[0] )
###
#
# Set up the training generator
#

batch_size = 64
generator = batch_generator( t1_fns,
        seg_fns,
        image_size=crop_size,
        batch_size = batch_size,
        group_labels_in=group_labels_target )

if False:
    for k in range(batch_size):
        t0= ants.from_numpy( generator[0][k,:,:,:,0] )
        t1= ants.from_numpy( generator[2][k,:,:,:] )
        t0=ants.copy_image_info( eximg,t0)
        t1=ants.copy_image_info( eximg,t1)
        ants.plot(t0,t1,axis=2)

np.save( outpre + "_Ximages.npy", generator[0] )
np.save( outpre + "_Xcc.npy", generator[1] )
np.save( outpre + "_Y.npy", generator[2] )
np.save( outpre + "_Ypts.npy", generator[3] )
np.save( outpre + "_Xprior.npy", generator[4])
