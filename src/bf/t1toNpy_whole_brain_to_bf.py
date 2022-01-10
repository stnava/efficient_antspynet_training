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

istest=True

def recode_labels( x, old_labels, new_labels ):
    xnew = x * 0.0
    for n in range(len(old_labels)):
        xnew[ x == old_labels[n] ] = new_labels[n]
    return xnew

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
    Y = np.zeros( (batch_size, *(image_size) ) )
    batch_count = 0
    print("BeginBatch")
    while batch_count < batch_size:
        reflect_it = random.choice([True,False])
        i = random.sample(list(range(len(image_filenames))), 1)[0]
        t1 = ants.image_read(image_filenames[i])
        seg = ants.image_read( segmentation_filenames[i] )
        seg = ants.mask_image( seg, seg, group_labels_in, binarize=False)
        if reflect_it:
            print("Reflect")
            reflectx=ants.reflect_image( eximg, axis=0, tx='Translation' )
            reflection_matrix=reflectx['fwdtransforms'][0]
            seg = ants.apply_transforms( seg, seg, reflection_matrix, interpolator='nearestNeighbor' )
            seg = recode_labels( seg, group_labels_target, reflection_labels )
            t1 = ants.apply_transforms( t1, t1, reflection_matrix, interpolator='linear' )
        centroids = ants.label_geometry_measures(seg)
        zz = pd.DataFrame({"Label":centroids['Label'],
                "x":centroids['WeightedCentroid_x'],
                "y":centroids['WeightedCentroid_y'],
                "z":centroids['WeightedCentroid_z']} )
        if batch_count == 0:
            npts = zz.shape[0]
            Ypts = np.zeros( ( batch_size,  npts, 3 ) )
        comMask = ants.mask_image( seg, seg, pt_labels, binarize=True )
        com = ants.get_center_of_mass( comMask )
        t1=special_crop( t1, com, image_size )
        seg=special_crop( seg, com, image_size )
        X[batch_count,:,:,:,0] = t1.numpy()
        Y[batch_count,:,:,:] = seg.numpy()
        Ypts[batch_count,:,0]=zz['x']
        Ypts[batch_count,:,1]=zz['y']
        Ypts[batch_count,:,2]=zz['z']
        batch_count = batch_count + 1
        if batch_count >= batch_size:
                break

    return X,  Y, Ypts

libdir = '/mnt/cluster/data/anatomicalLabels/basalforebrainlibrary/'
data_directory = libdir + "simulated_whole_brain/"
if istest:
    data_directory = "/tmp/testit/"
exfn = glob.glob( data_directory + "*img*sim_0.nii.gz" )[0]
eximg = ants.image_read( exfn )
group_labels_target = [0,1,2,3,4,5,6,7,8]
reflection_labels =   [0,2,1,6,7,8,3,4,5]
pt_labels = group_labels_target

crop_size = [128,128,96]

print("Loading brain data.")

t1_fns = glob.glob( data_directory + "*img*_sim_*.nii.gz" )
seg_fns = t1_fns.copy()
for k in range(len(t1_fns)):
    seg_fns[k] = re.sub( "img", "seg", seg_fns[k] )

print("Total training image files: ", len(t1_fns))

import random, string
def randword(length):
   letters = string.ascii_lowercase
   return ''.join(random.choice(letters) for i in range(length))
randstring = randword( 8 )
outpre = libdir + "numpystuff/BF_" + randstring
print( outpre )

###
#
# Set up the training generator
#
###

batch_size = 64
if istest:
    batch_size = 4

generator = batch_generator( t1_fns,
        seg_fns,
        image_size=crop_size,
        batch_size = batch_size,
        group_labels_in=group_labels_target )

if istest:
    for k in range(batch_size):
        print(k)
        t0= ants.from_numpy( generator[0][k,:,:,:,0] )
        t1= ants.from_numpy( generator[1][k,:,:,:] )
        t0=ants.copy_image_info( eximg,t0)
        t1=ants.copy_image_info( eximg,t1)
        ants.plot(t0,t1,axis=2)

if not istest:
    np.save( outpre + "_Ximages.npy", generator[0] )
    np.save( outpre + "_Y.npy", generator[2] )
    np.save( outpre + "_Ypts.npy", generator[3] )
