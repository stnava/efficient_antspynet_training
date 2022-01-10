import os
import glob
import sys
from os.path import exists
import ants
import antspynet
import antspyt1w
import numpy as np
import pandas as pd
import re
import random
import random, string

libdir = "/mnt/cluster/data/anatomicalLabels/basalforebrainlibrary/"
dtifns = glob.glob( libdir + "images_train/*SRnocsf.nii.gz" )
dtifns.sort()
segfns = dtifns.copy()
for x in range( len( dtifns )):
    segfns[x] = re.sub( "SRnocsf.nii.gz", "SRnbm3CH13.nii.gz" , dtifns[x] )
    segfns[x] = re.sub( "images_train", "segmentations" , segfns[x] )

spre = libdir + "simulated_whole_brain/"

if len( sys.argv ) > 1:
    fileindex = int(sys.argv[1])
else:
    fileindex=0

random.seed( fileindex )
def randword(length):
   letters = string.ascii_lowercase
   return ''.join(random.choice(letters) for i in range(length))

randstring = randword( 8 )

imgfn = dtifns[ fileindex ]
segfn = segfns[ fileindex ]
print( imgfn )
refimg = ants.image_read( antspyt1w.get_data( "CIT168_T1w_700um_pad", target_extension='.nii.gz' ))
refimg = ants.rank_intensity( refimg )
refimg = ants.resample_image( refimg, [0.5,0.5,0.5] )
refimgseg = ants.image_read( antspyt1w.get_data( "det_atlas_25_pad_LR", target_extension='.nii.gz' ))
refimgsmall = ants.resample_image( refimg, [2.0,2.0,2.0] )
ifnbase = randstring + "_img"
sfnbase = randstring + "_seg"
pfnbase = randstring + "_pri"
outipre = spre + ifnbase
outspre = spre + sfnbase
outppre = spre + pfnbase
print(outipre)
print(outspre)
print(outppre)
#############################
img = ants.image_read( imgfn )
seg = ants.image_read( segfn )
# NOTE: this image should be brain extracted by antspyt1w.brain_extraction( . , method='v1' )
imgr = ants.rank_intensity( img )
reg = ants.registration( refimgsmall, imgr, 'Similarity', verbose=False )
#############################
ilist = list()
ilist.append( [refimg] )
nsim = 64
# uu = antspynet.randomly_transform_image_data( refimg, ilist,
#    number_of_simulations = nsim,
#    transform_type='scaleShear', sd_affine=0.075 )

print("simulate")
uu = antspynet.randomly_transform_image_data( refimg, ilist,
    number_of_simulations = nsim, sd_affine=0.005,
    transform_type = "affineAndDeformation" )
print("simulate done")

fwdaffgd = ants.read_transform( reg['fwdtransforms'][0])
invaffgd = ants.invert_ants_transform( fwdaffgd )

for k in range( nsim ):
    print( "k: " + str(k) )
    simtx = uu['simulated_transforms'][k]
    cmptx = ants.compose_ants_transforms( [simtx,fwdaffgd] ) # good
    subjectsim = ants.apply_ants_transform_to_image( cmptx, img, refimg, interpolation='linear' )
    subjectsimseg = ants.apply_ants_transform_to_image( cmptx, seg, refimg, interpolation='nearestneighbor' )
    bias_field = antspynet.simulate_bias_field( subjectsim, number_of_points=10,
        sd_bias_field=0.10, number_of_fitting_levels=4, mesh_size=1)
    subjectsim = subjectsim * (bias_field + 1)
    subjectsim = ants.rank_intensity( subjectsim )
    # ants.plot( subjectsim, subjectsimseg, nslices=21, ncol=7, axis=1, crop=True )
    ants.image_write( subjectsimseg, outspre + "_sim_" + str(k) + ".nii.gz"  )
    ants.image_write( subjectsim, outipre + "_sim_" + str(k) + ".nii.gz"  )
    # ants.image_write( priorsim, outppre + "_sim_" + str(k) + ".nii.gz"  )
    print( outspre + "_sim_" + str(k) + ".nii.gz" )
    print( outipre + "_sim_" + str(k) + ".nii.gz"  )
    # print( outppre + "_sim_" + str(k) + ".nii.gz"  )

print("done")
