################################
import os
from os.path import exists
import glob
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

rootdir = "/mnt/cluster/data/"

t1fns = glob.glob( rootdir + "SRPBS_multidisorder_MRI/traveling_subjects/SRPBTravel/sub-*/anat/*_T1w.nii.gz" )
t1fns = t1fns + glob.glob( rootdir + "anatomicalLabels/Mindboggle101_volumes/*volumes/*/t1weighted.nii.gz" )
t1fns = t1fns + glob.glob( rootdir + "PPMI2/PPMI/*/*/T1w/*/dcm2niix/V0/*-dcm2niix-V0.nii.gz" )
t1fns = t1fns + glob.glob( rootdir + "PPMI1/*/*/*/*/*nii.gz" )

myqcd = pd.read_csv( "/mnt/cluster/data/QCStudy/data/brain_qc.csv" )

import sys
import re
fileindex=856
myoffset=0
dosr = True

if len( sys.argv ) > 1:
    fileindex = int(sys.argv[1])
if len( sys.argv ) > 2:
    dosr = eval(sys.argv[2])
if len( sys.argv ) > 3:
    myoffset = int(sys.argv[3])
if (fileindex + myoffset) > len( t1fns):
    sys.exit(0)
t1fn = t1fns[ fileindex + myoffset]
print( "target: " + t1fn + "  " + str(myoffset) + " " + str( fileindex ) )
myrw = str( fileindex )+"_"+str(myoffset)

outpre = "/mnt/cluster/data/T1wJoin/PPMI_RBP/zz" + myrw + "_" + \
    re.sub( ".nii.gz", "_RBP", os.path.basename( t1fn ) )

locid = os.path.basename( outpre ) + "_brain.png"
# print( (myqcd.ids.str.contains(locid) and  (not myqcd.grade.isna() ) ).sum() )

qcsub = myqcd.loc[myqcd['ids'] == locid]                       # Get rows with particular value

if qcsub.grade.isna().iloc[0]:
    print( locid + " no grade")
    sys.exit(0)
grade=str(qcsub.grade.iloc[0])
outpre = "/mnt/cluster/data/QCStudy/images/zz" + myrw + "_grade_" + grade + "_" + \
    re.sub( ".nii.gz", "_QCsim" , os.path.basename( t1fn ) )


# per grade simulations to balance the data
if grade == 'a':
    nsim=5
if grade == 'b':
    nsim=4
if grade == 'c':
    nsim=7
if grade == 'f':
    nsim=42
data={"id": [locid],  "grade":[grade],  "originalT1":[t1fn], "nsim": [nsim] }
outpd=pd.DataFrame( data )
outpd.to_csv( outpre + ".csv" )
# now we can do the processing we need ....
import ants
import antspynet
import antspymm
import tensorflow as tf
import tensorflow.keras.backend as K
K.set_floatx("float32")
import antspyt1w
import numpy as np


x=ants.image_read( t1fn )
bxt=antspyt1w.brain_extraction( x )
t1 = ants.iMath( x * bxt,  "Normalize" )
t1 = ants.rank_intensity( t1, get_mask=True )
pngfnb=outpre+'.png'
ants.plot( t1, axis=2, nslices=21, ncol=7, filename=pngfnb, crop=True )
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
uu = antspynet.randomly_transform_image_data( refimg, ilist,
    number_of_simulations = nsim,
    transform_type='scaleShear', sd_affine=0.075 )
print("simulate done")

fwdaffgd = ants.read_transform( reg['fwdtransforms'][0])
invaffgd = ants.invert_ants_transform( fwdaffgd )

for k in range( nsim ):
    print( "k: " + str(k) )
    simtx = uu['simulated_transforms'][k]
    cmptx = ants.compose_ants_transforms( [simtx,fwdaffgd] ) # good
    subjectsim = ants.apply_ants_transform_to_image( cmptx, t1, refimg, interpolation='linear' )
    # bias_field = antspynet.simulate_bias_field( subjectsim, number_of_points=10,
    #    sd_bias_field=0.10, number_of_fitting_levels=4, mesh_size=1)
    # subjectsim = subjectsim * (bias_field + 1)
    # subjectsim = ants.rank_intensity( subjectsim )
    # ants.plot( subjectsim, subjectsimseg, nslices=21, ncol=7, axis=1, crop=True )
    ants.image_write( subjectsim, outpre + str(k) + ".nii.gz"  )

print("done")
