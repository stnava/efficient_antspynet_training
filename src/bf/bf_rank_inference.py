import ants
import antspynet
import antspyt1w
import numpy as np
import tensorflow as tf
img=ants.image_read("PPMI-3810-20110913-MRI_T1-I269587-antspyt1w-V0-brain_n4_dnz-SR.nii.gz")
# img=ants.image_read("PPMI-58027-20210420-T1wHierarchical-I1495795-SR.nii.gz")
# img=ants.image_read("PPMI-107099-20210914-T1wHierarchical-I1498901-SR.nii.gz")
# img=ants.image_read("sub-021_T1wH_v1SR.nii.gz")

newfn="deepCIT168_sn_rank.h5"
print( newfn )
verbose=True

refimg = ants.image_read( antspyt1w.get_data( "CIT168_T1w_700um_pad", target_extension='.nii.gz' ))
refimg = ants.rank_intensity( refimg )
refimg = ants.resample_image( refimg, [0.5,0.5,0.5] )
refimgseg = ants.image_read( antspyt1w.get_data( "det_atlas_25_pad_LR", target_extension='.nii.gz' ))
refimgsmall = ants.resample_image( refimg, [2.0,2.0,2.0] )

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

def preprocess( img, bxt=False, returndef=False ):
    pt_labels = [7,9,23,25]
    crop_size = [96,96,64]
    if bxt:
        imgbxt = antspyt1w.brain_extraction( img, method='v1' )
        img = antspyt1w.preprocess_intensity( img, imgbxt, intensity_truncation_quantiles=[0.000001, 0.999999 ] )
    imgr = ants.rank_intensity( img )
    print("BeginReg")
    reg = ants.registration( refimgsmall, imgr, 'SyN',
        reg_iterations = [200,200,20],
#        syn_metric='CC', syn_sampling=2,
        verbose=False )
    print("EndReg")
    if not returndef:
        imgraff = ants.apply_transforms( refimg, imgr, reg['fwdtransforms'][1], interpolator='linear' )
        imgseg = ants.apply_transforms( refimg, refimgseg, reg['invtransforms'][1], interpolator='nearestNeighbor' )
    else:
        imgraff = ants.apply_transforms( refimg, imgr, reg['fwdtransforms'], interpolator='linear' )
        imgseg = ants.image_clone( refimgseg )
    binseg = ants.mask_image( imgseg, imgseg, pt_labels, binarize=True )
    imgseg = ants.mask_image( imgseg, imgseg, group_labels, binarize=False  )
    com = ants.get_center_of_mass( binseg )
    return {
        "img": imgraff,
        "seg": imgseg,
        "imgc": special_crop( imgraff, com, crop_size ),
        "segc": special_crop( imgseg, com, crop_size ),
        "reg" : reg
        }


group_labels = [0,7,8,9,23,24,25,33,34]
nLabels = len( group_labels )
number_of_classification_labels = len(group_labels)
number_of_channels = 1
################################################
unet0 = antspynet.create_unet_model_3d(
         [ None, None, None, number_of_channels ],
         number_of_outputs = 1, # number of landmarks must be known
         number_of_layers = 4, # should optimize this wrt criterion
         number_of_filters_at_base_layer = 32, # should optimize this wrt criterion
         convolution_kernel_size = 3, # maybe should optimize this wrt criterion
         deconvolution_kernel_size = 2,
         pool_size = 2,
         strides = 2,
         dropout_rate = 0.0,
         weight_decay = 0,
         additional_options = "nnUnetActivationStyle",
         mode =  "sigmoid" )

unet1 = antspynet.create_unet_model_3d(
    [None,None,None,2],
    number_of_outputs=number_of_classification_labels,
    mode="classification",
    number_of_filters=(32, 64, 96, 128, 256),
    convolution_kernel_size=(3, 3, 3),
    deconvolution_kernel_size=(2, 2, 2),
    dropout_rate=0.0,
    weight_decay=0,
    additional_options = "nnUnetActivationStyle")

# concat output to input and pass to 2nd net
nextin = tf.concat(  [ unet0.inputs[0], unet0.outputs[0] ], axis=4 )
unetonnet = unet1( nextin )
unet_model = tf.keras.models.Model(
        unet0.inputs,
        [ unetonnet,  unet0.outputs[0] ] )

unet_model.load_weights( newfn )

####################################################
returndef = True
imgprepro = preprocess( img, returndef = returndef )
####################################################
physspaceSN = imgprepro['imgc']
tfarr1 = tf.cast( physspaceSN.numpy() ,'float32' )
newshapeSN = list( tfarr1.shape )
newshapeSN.insert(0,1)
newshapeSN.insert(4,1)
tfarr1 = tf.reshape(tfarr1, newshapeSN )
snpred = unet_model.predict( tfarr1 )
segpred = snpred[0]
sigmoidpred = snpred[1]
snpred1_image = ants.from_numpy( sigmoidpred[0,:,:,:,0] )
snpred1_image = ants.copy_image_info( physspaceSN, snpred1_image )
bint = ants.threshold_image( snpred1_image, 0.5, 1.0 )
probability_images = []
for jj in range(number_of_classification_labels-1):
            temp = ants.from_numpy( segpred[0,:,:,:,jj+1] )
            probability_images.append( ants.copy_image_info( physspaceSN, temp ) )
image_matrix = ants.image_list_to_matrix(probability_images, bint)
segmentation_matrix = (np.argmax(image_matrix, axis=0) + 1)
segmentation_image = ants.matrix_to_images(np.expand_dims(segmentation_matrix, axis=0), bint)[0]
ants.plot( physspaceSN, segmentation_image, axis=2 )
relabeled_image = ants.image_clone(segmentation_image)
for i in range(1,len(group_labels)):
            relabeled_image[segmentation_image==(i)] = group_labels[i]
if not returndef:
    relabeled_image = ants.apply_transforms( img, relabeled_image,
                    imgprepro['reg']['invtransforms'][0], whichtoinvert=[True],
                    interpolator='genericLabel' )
else:
    relabeled_image = ants.apply_transforms( img, relabeled_image,
                    imgprepro['reg']['invtransforms'], interpolator='genericLabel' )
ants.image_write(relabeled_image, '/tmp/temp_sn.nii.gz' )


