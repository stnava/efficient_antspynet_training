# efficient_sn_training_antspynet

efficient antspynet training with sleek augmentation strategies based on a single template

for reproducible and efficient training of parcellating segmentation models

features:

* reflection in augmentation

* rank transform

* quick registration to localize region

* bias field simulation

* fixed train and test set for reproducible training

* parcellating unet models

the procedure is as follows:

* augment the `m` original images using registration, transform simulation and concatenation of these maps to generate `m*n` output images `t1sim_whole_brain.py` and the `docs`

* determine the subset of labels that you want to parcellate and decide upon the region over which to crop `t1toNpy_whole_brain_to_sn.py`

* write the test data function using independent samples following the same ideas used above `make_sn_test_data.py`

* train the network with fixed train and test `npy` files

This flexible system enables fair evaluation of network architectures, training strategies, etc.

To Do:

* single image training and testing example
