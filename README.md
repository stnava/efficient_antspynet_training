# efficient antspynet training

efficient antspynet training with sleek augmentation strategies based on single or multiple template(s)

for reproducible and efficient training of parcellating segmentation models

features:

* reflection in augmentation

* rank transform

* quick registration to localize region

* bias field simulation

* fixed train and test set for reproducible training

* parcellating unet models

the procedure is as follows ( using `src/sn` as example ):

* augment the `m` original images using registration, transform simulation and concatenation of these maps to generate `m*n` output images `t1sim_whole_brain.py` and the `docs`

* determine the subset of labels that you want to parcellate and decide upon the region over which to crop `t1toNpy_whole_brain_to_sn.py` --- this also generates the *fixed* numpy training files.

* write the test data function using independent samples following the same ideas used above `make_sn_test_data.py` --- this also generates the *fixed* numpy testing files.

* train the network with fixed train and test `npy` files

* all of the above makes writing the inference function very straightforward.

This flexible system enables fair evaluation of network architectures, training strategies, etc.

To Do:

* single image training and testing example

* figure out what the common and reusable functions are, if any.
