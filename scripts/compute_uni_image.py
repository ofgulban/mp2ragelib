"""Compute UNI image from the first and second inversions of MP2RAGE."""

import nibabel as nb
from mp2ragelib.core import compute_UNI

FILE1 = '/path/to/INV1_re.nii.gz'
FILE2 = '/path/to/INV1_im.nii.gz'
FILE3 = '/path/to/INV2_re.nii.gz'
FILE4 = '/path/to/INV2_im.nii.gz'

FILE_OUT = '/home/faruk/Data/test_pymp2rage/UNI_test.nii.gz'

# -----------------------------------------------------------------------------
# Load data
nii1 = nb.load(FILE1)
nii2 = nb.load(FILE2)
nii3 = nb.load(FILE3)
nii4 = nb.load(FILE4)

inv1_re = nii1.get_fdata()
inv1_im = nii2.get_fdata()

inv2_re = nii3.get_fdata()
inv2_im = nii4.get_fdata()

UNI = compute_UNI(inv1_re, inv1_im, inv2_re, inv2_im, scale=True)

# Save
img_out = nb.Nifti1Image(UNI, affine=nii1.affine)
nb.save(img_out, FILE_OUT)

print('Finished.')
