"""Plot MP2RAGE signal using equationn A1.3 from Marques et al. (2010)."""

import os
import numpy as np
import nibabel as nb
import matplotlib.pyplot as plt
from mp2ragelib.core import compute_T1_lookup_table, map_UNI_to_T1

# User-defined parameters
file_name = "/home/faruk/Data/test_mp2ragelib/sub-03_ses-T1_run-02_RL_UNI.nii.gz"
eff = 1
TR_MP2RAGE = 5.0
TI_1 = 0.8
TI_2 = 2.7
FA_1 = np.deg2rad(4.)
FA_2 = np.deg2rad(5.)
NR_RF = 216
TR_GRE = 0.00291 * 2

# WM/GM/CSF=1.05/1.85/3.35 s  (at 7T)
nr_timepoints = 1001
T1s = np.linspace(0.5, 5.5, nr_timepoints)

# =============================================================================
# Find UNI to T1 mapping using lookup table method
arr_UNI, arr_T1 = compute_T1_lookup_table(
    T1s=T1s, TR_MP2RAGE=TR_MP2RAGE, TR_GRE=TR_GRE, NR_RF=NR_RF,
    TI_1=TI_1, TI_2=TI_2, FA_1=FA_1, FA_2=FA_2, M0=1., eff=0.96,
    only_bijective_part=True)

# Plot UNI vs T1
fig = plt.plot(arr_UNI, arr_T1)
plt.xlabel("UNI")
plt.ylabel("T1")

# =============================================================================
# Load data
nii = nb.load(file_name)
data = nii.get_fdata()

# Scale range from uint12 to -0.5 to 0.5
data /= 4095
data -= 0.5

temp = map_UNI_to_T1(img_UNI=data, arr_UNI=arr_UNI, arr_T1=arr_T1)
temp *= 1000  # Seconds to milliseconds

# Save as nifti
img_out = nb.Nifti1Image(temp, affine=nii.affine)
basename = nii.get_filename().split(os.extsep, 1)[0]
out_name = "{}_T1.nii.gz".format(basename)
nb.save(img_out, out_name)

print("Finished")
