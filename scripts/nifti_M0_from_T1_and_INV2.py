"""Estimate M0 image from UNI and INV2 images."""

import numpy as np
import nibabel as nb
import matplotlib.pyplot as plt
from mp2ragelib.core import Mz_inv, Mz_0rf, Mz_nrf

# Parameters
eff = 1
TR_MP2RAGE = 5.0
TI_1 = 0.8
TI_2 = 2.7
FA_1 = np.deg2rad(4.)
FA_2 = np.deg2rad(5.)
NR_RF = 216
TR_GRE = 0.00291 * 2

# Derived
TA = TI_1 - (NR_RF * TR_GRE / 2)
TB = TI_2 - (TI_1 + (NR_RF * TR_GRE / 2))
TC = TR_MP2RAGE - (TI_1 + (NR_RF * TR_GRE / 2))

# WM/GM/CSF=1.05/1.85/3.35 s
T1s = np.linspace(0.5, 5.5, 101)
nr_timepoints = 1001
time = np.linspace(0, TR_MP2RAGE, nr_timepoints)
signal = np.zeros([T1s.size, nr_timepoints])
M0 = 1
for i, t1 in enumerate(T1s):
    # Step 0: Inversion
    signal[i, 0] = Mz_inv(eff=1, mz0=M0)

    # Step 1: Period with no pulses
    signal[i, 1] = Mz_0rf(mz0=signal[i, 0], t1=t1, t=TA, m0=M0)

    # Step 2: First GRE block
    signal[i, 2] = Mz_nrf(mz0=signal[i, 1], t1=t1, n_gre=NR_RF, tr_gre=TR_GRE,
                          alpha=FA_1, m0=M0)

    # Step 3: Prediod with no pulses
    signal[i, 3] = Mz_0rf(mz0=signal[i, 2], t1=t1, t=TB, m0=M0)

    # Step 4: Second GRE block
    signal[i, 4] = Mz_nrf(mz0=signal[i, 3], t1=t1, n_gre=NR_RF, tr_gre=TR_GRE,
                          alpha=FA_2, m0=M0)

    # Step 5: Final recovery with no pulses
    signal[i, 5] = Mz_0rf(mz0=signal[i, 4], t1=t1, t=TC, m0=M0)

# Compute uni
signal_gre1 = signal[:, 2]
signal_gre2 = signal[:, 4]
signal_uni = signal_gre1 * signal_gre2 / (signal_gre1**2 + signal_gre2**2)

# # Plot GRE1 vs GRE2
# fig = plt.plot(signal[:, 2], signal[:, 4])
# plt.xlabel("GRE1")
# plt.ylabel("GRE2")

# Plot UNI vs T1
fig = plt.plot(signal_uni, T1s)
plt.xlabel("UNI")
plt.ylabel("T1")

# NOTE(Faruk): Only take the bijective part
idx_min = np.argmin(signal_uni)
idx_max = np.argmax(signal_uni)
arr_UNI = signal_uni[range(idx_min, idx_max, -1)]
arr_T1 = T1s[range(idx_min, idx_max, -1)]

# Plot UNI vs T1
fig = plt.plot(arr_UNI, arr_T1)
plt.xlabel("UNI")
plt.ylabel("T1")

# =============================================================================
# Load data
nii = nb.load("/home/faruk/Data/test_mp2ragelib/sub-03_ses-T1_run-02_RL_UNI.nii.gz")
data = nii.get_fdata()

# Scale range from uint12 to -0.5 to 0.5
data /= 4095
data -= 0.5

temp = np.interp(data, xp=arr_UNI, fp=arr_T1)
temp *= 1000  # Seconds to milliseconds

img = nb.Nifti1Image(temp, affine=nii.affine)
nb.save(img, "/home/faruk/Data/test_mp2ragelib/sub-03_ses-T1_run-02_RL_UNI_T1_test.nii.gz")

# -----------------------------------------------------------------------------
# Estimate M0
nii = nb.load("/home/faruk/Data/test_mp2ragelib/sub-03_ses-T1_run-02_RL_INV2.nii.gz")
img_INV2 = nii.get_fdata() / 4095
arr_GRE2 = signal_gre2[range(idx_min, idx_max, -1)]
img_M0 = np.interp(temp/1000, xp=arr_T1[::-1], fp=arr_GRE2[::-1])
img_M0 = img_INV2 / img_M0

img = nb.Nifti1Image(img_M0, affine=nii.affine)
nb.save(img, "/home/faruk/Data/test_mp2ragelib/sub-03_ses-T1_run-02_RL_UNI_M0_test.nii.gz")

print("Finished")
