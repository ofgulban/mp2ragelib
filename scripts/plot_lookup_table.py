"""Plot MP2RAGE UNI to T1 lookup table."""

import numpy as np
import nibabel as nb
import matplotlib.pyplot as plt
from pymp2rage_redux.core import mz_ss_solved, signal_gre1, signal_gre2
from pymp2rage_redux.core import signal_gre

# Range of T1 values
NR_SAMPLES = 1000
arr_T1 = np.linspace(0.5, 3, NR_SAMPLES)

# From Marques et al. (2010), Table 1, 7T subjects 1-7
eff = 1
TR_MP2RAGE = 5
TI_1 = 0.8
TI_2 = 2.7
FA_1 = np.deg2rad(4.)
FA_2 = np.deg2rad(5.)
NR_RF = 120.
TR_GRE = 0.00291

arr_UNI = np.zeros(NR_SAMPLES)
for i, t1 in enumerate(arr_T1):
    mz_ss = mz_ss_solved(T1=t1, NR_RF=NR_RF, TR_GRE=TR_GRE,
                         TR_MP2RAGE=TR_MP2RAGE,
                         TI_1=TI_1, TI_2=TI_2, FA_1=FA_1, FA_2=FA_2, eff=eff)

    # s1, s2 = signal_gre(mz_ss=mz_ss, FA_1=FA_1, FA_2=FA_2, NR_RF=NR_RF,
    #                     TR_GRE=TR_GRE, TI_1=TI_1, TI_2=TI_2, T1=t1, eff=eff,
    #                     TR_MP2RAGE=TR_MP2RAGE)

    s1 = signal_gre1(mz_ss=mz_ss, FA_1=FA_1, NR_RF=NR_RF, TR_GRE=TR_GRE,
                     TI_1=TI_1, T1=t1, eff=eff)

    s2 = signal_gre2(mz_ss=mz_ss, FA_2=FA_2, NR_RF=NR_RF, TR_GRE=TR_GRE,
                     TR_MP2RAGE=TR_MP2RAGE, TI_1=TI_1, TI_2=TI_2, T1=t1)

    arr_UNI[i] = (s1 * s2) / (s1**2 + s2**2)

arr_UNI[np.isnan(arr_UNI)] = 0

# NOTE(Faruk): Only take the bijective part
idx_min = np.argmin(arr_UNI)
idx_max = np.argmax(arr_UNI)
arr_UNI = arr_UNI[range(idx_min, idx_max, -1)]
arr_T1 = arr_T1[range(idx_min, idx_max, -1)]

# Plot
fig = plt.plot(arr_UNI, arr_T1)
plt.xlabel("UNI")
plt.ylabel("T1 (ms)")
plt.xlim((-0.5, 0.5))
print("UNI Min={} Max={}".format(arr_UNI.min(), arr_UNI.max()))
print("T1 Min={} Max={}".format(arr_T1.min(), arr_T1.max()))

# Load data
nii = nb.load("/home/faruk/Data/test_pymp2rage/UNI_masked.nii.gz")
data = nii.get_fdata()

# Scale range from uint12 to -0.5 to 0.5
data /= 4095
data -= 0.5

temp = np.interp(data, xp=arr_UNI, fp=arr_T1)
temp *= 1000  # Seconds to milliseconds

img = nb.Nifti1Image(temp, affine=nii.affine)
nb.save(img, "/home/faruk/Data/test_pymp2rage/UNI_masked_T1_test.nii.gz")

print("Finished")

# =============================================================================
# TODO: Simulate signal over the whole MP2RAGE scan, stage by stage

# # Parameters
# nr_timepoints = 101
# time = np.linspace(0, 5, nr_timepoints)
# signal = np.zeros(nr_timepoints)
# T1 = 1.15
#
# T_GRE1_start = TI_1 - (NR_RF * TR_GRE / 2)
# T_GRE1_end = TI_1 + (NR_RF * TR_GRE / 2)
# T_GRE2_start = TI_2 - T_GRE1_end - (NR_RF * TR_GRE / 2)
# T_GRE2_end = TI_2 + (NR_RF * TR_GRE / 2)
#
# # Step 0: Inversion
# M0 = 1
# S0 = mz_inv(eff=1, mz0=M0)
# for i, t in enumerate(time):
#     if t < T_GRE1_start:  # Step 1: Period with no pulses
#         signal[i] = mz_0rf(mz0=M0, t1=T1, t=t, m0=S0)
#     elif t < T_GRE1_end:  # Step 2: First GRE block
#         signal[i] = mz_nrf(mz0=M0, t1=T1, n_gre, tr_gre, alpha, s1)
#     elif t < T_GRE2_start:  # Step 3: Prediod with no pulses
#         signal[i] = mz_0rf(mz0=M0, t1=T1, t=t, s2)
#     # Step 4: Second GRE block
#     s4 = mz_nrf(mz0, t1, n_gre, tr_gre, alpha, s3)
