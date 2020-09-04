"""Plot MP2RAGE UNI to T1 lookup table."""

import numpy as np
import nibabel as nb
import matplotlib.pyplot as plt
from mp2ragelib.core import Mz_ss_solved, signal_gre1, signal_gre2

# Range of T1 values
NR_SAMPLES = 1000
arr_T1 = np.linspace(0.05, 5, NR_SAMPLES)

# From Marques et al. (2010), Table 1, 7T subjects 1-7
eff = 0.96
TR_MP2RAGE = 5.0
TI_1 = 0.8
TI_2 = 2.7
FA_1 = np.deg2rad(4.)
FA_2 = np.deg2rad(5.)
NR_RF = 120.
# TR_GRE = 0.00291
TR_GRE = 0.00291

arr_UNI = np.zeros(NR_SAMPLES)
for i, t1 in enumerate(arr_T1):
    mz_ss = Mz_ss_solved(T1=t1, NR_RF=NR_RF, TR_GRE=TR_GRE,
                         TR_MP2RAGE=TR_MP2RAGE,
                         TI_1=TI_1, TI_2=TI_2, FA_1=FA_1, FA_2=FA_2, eff=eff)

    s1 = signal_gre1(mz_ss=mz_ss, FA_1=FA_1, NR_RF=NR_RF, TR_GRE=TR_GRE,
                     TI_1=TI_1, T1=t1, eff=eff)

    s2 = signal_gre2(mz_ss=mz_ss, FA_2=FA_2, NR_RF=NR_RF, TR_GRE=TR_GRE,
                     TR_MP2RAGE=TR_MP2RAGE, TI_1=TI_1, TI_2=TI_2, T1=t1)

    s1 = s1 + s1*1j
    s2 = s2 + s2*1j

    arr_UNI[i] = np.real((s1 * np.conj(s2)) / (np.abs(s1)**2 + np.abs(s2)**2))

arr_UNI[np.isnan(arr_UNI)] = 0

fig = plt.plot(arr_UNI, arr_T1)
plt.xlabel("UNI")
plt.ylabel("T1 (ms)")
plt.xlim((-0.5, 0.5))


# NOTE(Faruk): Only take the bijective part
idx_min = np.argmax(arr_UNI)
idx_max = np.argmin(arr_UNI)
arr_UNI = arr_UNI[idx_min:idx_max]
arr_T1 = arr_T1[idx_min:idx_max]

# Plot
fig = plt.plot(arr_UNI, arr_T1)
plt.xlabel("UNI")
plt.ylabel("T1 (ms)")
plt.xlim((-0.5, 0.5))
print("UNI Min={} Max={}".format(arr_UNI.min(), arr_UNI.max()))
print("T1 Min={} Max={}".format(arr_T1.min(), arr_T1.max()))

# # Load data
# nii = nb.load("/home/faruk/Data/test_mp2ragelib/sub-03_ses-T1_run-02_RL_UNI.nii.gz")
# data = nii.get_fdata()
#
# # Scale range from uint12 to -0.5 to 0.5
# data /= 4095
# data -= 0.5
#
# temp = np.interp(data, xp=arr_UNI, fp=arr_T1)
# temp *= 1000  # Seconds to milliseconds
#
# img = nb.Nifti1Image(temp, affine=nii.affine)
# nb.save(img, "/home/faruk/Data/test_mp2ragelib/sub-03_ses-T1_run-02_RL_UNI_T1_test.nii.gz")
#
# print("Finished")
