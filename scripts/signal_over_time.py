"""Plot MP2RAGE signal over time."""

import numpy as np
import matplotlib.pyplot as plt
from mp2ragelib.core import Mz_inv, Mz_0rf, Mz_nrf

# Parameters
eff = 1
TR_MP2RAGE = 5
TI_1 = 0.8
TI_2 = 2.7
FA_1 = np.deg2rad(4.)
FA_2 = np.deg2rad(5.)
NR_RF = 120.
# TR_GRE = 0.00291
TR_GRE = 0.00291

nr_timepoints = 1001
time = np.linspace(0, TR_MP2RAGE, nr_timepoints)
signal = np.zeros(nr_timepoints)

# WM/GM/CSF=1.05/1.85/3.35 s
T1s = [1.05, 1.85, 3.35]

T_GRE1_start = TI_1 - (NR_RF * TR_GRE / 2)
T_GRE1_end = TI_1 + (NR_RF * TR_GRE / 2)
T_GRE2_start = TI_2 - T_GRE1_end - (NR_RF * TR_GRE / 2)
T_GRE2_end = TI_2 + (NR_RF * TR_GRE / 2)

# Step 0: Inversion
M0 = 1
signal[0] = Mz_inv(eff=1, mz0=M0)
Mz0 = 0
for t1 in T1s:
    for i in range(1, nr_timepoints):
        t = time[i]

        # Step 1: Period with no pulses
        if t < T_GRE1_start:
            signal[i] = Mz_0rf(mz0=signal[i-1], t1=t1, t=t, m0=M0)

        # Step 2: First GRE block
        elif t < T_GRE1_end:
            signal[i] = Mz_nrf(mz0=signal[i-1], t1=t1, n_gre=NR_RF,
                               tr_gre=TR_GRE, alpha=FA_1, m0=M0)

        # Step 3: Prediod with no pulses
        elif t < T_GRE2_start:
            signal[i] = Mz_0rf(mz0=signal[i-1], t1=t1, t=t, m0=M0)

        # Step 4: Second GRE block
        elif t < T_GRE2_end:
            signal[i] = Mz_nrf(mz0=signal[i-1], t1=t1, n_gre=NR_RF,
                               tr_gre=TR_GRE, alpha=FA_2, m0=M0)

        # Step 5: Final recovery with no pulses
        else:
            signal[i] = Mz_0rf(mz0=signal[i-1], t1=t1, t=t, m0=M0)

    fig = plt.plot(time, signal)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Lngitudinal magnetization.")
