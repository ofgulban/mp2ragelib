"""Core functions."""
import numpy as np


def compute_UNI(inv1_re, inv1_im, inv2_re, inv2_im, scale=False):
    """Compute UNI image.

    Parameters
    ----------
    inv1_re: np.ndarray
        Real component of MP2RAGE first inversion complex image.
    inv1_im: np.ndarray
        Imaginary component of MP2RAGE first inversion complex image.
    inv2_re: np.ndarray
        Real component of MP2RAGE second inversion complex image.
    inv2_im: np.ndarray
        Imaginary component of MP2RAGE second inversion complex image.
    scale : bool
        Do not scale and clip the results when false.

    Returns
    -------
    uni: np.ndarray, 3D
        Unified image that looks similar to T1-weighted images.

    """
    inv1 = inv1_re + inv1_im * 1j
    inv2 = inv2_re + inv2_im * 1j

    # Marques et al. (2010), equation 3.
    uni = inv1.conj() * inv2 / (np.abs(inv1)**2 + np.abs(inv2)**2)
    uni = np.real(uni)

    if scale:  # Scale to 0-4095 (uin12) range
        # NOTE(Faruk): Looks redundant but keeping for backwards compatibility.
        uni *= 4095
        uni += 2048
        uni = np.clip(uni, 0, 4095)

    return uni


# Appendix a)
def Mz_inv(eff, mz0):
    """Magnetization after adiabatic inversion pulse.

    Marques et al. (2010), Appendix 1, Item A):
    'Longitudinal magnetization is inverted by means of an adiabatic pulse of
    a given efficiency.'

    Parameters
    ----------
    eff : float
        Efficiency of the adiabatic inversion pulse.
    mz0 : float
        Longitudinal magnetization at time=0.

    """
    return -eff * mz0


# Appendix b)
def Mz_nrf(mz0, t1, n_gre, tr_gre, alpha, m0):
    """Magnetization during the GRE block.

    Marques et al. (2010), Appendix 1, Item B):
    'During the GRE blocks of n RF pulses with constant flip angles (alpha),
    separated by an interval TR, the longitudinal magnetization evolves in
    the following way (Deichmann and Haase, 1992).'

    Parameters
    ----------
    mz0 : float
        Longitudinal magnetization at time=0.
    t1 : float, np.ndarray
        T1 time in seconds.
    n_gre : int
        Number of radio frequency (RF) pulses in each GRE block.
    tr_gre : float
        Repetition time (TR) of gradient recalled echo (GRE) pulses in
        seconds.
    alpha : float
        Flip angle in radians.
    m0 : float
        Longitudinal magnetization at the start of the RF free periods.

    """
    from sympy import exp, cos
    return (mz0 * (cos(alpha) * exp(-tr_gre / t1)) ** n_gre
            + m0 * (1 - exp(-tr_gre / t1))
            * (1 - (cos(alpha) * exp(-tr_gre / t1)) ** n_gre)
            / (1 - cos(alpha) * exp(-tr_gre / t1))
            )


# Appendix c)
def Mz_0rf(mz0, t1, t, m0):
    """Magnetization during the period with no pulses.

    Marques et al. (2010), Appendix 1, Item C):
    'During the periods with no RF pulses, the longitudinal magnetization
    relaxes freely towards equilibrium following the conventional T1
    relaxation expression.'

    Parameters
    ----------
    mz0 : float
        Longitudinal magnetization at time=0.
    t1 : float, np.ndarray
        T1 time in seconds.
    t : float
        Time in seconds.
    m0 : float
        Longitudinal magnetization at the start of the RF free periods.

    """
    from sympy import exp
    return mz0 * exp(-t / t1) + m0 * (1 - exp(-t / t1))


def Mz_ss(eff, mz0, t1, t, n_gre, tr_gre, alpha):
    """MP2RAGE signal.

    A full account of the signal resulting from the MP2RAGE sequence has to
    take into account the steady-state condition. This implies that the
    longitudinal magnetization before successive inversions, mz,ss, has to be
    the same. Between two successive inversions the mz,ss undergoes first an
    inversion (a), followed by recovery for a period TA (c), a first GRE
    block (b), a free recovery for a period TB (c), a second GRE block (b),
    and a final recovery for a period TC (c) by the end of which it should be
    back to its initial value

    """
    # Step 0: Inversion
    s0 = Mz_inv(eff, mz0)
    # Step 1: Period with no pulses
    s1 = Mz_0rf(mz0, t1, t, s0)
    # Step 2: First GRE block
    s2 = Mz_nrf(mz0, t1, n_gre, tr_gre, alpha, s1)
    # Step 3: Prediod with no pulses
    s3 = Mz_0rf(mz0, t1, t, s2)
    # Step 4: Second GRE block
    s4 = Mz_nrf(mz0, t1, n_gre, tr_gre, alpha, s3)
    # Step 5: Final recovery with no pulses
    s5 = Mz_0rf(mz0, t1, t, s4)

    return s5


def Mz_ss_solved(T1, NR_RF, TR_GRE, TR_MP2RAGE, TI_1, TI_2, FA_1, FA_2,
                 M0=1., eff=0.96):
    """Compute steadt state longitudinal magnetization.

    Parameters
    ----------
    M0 : float
        Longitudinal signal at time=0.
    T1 : float
        T1 time in seconds.
    NR_RF : int
        Number of radio frequency pulses in one GRE readout.
    TR_GRE : float
        Time between successive excitation pulses in the GRE kernel in
        seconds, which is composed of NR_RF pulses.
    TR_MP2RAGE : float
        Time between two successive inversion pulses in seconds.
    TI_1 : float
        First inversion time in seconds.
    TI_2 : float
        Second inversion time in seconds.
    FA_1 : float
        Flip angle 1 in radians.
    FA_2 : float
        Flip angle 2 in radians.
    eff: float
        Inversion efficiency of the adiabatic pulse. Default is 0.96 as used in
        Marques et al. (2010), mentioned below Equation 3.

    Notes
    -----
    In order to fully understand what is happening in this function, study
    Marques et al (2010), Appendix 1, Equation A1.4 and see Figure 1 for the
    parameter definitions.

    """
    # See Marques et al. (2010) Figure 1 where TA, TB, TC are defined.
    T_GRE = NR_RF * TR_GRE  # Duration of one readout
    TA = TI_1 - (T_GRE / 2.)  # First no pulse period
    TB = TI_2 - (TA + T_GRE + (T_GRE / 2.))  # Second no pulse period
    TC = TR_MP2RAGE - (TA + T_GRE + TB + T_GRE)  # Final no pulse period

    # print("T_GRE={} TA={} TB={} TC={}".format(T_GRE, TA, TB, TC))

    # Following handy definitions below Equation A1.4 in Marques et al. (2010)
    E1 = np.exp(-TR_GRE / T1)
    EA = np.exp(-TA / T1)
    EB = np.exp(-TB / T1)
    EC = np.exp(-TC / T1)

    # print("E1={} EA={} EB={} EC={}".format(E1, EA, EB, EC))

    # Pre-compute cosine terms
    C1 = np.cos(FA_1) * E1
    C2 = np.cos(FA_2) * E1

    # print("C1={} C2={}".format(C1, C2))

    # Numerator part
    term1 = (1 - EA) * C1**NR_RF
    term2 = (1 - E1) * (1 - C1**NR_RF) / (1 - C1)
    term3 = (term1 + term2) * EB + (1 - EB)
    term4 = term3 * C2**NR_RF
    term5 = (1 - E1) * (1 - C2**NR_RF) / (1 - C2)
    term6 = M0 * (term4 + term5) * EC + (1 - EC)

    # print("term1={} term2={} term3={}".format(term1, term2, term3))
    # print("term4={} term5={} term6={}".format(term4, term5, term6))

    # Denominator part
    term7 = (np.cos(FA_1) * np.cos(FA_2))**NR_RF
    term8 = 1 + eff * term7 * np.exp(-TR_MP2RAGE / T1)

    # print("term7={} term8={}".format(term7, term8))

    return term6 / term8


def signal_gre1(mz_ss, FA_1, NR_RF, TR_GRE, TI_1, T1, M0=1., eff=0.96):
    """Signal of the first inversion."""
    # Handy definitions
    T_GRE = NR_RF * TR_GRE  # Duration of one readout
    TA = TI_1 - (T_GRE / 2.)  # First no pulse period
    E1 = np.exp(-TR_GRE / T1)
    EA = np.exp(-TA / T1)
    C1 = np.cos(FA_1) * E1

    term1 = ((-eff * mz_ss) / M0) * EA + (1 - EA)
    term2 = C1 ** (NR_RF / 2 - 1)
    term3 = (1 - E1) * (1 - C1**(NR_RF / 2 - 1)) / (1 - C1)

    return np.sin(FA_1) * (term1 * term2 / term3)


def signal_gre2(mz_ss, FA_2, NR_RF, TR_GRE, TR_MP2RAGE, TI_1, TI_2, T1, M0=1.):
    """Signal of the second inversion."""
    # Handy definitions
    T_GRE = NR_RF * TR_GRE  # Duration of one readout
    TA = TI_1 - (T_GRE / 2.)  # First no pulse period
    TB = TI_2 - (TA + T_GRE + (T_GRE / 2.))  # Second no pulse period
    TC = TR_MP2RAGE - (TA + T_GRE + TB + T_GRE)  # Final no pulse period
    E1 = np.exp(-TR_GRE / T1)
    EC = np.exp(-TC / T1)
    C2 = np.cos(FA_2) * E1

    term1 = (mz_ss / M0) - (1 - EC)
    term2 = EC * C2**(NR_RF / 2.)
    term3 = (1 - E1) * (C2**(-NR_RF / 2.) - 1) / (1 - C2)

    return np.sin(FA_2) * (term1 / term2 - term3)


# def signal_gre(mz_ss, FA_1, FA_2, NR_RF, TR_GRE, TI_1, TI_2, T1, TR_MP2RAGE,
#                M0=1., eff=0.96):
#     """TEMP!!! Signal of the first inversion."""
#     # Handy definitions
#     T_GRE = NR_RF * TR_GRE  # Duration of one readout
#     TA = TI_1 - (T_GRE / 2.)  # First no pulse period
#     TB = TI_2 - (TA + T_GRE + (T_GRE / 2.))  # Second no pulse period
#     TC = TR_MP2RAGE - (TA + T_GRE + TB + T_GRE)  # Final no pulse period
#     E1 = np.exp(-TR_GRE / T1)
#     EA = np.exp(-TA / T1)
#     EB = np.exp(-TB / T1)
#     EC = np.exp(-TC / T1)
#     C1 = np.cos(FA_1) * E1
#     C2 = np.cos(FA_2) * E1
#
#     term1 = ((-eff * mz_ss) / M0) * EA + (1 - EA)
#     term2 = C1 ** (NR_RF / 2 - 1)
#     term3 = (1 - E1) * (1 - C1**(NR_RF / 2 - 1)) / (1 - C1)
#
#     temp = (term1 * term2 + term3)
#     signal1 = np.sin(FA_1) * temp
#
#     #
#     term4 = temp * C1**(NR_RF / 2 - 1)
#     term5 = M0 * (1 - E1) * (1 - C1**(NR_RF / 2 - 1)) / (1 - C1)
#     temp = term4 + term5
#
#     term6 = temp * EC + M0 * (1 - EC) * C1**(NR_RF / 2 - 1)
#     term7 = M0 * (1 - E1) * (1 - C2)**(NR_RF / 2 - 1) / (1 - C2)
#
#     temp = term6 + term7
#     signal2 = np.sin(FA_2) * temp
#
# # temp=temp * (cosalfaE1(m-1))^(nZ_aft) + ...
# #     M0.* (1 - E_1) .* (1 - (cosalfaE1(m-1))^(nZ_aft))...
# #     ./ (1-(cosalfaE1(m-1)));
# #
# # temp=(temp*E_TD(m) + M0 * ( 1-E_TD(m))).*(cosalfaE1(m))^(nZ_bef)+...
# #     M0.*(1-E_1).*(1-(cosalfaE1(m))^(nZ_bef))...
# #     ./(1-(cosalfaE1(m)));
#
#     return signal1, signal2
