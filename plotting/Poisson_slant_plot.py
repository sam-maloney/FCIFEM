# -*- coding: utf-8 -*-
"""
Created on Mon May 17 17:46:29 2021

@author: samal
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# For all of the below, unless otherwise noted
# uniform spacing, NQY=NY, NQX=6, Qord=3, quadType='gauss', massLumping=False
NY_array = np.array([  4,   8,  16,  32,  64, 128, 256, 512])
start = int(np.log2(NY_array[0]))
stop = int(np.log2(NY_array[-1]))

##### Left and bottom borders and centre point constrained #####

# Straight mapping, NY = NX
# NX_array = np.array([  4,   8,  16,  32,  64, 128, 256])
# NY_array = np.array([  4,   8,  16,  32,  64, 128, 256])
E_2_1N_str = np.array([1.24854595e-04, 1.13034564e-04, 5.36187355e-05, 3.61905545e-05,
       9.19518906e-06, 2.25601602e-06, 5.60632125e-07, np.nan])
E_inf_1N_str = np.array([2.93155729e-04, 3.19461015e-04, 1.35666951e-04, 8.51800106e-05,
       2.34681268e-05, 5.43782230e-06, 1.31899050e-06, np.nan])
t_setup_1N_str = np.array([1.21315800e-01, 4.54591600e-01, 1.97989810e+00, 7.94103160e+00,
       3.19645893e+01, 1.14166103e+02, 4.37051618e+02, np.nan])
t_solve_1N_str = np.array([0.0014533, 0.0027783, 0.0062069, 0.0179855, 0.0437613, 0.2242696,
       1.283171, np.nan])
t_sim_1N_str = t_setup_1N_str + t_solve_1N_str
eta_1N_str = (t_sim_1N_str/t_sim_1N_str) / (E_2_1N_str/E_2_1N_str)

# Aligned-linear mapping, NY = NX
# NX_array = np.array([  4,   8,  16,  32,  64, 128, 256])
# NY_array = np.array([  4,   8,  16,  32,  64, 128, 256])
E_2_1N_lin = np.array([6.80978330e-05, 3.34541181e-06, 1.13919640e-06, 2.53629306e-07,
       6.23542020e-08, 1.55320678e-08, 3.88466425e-09, np.nan])
E_inf_1N_lin = np.array([1.58884385e-04, 8.54154172e-06, 2.96035339e-06, 5.58509505e-07,
       1.27822751e-07, 3.03057707e-08, 7.53510059e-09, np.nan])
t_setup_1N_lin = np.array([1.29583700e-01, 4.73063600e-01, 1.94695980e+00, 7.38590050e+00,
       2.97660262e+01, 1.18776700e+02, 4.94964127e+02, np.nan])
t_solve_1N_lin = np.array([0.0102117, 0.0043037, 0.0066012, 0.0145141, 0.0469667, 0.2109226,
       1.242389, np.nan])
t_sim_1N_lin = t_setup_1N_lin + t_solve_1N_lin
eta_1N_lin = (t_sim_1N_str/t_sim_1N_lin) / (E_2_1N_lin/E_2_1N_str)

# Aligned-linear mapping, NY=8*NX
# NX_array = np.array([  2,   4,   8,  16,  32,  64])
# NY_array = np.array([ 16,  32,  64, 128, 256, 512])
E_2_8N_lin = np.array([np.nan, np.nan, 5.25375009e-06, 1.38000292e-05, 3.46288388e-06, 8.52320168e-07,
       2.10901674e-07, 5.26099334e-08])
E_inf_8N_lin = np.array([np.nan, np.nan, 1.02423035e-05, 3.14426096e-05, 8.89229498e-06, 2.21081229e-06,
       5.23471889e-07, 1.25446896e-07])
t_setup_8N_lin = np.array([np.nan, np.nan,   0.2725781,   1.0872462,   3.7958009,  15.0327328,  60.7268843,
       252.6356505])
t_solve_8N_lin = np.array([np.nan, np.nan, 8.85099871e-04, 2.13967999e-02, 2.96226998e-02, 6.72756999e-02,
       3.38067900e-01, 1.48106630e+00])
t_sim_8N_lin = t_setup_8N_lin + t_solve_8N_lin
eta_8N_lin = (t_sim_1N_str/t_sim_8N_lin) / (E_2_8N_lin/E_2_1N_str)

# Aligned-linear mapping, NY=16*NX
# NX_array = np.array([  2,   4,   8,  16,  32])
# NY_array = np.array([ 32,  64, 128, 256, 512])
E_2_16N_lin = np.array([np.nan, np.nan, np.nan, 4.95322139e-06, 1.34090761e-05, 3.37345988e-06, 8.30703195e-07,
       2.05578962e-07])
E_inf_16N_lin = np.array([np.nan, np.nan, np.nan, 1.37547994e-05, 3.36564826e-05, 8.75730417e-06, 2.19127303e-06,
       5.15289696e-07])
t_setup_16N_lin = np.array([np.nan, np.nan, np.nan,   0.5231074,   2.1668616,   8.0369016,  31.1588809, 120.1277179])
t_solve_16N_lin = np.array([np.nan, np.nan, np.nan, 0.0017952, 0.029886 , 0.0624809, 0.2081966, 1.1798728])
t_sim_16N_lin = t_setup_16N_lin + t_solve_16N_lin
eta_16N_lin = (t_sim_1N_str/t_sim_16N_lin) / (E_2_16N_lin/E_2_1N_str)


##### Begin Plotting Routines #####

# from matplotlib import rcParams
mpl.rcParams['pdf.fonttype'] = 42
mpl.rc("font", **{"family": "serif", "serif": ["Latin Modern Roman"]})
# mpl.rc("font", **{"family": "serif", "serif": ["Palatino"]})
mpl.rc("text", usetex = True)

# clear the current figure, if opened, and set parameters
fig = plt.gcf()
fig.clf()
fig.set_size_inches(7.75,3)
plt.subplots_adjust(hspace = 0.5, wspace = 0.5)

# SMALL_SIZE = 7
# MEDIUM_SIZE = 8
# BIGGER_SIZE = 10
# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=MEDIUM_SIZE)    # fontsize of the axes title
# plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

axL1 = plt.subplot(121)
plt.loglog(NY_array, E_2_1N_str, '.-', label=r'unaligned 1:1')
plt.loglog(NY_array, E_2_1N_lin, '.-', label=r'alinged 1:1')
plt.loglog(NY_array, E_2_8N_lin, '.-', label=r'aligned 1:8')
plt.loglog(NY_array, E_2_16N_lin, '.-', label=r'aligned 1:16')
plt.minorticks_off()
# plt.ylim(top=1.5e-2)
plt.xticks(NY_array, NY_array)
plt.xlabel(r'$NY$')
plt.ylabel(r'$|E_2|$', rotation=0, labelpad=10)
axL1.legend()
plt.title('Aligned vs Unaligned')

# # plot the intra-step order of convergence
# axL2 = axL1.twinx()
# logN = np.log(NY_array)
# intraN = np.logspace(start+0.5, stop-0.5, num=stop-start, base=2.0)
# # logE_str = np.log(E_2_u_str)
# # order_str = (logE_str[0:-1] - logE_str[1:])/(logN[1:] - logN[0:-1])
# # plt.plot(intraN, order_str, '.:', linewidth=1, label=r'Str/Uniform order')
# logE_sin = np.log(E_2_u_sin)
# order_sin = (logE_sin[0:-1] - logE_sin[1:])/(logN[1:] - logN[0:-1])
# plt.plot(intraN, order_sin, '.:', linewidth=1, label=r'Uniform order')
# # logE_p1_str = np.log(E_2_p1_str)
# # order_p1_str = (logE_p1_str[0:-1] - logE_p1_str[1:])/(logN[1:] - logN[0:-1])
# # plt.plot(intraN, order_p1_str, '.:', linewidth=1, label=r'Str/10% pert order')
# logE_p1_sin = np.log(E_2_p1_sin)
# order_p1_sin = (logE_p1_sin[0:-1] - logE_p1_sin[1:])/(logN[1:] - logN[0:-1])
# plt.plot(intraN, order_p1_sin, '.:', linewidth=1, label=r'10\% pert order')

# plt.plot(plt.xlim(), [2, 2], 'k:', linewidth=1, label='Expected ord')
# ordb = 0
# ordt = 2.5
# plt.ylim(ordb, ordt)
# # plt.yticks(np.linspace(ordb, ordt, ordt - ordb + 1)) # unit spacing
# plt.yticks(np.linspace(ordb, ordt, int((ordt - ordb)*2 + 1))) # 0.5 spacing
# plt.ylabel(r'Intra-step Order of Convergence')
# lines, labels = axL1.get_legend_handles_labels()
# lines2, labels2 = axL2.get_legend_handles_labels()
# leg = axL2.legend(lines, labels, loc='lower left')
# # leg = axL2.legend(lines + lines2, labels + labels2, loc='best')
# # leg = axL2.legend(lines + [lines2[-1]], labels + [labels2[-1]], loc='lower left')
# plt.margins(0,0)


axR1 = plt.subplot(122)
plt.loglog(NY_array[:-1], eta_1N_str[:-1], '.-', label=r'unaligned 1:1')
plt.loglog(NY_array[:-1], eta_1N_lin[:-1], '.-', label=r'alinged 1:1')
plt.loglog(NY_array[:-1], eta_8N_lin[:-1], '.-', label=r'aligned 1:8')
plt.loglog(NY_array[:-1], eta_16N_lin[:-1], '.-', label=r'aligned 1:16')
plt.minorticks_off()
# plt.ylim(top=1.5e-2)
plt.xticks(NY_array[:-1], NY_array[:-1])
plt.xlabel(r'$NY$')
plt.ylabel(r'$\eta$', rotation=0, labelpad=10)
axR1.legend(loc='lower right')
plt.title('Time Efficiency')

# # plot the intra-step order of convergence
# axR2 = axR1.twinx()
# logN = np.log(NY_array)
# # logE_p1_sin = np.log(E_2_p1_sin)
# # order_p1_sin = (logE_p1_sin[0:-1] - logE_p1_sin[1:])/(logN[1:] - logN[0:-1])
# # plt.plot(intraN, order_p1_sin, '.:', linewidth=1, label=r'Q3 order')
# logE_sin10 = np.log(E_2_p1_sin10)
# order_sin10 = (logE_sin10[0:-1] - logE_sin10[1:])/(logN[1:] - logN[0:-1])
# plt.plot(intraN, order_sin10, '.:', linewidth=1, label=r'Q10 order')
# logE_p1_sin_L = np.log(E_2_p1_sin_L)
# order_p1_sin_L = (logE_p1_sin_L[0:-1] - logE_p1_sin_L[1:])/(logN[1:] - logN[0:-1])
# plt.plot(intraN, order_p1_sin_L, '.:', linewidth=1, label=r'Q3 VC1 order')
# logE_p1_sin_Q = np.log(E_2_p1_sin_Q)
# order_p1_sin_Q = (logE_p1_sin_Q[0:-1] - logE_p1_sin_Q[1:])/(logN[1:] - logN[0:-1])
# plt.plot(intraN, order_p1_sin_Q, '.:', linewidth=1, label=r'Q3 VC2 order')
# logE_p1_sin_CN = np.log(E_2_p1_sin_CN)
# order_p1_sin_CN = (logE_p1_sin_CN[0:-1] - logE_p1_sin_CN[1:])/(logN[1:] - logN[0:-1])
# plt.plot(intraN, order_p1_sin_CN, '.:', linewidth=1, label=r'Q3 VC1-C order')

# plt.plot(plt.xlim(), [2, 2], 'k:', linewidth=1, label='Expected ord')
# ordb = 0
# ordt = 2.5
# plt.ylim(ordb, ordt)
# # plt.yticks(np.linspace(ordb, ordt, ordt - ordb + 1))
# plt.yticks(np.linspace(ordb, ordt, int((ordt - ordb)*2 + 1)))
# plt.ylabel(r'Intra-step Order of Convergence')
# lines, labels = axR1.get_legend_handles_labels()
# lines2, labels2 = axR2.get_legend_handles_labels()
# leg = axR2.legend(lines, labels, loc='lower left')
# # leg = axR2.legend(lines + lines2, labels + labels2, loc='best')
# # leg = axR2.legend(lines + [lines2[-1]], labels + [labels2[-1]], loc='lower left')
# plt.margins(0,0)

plt.savefig("Poisson_slant.pdf", bbox_inches = 'tight', pad_inches = 0)
