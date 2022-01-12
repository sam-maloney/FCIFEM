# -*- coding: utf-8 -*-
"""
Created on Mon May 17 17:46:29 2021

@author: samal
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


# For all of the below
NX_array = np.array([  4,   8,  16,  32,  64, 128, 256])
start = int(np.log2(NX_array[0]))
stop = int(np.log2(NX_array[-1]))
# NY = NX, NQY=NY, NQX=6, Qord=3, quadType='g', massLumping=False
# D = 0.01, n = 2, dt = 0.005, nSteps = 100

# Straight mapping, Uniform spacing, CN
E_2u_str = np.array([1.74572355e-02, 4.30716118e-03, 1.06620253e-03, 2.65830623e-04,
        6.64313804e-05, 1.66260242e-05, 4.17748043e-06])
E_infu_str = np.array([3.49144717e-02, 8.61432242e-03, 2.13240597e-03, 5.31662194e-04,
        1.32863867e-04, 3.32525489e-05, 8.35496139e-06])

# Straight mapping, 0.1 perturbation, CN
E_2p1_str = np.array([0.01873866, 0.01795057, 0.01439723, 0.00723706, 0.00369697,
        0.00200346, 0.00128445])
E_infp1_str = np.array([0.04203476, 0.04831195, 0.05343568, 0.02944521, 0.01607812,
        0.00946864, 0.0049019 ])

# Sinusoidal mapping, Uniform spacing, CN
E_2u_sin = np.array([2.00401406e-02, 5.33575655e-03, 1.16170918e-03, 2.72763750e-04,
        7.38494580e-05, 1.78124482e-05, 4.46183625e-06])
E_infu_sin = np.array([4.00802813e-02, 1.22987430e-02, 2.75653138e-03, 6.09220904e-04,
        1.98373607e-04, 4.47375699e-05, 1.42873479e-05])

# Sinusoidal mapping, 0.1 perturbation, CN
E_2p1_sin = np.array([0.02097072, 0.00730126, 0.0023875 , 0.00101677, 0.00050876,
        0.00024956, 0.00013334])
E_infp1_sin = np.array([0.0458054 , 0.01557467, 0.01064567, 0.00574506, 0.00302401,
        0.00162349, 0.00095905])


##### Begin Plotting Routines #####

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
plt.loglog(NX_array, E_infu_str, '.-', label=r'Straight')
plt.loglog(NX_array, E_infu_sin, '.-', label=r'Sinusoid')
plt.minorticks_off()
plt.xticks(NX_array, NX_array)
plt.xlabel(r'$NX$')
plt.ylabel(r'$|E_\infty|$', rotation=0, labelpad=10)
plt.title('Uniform Node Spacing')

# plot the intra-step order of convergence
axL2 = axL1.twinx()
logN = np.log(NX_array)
logE_str = np.log(E_infu_str)
logE_sin = np.log(E_infu_sin)
order_str = (logE_str[0:-1] - logE_str[1:])/(logN[1:] - logN[0:-1])
order_sin = (logE_sin[0:-1] - logE_sin[1:])/(logN[1:] - logN[0:-1])
intraN = np.logspace(start+0.5, stop-0.5, num=stop-start, base=2.0)
plt.plot(intraN, order_str, '.:', linewidth=1, label=r'Straight order')
plt.plot(intraN, order_sin, '.:', linewidth=1, label=r'Sinusoid order')
plt.plot(plt.xlim(), [2, 2], 'k:', linewidth=1, label='Expected')
ordb = 0
ordt = 5
plt.ylim(ordb, ordt)
plt.yticks(np.linspace(ordb, ordt, ordt - ordb + 1))
plt.ylabel(r'Intra-step Order of Convergence')
lines, labels = axL1.get_legend_handles_labels()
lines2, labels2 = axL2.get_legend_handles_labels()
axL2.legend(lines + lines2, labels + labels2, loc='best')
plt.margins(0,0)

# plot the error convergence
axR1 = plt.subplot(122)
plt.loglog(NX_array, E_infp1_str, '.-', label=r'Straight')
plt.loglog(NX_array, E_infp1_sin, '.-', label=r'Sinusoid')
plt.minorticks_off()
plt.xticks(NX_array, NX_array)
plt.xlabel(r'$NX$')
plt.ylabel(r'$|E_\infty|$', rotation=0, labelpad=10)
plt.title('10% Perturbed Node Spacing')

# plot the intra-step order of convergence
axR2 = axR1.twinx()
logN = np.log(NX_array)
logE_str = np.log(E_infp1_str)
logE_sin = np.log(E_infp1_sin)
order_str = (logE_str[0:-1] - logE_str[1:])/(logN[1:] - logN[0:-1])
order_sin = (logE_sin[0:-1] - logE_sin[1:])/(logN[1:] - logN[0:-1])
intraN = np.logspace(start+0.5, stop-0.5, num=stop-start, base=2.0)
plt.plot(intraN, order_str, '.:', linewidth=1, label=r'Straight order')
plt.plot(intraN, order_sin, '.:', linewidth=1, label=r'Sinusoid order')
plt.plot(plt.xlim(), [2, 2], 'k:', linewidth=1, label='Expected')
ordb = 0
ordt = 5
plt.ylim(ordb, ordt)
plt.yticks(np.linspace(ordb, ordt, ordt - ordb + 1))
plt.ylabel(r'Intra-step Order of Convergence')
lines, labels = axR1.get_legend_handles_labels()
lines2, labels2 = axR2.get_legend_handles_labels()
axR2.legend(lines + lines2, labels + labels2, loc='best')
plt.margins(0,0)

# plt.savefig("Diff_conv.pdf", bbox_inches = 'tight', pad_inches = 0)