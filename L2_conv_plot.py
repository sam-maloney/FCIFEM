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
# NY = NX

# Uniform spacing, Nquad=5
E_2u = np.array([1.17817353e-01, 9.85576979e-02, 2.45978802e-02, 6.02352877e-03,
        1.49687962e-03, 3.73653269e-04, 9.33777137e-05])
# E_infu = np.array([2.35634705e-01, 1.74694642e-01, 4.76311914e-02, 1.16460444e-02,
#         2.89217058e-03, 7.21715652e-04, 1.80346973e-04])

# 0.1 perturbation, Nquad=5
E_2p1 = np.array([1.34140716e-01, 1.02430887e-01, 2.53661338e-02, 6.15948128e-03,
        1.52570919e-03, 3.80520484e-04, 9.52952878e-05])
# E_infp1 = np.array([2.42670205e-01, 1.84218885e-01, 5.24279763e-02, 1.34969241e-02,
#         3.40146362e-03, 9.10205975e-04, 2.26928637e-04])

# 0.5 perturbation, Nquad=5
E_2p5 = np.array([3.32304951e-01, 1.58513739e-01, 3.94981765e-02, 9.56499954e-03,
        2.23580423e-03, 5.46624171e-04, 1.40440403e-04])
# E_infp5 = np.array([5.36106904e-01, 3.84477693e-01, 1.21872302e-01, 3.14200379e-02,
#         7.67059713e-03, 2.10334214e-03, 5.30771969e-04])

##### Begin Plotting Routines #####

# from matplotlib import rcParams
mpl.rcParams['pdf.fonttype'] = 42
mpl.rc("font", **{"family": "serif", "serif": ["Latin Modern Roman"]})
# mpl.rc("font", **{"family": "serif", "serif": ["Palatino"]})
mpl.rc("text", usetex = True)

# clear the current figure, if opened, and set parameters
fig = plt.gcf()
fig.clf()
fig.set_size_inches(3.875,3)
# plt.subplots_adjust(hspace = 0.5, wspace = 0.5)
plt.subplots_adjust(left = 0.2, right = 0.85)

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

axL1 = plt.subplot(111)
plt.loglog(NX_array, E_2u, '.-', label=r'uniform')
plt.loglog(NX_array, E_2p1, '.-', label=r'10\%')
plt.loglog(NX_array, E_2p5, '.-', label=r'50\%')
plt.minorticks_off()
plt.xticks(NX_array, NX_array)
plt.xlabel(r'$NX$')
plt.ylabel(r'$|E_2|$', rotation=0, labelpad=10)
plt.title(r'$L_2$ Projection')

# plot the intra-step order of convergence
axL2 = axL1.twinx()
logN = np.log(NX_array)
# logE_inf = np.log(E_infu)
logE_2u = np.log(E_2u)
logE_2p1 = np.log(E_2p1)
logE_2p5 = np.log(E_2p5)
# order_inf = (logE_inf[0:-1] - logE_inf[1:])/(logN[1:] - logN[0:-1])
order_2u = (logE_2u[0:-1] - logE_2u[1:])/(logN[1:] - logN[0:-1])
order_2p1 = (logE_2p1[0:-1] - logE_2p1[1:])/(logN[1:] - logN[0:-1])
order_2p5 = (logE_2p5[0:-1] - logE_2p5[1:])/(logN[1:] - logN[0:-1])
intraN = np.logspace(start+0.5, stop-0.5, num=stop-start, base=2.0)
# plt.plot(intraN, order_inf, '.:', linewidth=1, label=r'$E_\infty$ order')
plt.plot(intraN, order_2u, '.:', linewidth=1, label=r'uniform order')
plt.plot(intraN, order_2p1, '.:', linewidth=1, label=r'10\% order')
plt.plot(intraN, order_2p5, '.:', linewidth=1, label=r'50\% order')
plt.plot(plt.xlim(), [2, 2], 'k:', linewidth=1, label='Expected')
plt.ylim(0, 5)
plt.yticks(np.linspace(0,5,6))
plt.ylabel(r'Intra-step Order of Convergence')
lines, labels = axL1.get_legend_handles_labels()
# lines2, labels2 = axL2.get_legend_handles_labels()
# axL2.legend(lines + lines2, labels + labels2, loc='best')
axL2.legend(lines, labels, loc='best')
plt.margins(0,0)

# # plot the error convergence
# axR1 = plt.subplot(122)
# plt.loglog(NX_array, E_infp1, '.-', label=r'10% $|E_\infty|$')
# plt.loglog(NX_array, E_infp5, '.-', label=r'50% $|E_\infty|$')
# plt.minorticks_off()
# plt.xticks(NX_array, NX_array)
# plt.xlabel(r'$NX$')
# plt.ylabel(r'$|E_\infty|$', rotation=0, labelpad=10)
# plt.title('Perturbed Node Spacing')

# # plot the intra-step order of convergence
# axR2 = axR1.twinx()
# logN = np.log(NX_array)
# logE_infp1 = np.log(E_infp1)
# logE_infp5 = np.log(E_infp5)
# order_inf = (logE_infp1[0:-1] - logE_infp1[1:])/(logN[1:] - logN[0:-1])
# order_2 = (logE_infp5[0:-1] - logE_infp5[1:])/(logN[1:] - logN[0:-1])
# intraN = np.logspace(start+0.5, stop-0.5, num=stop-start, base=2.0)
# plt.plot(intraN, order_inf, '.:', linewidth=1, label=r'10% order')
# plt.plot(intraN, order_2, '.:', linewidth=1, label=r'50% order')
# plt.plot(plt.xlim(), [2, 2], 'k:', linewidth=1, label='Expected')
# plt.ylim(0, 5)
# plt.yticks(np.linspace(0,5,6))
# plt.ylabel(r'Intra-step Order of Convergence')
# lines, labels = axR1.get_legend_handles_labels()
# lines2, labels2 = axR2.get_legend_handles_labels()
# axR2.legend(lines + lines2, labels + labels2, loc='best')
# plt.margins(0,0)

# plt.savefig("L2_conv.pdf", bbox_inches = 'tight', pad_inches = 0)