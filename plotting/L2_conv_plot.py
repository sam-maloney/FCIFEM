# -*- coding: utf-8 -*-
"""
Created on Mon May 17 17:46:29 2021

@author: samal
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

##### Doubly-Periodic BCs #####
# f(x,y) = sin(x)sin(2pi*y)
# sinusoidal mapping
NX_array = np.array([  4,   8,  16,  32,  64, 128, 256])
start = int(np.log2(NX_array[0]))
stop = int(np.log2(NX_array[-1]))
# NY = NX
# Nquad = 5

# Uniform spacing
E_2u = np.array([1.17817353e-01, 9.85576979e-02, 2.45978802e-02, 6.02352877e-03,
        1.49687962e-03, 3.73653269e-04, 9.33777137e-05])
E_infu = np.array([2.35634705e-01, 1.74694642e-01, 4.76311914e-02, 1.16460444e-02,
        2.89217058e-03, 7.21715652e-04, 1.80346973e-04])

# 0.1 perturbation
E_2p1 = np.array([1.34140716e-01, 1.02430887e-01, 2.53661338e-02, 6.15948128e-03,
        1.52570919e-03, 3.80520484e-04, 9.52952878e-05])
E_infp1 = np.array([2.42670205e-01, 1.84218885e-01, 5.24279763e-02, 1.34969241e-02,
        3.40146362e-03, 9.10205975e-04, 2.26928637e-04])

# 0.5 perturbation
E_2p5 = np.array([3.32304951e-01, 1.58513739e-01, 3.94981765e-02, 9.56499954e-03,
        2.23580423e-03, 5.46624171e-04, 1.40440403e-04])
E_infp5 = np.array([5.36106904e-01, 3.84477693e-01, 1.21872302e-01, 3.14200379e-02,
        7.67059713e-03, 2.10334214e-03, 5.30771969e-04])


##### Dirichlet BCs #####
# f(x,y) = x*sin(n(y - a*x^2 - b*x))
# n = 16, a = (1 - 2pi*b)/(2pi)^2, b = 0.05
# quadratic mapping
# NX_array = np.array([  4,   8,  16,  32,  64, 128, 256])
# NY = NX
# NDX = 1
# Qord = 2

# Uniform spacing
E_2Du = np.array([6.62029892e-01, 1.08962737e+00, 2.37259506e-01, 5.54249952e-02,
       1.32975974e-02, 3.25784140e-03, 8.06394720e-04])
E_infDu = np.array([1.28571509e+00, 3.28373583e+00, 9.16018219e-01, 2.77705915e-01,
       6.73241828e-02, 1.69879717e-02, 4.27389169e-03])

# 0.1 perturbation
E_2Dp1 = np.array([6.54334475e-01, 1.05508331e+00, 2.43934794e-01, 5.79118389e-02,
       1.38489205e-02, 3.39191323e-03, 8.41864925e-04])
E_infDp1 = np.array([1.26739623e+00, 3.03331083e+00, 8.00084991e-01, 2.90610896e-01,
       8.19225501e-02, 1.74032848e-02, 5.80826892e-03])

# 0.5 perturbation
E_2Dp5 = np.array([2.07743225e+00, 1.77603131e+00, 3.78782599e-01, 9.55259703e-02,
       2.29909443e-02, 5.71106468e-03, 1.39841124e-03])
E_infDp5 = np.array([3.88526713e+00, 5.73392763e+00, 1.12991635e+00, 5.40816765e-01,
       2.16686670e-01, 4.24504807e-02, 1.21522208e-02])


# NX_array = np.array([ 2,  4,  8, 16, 32, 64])
# NY = 16*NX
# NDX = 28
# Qord = 1

# Uniform spacing
E_2D16u = np.array([7.17688450e-02, 1.58587994e-02, 3.71731020e-03, 8.85208769e-04,
       2.15278502e-04, 5.30074055e-05])
E_infD16u = np.array([1.61561692e-01, 3.97243894e-02, 1.10150789e-02, 3.59195811e-03,
       7.16963378e-04, 1.57547376e-04])

# 0.1 perturbation
E_2D16p1 = np.array([7.76002544e-02, 1.62746838e-02, 3.80434589e-03, 9.10042413e-04,
       2.13957137e-04, 5.39539566e-05])
E_infD16p1 = np.array([1.59802520e-01, 4.53688113e-02, 1.21431025e-02, 3.56711723e-03,
       8.78431975e-04, 2.10000834e-04])

# 0.5 perturbation
E_2D16p5 = np.array([1.27309374e-01, 2.72171775e-02, 6.37504188e-03, 1.49426351e-03,
       3.63508163e-04, 8.69780998e-05])
E_infD16p5 = np.array([3.60513171e-01, 1.11728717e-01, 2.64270520e-02, 1.13455086e-02,
       2.80353131e-03, 6.13239932e-04])


##### Begin Plotting Routines #####

mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['markers.fillstyle'] = 'full'
mpl.rcParams['lines.markersize'] = 5.0
mpl.rcParams['lines.linewidth'] = 1.25
mpl.rcParams['pdf.fonttype'] = 42
# mpl.rc('font', **{'family': 'serif', 'serif': ['Latin Modern Roman']})
# mpl.rc('font', **{'family': 'serif', 'serif': ['Palatino']})
mpl.rcParams['text.usetex'] = True
colors = mpl.rcParams['axes.prop_cycle'].by_key()['color']
blue = colors[0]
orange = colors[1]
green = colors[2]
red = colors[3]
# next are: purple, brown, pink, grey, yellow, cyan

# clear the current figure, if opened, and set parameters
fig = plt.gcf()
fig.clf()
fig.set_size_inches(7.75,3)
plt.subplots_adjust(hspace = 0.5, wspace = 0.5)
# fig.set_size_inches(3.875,3)
# plt.subplots_adjust(left = 0.2, right = 0.85)

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

# plot the error convergence for periodic BCs
axL1 = plt.subplot(121)
N_array = np.log2(NX_array**2).astype('int')
plt.semilogy(N_array, E_2u, 'o-', color=blue, label=r'uniform')
# plt.semilogy(N_array, E_2p1, '.-', label=r'10\%')
plt.semilogy(N_array, E_2p5, 's-', color=red, label=r'50\% pert.')
plt.minorticks_off()
plt.xticks(N_array, N_array)
plt.xlabel(r'$\log_2(N_xN_y)$')
plt.ylabel(r'$|E_2|$', rotation=0, labelpad=10)
plt.title(r'Doubly-Periodic BCs')

# plot the intra-step order of convergence for periodic BCs
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
# intraN = np.logspace(start+0.5, stop-0.5, num=stop-start, base=2.0)
intraN = np.arange(2*start + 1, 2*stop, 2)
plt.autoscale(False)
plt.plot(plt.xlim(), [2, 2], ':k', linewidth=1, label='Expected')
# plt.plot(intraN, order_inf, '.:', linewidth=1, label=r'$E_\infty$ order')
plt.plot(intraN, order_2u, 'o:', linewidth=1, label=r'uniform order')
# plt.plot(intraN, order_2p1, '.:', linewidth=1, label=r'10\% order')
plt.plot(intraN, order_2p5, 's:', linewidth=1, color=red, label=r'50\% order')
plt.ylim(0, 4)
plt.yticks(np.linspace(0,4,5))
plt.ylabel(r'Intra-step Order of Convergence')
lines, labels = axL1.get_legend_handles_labels()
# lines2, labels2 = axL2.get_legend_handles_labels()
# axL2.legend(lines + lines2, labels + labels2, loc='best')
axL2.legend(lines, labels, loc='best')
# plt.margins(0,0)

# plot the error convergence for Dirichlet BCs
axR1 = plt.subplot(122)
plt.autoscale(True)
N_array = np.log2(NX_array**2).astype('int')
# N16_array = 16*np.array([ 2,  4,  8, 16, 32, 64])**2
N16_array = np.array([ 6,  8, 10, 12, 14, 16])
plt.semilogy(N_array, E_2Du, 'o-', color=blue, label=r'uniform 1:1')
# plt.semilogy(N_array, E_2Dp1, '.-', label=r'10\%')
plt.semilogy(N_array, E_2Dp5, 's-', color=red, label=r'50\% pert.~1:1')
mpl.rcParams['markers.fillstyle'] = 'none'
plt.semilogy(N16_array, E_2D16u, 'o-', color=blue, label=r'uniform 1:16')
# plt.semilogy(N16_array, E_2D16p1, '.-', label=r'10\%')
plt.semilogy(N16_array, E_2D16p5, 's-', color=red, label=r'50\% pert.~1:16')
mpl.rcParams['markers.fillstyle'] = 'full'
plt.minorticks_off()
plt.xticks(N_array, N_array)
plt.ylim(top = 5*plt.ylim()[1])
plt.xlabel(r'$\log_2(N_xN_y)$')
plt.ylabel(r'$|E_2|$', rotation=0, labelpad=10)
plt.title('Dirichlet BCs')

# plot the intra-step order of convergence for Dirichlet BCs
axR2 = axR1.twinx()
logN = np.log2(NX_array)
logN16 = np.log2(np.array([ 2,  4,  8, 16, 32, 64]))
logE_2Du = np.log2(E_2Du)
logE_2Dp1 = np.log2(E_2Dp1)
logE_2Dp5 = np.log2(E_2Dp5)
logE_2D16u = np.log2(E_2D16u)
logE_2D16p1 = np.log2(E_2D16p1)
logE_2D16p5 = np.log2(E_2D16p5)
order_2Du = (logE_2Du[0:-1] - logE_2Du[1:])/(logN[1:] - logN[0:-1])
order_2Dp1 = (logE_2Dp1[0:-1] - logE_2Dp1[1:])/(logN[1:] - logN[0:-1])
order_2Dp5 = (logE_2Dp5[0:-1] - logE_2Dp5[1:])/(logN[1:] - logN[0:-1])
order_2D16u = (logE_2D16u[0:-1] - logE_2D16u[1:])/(logN16[1:] - logN16[0:-1])
order_2D16p1 = (logE_2D16p1[0:-1] - logE_2D16p1[1:])/(logN16[1:] - logN16[0:-1])
order_2D16p5 = (logE_2D16p5[0:-1] - logE_2D16p5[1:])/(logN16[1:] - logN16[0:-1])
# intraN = np.logspace(start+0.5, stop-0.5, num=stop-start, base=2.0)**2
intraN = np.arange(2*start + 1, 2*stop, 2)
plt.plot(plt.xlim(), [2, 2], 'k:', linewidth=1, label='Expected')
plt.plot(intraN, order_2Du, 'o:', linewidth=1, color=blue, label=r'uniform order')
# plt.plot(intraN, order_2Dp1, '.:', linewidth=1, label=r'10% order')
plt.plot(intraN, order_2Dp5, 's:', linewidth=1, color=red, label=r'50% order')
mpl.rcParams['markers.fillstyle'] = 'none'
plt.plot(intraN[1:], order_2D16u, 'o:', linewidth=1, label=r'uniform order')
# plt.plot(intraN[1:], order_2D16p1, '.:', linewidth=1, label=r'10% order')
plt.plot(intraN[1:], order_2D16p5, 's:', linewidth=1, color=red, label=r'50% order')
mpl.rcParams['markers.fillstyle'] = 'full'
plt.ylim(0, 4)
plt.yticks(np.linspace(0,4,5))
plt.ylabel(r'Intra-step Order of Convergence')
lines, labels = axR1.get_legend_handles_labels()
axR2.legend(lines, labels, loc='best')
# lines2, labels2 = axR2.get_legend_handles_labels()
# axR2.legend(lines + lines2, labels + labels2, loc='best')
plt.margins(0,0)

# plt.savefig('L2_conv.pdf', bbox_inches = 'tight', pad_inches = 0)