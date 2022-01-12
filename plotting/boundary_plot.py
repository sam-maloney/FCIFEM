# -*- coding: utf-8 -*-
"""
Created on Mon May 17 17:46:29 2021

@author: samal
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# For all of the below, unless otherwise noted
NX_array = np.array([  2,   4,   8,  16,  32,  64, 128])
start = int(np.log2(NX_array[0]))
stop = int(np.log2(NX_array[-1]))

# # ##### DirichletBoundary
# # n = 20, a = 0.02, b = 0
# # NY = 20*NX, NDX = 32, Qord = 1
# E_2_D_p0_q1 = np.array([3.19128962e-02, 1.66174040e-02, 7.35715866e-03, 5.05400243e-03,
#         1.37505435e-03, 6.28625444e-04, 1.49246335e-04])
# E_inf_D_p0_q1 = np.array([6.01267696e-02, 3.91689608e-02, 2.59411522e-02, 2.42221312e-02,
#         8.78515282e-03, 6.51887251e-03, 3.40038087e-03])
# t_setup_D_p0_q1 = np.array([4.92162494e-01, 2.07696175e+00, 8.58752604e+00, 3.48628305e+01,
#         1.41986197e+02, 5.63175996e+02, 2.29591821e+03])
# t_solve_D_p0_q1 = np.array([6.79264893e-03, 2.44334240e-02, 4.95543720e-02, 1.81298157e-01,
#         1.14694606e+00, 8.82098613e+00, 6.88338451e+01])

# # ##### DirichletBoundary
# # n = 20, a = 0.02, b = 0
# # NY = 20*NX, NDX = 32, Qord = 2, px = py = 0
# # NX_array = np.array([ 2,  4,  8, 16, 32, 64])
# E_2_D_p0_q2 = np.array([3.20631362e-02, 1.38357481e-02, 4.66860478e-03, 2.56289445e-03,
#         9.61139994e-03, 1.37399231e-03])
# E_inf_D_p0_q2 = np.array([9.04214930e-02, 5.24540041e-02, 1.64607034e-02, 1.43771064e-02,
#         9.37708754e-02, 1.07042116e-02])
# t_setup_D_p0_q2 = np.array([1.93709404e+00, 8.34575381e+00, 3.42601427e+01, 1.41035787e+02,
#         5.65862695e+02, 2.32414412e+03])
# t_solve_D_p0_q2 = np.array([7.75411818e-03, 2.43537831e-02, 4.95607180e-02, 1.78673380e-01,
#         1.11589643e+00, 9.6233015e+00])

# ##### DirichletBoundary
# xmax = 2*np.pi, n = 2*np.pi, b = 0.05, a = (1 - b*xmax)/xmax**2
# NY = 6*NX, NQX = NDX = 11, Qord = 1, px = py = 0
E_2_D_n1_p0_q1 = np.array([8.40270876e-02, 3.40794125e-02, 1.36117243e-02, 5.56202646e-03,
        1.58087841e-03, 4.49679970e-04, 1.38338765e-04])
E_inf_D_n1_p0_q1 = np.array([1.71750701e-01, 8.09312973e-02, 5.17261148e-02, 2.18594446e-02,
        1.29669730e-02, 6.80857945e-03, 3.54536785e-03])
t_setup_D_n1_p0_q1 = np.array([7.64369000e-02, 3.38252900e-01, 1.60485300e+00, 5.59511730e+00,
        2.16920194e+01, 8.66802058e+01, 3.43453451e+02])
t_solve_D_n1_p0_q1 = np.array([1.42950000e-03, 1.35460000e-02, 2.68953000e-02, 6.57390000e-02,
        2.87634900e-01, 1.60251920e+00, 1.33176866e+01])

# ##### DirichletXPeriodicYBoundary
# xmax = 2*np.pi, n = 2*np.pi, b = 0.05, a = (1 - b*xmax)/xmax**2
# NY = 6*NX, NQX = NDX = 11, Qord = 1, px = py = 0
E_2_DXPY_n1_p0_q1 = np.array([4.44963164e-02, 1.24041757e-02, 6.71724548e-03, 3.68103559e-03,
        7.03944140e-04, 1.62280865e-04, 3.69943998e-05])
E_inf_DXPY_n1_p0_q1 = np.array([6.23595076e-02, 2.85233499e-02, 1.56946389e-02, 1.40071560e-02,
        2.33786277e-03, 5.42100080e-04, 1.23390303e-04])
t_setup_DXPY_n1_p0_q1 = np.array([6.65398000e-02, 2.46649600e-01, 1.07145090e+00, 4.08933360e+00,
        1.70970712e+01, 6.73706934e+01, 2.73834063e+02])
t_solve_DXPY_n1_p0_q1 = np.array([8.81799962e-04, 1.28149998e-03, 4.12850000e-03, 1.87049000e-02,
        9.95074000e-02, 1.16650010e+00, 9.00375790e+00])

# ##### DirichletXPeriodicYBoundary
# xmax = 2*np.pi, n = 2*2*np.pi, b = 0.05, a = (1 - b*xmax)/xmax**2
# NY = 12*NX, NQX = NDX = 21, Qord = 1, px = py = 0
# NX_array = np.array([  2,   4,   8,  16,  32,  64, 128])
E_2_DXPY_n2_p0_q1 = np.array([1.86588223e-02, 3.05002326e-02, 8.20772237e-03, 2.09824270e-03,
        1.08068762e-03, 2.42351627e-04, 5.99447534e-05])
E_inf_DXPY_n2_p0_q1 = np.array([2.63846173e-02, 5.06978962e-02, 2.27667358e-02, 5.64787150e-03,
        4.60911635e-03, 1.19323014e-03, 3.18186002e-04])
t_setup_DXPY_n2_p0_q1 = np.array([1.98775000e-01, 9.12010200e-01, 3.65064190e+00, 1.66881422e+01,
        6.86359620e+01, 2.58258212e+02, 9.96503545e+02])
t_solve_DXPY_n2_p0_q1 = np.array([1.50160003e-03, 1.19340001e-03, 5.79810003e-03, 2.95100000e-02,
        1.29468000e-01, 1.69622240e+00, 1.64209980e+01])

# ##### DirichletXPeriodicYBoundary
# xmax = 2*np.pi, n = 2*2*np.pi, b = 0.05, a = (1 - b*xmax)/xmax**2
# NY = 12*NX, NQX = NDX = 21, Qord = 1, px = py = 0.1
# NX_array = np.array([  2,   4,   8,  16,  32,  64, 128])
E_2_DXPY_n2_p1_q1 = np.array([6.07151990e-02, 3.19257550e-02, 1.79889363e-02, 8.95151686e-03,
       1.05009337e-01, 8.87828389e-03, 6.64412102e-02])
E_inf_DXPY_n2_p1_q1 = np.array([1.08287752e-01, 1.22302049e-01, 7.73200531e-02, 4.07025206e-02,
       8.02995556e-01, 9.11946225e-02, 5.79365054e-01])
t_setup_DXPY_n2_p1_q1 = np.array([1.99418800e-01, 9.80456400e-01, 3.97551440e+00, 1.58233391e+01,
       6.51082619e+01, 2.49638775e+02, 1.00581255e+03])
t_solve_DXPY_n2_p1_q1 = np.array([3.38469999e-03, 6.49430000e-02, 1.38907400e-01, 4.09027600e-01,
       1.55531940e+00, 1.72606012e+01, 1.45715785e+02])

# ##### DirichletXPeriodicYBoundary with VCI-C
# xmax = 2*np.pi, n = 2*2*np.pi, b = 0.05, a = (1 - b*xmax)/xmax**2
# NY = 12*NX, NQX = NDX = 21, Qord = 1, px = py = 0.1
# NX_array = np.array([  2,   4,   8,  16,  32,  64,  128])
E_2_DXPY_n2_p1_q1_VCI_C = np.array([3.65927318e+00, 3.11706802e-01, 2.11781068e-01, 5.77578695e-02,
       5.98882897e-03, 9.10699940e-04, 8.89662909e-04])
E_inf_DXPY_n2_p1_q1_VCI_C = np.array([3.71732987e+00, 6.04125720e-01, 3.86132420e-01, 1.00654121e-01,
       2.78458966e-02, 5.73213804e-03, 5.46466299e-03])
t_setup_DXPY_n2_p1_q1_VCI_C = np.array([4.37193700e-01, 1.70339670e+00, 7.51503080e+00, 3.39897945e+01,
       1.40137065e+02, 6.17224359e+02, 2.99862325e+03])
t_solve_DXPY_n2_p1_q1_VCI_C = np.array([3.2311000e-03, 8.1359900e-02, 1.7258050e-01, 4.9226420e-01,
       2.4544921e+00, 1.8101165e+01, 1.26678911e+02])

# ##### DirichletXPeriodicYBoundary
# xmax = 2*np.pi, n = 2*2*np.pi, b = 0.05, a = (1 - b*xmax)/xmax**2
# NY = 12*NX, NQX = NDX = 21, Qord = 2, px = py = 0.1
# NX_array = np.array([  2,   4,   8,  16,  32,  64])
E_2_DXPY_n2_p1_q2 = np.array([4.62734961e-02, 1.59876082e-02, 6.37444733e-03, 4.14246114e-03,
       1.50470528e-02, 1.63164576e-03, np.nan])
E_inf_DXPY_n2_p1_q2 = np.array([8.88986539e-02, 4.45647259e-02, 3.07720409e-02, 2.52250026e-02,
       1.24668247e-01, 1.26571724e-02, np.nan])
t_setup_DXPY_n2_p1_q2 = np.array([8.04077100e-01, 3.50453490e+00, 1.47012774e+01, 5.93644995e+01,
       2.41609634e+02, 9.91597784e+02, np.nan])
t_solve_DXPY_n2_p1_q2 = np.array([1.0633800e-02, 6.4964800e-02, 1.5490770e-01, 3.4264040e-01,
       1.8249460e+00, 1.1010249e+01, np.nan])


##### Begin Plotting Routines #####

# from matplotlib import rcParams
mpl.rcParams['pdf.fonttype'] = 42
mpl.rc("font", **{"family": "serif", "serif": ["Latin Modern Roman"]})
# mpl.rc("font", **{"family": "serif", "serif": ["Palatino"]})
# mpl.rc("text", usetex = True)

# clear the current figure, if opened, and set parameters
# fig = plt.gcf()
fig = plt.figure()
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
plt.loglog(NX_array, E_2_D_n1_p0_q1, '.-', label=r'D_n1_p0_q1')
plt.loglog(NX_array, E_2_DXPY_n1_p0_q1, '.-', label=r'DXPY_n1_p0_q1')
plt.loglog(NX_array, E_2_DXPY_n2_p0_q1, '.-', label=r'DXPY_n2_p0_q1')
plt.loglog(NX_array, E_2_DXPY_n2_p1_q1, '.-', label=r'DXPY_n2_p1_q1')
plt.loglog(NX_array, E_2_DXPY_n2_p1_q2, '.-', label=r'DXPY_n2_p1_q2')
plt.loglog(NX_array, E_2_DXPY_n2_p1_q1_VCI_C, '.-', label=r'DXPY_n2_p1_q1_VCI_C')
plt.minorticks_off()
# plt.ylim(top=1.5e-2)
plt.xticks(NX_array, NX_array)
plt.xlabel(r'$NX$')
plt.ylabel(r'$|E_2|$', rotation=0, labelpad=10)
# plt.title('Uniform vs. Perturbed Grid')
plt.legend()

# plot the intra-step order of convergence
# axL2 = axL1.twinx()
axL2 = plt.subplot(122)
logN = np.log(NX_array)
intraN = np.logspace(start+0.5, stop-0.5, num=stop-start, base=2.0)
logE_D_n1_p0_q1 = np.log(E_2_D_n1_p0_q1)
order_D_n1_p0_q1 = (logE_D_n1_p0_q1[0:-1] - logE_D_n1_p0_q1[1:])/(logN[1:] - logN[0:-1])
plt.plot(intraN, order_D_n1_p0_q1, '.:', linewidth=1, label=r'D_n1_p0_q1 order')
logE_DXPY_n1_p0_q1 = np.log(E_2_DXPY_n1_p0_q1)
order_DXPY_n1_p0_q1 = (logE_DXPY_n1_p0_q1[0:-1] - logE_DXPY_n1_p0_q1[1:])/(logN[1:] - logN[0:-1])
plt.plot(intraN, order_DXPY_n1_p0_q1, '.:', linewidth=1, label=r'DXPY_n1_p0_q1 order')
logE_DXPY_n2_p0_q1 = np.log(E_2_DXPY_n2_p0_q1)
order_DXPY_n2_p0_q1 = (logE_DXPY_n2_p0_q1[0:-1] - logE_DXPY_n2_p0_q1[1:])/(logN[1:] - logN[0:-1])
plt.plot(intraN, order_DXPY_n2_p0_q1, '.:', linewidth=1, label=r'DXPY_n2_p0_q1 order')
logE_DXPY_n2_p1_q1 = np.log(E_2_DXPY_n2_p1_q1)
order_DXPY_n2_p1_q1 = (logE_DXPY_n2_p1_q1[0:-1] - logE_DXPY_n2_p1_q1[1:])/(logN[1:] - logN[0:-1])
plt.plot(intraN, order_DXPY_n2_p1_q1, '.:', linewidth=1, label=r'DXPY_n2_p1_q1')
logE_DXPY_n2_p1_q2 = np.log(E_2_DXPY_n2_p1_q2)
order_DXPY_n2_p1_q2 = (logE_DXPY_n2_p1_q2[0:-1] - logE_DXPY_n2_p1_q2[1:])/(logN[1:] - logN[0:-1])
plt.plot(intraN, order_DXPY_n2_p1_q2, '.:', linewidth=1, label=r'DXPY_n2_p1_q2')
logE_DXPY_n2_p1_q1_VCI_C = np.log(E_2_DXPY_n2_p1_q1_VCI_C)
order_DXPY_n2_p1_q1_VCI_C = (logE_DXPY_n2_p1_q1_VCI_C[0:-1] - logE_DXPY_n2_p1_q1_VCI_C[1:])/(logN[1:] - logN[0:-1])
plt.plot(intraN, order_DXPY_n2_p1_q1_VCI_C, '.:', linewidth=1, label=r'DXPY_n2_p1_q1_VCI_C')

plt.plot(plt.xlim(), [2, 2], 'k:', linewidth=1, label='Expected ord')
ordb = 0
ordt = 4
plt.ylim(ordb, ordt)
plt.yticks(np.linspace(ordb, ordt, ordt - ordb + 1)) # unit spacing
# plt.yticks(np.linspace(ordb, ordt, int((ordt - ordb)*2 + 1))) # 0.5 spacing
plt.ylabel(r'Intra-step Order of Convergence')
lines, labels = axL1.get_legend_handles_labels()
lines2, labels2 = axL2.get_legend_handles_labels()
# leg = axL2.legend(lines, labels, loc='lower left')
# leg = axL2.legend(lines + lines2, labels + labels2, loc='best')
# leg = axL2.legend(lines + [lines2[-1]], labels + [labels2[-1]], loc='lower left')
plt.margins(0,0)

# # plot the error convergence
# axR1 = plt.subplot(122)
# # plt.loglog(NX_array, E_2_p1_sin, '.-', label=r'Q3')
# # plt.loglog(NX_array, E_2_p1_sin10, '.-', label=r'Q10')
# # plt.loglog(NX_array, E_2_p1_sin_L, '.-', label=r'Q3 VC1')
# # plt.loglog(NX_array, E_2_p1_sin_Q, '.-', label=r'Q3 VC2')
# # plt.loglog(NX_array, E_2_p1_sin_CN, '.-', label=r'Q3 VC1-C')
# plt.minorticks_off()
# plt.ylim(top=1.5e-2)
# plt.xticks(NX_array, NX_array)
# plt.xlabel(r'$NX$')
# plt.ylabel(r'$|E_2|$', rotation=0, labelpad=10)
# plt.title(r'Improved Quadrature')

# # plot the intra-step order of convergence
# axR2 = axR1.twinx()
# logN = np.log(NX_array)
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

# plt.savefig("boundary_conv.pdf", bbox_inches = 'tight', pad_inches = 0)