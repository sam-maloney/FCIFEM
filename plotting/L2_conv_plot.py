# -*- coding: utf-8 -*-
"""
Created on Mon May 17 17:46:29 2021

@author: samal
"""

import numpy as np
import matplotlib.pyplot as plt

E_2_L = []
E_inf_L = []
labels_L = []
NX_L = []
NY_L = []

E_2_R = []
E_inf_R = []
labels_R = []
NX_R = []
NY_R = []

##### Doubly-Periodic BCs #####
# f(x,y) = sin(2pi*x)sin(2pi*y)
# Omega = (0,1) X (0,1)
# SinusoidalMapping(0.2, -0.25, 1.0)
# NQX = NDX = 1
# Qord = 2

E_2_L.append(np.array([1.30981434e-01, 9.78312351e-02, 2.40098080e-02, 5.85955061e-03,
       1.45497063e-03, 3.63129526e-04, 9.07433908e-05]))
E_inf_L.append(np.array([2.61962869e-01, 1.77564364e-01, 4.69223848e-02, 1.13783076e-02,
       2.82079514e-03, 7.03690530e-04, 1.75827983e-04]))
labels_L.append('uniform')
NX_L.append(np.array([  4,   8,  16,  32,  64, 128, 256]))
NY_L.append(1)

# E_2_L.append(np.array([1.47894564e-01, 1.03151297e-01, 2.49435881e-02, 6.04319111e-03,
#        1.49513129e-03, 3.72760403e-04, 9.33709635e-05]))
# E_inf_L.append(np.array([2.80911002e-01, 1.88679996e-01, 5.31156095e-02, 1.33446321e-02,
#        3.46509508e-03, 8.92803465e-04, 2.29498729e-04]))
# labels_L.append(r'10\% pert.')
# NX_L.append(np.array([  4,   8,  16,  32,  64, 128, 256]))
# NY_L.append(1)

E_2_L.append(np.array([3.78624292e-01, 1.62348878e-01, 3.97497535e-02, 9.58373158e-03,
       2.24033007e-03, 5.47961210e-04, 1.40723140e-04]))
E_inf_L.append(np.array([7.82913593e-01, 3.75742158e-01, 1.26069901e-01, 3.47098278e-02,
       8.34303714e-03, 2.10856595e-03, 5.40784069e-04]))
labels_L.append(r'50\% pert.')
NX_L.append(np.array([  4,   8,  16,  32,  64, 128, 256]))
NY_L.append(1)


##### Dirichlet BCs #####
# f(x,y) = x*sin(2pi*n(y - a*x^2 - b*x))
# Omega = (0,1) X (0,1)
# n = 3, a = (1 - xmax*b)/xmax^2 = 0.95, b = 0.05
# QuadraticMapping(0.95, 0.05)
# NQX = NDX = 1
# Qord = 2

E_2_R.append(np.array([3.08346777e-01, 2.32430829e-01, 5.88064927e-02, 1.36571113e-02,
       3.22287817e-03, 7.82371947e-04, 1.92750314e-04]))
E_inf_R.append(np.array([6.14369030e-01, 7.48136301e-01, 2.88660465e-01, 9.31112471e-02,
       2.05478114e-02, 5.18048007e-03, 1.29888727e-03]))
labels_R.append('uniform 1:1')
NX_R.append(np.array([  4,   8,  16,  32,  64, 128, 256]))
NY_R.append(1)

# E_2_R.append(np.array([2.66207146e-01, 2.47793003e-01, 5.85928829e-02, 1.39529055e-02,
#        3.29927584e-03, 7.99118679e-04, 1.97558925e-04]))
# E_inf_R.append(np.array([5.71516210e-01, 7.41036645e-01, 2.86612019e-01, 9.41934134e-02,
#        2.31979770e-02, 5.91087548e-03, 1.51081290e-03]))
# labels_R.append(r'10\% pert. 1:1')
# NX_R.append(np.array([  4,   8,  16,  32,  64, 128, 256]))
# NY_R.append(1)

E_2_R.append(np.array([3.90672230e-01, 3.20336814e-01, 8.86465588e-02, 2.16625267e-02,
       5.11564986e-03, 1.27730929e-03, 3.12497767e-04]))
E_inf_R.append(np.array([9.00368168e-01, 8.18835227e-01, 5.21530746e-01, 1.08311258e-01,
       4.52095486e-02, 1.75038041e-02, 5.69974895e-03]))
labels_R.append(r'50\% pert. 1:1')
NX_R.append(np.array([  4,   8,  16,  32,  64, 128, 256]))
NY_R.append(1)


# NQX = NDX = 32
# Qord = 1

E_2_R.append(np.array([1.57794426e-02, 3.55243033e-03, 8.27219595e-04, 1.95167634e-04,
       4.76499550e-05, 1.17258530e-05]))
E_inf_R.append(np.array([3.59693799e-02, 1.03625819e-02, 3.00425880e-03, 6.13416709e-04,
       1.91138084e-04, 3.81229784e-05]))
labels_R.append('uniform 1:16')
NX_R.append(np.array([ 2,  4,  8, 16, 32, 64]))
NY_R.append(16)

# E_2_R.append(np.array([1.70188283e-02, 3.68581977e-03, 8.45433113e-04, 1.99397928e-04,
#        4.75289560e-05, 1.19705055e-05]))
# E_inf_R.append(np.array([3.58748946e-02, 1.34342268e-02, 2.69182451e-03, 7.22245330e-04,
#        2.06401604e-04, 4.55779360e-05]))
# labels_R.append(r'10\% pert. 1:16')
# NX_R.append(np.array([ 2,  4,  8, 16, 32, 64]))
# NY_R.append(16)

E_2_R.append(np.array([2.45711754e-02, 6.03620664e-03, 1.37030952e-03, 3.32210140e-04,
       8.04840556e-05, 1.91223603e-05]))
E_inf_R.append(np.array([5.10870829e-02, 2.38641372e-02, 5.85909422e-03, 1.97354486e-03,
       5.41506570e-04, 1.31599676e-04]))
labels_R.append(r'50\% pert. 1:16')
NX_R.append(np.array([ 2,  4,  8, 16, 32, 64]))
NY_R.append(16)


##### Begin Plotting Routines #####

solid_linewidth = 1.25
dashed_linewidth = 1.0

plt.rc('markers', fillstyle='full')
plt.rc('lines', markersize=5.0, linewidth=solid_linewidth)
plt.rc('pdf', fonttype=42)
plt.rc('text', usetex=True)
# fontsize : int or {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}
# plt.rc('font', size='small')
plt.rc('legend', fontsize='small')
# plt.rc('axes', titlesize='medium', labelsize='medium')
# plt.rc('xtick', labelsize='small')
# plt.rc('ytick', labelsize='small')
# plt.rc('figure', titlesize='large')

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
blue = '#1f77b4'
orange = '#ff7f0e'
green = '#2ca02c'
red = '#d62728'
purple = '#9467bd'
brown = '#8c564b'
pink = '#e377c2'
grey = '#7f7f7f'
yellow = '#bcbd22'
cyan = '#17becf'
black = '#000000'

if len(E_2_L) == 1:
    cycler = plt.cycler(color=[black], marker=['d'])
elif len(E_2_L) == 2:
    cycler = plt.cycler(color=[blue, red], marker=['o', 's'])
else:
    cycler = plt.cycler(color=[blue, red, black], marker=['o', 's', 'd'])

fig = plt.figure(figsize=(7.75, 3))
fig.subplots_adjust(hspace=0.5, wspace=0.5)

axL1, axR1 = fig.subplots(1, 2)
axL2 = axL1.twinx()
axR2 = axR1.twinx()
axL1.set_prop_cycle(cycler)
axL2.set_prop_cycle(cycler)
axR1.set_prop_cycle(cycler)
axR2.set_prop_cycle(cycler)

N = []
inds = []
for i, error in enumerate(E_2_L):
    N.append(np.log2(NY_L[i]*NX_L[i]**2).astype('int'))
    inds.append(N[i] >= 2)
    axL1.semilogy(N[i][inds[i]], error[inds[i]], label=labels_L[i],
                  linewidth=solid_linewidth)
    
    logE = np.log(error[inds[i]])
    logN = np.log(NX_L[i][inds[i]])
    order = (logE[:-1] - logE[1:])/(logN[1:] - logN[:-1])
    intraN = 0.5 * (N[i][inds[i]][:-1] + N[i][inds[i]][1:])
    axL2.plot(intraN, order, linestyle=':', label=labels_L[i],
              linewidth=dashed_linewidth)
axL2.axhline(2, linestyle=':', color=black, label='Expected order', zorder=0,
             linewidth=dashed_linewidth)
axL1.minorticks_off()
axL1.set_xlabel(r'$\log_2(N_xN_y)$')
axL1.set_ylabel(r'$|E_2|$', rotation=0, labelpad=10)
axL2.set_ylabel(r'Intra-step Order of Convergence')
axL1.legend()
Nmin = min([min(N[i]) for i in range(len(N))])
Nmax = max([max(N[i]) for i in range(len(N))])
Nstep = 2
axL1.set_xticks(np.linspace(Nmin, Nmax, (Nmax - Nmin)/Nstep + 1))
ordb = 0
ordt = 4
ordstep = 1
axL2.set_ylim(ordb, ordt)
axL2.set_yticks(np.linspace(ordb, ordt, (ordt - ordb)/ordstep + 1))

N = []
inds = []
for i, error in enumerate(E_2_R):
    N.append(np.log2(NY_R[i]*NX_R[i]**2).astype('int'))
    inds.append(N[i] >= 2)
    if i < len(E_2_R)/2:
        fillstyle = 'full'
    else:
        fillstyle = 'none'
    axR1.semilogy(N[i][inds[i]], error[inds[i]], label=labels_R[i],
                  linewidth=solid_linewidth, fillstyle=fillstyle)
    
    logE = np.log(error[inds[i]])
    logN = np.log(NX_R[i][inds[i]])
    order = (logE[:-1] - logE[1:])/(logN[1:] - logN[:-1])
    intraN = 0.5 * (N[i][inds[i]][:-1] + N[i][inds[i]][1:])
    axR2.plot(intraN, order, linestyle=':', label=labels_R[i],
              linewidth=dashed_linewidth, fillstyle=fillstyle)
axR2.axhline(2, linestyle=':', color=black, label='Expected order', zorder=0,
             linewidth=dashed_linewidth)
axR1.minorticks_off()
axR1.set_xlabel(r'$\log_2(N_xN_y)$')
axR1.set_ylabel(r'$|E_2|$', rotation=0, labelpad=10)
axR2.set_ylabel(r'Intra-step Order of Convergence')
axR1.legend(loc='upper right')
Nmin = min([min(N[i]) for i in range(len(N))])
Nmax = max([max(N[i]) for i in range(len(N))])
Nstep = 2
axR1.set_xticks(np.linspace(Nmin, Nmax, (Nmax - Nmin)/Nstep + 1))
axR2.set_ylim(ordb, ordt)
axR2.set_yticks(np.linspace(ordb, ordt, (ordt - ordb)/ordstep + 1))
axR1.set_ylim(top=2)

# fig.savefig('L2_conv.pdf', bbox_inches='tight', pad_inches=0)
