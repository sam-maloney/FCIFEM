# -*- coding: utf-8 -*-
"""
Created on Mon May 17 17:46:29 2021

@author: samal
"""

import numpy as np
import matplotlib.pyplot as plt

# For all of the below, unless otherwise noted
# Left and bottom borders and centre point constrained
# u(x,y) = x*sin(2pi*n*[y - a*x^2 - b*x])
# n = 3, a = 0.95, b = 0.05
# Omega = (0,1) X (0,1)
# DirichletBoundary with VCI-C
# px = py = 0.1
# NQY = NY, quadType = 'gauss', massLumping = False

E_2 = []
E_inf = []
t_setup = []
t_solve = []
labels = []
NX = []
NY = []

# StraightMapping()
# NQX = NDX = 1, Qord = 3
E_2.append(np.array([4.06673103e+00, 5.87982631e-01, 1.81038252e-01, 1.72990902e-01,
       3.52738228e-02, 5.80219359e-03, 1.25770637e-03, 3.33958611e-04]))
E_inf.append(np.array([4.06673103e+00, 1.27645289e+00, 5.48873588e-01, 5.24559361e-01,
       1.41505101e-01, 3.49764170e-02, 8.78641178e-03, 2.06240798e-03]))
t_setup.append(np.array([1.47075669e-02, 5.07215180e-02, 1.98696069e-01, 7.83551860e-01,
       3.19821026e+00, 1.29745868e+01, 5.22686873e+01, 2.11342415e+02]))
t_solve.append(np.array([4.18831129e-04, 7.01643992e-04, 1.49277295e-03, 2.86228815e-03,
       7.44230812e-03, 2.11537471e-02, 1.17465902e-01, 6.86309253e-01]))
labels.append('unaligned 1:1')
NX.append(np.array([  2,   4,   8,  16,  32,  64, 128, 256]))
NY.append(1)

# QuadraticMapping(0.95, 0.05)
# NQX = NDX = 1, Qord = 3
E_2.append(np.array([2.75177924e-02, 4.66436584e-01, 1.95301545e-01, 7.54000679e-02,
       1.13931278e-02, 3.80245215e-03, 9.17835917e-04, 2.95894260e-04]))
E_inf.append(np.array([2.75177924e-02, 9.07245700e-01, 5.70754468e-01, 2.17962142e-01,
       3.73445243e-02, 1.68892865e-02, 5.76069809e-03, 2.00070348e-03]))
t_setup.append(np.array([1.57121969e-02, 5.50243820e-02, 2.21912339e-01, 8.73713755e-01,
       3.48041777e+00, 1.43380649e+01, 5.71902653e+01, 2.32448775e+02]))
t_solve.append(np.array([3.90805071e-04, 7.43536977e-04, 2.13289703e-03, 6.59624394e-03,
       1.71172130e-02, 4.74065030e-02, 2.69226695e-01, 1.95620522e+00]))
labels.append('aligned 1:1')
NX.append(np.array([  2,   4,   8,  16,  32,  64, 128, 256]))
NY.append(1)

# QuadraticMapping(0.95, 0.05)
# NQX = 32, NDX = 0, Qord = 2
# NX_array = np.array([ 2,  4,  8, 16, 32])
E_2.append(np.array([1.76617847e-02, 4.79615988e-03, 1.24278636e-03, 3.11981261e-04,
       3.63605827e-04]))
E_inf.append(np.array([2.92235110e-02, 1.77732723e-02, 3.56512101e-03, 1.07145565e-03,
       2.45068635e-03]))
t_setup.append(np.array([3.35879835e+00, 1.29466946e+01, 4.96347048e+01, 1.99090640e+02,
       1.12483076e+03]))
t_solve.append(np.array([3.52571113e-03, 1.94021410e-02, 4.13835919e-02, 1.16801437e-01,
       1.22739806e+00]))
labels.append('aligned 1:16')
NX.append(np.array([ 2,  4,  8, 16, 32]))
NY.append(16)


##### Begin Plotting Routines #####

solid_linewidth = 1.25
dashed_linewidth = 1.0

plt.rc('markers', fillstyle='full')
plt.rc('lines', markersize=5.0)
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

if len(E_2) == 1:
    cycler = plt.cycler(color=[black], marker=['d'])
elif len(E_2) < 4: # 2 and 3
    cycler = plt.cycler(color=[blue, red, black], marker=['o', 's', 'd'])
else: # 4 or more
    cycler = plt.cycler(color=[blue, red, orange, black, green] + colors[4:],
        marker=['o', 's', '^', 'd', 'x', '+', 'v', '<', '>', '*', 'p'])
    
fig = plt.figure(figsize=(7.75, 3))
fig.subplots_adjust(hspace=0.5, wspace=0.5)

axL1, axR1 = fig.subplots(1, 2)
axL1.set_prop_cycle(cycler)
axR1.set_prop_cycle(cycler)
N = []
inds = []
for i, error in enumerate(E_2):
    N.append(np.log2(NY[i]*NX[i]**2).astype('int'))
    inds.append(N[i] >= 2)
    axL1.semilogy(N[i][inds[i]], error[inds[i]]/(2*np.pi), label=labels[i],
                  linewidth=solid_linewidth)
axL1.minorticks_off()
Nmin = min([min(N[i]) for i in range(len(N))])
Nmax = max([max(N[i]) for i in range(len(N))])
Nstep = 2
axL1.set_xticks(np.linspace(Nmin, Nmax, (Nmax - Nmin)/Nstep + 1))
axL1.set_xlabel(r'$\log_2(N_xN_y)$')
axL1.set_ylabel(r'$|E_2|$', rotation=0, labelpad=10)
axL1.legend(loc='lower left')
xlim = axL1.get_xlim()

axR1.axhline(2, linestyle=':', color=black, label='Expected order',
             linewidth=dashed_linewidth)
for i, error in enumerate(E_2):
    logE = np.log(error[inds[i]])
    logN = np.log(NX[i][inds[i]])
    order = (logE[:-1] - logE[1:])/(logN[1:] - logN[:-1])
    intraN = 0.5 * (N[i][inds[i]][:-1] + N[i][inds[i]][1:])
    axR1.plot(intraN, order, linestyle=':', label=labels[i],
              linewidth=dashed_linewidth)
axL1.set_xticks(np.linspace(Nmin, Nmax, (Nmax - Nmin)/Nstep + 1))
axR1.set_xlim(xlim)
axR1.set_xlabel(r'$\log_2(N_xN_y)$')
axR1.set_ylabel(r'Intra-step Order of Convergence')
ordb = 0
ordt = 3
ordstep = 0.5
axR1.set_ylim(ordb, ordt)
axR1.set_yticks(np.linspace(ordb, ordt, (ordt - ordb)/ordstep + 1))
lines, labels = axR1.get_legend_handles_labels()
axR1.legend(lines[1:], labels[1:], loc='lower right')

# fig.savefig("boundary_conv.pdf", bbox_inches = 'tight', pad_inches = 0)