# -*- coding: utf-8 -*-
"""
Created on Mon May 17 17:46:29 2021

@author: Samuel A. Maloney
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

##### Old VCI-C #####

# # StraightMapping()
# # NQX = NDX = 1, Qord = 3
# E_2.append(np.array([4.06673103e+00, 5.87982631e-01, 1.81038252e-01, 1.72990902e-01,
#        3.52738228e-02, 5.80219359e-03, 1.25770637e-03, 3.33958611e-04]))
# E_inf.append(np.array([4.06673103e+00, 1.27645289e+00, 5.48873588e-01, 5.24559361e-01,
#        1.41505101e-01, 3.49764170e-02, 8.78641178e-03, 2.06240798e-03]))
# t_setup.append(np.array([1.47075669e-02, 5.07215180e-02, 1.98696069e-01, 7.83551860e-01,
#        3.19821026e+00, 1.29745868e+01, 5.22686873e+01, 2.11342415e+02]))
# t_solve.append(np.array([4.18831129e-04, 7.01643992e-04, 1.49277295e-03, 2.86228815e-03,
#        7.44230812e-03, 2.11537471e-02, 1.17465902e-01, 6.86309253e-01]))
# labels.append('unaligned 1:1')
# NX.append(np.array([  2,   4,   8,  16,  32,  64, 128, 256]))
# NY.append(1)

# # QuadraticMapping(0.95, 0.05)
# # NQX = NDX = 1, Qord = 3
# E_2.append(np.array([2.75177924e-02, 4.66436584e-01, 1.95301545e-01, 7.54000679e-02,
#        1.13931278e-02, 3.80245215e-03, 9.17835917e-04, 2.95894260e-04]))
# E_inf.append(np.array([2.75177924e-02, 9.07245700e-01, 5.70754468e-01, 2.17962142e-01,
#        3.73445243e-02, 1.68892865e-02, 5.76069809e-03, 2.00070348e-03]))
# t_setup.append(np.array([1.57121969e-02, 5.50243820e-02, 2.21912339e-01, 8.73713755e-01,
#        3.48041777e+00, 1.43380649e+01, 5.71902653e+01, 2.32448775e+02]))
# t_solve.append(np.array([3.90805071e-04, 7.43536977e-04, 2.13289703e-03, 6.59624394e-03,
#        1.71172130e-02, 4.74065030e-02, 2.69226695e-01, 1.95620522e+00]))
# labels.append('aligned 1:1')
# NX.append(np.array([  2,   4,   8,  16,  32,  64, 128, 256]))
# NY.append(1)

# # QuadraticMapping(0.95, 0.05)
# # NQX = 32, NDX = 0, Qord = 2
# # NX_array = np.array([ 2,  4,  8, 16, 32])
# E_2.append(np.array([1.76617847e-02, 4.79615988e-03, 1.24278636e-03, 3.11981261e-04,
#         3.63605827e-04]))
# E_inf.append(np.array([2.92235110e-02, 1.77732723e-02, 3.56512101e-03, 1.07145565e-03,
#         2.45068635e-03]))
# t_setup.append(np.array([3.35879835e+00, 1.29466946e+01, 4.96347048e+01, 1.99090640e+02,
#         1.12483076e+03]))
# t_solve.append(np.array([3.52571113e-03, 1.94021410e-02, 4.13835919e-02, 1.16801437e-01,
#         1.22739806e+00]))
# labels.append('aligned 1:16')
# NX.append(np.array([ 2,  4,  8, 16, 32]))
# NY.append(16)


##### New VCI-C #####

# # ssqr
# # StraightMapping()
# # NQX = NDX = 1, Qord = 3
# E_2.append(np.array([3.81366039e+00, 7.46629345e-01, 1.90601415e-01, 1.48045476e-01,
#         2.12837969e-02, 4.33957685e-03, 1.05783794e-03, 2.62457496e-04]))
# E_inf.append(np.array([3.81366039e+00, 1.24967795e+00, 6.76226261e-01, 4.57574560e-01,
#         1.07231639e-01, 2.62044413e-02, 5.61711410e-03, 1.44143779e-03]))
# t_setup.append(np.array([1.64840640e-02, 5.73141780e-02, 2.32370672e-01, 9.28266249e-01,
#         3.69077779e+00, 1.49703802e+01, 5.93579917e+01, 2.38802199e+02]))
# t_solve.append(np.array([3.87747001e-04, 6.76597992e-04, 1.48744301e-03, 3.03357700e-03,
#         7.42373800e-03, 1.95542670e-02, 1.17016165e-01, 7.32202329e-01]))
# labels.append('unaligned 1:1')
# NX.append(np.array([  2,   4,   8,  16,  32,  64, 128, 256]))
# NY.append(1)

# scipy.lsqr (no diagonal preconditioning)
# StraightMapping()
# NQX = NDX = 1, Qord = 3
E_2.append(np.array([4.16885445e+00, 2.22473608e-01, 2.14675721e-01, 1.54902510e-01,
       2.19827629e-02, 4.53306502e-03, 1.09838952e-03, 2.72538762e-04]))
E_inf.append(np.array([4.16885445e+00, 4.57469501e-01, 5.30385138e-01, 4.63156750e-01,
       1.08819627e-01, 2.60713352e-02, 5.45519528e-03, 1.44733311e-03]))
t_setup.append(np.array([1.46479460e-02, 5.23102210e-02, 2.07176055e-01, 8.34646109e-01,
       3.38998371e+00, 1.39655117e+01, 6.29699045e+01, 3.14762068e+02]))
t_solve.append(np.array([4.26808008e-04, 7.18028998e-04, 1.45784301e-03, 2.87212900e-03,
       7.01781501e-03, 1.95289790e-02, 9.56007880e-02, 9.91174069e-01]))
labels.append('unaligned 1:1')
NX.append(np.array([  2,   4,   8,  16,  32,  64, 128, 256]))
NY.append(1)

# scipy.lsqr (no diagonal preconditioning)
# QuadraticMapping(0.95, 0.05)
# NQX = NDX = 1, Qord = 3
E_2.append(np.array([2.83330382e-01, 3.83807315e-01, 1.86721611e-01, 3.15730146e-02,
        7.10138237e-03, 2.00821396e-03, 5.32566849e-04, 1.46033121e-04]))
E_inf.append(np.array([2.83330382e-01, 8.14362311e-01, 4.14945003e-01, 9.95680826e-02,
        4.01769321e-02, 1.27785489e-02, 4.94996104e-03, 1.59393790e-03]))
t_setup.append(np.array([1.52560570e-02, 5.85185960e-02, 2.37768943e-01, 9.64615982e-01,
        3.98198445e+00, 1.70124730e+01, 8.64449543e+01, 5.32962180e+02]))
t_solve.append(np.array([4.74400003e-04, 6.46193002e-04, 2.11497801e-03, 6.02916599e-03,
        1.30642260e-02, 4.36630650e-02, 2.47447900e-01, 1.94757541e+00]))
labels.append('aligned 1:1')
NX.append(np.array([  2,   4,   8,  16,  32,  64, 128, 256]))
NY.append(1)

# scipy.lsqr (no diagonal preconditioning)
# QuadraticMapping(0.95, 0.05)
# NQX = 32, NDX = 0, Qord = 2
# NX_array = np.array([ 2,  4,  8, 16, 32])
E_2.append(np.array([1.73029870e-02, 3.06178064e-03, 1.34694047e-03, 2.30507442e-04,
        1.5968434e-02]))
E_inf.append(np.array([2.68872602e-02, 1.37276762e-02, 3.17286845e-03, 9.78016511e-04,
        6.76976813e-01]))
t_setup.append(np.array([3.13443866e+00, 1.58049735e+01, 9.47831106e+01, 6.48815792e+02,
        4.52048285e+03]))
t_solve.append(np.array([3.39838200e-03, 1.87457300e-02, 4.23197700e-02, 1.16615102e-01,
        9.61501099e+00]))
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
    inds.append(N[i] >= 4)
    axL1.semilogy(N[i][inds[i]], error[inds[i]]/(2*np.pi), label=labels[i],
                  linewidth=solid_linewidth)
# axL1.minorticks_off()
Nmin = min([min(N[i][inds[i]]) for i in range(len(N))])
Nmax = max([max(N[i][inds[i]]) for i in range(len(N))])
Nstep = 2
axL1.set_xticks(np.linspace(Nmin, Nmax, (Nmax - Nmin)//Nstep + 1))
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
axR1.set_xticks(np.linspace(Nmin, Nmax, (Nmax - Nmin)//Nstep + 1))
axR1.set_xlim(xlim)
axR1.set_xlabel(r'$\log_2(N_xN_y)$')
axR1.set_ylabel(r'Intra-step Order of Convergence')
ordb = 0
ordt = 3
ordstep = 0.5
axR1.set_ylim(ordb, ordt)
axR1.set_yticks(np.linspace(ordb, ordt, int((ordt - ordb)/ordstep) + 1))
lines, labels = axR1.get_legend_handles_labels()
axR1.legend(lines[1:], labels[1:], loc='lower right')

# fig.savefig("boundary_conv.pdf", bbox_inches = 'tight', pad_inches = 0)