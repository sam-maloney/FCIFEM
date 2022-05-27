# -*- coding: utf-8 -*-
"""
Created on Mon May 17 17:46:29 2021

@author: Samuel A. Maloney
"""

import numpy as np
import matplotlib.pyplot as plt

# For all of the below, unless otherwise noted
# Left and bottom borders and centre point constrained
# f(x,y) = sin(2pi*x)sin(2pi*y)
# Omega = (0,1) X (0,1)
# periodic BCs
# SinusoidalMapping(0.2, -0.25, 1.0)
# NY = NX, NQY = NY, NQX = 1, quadType = 'gauss', massLumping = False
NX = np.array([  4,   8,  16,  32,  64, 128, 256])


E_2_L = []
E_inf_L = []
t_setup_L = []
t_solve_L = []
labels_L = []

E_2_R = []
E_inf_R = []
t_setup_R = []
t_solve_R = []
labels_R = []

##### Standard integration #####
# Qord = 3

E_2_L.append(np.array([2.24286716e-01, 4.60214043e-02, 1.87935993e-02, 4.84572274e-03,
       1.17576373e-03, 3.02809766e-04, 7.59481226e-05]))
E_inf_L.append(np.array([4.48573433e-01, 1.07012181e-01, 4.44141856e-02, 1.11497373e-02,
       2.74568161e-03, 7.01529053e-04, 1.76669868e-04]))
t_setup_L.append(np.array([1.51476361e-02, 4.95175851e-02, 1.83443217e-01, 7.26344558e-01,
       2.74937745e+00, 1.10570457e+01, 4.43386749e+01]))
t_solve_L.append(np.array([4.78058937e-04, 8.60122964e-04, 1.79808296e-03, 4.07816202e-03,
       1.21440480e-02, 3.27576796e-01, 3.85609095e-01]))
labels_L.append('uniform')

E_2_L.append(np.array([2.06541260e-01, 5.21962940e-02, 1.52260626e-02, 4.51469788e-03,
       2.20271859e-03, 1.70464420e-03, 1.61290836e-03]))
E_inf_L.append(np.array([4.26536539e-01, 1.50667131e-01, 4.17292356e-02, 2.10290127e-02,
       1.18631488e-02, 7.75690221e-03, 4.79163657e-03]))
t_setup_L.append(np.array([1.41061100e-02, 5.04285559e-02, 1.89393934e-01, 7.37498919e-01,
       2.87805291e+00, 1.12869475e+01, 4.43945805e+01]))
t_solve_L.append(np.array([9.66881984e-04, 1.68331200e-03, 3.99692997e-03, 9.65317397e-03,
       2.50615481e-02, 1.44141595e-01, 9.14509741e-01]))
labels_L.append(r'10\% pert.')


##### VCI integration #####

E_2_R.append(np.array([1.90595923e-01, 5.21377809e-02, 1.36890773e-02, 3.54037313e-03,
       8.85166925e-04, 2.50577187e-04, 9.84975196e-05]))
E_inf_R.append(np.array([4.02166983e-01, 1.33612711e-01, 3.32178775e-02, 9.52987688e-03,
       2.79818406e-03, 1.49244249e-03, 7.65459181e-04]))
t_setup_R.append(np.array([1.21972977e-01, 4.93491575e-01, 1.97411507e+00, 7.92049338e+00,
       3.15406756e+01, 1.24353144e+02, 4.95655412e+02]))
t_solve_R.append(np.array([8.13680934e-04, 1.72615703e-03, 4.00521699e-03, 8.56902299e-03,
       2.51373600e-02, 1.28918873e-01, 6.55384905e-01]))
labels_R.append('Q10')

# E_2_R.append(np.array([2.05260518e-01, 5.10819359e-02, 1.32456999e-02, 3.08714591e-03,
#        2.14797081e-03, 2.44018172e-03, 2.53156610e-03]))
# E_inf_R.append(np.array([4.18435371e-01, 1.47342655e-01, 3.65556691e-02, 1.04120560e-02,
#        7.12587130e-03, 5.23476167e-03, 5.11442128e-03]))
# t_setup_R.append(np.array([1.51236879e-02, 5.29311310e-02, 1.96780006e-01, 7.76047459e-01,
#        2.98963896e+00, 1.18895803e+01, 4.71364832e+01]))
# t_solve_R.append(np.array([9.14165983e-04, 1.59140700e-03, 4.00817895e-03, 8.78759602e-03,
#        2.37668321e-02, 8.46175553e-01, 9.53274111e-01]))
# labels_R.append(r'Q3 VC1')

E_2_R.append(np.array([2.06146000e-01, 5.45164776e-02, 1.53307592e-02, 2.82187428e-03,
       7.24008829e-04, 6.84379791e-04, 8.12644053e-04]))
E_inf_R.append(np.array([4.31195421e-01, 1.54618550e-01, 4.24866287e-02, 9.75570087e-03,
       2.50746758e-03, 1.63339112e-03, 1.62358545e-03]))
t_setup_R.append(np.array([4.19849210e-02, 1.58836655e-01, 6.25481153e-01, 2.44788010e+00,
       9.80438881e+00, 3.93305919e+01, 1.56269683e+02]))
t_solve_R.append(np.array([7.15006958e-04, 1.95295701e-03, 3.98603699e-03, 9.36475908e-03,
       2.52552710e-02, 1.49755277e-01, 9.61586514e-01]))
labels_R.append(r'Q3 VC2')

# E_2_R.append(np.array([2.09907986e-01, 5.46742486e-02, 1.47418030e-02, 3.70483173e-03,
#        1.02687456e-03, 2.10687836e-04, 5.90265833e-05]))
# E_inf_R.append(np.array([4.57241765e-01, 1.58905612e-01, 4.21797726e-02, 1.44004039e-02,
#        3.55452072e-03, 8.41678939e-04, 2.22708221e-04]))
# t_setup_R.append(np.array([6.62901850e-02, 6.34597780e-02, 2.37282159e-01, 1.03900382e+00,
#        4.21067669e+00, 1.82577712e+01, 8.75533591e+01]))
# t_solve_R.append(np.array([1.32109004e-03, 1.66552595e-03, 4.07421007e-03, 8.94560793e-03,
#        2.54322201e-02, 1.44350352e-01, 1.02197056e+00]))
# labels_R.append(r'Q3 VC1-C')

# n.b. these timings use conda spyder install and are much slower/not comparable to others
# Using slice-by-slice VCI-C method
E_2_R.append(np.array([2.24730967e-01, 5.78038318e-02, 1.54399525e-02, 3.78086314e-03,
        1.03981626e-03, 2.68411373e-04, 6.98601365e-05]))
E_inf_R.append(np.array([5.18181108e-01, 1.70584387e-01, 3.88742527e-02, 1.11035762e-02,
        3.34560401e-03, 9.07360842e-04, 2.28819525e-04]))
t_setup_R.append(np.array([4.50561600e-02, 1.68772199e-01, 6.39779512e-01, 2.53059033e+00,
        1.00745427e+01, 4.04720636e+01, 1.62557906e+02]))
t_solve_R.append(np.array([7.62570999e-04, 1.73928900e-03, 3.85338700e-03, 8.39876299e-03,
        2.43427050e-02, 1.20427147e-01, 8.17088706e-01]))
labels_R.append(r'Q3 VC1-C')


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

if len(E_2_R) == 1:
    cycler = plt.cycler(color=[black], marker=['d'])
elif len(E_2_R) < 4: # 2 and 3
    cycler = plt.cycler(color=[blue, red, black], marker=['o', 's', 'd'])
else: # 4 or more
    cycler = plt.cycler(color=[blue, red, orange, black, green] + colors[4:],
        marker=['o', 's', '^', 'd', 'x', '+', 'v', '<', '>', '*', 'p'])

fig = plt.figure(figsize=(7.75, 3))
fig.subplots_adjust(hspace=0.5, wspace=0.5)

axL1, axR1 = fig.subplots(1, 2)
axL2 = axL1.twinx()
axR2 = axR1.twinx()
axL1.set_prop_cycle(cycler)
axL2.set_prop_cycle(cycler)
axR1.set_prop_cycle(cycler)
axR2.set_prop_cycle(cycler)

N = np.log2(NX**2).astype('int')
inds = (N >= 2)
intraN = 0.5 * (N[inds][:-1] + N[inds][1:])
logN = np.log(NX[inds])
Nmin = min(N)
Nmax = max(N)
Nstep = 2
for i, error in enumerate(E_2_L):
    axL1.semilogy(N[inds], error[inds], label=labels_L[i],
                  linewidth=solid_linewidth)

    logE = np.log(error[inds])
    order = (logE[:-1] - logE[1:])/(logN[1:] - logN[:-1])
    axL2.plot(intraN, order, linestyle=':', label=labels_L[i],
              linewidth=dashed_linewidth)
axL2.axhline(2, linestyle=':', color=black, label='Expected order', zorder=0,
             linewidth=dashed_linewidth)
# axL1.minorticks_off()
axL1.set_xlabel(r'$\log_2(N_xN_y)$')
axL1.set_ylabel(r'$|E_2|$', rotation=0, labelpad=10)
axL2.set_ylabel(r'Intra-step Order of Convergence')
axL1.legend(loc='lower left')
axL1.set_xticks(np.linspace(Nmin, Nmax, (Nmax - Nmin)//Nstep + 1))
ordb = 0
ordt = 2.5
ordstep = 0.5
axL2.set_ylim(ordb, ordt)
axL2.set_yticks(np.linspace(ordb, ordt, int((ordt - ordb)/ordstep) + 1))

for i, error in enumerate(E_2_R):
    axR1.semilogy(N[inds], error[inds], label=labels_R[i],
                  linewidth=solid_linewidth)

    logE = np.log(error[inds])
    order = (logE[:-1] - logE[1:])/(logN[1:] - logN[:-1])
    axR2.plot(intraN, order, linestyle=':', label=labels_R[i],
              linewidth=dashed_linewidth)
axR2.axhline(2, linestyle=':', color=black, label='Expected order', zorder=0,
             linewidth=dashed_linewidth)
# axR1.minorticks_off()
axR1.set_xlabel(r'$\log_2(N_xN_y)$')
axR1.set_ylabel(r'$|E_2|$', rotation=0, labelpad=10)
axR2.set_ylabel(r'Intra-step Order of Convergence')
axR1.legend(loc='lower left')
axR1.set_xticks(np.linspace(Nmin, Nmax, (Nmax - Nmin)//Nstep + 1))
axR2.set_ylim(ordb, ordt)
axL2.set_yticks(np.linspace(ordb, ordt, int((ordt - ordb)/ordstep) + 1))

ylimL = axL1.get_ylim()
ylimR = axR1.get_ylim()
axL1.set_ylim(min(ylimL[0],ylimR[0]), max(ylimL[1],ylimR[1]))
axR1.set_ylim(min(ylimL[0],ylimR[0]), max(ylimL[1],ylimR[1]))

# axL1.set_ylim(1e-6, 1e-2)
# axR1.set_ylim(1e-6, 1e-2)

# fig.savefig("Poisson_VCI_conv.pdf", bbox_inches = 'tight', pad_inches = 0)
