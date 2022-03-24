# -*- coding: utf-8 -*-
"""
Created on Mon May 17 17:46:29 2021

@author: Samuel A. Maloney
"""

import numpy as np
import matplotlib.pyplot as plt

# For all of the below, unless otherwise noted
# Left and bottom borders and centre point constrained
# f(x,y) = 0.5*sin(n*(2pi*y - 2pi*x))*(1 + sin(2pi*y))
# n = 8
# Omega = (0,1) X (0,1)
# Periodic BCs with slice VCI-C
# NQX = 1, NQY = NY, Qord = 3, quadType = 'gauss', massLumping = False

E_2_L = []
E_inf_L = []
t_setup_L = []
t_solve_L = []
labels_L = []
NX_L = []
NY_L = []

E_2_R = []
E_inf_R = []
t_setup_R = []
t_solve_R = []
labels_R = []
NX_R = []
NY_R = []

##### Uniform grid spacing #####

# StraightMapping()
E_2_L.append(np.array([2.73317422e-01, 1.07220109e-02, 3.66400910e-02, 8.59797844e-02,
        2.08355376e-02, 5.13614469e-03, 1.27914136e-03]))
E_inf_L.append(np.array([7.25262020e-01, 2.38691930e-02, 8.44773077e-02, 2.05546489e-01,
        5.17351880e-02, 1.28034753e-02, 3.19152003e-03]))
t_setup_L.append(np.array([4.43934550e-02, 1.56605728e-01, 6.14636523e-01, 2.44001645e+00,
        9.62582355e+00, 3.97420921e+01, 1.57916810e+02]))
t_solve_L.append(np.array([5.55408013e-04, 8.35318002e-04, 1.38938800e-03, 3.52759799e-03,
        9.27456701e-03, 4.41565860e-02, 4.09790143e-01]))
labels_L.append('unaligned 1:1')
NX_L.append(np.array([  4,   8,  16,  32,  64, 128, 256]))
NY_L.append(1)

# LinearMapping(1.0)
E_2_L.append(np.array([2.94343115e-01, 1.86838701e+00, 7.31461781e-02, 1.50512314e-02,
        4.17501343e-03, 1.11750114e-03, 5.27058899e-04]))
E_inf_L.append(np.array([7.59346093e-01, 5.28205981e+00, 2.99162921e-01, 5.35721252e-02,
        1.88204639e-02, 8.76689662e-03, 2.66542336e-03]))
t_setup_L.append(np.array([4.37268890e-02, 1.70551610e-01, 6.84369751e-01, 2.79177382e+00,
        1.18696000e+01, 6.72806054e+01, 4.91191352e+02]))
t_solve_L.append(np.array([5.51581965e-04, 1.60644599e-03, 3.03670502e-03, 6.14116300e-03,
        1.56168690e-02, 7.66609090e-02, 6.03772398e-01]))
labels_L.append('aligned 1:1')
NX_L.append(np.array([  4,   8,  16,  32,  64, 128, 256]))
NY_L.append(1)

# # LinearMapping(1.0)
# E_2_L.append(np.array([3.66295066e-02, 7.11117489e-02, 1.63786688e-02, 3.63824348e-03,
#         8.93386567e-04, 2.21801641e-04, 5.52309137e-05]))
# E_inf_L.append(np.array([7.32590132e-02, 1.71084837e-01, 3.49591059e-02, 7.59348966e-03,
#         1.75778561e-03, 4.36220524e-04, 1.07802739e-04]))
# t_setup_L.append(np.array([4.26027420e-02, 1.59459732e-01, 6.25250027e-01, 2.45022827e+00,
#         9.74463481e+00, 3.97839780e+01, 1.58308334e+02]))
# t_solve_L.append(np.array([3.79103993e-04, 1.54654600e-03, 5.19639696e-03, 1.01869120e-02,
#         2.25607480e-02, 1.20968345e-01, 1.12225949e+00]))
# labels_L.append('aligned 1:4')
# NX_L.append(np.array([  2,   4,   8,  16,  32,  64, 128]))
# NY_L.append(4)

# # LinearMapping(1.0)
# E_2_L.append(np.array([3.10433622e-02, 5.14015467e-02, 1.29429803e-02, 3.13668008e-03,
#         7.79365287e-04, 1.94315497e-04]))
# E_inf_L.append(np.array([8.15554956e-02, 1.37896376e-01, 3.45950235e-02, 7.75987017e-03,
#         1.86172146e-03, 4.62880009e-04]))
# t_setup_L.append(np.array([7.99738060e-02, 3.09326116e-01, 1.21266577e+00, 4.83178470e+00,
#         1.96829633e+01, 7.80877134e+01]))
# t_solve_L.append(np.array([5.54586994e-04, 5.09201200e-03, 1.00522140e-02, 2.16789820e-02,
#         6.68566270e-02, 5.19047711e-01]))
# labels_L.append('aligned 1:8')
# NX_L.append(np.array([  2,  4,  8, 16, 32, 64]))
# NY_L.append(8)

# # LinearMapping(1.0)
# E_2_L.append(np.array([3.79285860e-02, 4.51282374e-02, 1.16327878e-02, 2.85512024e-03,
#         7.10634169e-04, 1.77250267e-04]))
# E_inf_L.append(np.array([8.61569372e-02, 1.08027035e-01, 2.83953209e-02, 6.56699440e-03,
#         1.54869979e-03, 3.84958003e-04]))
# t_setup_L.append(np.array([1.55809037e-01, 6.12251126e-01, 2.43680181e+00, 9.74610735e+00,
#         3.97854687e+01, 1.59057542e+02]))
# t_solve_L.append(np.array([6.88656000e-04, 1.00071870e-02, 1.99758390e-02, 4.48078190e-02,
#         2.46239777e-01, 2.37466906e+00]))
# labels_L.append('aligned 1:16')
# NX_L.append(np.array([ 2,  4,  8, 16, 32, 64]))
# NY_L.append(16)


# LinearMapping(1.0)
# NQX = NY/NX = 4
E_2_L.append(np.array([1.57054623e-01, 1.27288028e-01, 9.58635730e-02, 7.40382804e-03,
        2.41160097e-03, 6.10297904e-04, 1.67826069e-04]))
E_inf_L.append(np.array([5.30136499e-01, 3.95404510e-01, 3.01588780e-01, 2.81301876e-02,
        1.85333293e-02, 5.29599063e-03, 1.41938236e-03]))
t_setup_L.append(np.array([1.55342407e-01, 6.28205353e-01, 2.58381043e+00, 1.08982816e+01,
        4.69740618e+01, 2.54630817e+02, 1.90288715e+03]))
t_solve_L.append(np.array([5.82335051e-04, 2.79350998e-03, 6.39753405e-03, 1.10273759e-02,
        2.85668761e-02, 1.46424473e-01, 1.19165149e+00]))
labels_L.append('aligned 1:4')
NX_L.append(np.array([  2,   4,   8,  16,  32,  64, 128]))
NY_L.append(4)

# LinearMapping(1.0)
# NQX = NY/NX = 8
E_2_L.append(np.array([1.39848014e-01, 8.19874434e-02, 2.14054811e-02, 5.27441998e-03,
        1.39361544e-03, 3.28512237e-04]))
E_inf_L.append(np.array([3.79649311e-01, 3.02690598e-01, 7.45753039e-02, 2.56121129e-02,
        9.29776707e-03, 2.45576067e-03]))
t_setup_L.append(np.array([6.31260964e-01, 2.54500607e+00, 1.06838160e+01, 4.69587301e+01,
        2.52567668e+02, 1.69888114e+03]))
t_solve_L.append(np.array([1.08584506e-03, 5.82904508e-03, 1.11910290e-02, 2.43627781e-02,
        8.34683479e-02, 5.88462009e-01]))
labels_L.append('aligned 1:8')
NX_L.append(np.array([  2,  4,  8, 16, 32, 64]))
NY_L.append(8)

# LinearMapping(1.0)
# NQX = NY/NX = 16
E_2_L.append(np.array([3.66575932e-02, 5.26842510e-02, 1.69581375e-02, 3.75274324e-03,
        9.12726789e-04, 2.35790509e-04]))
E_inf_L.append(np.array([1.15886012e-01, 1.61444070e-01, 5.39364735e-02, 1.47443508e-02,
        4.79487455e-03, 1.83943293e-03]))
t_setup_L.append(np.array([2.53196687e+00, 1.07588701e+01, 5.07619852e+01, 2.92537090e+02,
        2.32432694e+03, 1.95460627e+04]))
t_solve_L.append(np.array([1.83609396e-03, 1.06842071e-02, 2.30856950e-02, 5.75634730e-02,
        2.76830368e-01, 2.28852086e+00]))
labels_L.append('aligned 1:16')
NX_L.append(np.array([ 2,  4,  8, 16, 32, 64]))
NY_L.append(16)


##### px = py = 0.1 #####

# StraightMapping()
E_2_R.append(np.array([5.65605756e+00, 5.71803490e-01, 1.07925536e+00, 1.34975486e-01,
       2.50190204e-02, 5.69054413e-03, 1.35525260e-03]))
E_inf_R.append(np.array([1.20542306e+01, 1.78702131e+00, 3.21913316e+00, 6.15480925e-01,
       1.14821438e-01, 2.66060246e-02, 6.07068279e-03]))
t_setup_R.append(np.array([4.41093179e-02, 1.65152141e-01, 6.54403051e-01, 2.69359705e+00,
       1.09823763e+01, 5.38209426e+01, 2.90464745e+02]))
t_solve_R.append(np.array([4.41093179e-02, 1.65152141e-01, 6.54403051e-01, 2.69359705e+00,
       1.09823763e+01, 5.38209426e+01, 2.90464745e+02]))
labels_R.append('unaligned 1:1')
NX_R.append(np.array([  4,   8,  16,  32,  64, 128, 256]))
NY_R.append(1)

# LinearMapping(1.0)
E_2_R.append(np.array([1.14927325e+00, 1.37256646e+00, 1.20356283e+00, 8.37506735e-02,
       1.23340675e-02, 2.60364795e-03, 5.82840804e-04]))
E_inf_R.append(np.array([3.85321838e+00, 4.80387343e+00, 4.03600553e+00, 4.19738035e-01,
       6.65410180e-02, 1.56026730e-02, 4.12627696e-03]))
t_setup_R.append(np.array([4.68132300e-02, 1.90119803e-01, 7.54799766e-01, 3.09830493e+00,
       1.31254593e+01, 7.10620103e+01, 5.54923232e+02]))
t_solve_R.append(np.array([7.33752036e-04, 1.56024902e-03, 3.75191693e-03, 9.48439399e-03,
       1.99245029e-02, 1.09904737e-01, 8.03978422e-01]))
labels_R.append('aligned 1:1')
NX_R.append(np.array([  4,   8,  16,  32,  64, 128, 256]))
NY_R.append(1)

# # LinearMapping(1.0)
# E_2_R.append(np.array([5.21067337e+00, 6.22774311e-01, 2.73094276e-01, 4.22059139e-02,
#        9.85303158e-03, 1.98549585e-03, 4.40299466e-04]))
# E_inf_R.append(np.array([1.44942657e+01, 1.88536084e+00, 1.18084847e+00, 2.10525344e-01,
#        4.67228682e-02, 1.66595239e-02, 3.45457562e-03]))
# t_setup_R.append(np.array([4.80882460e-02, 1.87727493e-01, 7.45778577e-01, 3.06630655e+00,
#        1.31194371e+01, 7.39356792e+01, 5.55160143e+02]))
# t_solve_R.append(np.array([6.60517020e-04, 2.84689607e-03, 6.40120404e-03, 1.28399871e-02,
#        3.49453279e-02, 1.85368782e-01, 1.51815793e+00]))
# labels_R.append('aligned 1:4')
# NX_R.append(np.array([  2,   4,   8,  16,  32,  64, 128]))
# NY_R.append(4)

# # LinearMapping(1.0)
# E_2_R.append(np.array([1.59520803e+00, 5.86965808e-01, 6.24328691e-02, 1.78520827e-02,
#        4.92260153e-03, 8.67773928e-04]))
# E_inf_R.append(np.array([4.66516449e+00, 2.00237853e+00, 2.14842838e-01, 9.48346543e-02,
#        3.13195037e-02, 5.22295634e-03]))
# t_setup_R.append(np.array([1.00016563e-01, 3.82668850e-01, 1.55976420e+00, 6.43926618e+00,
#        3.26876116e+01, 1.95483812e+02]))
# t_solve_R.append(np.array([1.09983899e-03, 6.55498297e-03, 1.19966620e-02, 3.00537290e-02,
#        1.08743327e-01, 6.47840927e-01]))
# labels_R.append('aligned 1:8')
# NX_R.append(np.array([  2,  4,  8, 16, 32, 64]))
# NY_R.append(8)

# # LinearMapping(1.0)
# E_2_R.append(np.array([1.92836973e-01, 2.29565763e-01, 3.80961279e-02, 7.35742303e-03,
#        1.64210265e-03, 3.70110519e-04]))
# E_inf_R.append(np.array([5.10917669e-01, 8.15336938e-01, 2.03375363e-01, 3.57510144e-02,
#        8.78621206e-03, 2.19528669e-03]))
# t_setup_R.append(np.array([1.92806849e-01, 7.88583498e-01, 3.20526147e+00, 1.39824541e+01,
#        8.07944004e+01, 6.18632584e+02]))
# t_solve_R.append(np.array([2.56793795e-03, 1.21479930e-02, 2.52503000e-02, 7.18028050e-02,
#        3.59592133e-01, 3.01512412e+00]))
# labels_R.append('aligned 1:16')
# NX_R.append(np.array([ 2,  4,  8, 16, 32, 64]))
# NY_R.append(16)

# LinearMapping(1.0)
# NQX = NY/NX = 4
E_2_R.append(np.array([4.75676417e-01, 8.61391826e-01, 9.66977895e-02, 2.54383958e-02,
        6.40352902e-03, 1.45266992e-03, 3.03146472e-04]))
E_inf_R.append(np.array([1.18693690e+00, 2.89128914e+00, 3.91359899e-01, 1.19466263e-01,
        3.17295702e-02, 9.60271899e-03, 2.72710826e-03]))
t_setup_R.append(np.array([1.64748723e-01, 6.96355763e-01, 3.09706024e+00, 1.33779816e+01,
        6.19731139e+01, 3.74446462e+02, 2.93865206e+03]))
t_solve_R.append(np.array([6.95285038e-04, 2.73545797e-03, 5.93674497e-03, 1.13483320e-02,
        3.20504310e-02, 1.71405263e-01, 1.26482569e+00]))
labels_R.append('aligned 1:4')
NX_R.append(np.array([  2,   4,   8,  16,  32,  64, 128]))
NY_R.append(4)

# LinearMapping(1.0)
# NQX = NY/NX = 8
E_2_R.append(np.array([3.99830541e-01, 1.25570821e-01, 2.66998721e-02, 8.75755340e-03,
        1.91949518e-03, 4.62730505e-04]))
E_inf_R.append(np.array([1.10874879e+00, 4.36491957e-01, 9.95892977e-02, 3.83688835e-02,
        1.24765893e-02, 3.94359545e-03]))
t_setup_R.append(np.array([6.51105676e-01, 2.82038390e+00, 1.47159874e+01, 7.19584263e+01,
        4.65288965e+02, 3.72376342e+03]))
t_solve_R.append(np.array([1.06475095e-03, 5.67114609e-03, 1.08432770e-02, 2.63934410e-02,
        9.25234440e-02, 6.40935463e-01]))
labels_R.append('aligned 1:8')
NX_R.append(np.array([  2,  4,  8, 16, 32, 64]))
NY_R.append(8)

# LinearMapping(1.0)
# NQX = NY/NX = 16
E_2_R.append(np.array([4.80954593e-02, 5.50201937e-02, 1.44564871e-02, 4.85386070e-03,
        9.48869425e-04, np.nan]))
E_inf_R.append(np.array([1.53487332e-01, 1.87176664e-01, 4.81372300e-02, 2.82831253e-02,
        5.38265125e-03, np.nan]))
# these timings run on tabuchi
t_setup_R.append(np.array([3.84321927e+00, 1.88315809e+01, 2.26784287e+02, 1.56933466e+03,
        1.23284896e+04, np.nan]))
t_solve_R.append(np.array([3.74156570e-02, 1.22569010e-02, 2.44721780e-02, 9.08271100e-02,
        4.13082022e-01, np.nan]))
labels_R.append('aligned 1:16')
NX_R.append(np.array([ 2,  4,  8, 16, 32, 64]))
NY_R.append(16)


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

if len(E_2_L) == 1:
    cycler = plt.cycler(color=[black], marker=['d'])
elif len(E_2_L) < 4: # 2 and 3
    cycler = plt.cycler(color=[blue, red, black], marker=['o', 's', 'd'])
else: # 4 or more
    cycler = plt.cycler(color=[blue, red, orange, black, green] + colors[4:],
        marker=['o', 's', '^', 'd', 'x', '+', 'v', '<', '>', '*', 'p'])

fig = plt.figure(figsize=(7.75, 3))
fig.subplots_adjust(hspace=0.5, wspace=0.5)

axL1, axR1 = fig.subplots(1, 2)
axL1.set_prop_cycle(cycler)
axR1.set_prop_cycle(cycler)
N_L = []
inds_L = []
for i, error in enumerate(E_2_L):
    N_L.append(np.log2(NY_L[i]*NX_L[i]**2).astype('int'))
    inds_L.append(N_L[i] >= 2)
    axL1.semilogy(N_L[i][inds_L[i]], error[inds_L[i]]/(2*np.pi), label=labels_L[i],
                  linewidth=solid_linewidth)
# axL1.minorticks_off()
Nmin = min([min(N_L[i]) for i in range(len(N_L))])
Nmax = max([max(N_L[i]) for i in range(len(N_L))])
Nstep = 2
axL1.set_xticks(np.linspace(Nmin, Nmax, (Nmax - Nmin)//Nstep + 1))
axL1.set_xlabel(r'$\log_2(N_xN_y)$')
axL1.set_ylabel(r'$|E_2|$', rotation=0, labelpad=10)
axL1.legend(loc='lower left')
xlim = axL1.get_xlim()

N_R = []
inds_R = []
for i, error in enumerate(E_2_R):
    N_R.append(np.log2(NY_L[i]*NX_L[i]**2).astype('int'))
    inds_R.append(N_R[i] >= 2)
    axR1.semilogy(N_R[i][inds_R[i]], error[inds_R[i]]/(2*np.pi), label=labels_R[i],
                  linewidth=solid_linewidth)
axR1.set_xticks(np.linspace(Nmin, Nmax, (Nmax - Nmin)//Nstep + 1))
axR1.set_xlim(xlim)
axR1.set_xlabel(r'$\log_2(N_xN_y)$')
axR1.set_ylabel(r'$|E_2|$', rotation=0, labelpad=10)
axR1.legend(loc='lower left')

# ordb = 0
# ordt = 3
# ordstep = 0.5
# axR1.set_ylim(ordb, ordt)
# axR1.set_yticks(np.linspace(ordb, ordt, int((ordt - ordb)/ordstep) + 1))
# lines, labels = axR1.get_legend_handles_labels()
# axR1.legend(lines[1:], labels[1:], loc='lower right')

# fig.savefig("boundary_conv.pdf", bbox_inches = 'tight', pad_inches = 0)