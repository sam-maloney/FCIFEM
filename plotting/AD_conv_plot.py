# -*- coding: utf-8 -*-
"""
Created on Mon May 17 17:46:29 2021

@author: Samuel A. Maloney
"""

import numpy as np
import matplotlib.pyplot as plt

# For all of the below, unless otherwise noted
# u0(x,y) = sin(2*pi*x - 2*pi*y)
# dt = 0.01, t_final = 1, theta = 0.7853981633974483
# Omega = (0,1) X (0,1)
# Periodic BCs with VCI-C (slice-by-slice) using ssqr.min2norm
# NQY = NY, Qord = 3, quadType = 'gauss', massLumping = False

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

##### Uniform grid, VCI-C (slice-by-slice) using ssqr.min2norm #####

# # StraightMapping()
# # NQX = 1
# E_2_L.append(np.array([7.07102499e-01, 6.32402419e-01, 2.86940209e-01, 8.47400108e-02,
#         2.21045970e-02, 5.58543215e-03, 1.40009207e-03]))
# E_inf_L.append(np.array([9.99993944e-01, 8.94352078e-01, 4.05794735e-01, 1.19840473e-01,
#        3.12606208e-02, 7.89899390e-03, 1.98002923e-03]))
# t_setup_L.append(np.array([4.70260930e-02, 1.81292197e-01, 6.96564461e-01, 1.93102438e+00,
#        7.44859410e+00, 3.60649786e+01, 1.48865582e+02]))
# t_solve_L.append(np.array([2.30894950e-02, 3.65550540e-02, 4.17699400e-02, 3.36732130e-02,
#        1.08683866e-01, 5.07320221e-01, 4.05322921e+00]))
# labels_L.append('unaligned 1:1')
# NX_L.append(np.array([  4,   8,  16,  32,  64, 128, 256]))
# NY_L.append(1)

# # StraightMapping()
# # NQX = NY/NX = 4
# E_2_L.append(np.array([7.07106781e-01, 7.05811181e-01, 4.91497907e-01, 1.70622808e-01,
#        4.63469303e-02, 1.18294095e-02, 2.97270992e-03]))
# E_inf_L.append(np.array([1.00000000e+00, 9.98167745e-01, 6.95083006e-01, 2.41297090e-01,
#        6.55444575e-02, 1.67293114e-02, 4.20404680e-03]))
# t_setup_L.append(np.array([5.93563080e-02, 2.31001661e-01, 9.21408252e-01, 3.59988035e+00,
#        2.83620386e+01, 1.25452350e+02, 5.52517975e+02]))
# t_solve_L.append(np.array([7.45416700e-03, 1.57737530e-02, 1.94822680e-02, 3.32286810e-02,
#        1.92044193e-01, 3.80585466e-01, 3.04666002e+00]))
# labels_L.append('aligned 1:4')
# NX_L.append(np.array([  2,   4,   8,  16,  32,  64, 128]))
# NY_L.append(4)

# # LinearMapping(1.0)
# # NQX = 1
# E_2_L.append(np.array([0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
#        0.00000000e+00, 0.00000000e+00, 0.00000000e+00]))
# E_inf_L.append(np.array([0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
#        0.00000000e+00, 0.00000000e+00, 0.00000000e+00]))
# t_setup_L.append(np.array([4.88730690e-02, 1.86418051e-01, 7.25021679e-01, 2.90234456e+00,
#        1.17377452e+01, 3.99592889e+01, 1.64007416e+02]))
# t_solve_L.append(np.array([6.50436699e-03, 6.97252101e-03, 7.31036499e-03, 1.00214330e-02,
#        2.03012190e-02, 2.47333640e-02, 2.10771298e-01]))
# labels_L.append('aligned 1:1')
# NX_L.append(np.array([  4,   8,  16,  32,  64, 128, 256]))
# NY_L.append(1)

# # LinearMapping(1.0)
# # NQX = 1
# E_2_L.append(np.array([0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
#        0.00000000e+00, 0.00000000e+00, 0.00000000e+00]))
# E_inf_L.append(np.array([0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
#        0.00000000e+00, 0.00000000e+00, 0.00000000e+00]))
# t_setup_L.append(np.array([4.93491310e-02, 1.84336010e-01, 7.29488004e-01, 2.93562295e+00,
#        9.31869497e+00, 3.79475151e+01, 1.47018570e+02]))
# t_solve_L.append(np.array([6.50308900e-03, 6.69645800e-03, 7.20020701e-03, 8.57191700e-03,
#        7.63390699e-03, 2.74969240e-02, 8.70193770e-02]))
# labels_L.append('aligned 1:4')
# NX_L.append(np.array([  2,   4,   8,  16,  32,  64, 128]))
# NY_L.append(4)


# # LinearMapping(1.0)
# # NQX = NY/NX = 4
# E_2_L.append(np.)
# E_inf_L.append(np.)
# t_setup_L.append(np.)
# t_solve_L.append(np.)
# labels_L.append('aligned 1:4')
# NX_L.append(np.array([  2,   4,   8,  16,  32,  64, 128]))
# NY_L.append(4)

# # LinearMapping(1.0)
# # NQX = NY/NX = 8
# E_2_L.append(np.)
# E_inf_L.append(np.)
# t_setup_L.append(np.)
# t_solve_L.append(np.)
# labels_L.append('aligned 1:8')
# NX_L.append(np.array([  2,  4,  8, 16, 32, 64]))
# NY_L.append(8)

# # LinearMapping(1.0)
# # NQX = NY/NX = 16
# E_2_L.append(np.)
# E_inf_L.append(np.)
# t_setup_L.append(np.)
# t_solve_L.append(np.)
# labels_L.append('aligned 1:16')
# NX_L.append(np.array([ 2,  4,  8, 16, 32, 64]))
# NY_L.append(16)


##### Uniform grid, no VCI #####

# StraightMapping()
# NQX = 1
E_2_L.append(np.array([7.07102499e-01, 6.32402419e-01, 2.86940209e-01, 8.47400108e-02,
       2.21045970e-02, 5.58543215e-03, 1.40009207e-03]))
E_inf_L.append(np.array([9.99993944e-01, 8.94352078e-01, 4.05794735e-01, 1.19840473e-01,
       3.12606208e-02, 7.89899390e-03, 1.98002922e-03]))
t_setup_L.append(np.array([2.59653440e-02, 9.60321950e-02, 3.72480096e-01, 1.51662188e+00,
       5.96279402e+00, 2.38240262e+01, 9.32769962e+01]))
t_solve_L.append(np.array([2.44251360e-02, 3.58391850e-02, 4.18269810e-02, 6.73743070e-02,
       1.85792999e-01, 6.89515132e-01, 3.60605242e+00]))
labels_L.append('unaligned 1:1')
NX_L.append(np.array([  4,   8,  16,  32,  64, 128, 256]))
NY_L.append(1)

# StraightMapping()
# NQX = 4
E_2_L.append(np.array([7.07106781e-01, 7.05811181e-01, 4.91497907e-01, 1.70622808e-01,
       4.63469304e-02, 1.18294095e-02, 2.97270993e-03]))
E_inf_L.append(np.array([1.00000000e+00, 9.98167745e-01, 6.95083006e-01, 2.41297090e-01,
       6.55444575e-02, 1.67293114e-02, 4.20404681e-03]))
t_setup_L.append(np.array([3.04938000e-02, 1.15963849e-01, 4.74950790e-01, 1.89672524e+00,
       7.51895720e+00, 6.01483583e+01, 3.08996843e+02]))
t_solve_L.append(np.array([7.36359500e-03, 1.51947810e-02, 1.91468110e-02, 3.27766000e-02,
       9.90197870e-02, 6.96034517e-01, 2.68265982e+00]))
labels_L.append('unaligned 1:4')
NX_L.append(np.array([  2,   4,   8,  16,  32,  64, 128]))
NY_L.append(4)

# LinearMapping(1.0)
# NQX = 1
E_2_L.append(np.array([0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00]))
E_inf_L.append(np.array([0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00]))
t_setup_L.append(np.array([2.73262700e-02, 1.01799461e-01, 4.06831057e-01, 1.62005424e+00,
       6.42052850e+00, 2.57952718e+01, 6.04735972e+01]))
t_solve_L.append(np.array([6.54689601e-03, 7.07215800e-03, 7.35113501e-03, 9.95931601e-03,
       2.04455920e-02, 7.57735950e-02, 2.65873768e-01]))
labels_L.append('aligned 1:1')
NX_L.append(np.array([  4,   8,  16,  32,  64, 128, 256]))
NY_L.append(1)

# LinearMapping(1.0)
# NQX = 
E_2_L.append(np.array([0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00]))
E_inf_L.append(np.array([0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00]))
t_setup_L.append(np.array([2.73053050e-02, 1.02500677e-01, 4.06507570e-01, 1.62117526e+00,
       6.49584998e+00, 2.59842743e+01, 9.79202378e+01]))
t_solve_L.append(np.array([6.44853500e-03, 6.62292600e-03, 7.28711300e-03, 8.65784100e-03,
       1.61171100e-02, 4.25822420e-02, 1.72924669e-01]))
labels_L.append('aligned 1:4')
NX_L.append(np.array([  2,   4,   8,  16,  32,  64, 128]))
NY_L.append(4)



##### px = py = 0.1, seed = 42, no VCI #####

# StraightMapping()
# NQX = 1
E_2_R.append(np.array([7.07448695e-01, 6.35011387e-01, 2.90646457e-01, 8.61810504e-02,
       2.24596145e-02, 5.67433074e-03, 1.42485852e-03]))
E_inf_R.append(np.array([1.00477395e+00, 9.00660009e-01, 4.15355527e-01, 1.23879062e-01,
       3.30582064e-02, 8.39822500e-03, 2.12766492e-03]))
t_setup_R.append(np.array([1.72929108e-01, 9.51024580e-02, 3.74753928e-01, 1.50016260e+00,
       6.10599585e+00, 2.40968839e+01, 9.49774581e+01]))
t_solve_R.append(np.array([8.93239460e-02, 5.54534370e-02, 4.34385540e-02, 7.00512000e-02,
       2.03555422e-01, 7.72443960e-01, 3.62231795e+00]))
labels_R.append('unaligned 1:1')
NX_R.append(np.array([  4,   8,  16,  32,  64, 128, 256]))
NY_R.append(1)

# LinearMapping(1.0)
# NQX = 1
E_2_R.append(np.array([4.17828953e-02, 1.16959925e-02, 4.84938436e-03, 1.83840883e-03,
       5.69563649e-04, 1.65625100e-04, 4.48447058e-05]))
E_inf_R.append(np.array([7.90334238e-02, 2.81271867e-02, 1.38725295e-02, 5.99212924e-03,
       1.98861723e-03, 6.45249417e-04, 2.11720070e-04]))
t_setup_R.append(np.array([2.78115400e-02, 1.04515585e-01, 4.06629675e-01, 1.62032077e+00,
       6.50896160e+00, 2.60520799e+01, 9.71830533e+01]))
t_solve_R.append(np.array([2.87081700e-02, 3.01700220e-02, 3.62924990e-02, 6.13144450e-02,
       2.27313199e-01, 8.35398510e-01, 2.95626823e+00]))
labels_R.append('aligned 1:1')
NX_R.append(np.array([  4,   8,  16,  32,  64, 128, 256]))
NY_R.append(1)

# LinearMapping(1.0)
# NQX = 4
E_2_R.append(np.array([8.64329058e-03, 2.75222621e-03, 1.05897604e-03, 2.54197731e-04,
       6.54424224e-05, 2.04380058e-05, 5.53069160e-06]))
E_inf_R.append(np.array([1.51876258e-02, 7.95694310e-03, 3.70298310e-03, 9.74585857e-04,
       2.96034005e-04, 1.13593687e-04, 3.05105083e-05]))
t_setup_R.append(np.array([9.99896600e-02, 3.97597781e-01, 7.65955867e-01, 2.04888877e+00,
       8.44476181e+00, 3.34125533e+01, 2.72867091e+02]))
t_solve_R.append(np.array([2.83708200e-02, 3.03293850e-02, 1.63050570e-02, 2.78716960e-02,
       9.28489360e-02, 3.32388050e-01, 2.83546340e+00]))
labels_R.append('aligned 1:4')
NX_R.append(np.array([  2,   4,   8,  16,  32,  64, 128]))
NY_R.append(4)

# # LinearMapping(1.0)
# # NQX = 8
# E_2_R.append(np.)
# E_inf_R.append(np.)
# t_setup_R.append(np.)
# t_solve_R.append(np.)
# labels_R.append('aligned 1:8')
# NX_R.append(np.array([ 2,  4,  8, 16, 32, 64]))
# NY_R.append(8)

# # LinearMapping(1.0)
# # NQX = 16
# E_2_R.append(np.)
# E_inf_R.append(np.)
# t_setup_R.append(np.)
# t_solve_R.append(np.)
# labels_R.append('aligned 1:16')
# NX_R.append(np.array([ 2,  4,  8, 16, 32, 64]))
# NY_R.append(16)


##### px = py = 0.1, seed = 42, VC1-C (whole domain) using ssqr.min2norm #####

# # StraightMapping()
# # NQX = 1
# E_2_R.append(np.array([7.07524521e-01, 6.33525202e-01, 2.87448749e-01, 8.49683404e-02,
#        2.21307106e-02, 5.58993718e-03, 1.40338321e-03]))
# E_inf_R.append(np.array([1.00002761e+00, 9.00786545e-01, 4.07545401e-01, 1.21710495e-01,
#        3.21437603e-02, 8.14469363e-03, 2.04940777e-03]))
# t_setup_R.append(np.array([1.55646440e-02, 5.53931360e-02, 2.32634438e-01, 2.05574035e+00,
#        9.34149883e+00, 4.46373785e+01, 1.36979815e+02]))
# t_solve_R.append(np.array([1.02515560e-02, 1.57257540e-02, 2.06208690e-02, 6.84402180e-02,
#        1.97079202e-01, 7.34847391e-01, 2.89280845e+00]))
# labels_R.append('unaligned 1:1')
# NX_R.append(np.array([  4,   8,  16,  32,  64, 128, 256]))
# NY_R.append(1)

# # LinearMapping(1.0)
# # NQX = 1
# E_2_R.append(np.array([4.20588066e-02, 1.20344639e-02, 4.62066701e-03, 1.81373936e-03,
#        5.69312090e-04, 1.69265068e-04, 4.56623421e-05]))
# E_inf_R.append(np.array([8.03316995e-02, 2.93828184e-02, 1.36057581e-02, 6.23217764e-03,
#        2.08326721e-03, 7.01409572e-04, 2.23593090e-04]))
# t_setup_R.append(np.array([1.71393950e-02, 6.11791060e-02, 2.55736097e-01, 9.90431428e-01,
#        3.91592057e+00, 1.75384460e+01, 8.55581658e+01]))
# t_solve_R.append(np.array([1.30258670e-02, 1.38500570e-02, 1.72923100e-02, 3.24152890e-02,
#        1.26708600e-01, 8.32324660e-01, 3.83949834e+00]))
# labels_R.append('aligned 1:1')
# NX_R.append(np.array([  4,   8,  16,  32,  64, 128, 256]))
# NY_R.append(1)

# # LinearMapping(1.0)
# # NQX = 1
# E_2_R.append(np.array([8.42483119e-03, 2.79679089e-03, 1.03154046e-03, 2.54227843e-04,
#        6.35208213e-05, 2.05660374e-05, 5.49806304e-06]))
# E_inf_R.append(np.array([1.54179560e-02, 7.93045625e-03, 3.66067429e-03, 9.91423646e-04,
#        2.83100128e-04, 1.20429956e-04, 2.96728190e-05]))
# t_setup_R.append(np.array([6.19423580e-02, 5.24986616e-01, 2.88658111e+00, 1.16108029e+01,
#        4.73325182e+01, 1.30713728e+02, 3.17532813e+02]))
# t_solve_R.append(np.array([1.24272890e-02, 3.07168760e-02, 3.54864830e-02, 5.71163310e-02,
#        1.70405363e-01, 3.56370156e-01, 2.24756884e+00]))
# labels_R.append('aligned 1:4')
# NX_R.append(np.array([  2,   4,   8,  16,  32,  64, 128]))
# NY_R.append(4)

# # # LinearMapping(1.0)
# # # NQX = 1
# # E_2_R.append(np.)
# # E_inf_R.append(np.)
# # t_setup_R.append(np.)
# # t_solve_R.append(np.)
# # labels_R.append('aligned 1:8')
# # NX_R.append(np.array([  2,  4,  8, 16, 32, 64]))
# # NY_R.append(8)

# # # LinearMapping(1.0)
# # # NQX = 1
# # E_2_R.append(np.)
# # E_inf_R.append(np.)
# # t_setup_R.append(np.)
# # t_solve_R.append(np.)
# # labels_R.append('aligned 1:16')
# # NX_R.append(np.array([ 2,  4,  8, 16, 32, 64]))
# # NY_R.append(16)

# # LinearMapping(1.0)
# # NQX = NY/NX = 4
# E_2_R.append(np.)
# E_inf_R.append(np.)
# t_setup_R.append(np.)
# t_solve_R.append(np.)
# labels_R.append('aligned 1:4')
# NX_R.append(np.array([  2,   4,   8,  16,  32,  64, 128]))
# NY_R.append(4)

# # LinearMapping(1.0)
# # NQX = NY/NX = 8
# E_2_R.append(np.)
# E_inf_R.append(np.)
# t_setup_R.append(np.)
# t_solve_R.append(np.)
# labels_R.append('aligned 1:8')
# NX_R.append(np.array([  2,  4,  8, 16, 32, 64]))
# NY_R.append(8)

# # LinearMapping(1.0)
# # NQX = NY/NX = 16
# E_2_R.append(np.)
# E_inf_R.append(np.)
# t_setup_R.append(np.)
# t_solve_R.append(np.)
# labels_R.append('aligned 1:16')
# NX_R.append(np.array([ 2,  4,  8, 16, 32, 64]))
# NY_R.append(16)


##### Begin Plotting Routines #####

solid_linewidth = 1.25
dashed_linewidth = 1.0

plt.rc('markers', fillstyle='full')
plt.rc('lines', markersize=5.0)
plt.rc('pdf', fonttype=42)
plt.rc('text', usetex=True)
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
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
axL1.set_title(r'Uniform Grid')
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
axR1.set_title(r'Perturbed Grid (up to 10\%)')
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

fig.savefig("Poisson_slant_conv.pdf", bbox_inches = 'tight', pad_inches = 0)