# -*- coding: utf-8 -*-
"""
Created on Mon May 17 17:46:29 2021

@author: samal
"""

import numpy as np
import matplotlib.pyplot as plt

# For all of the below, unless otherwise noted
# uniform spacing, NQY=NY, NQX=6, Qord=3, quadType='gauss', massLumping=False
# NY_array = np.array([  4,   8,  16,  32,  64, 128, 256, 512])
# start = int(np.log2(NY_array[0]))
# stop = int(np.log2(NY_array[-1]))

##### Left and bottom borders and centre point constrained #####

# Straight mapping, NY = NX
NX_1N_str = np.array([  4,   8,  16,  32,  64, 128, 256, np.nan])
NY_1N_str = np.array([  4,   8,  16,  32,  64, 128, 256, np.nan])
E_2_1N_str = np.array([1.24854595e-04, 1.13034564e-04, 5.36187355e-05, 3.61905545e-05,
        9.19518906e-06, 2.25601602e-06, 5.60632125e-07, np.nan])
E_inf_1N_str = np.array([2.93155729e-04, 3.19461015e-04, 1.35666951e-04, 8.51800106e-05,
        2.34681268e-05, 5.43782230e-06, 1.31899050e-06, np.nan])
t_setup_1N_str = np.array([1.21315800e-01, 4.54591600e-01, 1.97989810e+00, 7.94103160e+00,
        3.19645893e+01, 1.14166103e+02, 4.37051618e+02, np.nan])
t_solve_1N_str = np.array([0.0014533, 0.0027783, 0.0062069, 0.0179855, 0.0437613, 0.2242696,
        1.283171, np.nan])
# t_sim_1N_str = t_setup_1N_str + t_solve_1N_str
# eta_1N_str = (t_sim_1N_str/t_sim_1N_str) / (E_2_1N_str/E_2_1N_str)

# Aligned-linear mapping, NY = NX
NX_1N_lin = np.array([  4,   8,  16,  32,  64, 128, 256, np.nan])
NY_1N_lin = np.array([  4,   8,  16,  32,  64, 128, 256, np.nan])
E_2_1N_lin = np.array([6.80978330e-05, 3.34541181e-06, 1.13919640e-06, 2.53629306e-07,
        6.23542020e-08, 1.55320678e-08, 3.88466425e-09, np.nan])
E_inf_1N_lin = np.array([1.58884385e-04, 8.54154172e-06, 2.96035339e-06, 5.58509505e-07,
        1.27822751e-07, 3.03057707e-08, 7.53510059e-09, np.nan])
t_setup_1N_lin = np.array([1.29583700e-01, 4.73063600e-01, 1.94695980e+00, 7.38590050e+00,
        2.97660262e+01, 1.18776700e+02, 4.94964127e+02, np.nan])
t_solve_1N_lin = np.array([0.0102117, 0.0043037, 0.0066012, 0.0145141, 0.0469667, 0.2109226,
        1.242389, np.nan])
# t_sim_1N_lin = t_setup_1N_lin + t_solve_1N_lin
# eta_1N_lin = (t_sim_1N_str/t_sim_1N_lin) / (E_2_1N_lin/E_2_1N_str)

# Aligned-linear mapping, NY=8*NX
NX_8N_lin = np.array([np.nan, np.nan,   2,   4,   8,  16,  32,  64])
NY_8N_lin = np.array([np.nan, np.nan,  16,  32,  64, 128, 256, 512])
E_2_8N_lin = np.array([np.nan, np.nan, 5.25375009e-06, 1.38000292e-05, 3.46288388e-06, 8.52320168e-07,
        2.10901674e-07, 5.26099334e-08])
E_inf_8N_lin = np.array([np.nan, np.nan, 1.02423035e-05, 3.14426096e-05, 8.89229498e-06, 2.21081229e-06,
        5.23471889e-07, 1.25446896e-07])
t_setup_8N_lin = np.array([np.nan, np.nan,   0.2725781,   1.0872462,   3.7958009,  15.0327328,  60.7268843,
        252.6356505])
t_solve_8N_lin = np.array([np.nan, np.nan, 8.85099871e-04, 2.13967999e-02, 2.96226998e-02, 6.72756999e-02,
        3.38067900e-01, 1.48106630e+00])
# t_sim_8N_lin = t_setup_8N_lin + t_solve_8N_lin
# eta_8N_lin = (t_sim_1N_str/t_sim_8N_lin) / (E_2_8N_lin/E_2_1N_str)

# Aligned-linear mapping, NY=16*NX
NX_16N_lin = np.array([np.nan, np.nan, np.nan,    2,    4,    8,   16,   32,   64])
NY_16N_lin = np.array([np.nan, np.nan, np.nan,   32,   64,  128,  256,  512, 1024])
E_2_16N_lin = np.array([np.nan, np.nan, np.nan, 4.95322139e-06, 1.34090761e-05, 3.37345988e-06, 8.30703195e-07,
        2.05578962e-07, 5.12833202e-08])
E_inf_16N_lin = np.array([np.nan, np.nan, np.nan, 1.37547994e-05, 3.36564826e-05, 8.75730417e-06, 2.19127303e-06,
        5.15289696e-07, 1.23332301e-07])
# final timings for 64X1024 case would need to be added from laptop if needed
t_setup_16N_lin = np.array([np.nan, np.nan, np.nan,   0.5231074,   2.1668616,   8.0369016,  31.1588809, 120.1277179])
t_solve_16N_lin = np.array([np.nan, np.nan, np.nan, 0.0017952, 0.029886 , 0.0624809, 0.2081966, 1.1798728])
# t_sim_16N_lin = t_setup_16N_lin + t_solve_16N_lin
# eta_16N_lin = (t_sim_1N_str/t_sim_16N_lin) / (E_2_16N_lin/E_2_1N_str)


##### perturbation = 0.1, VCI-CP #####
# # NQX=1, Qord=3

# # Straight mapping, NY = NX
# NX_1N_str_p1 = np.array([  4,   8,  16,  32,  64, 128, 256])
# NY_1N_str_p1 = np.array([  4,   8,  16,  32,  64, 128, 256])
# E_2_1N_str_p1 = np.array([4.07133752e-04, 1.76239938e-04, 2.55988820e-04, 2.22258930e-04,
#        4.63885113e-05, 1.61082817e-05, 3.13962582e-06])
# E_inf_1N_str_p1 = np.array([1.01996241e-03, 6.02332439e-04, 8.89386556e-04, 9.37991884e-04,
#        1.73720694e-04, 7.25883884e-05, 1.43482136e-05])
# t_setup_1N_str_p1 = np.array([1.74447460e-02, 6.12895360e-02, 2.64487503e-01, 1.07360594e+00,
#        4.23882521e+00, 1.76971135e+01, 7.81988461e+01])
# t_solve_1N_str_p1 = np.array([7.63233998e-04, 2.33575499e-03, 5.53843200e-03, 1.14164030e-02,
#        3.29754300e-02, 1.77850160e-01, 1.05456406e+00])

# # Aligned-linear mapping, NY = NX
# NX_1N_lin_p1 = np.array([  4,   8,  16,  32,  64, 128, 256])
# NY_1N_lin_p1 = np.array([  4,   8,  16,  32,  64, 128, 256])
# E_2_1N_lin_p1 = np.array([4.44485022e-03, 2.92433679e-04, 1.17801393e-03, 2.89769087e-04,
#        7.78266256e-05, 1.66114352e-05, 7.66591472e-06])
# E_inf_1N_lin_p1 = np.array([1.21134833e-02, 9.36408974e-04, 3.21628024e-03, 1.02816316e-03,
#        2.87397694e-04, 7.88818203e-05, 3.35874758e-05])
# t_setup_1N_lin_p1 = np.array([1.09086467e-01, 6.19416860e-02, 2.44238988e-01, 1.07897191e+00,
#        4.27369080e+00, 1.80254039e+01, 8.28526854e+01])
# t_solve_1N_lin_p1 = np.array([4.19278300e-02, 2.39465800e-03, 5.65276001e-03, 1.22080460e-02,
#        3.57189640e-02, 2.05928853e-01, 1.54801015e+00])

# # Aligned-linear mapping, NY=8*NX
# NX_8N_lin_p1 = np.array([  2,   4,   8,  16,  32,  64])
# NY_8N_lin_p1 = np.array([ 16,  32,  64, 128, 256, 512])
# E_2_8N_lin_p1 = np.array([1.31263982e-04, 1.14080890e-04, 1.74203911e-04, 4.31021213e-05,
#        1.05024158e-05, 2.96518807e-06])
# E_inf_8N_lin_p1 = np.array([3.88049059e-04, 2.87667338e-04, 4.22666280e-04, 1.82333027e-04,
#        5.68914958e-05, 1.45895829e-05])
# t_setup_8N_lin_p1 = np.array([3.22044950e-02, 1.29852966e-01, 5.39043259e-01, 2.20000726e+00,
#        9.09122828e+00, 4.20618570e+01])
# t_solve_8N_lin_p1 = np.array([1.07768401e-03, 9.15430499e-03, 2.05701570e-02, 5.20889770e-02,
#        1.82985017e-01, 1.20948216e+00])

# # Aligned-linear mapping, NY=16*NX
# NX_16N_lin_p1 = np.array([   2,    4,    8,   16,   32,   64])
# NY_16N_lin_p1 = np.array([  32,   64,  128,  256,  512, 1024])
# E_2_16N_lin_p1 = np.array([1.43128704e-04, 3.47062799e-04, 6.28617145e-05, 1.70808555e-05,
#        5.10453996e-06, 1.17231719e-06])
# E_inf_16N_lin_p1 = np.array([3.35984871e-04, 1.04636731e-03, 1.93035748e-04, 8.45271419e-05,
#        2.66027131e-05, 4.28266621e-06])
# t_setup_16N_lin_p1 = np.array([6.11305910e-02, 2.52988497e-01, 1.08350677e+00, 4.41740565e+00,
#        1.88484056e+01, 8.68899701e+01])
# t_solve_16N_lin_p1 = np.array([3.43930701e-03, 2.11478050e-02, 4.22471340e-02, 1.32442926e-01,
#        7.67880989e-01, 4.61033751e+01])

# # NQX=2, Qord=2

# # Straight mapping, NY = NX
# NX_1N_str_p1 = np.array([  4,   8,  16,  32,  64, 128, 256])
# NY_1N_str_p1 = np.array([  4,   8,  16,  32,  64, 128, 256])
# E_2_1N_str_p1 = np.array([6.67575581e-04, 2.42628697e-04, 2.62991120e-04, 2.28279717e-04,
#        4.64418957e-05, 1.61272628e-05, 3.13996083e-06])
# E_inf_1N_str_p1 = np.array([2.15736226e-03, 6.30096351e-04, 1.00123575e-03, 9.64236990e-04,
#        1.73826979e-04, 7.27799755e-05, 1.43503849e-05])
# t_setup_1N_str_p1 = np.array([1.54788630e-02, 5.44259900e-02, 2.31327502e-01, 9.28981009e-01,
#        3.76328145e+00, 1.54313745e+01, 6.68889811e+01])
# t_solve_1N_str_p1 = np.array([7.59250004e-04, 2.26351300e-03, 5.00670100e-03, 1.25432670e-02,
#        3.28775100e-02, 1.86007991e-01, 1.09767419e+00])

# # Aligned-linear mapping, NY = NX
# NX_1N_lin_p1 = np.array([  4,   8,  16,  32,  64, 128, 256])
# NY_1N_lin_p1 = np.array([  4,   8,  16,  32,  64, 128, 256])
# E_2_1N_lin_p1 = np.array([1.28003818e-03, 5.33032659e-04, 5.46569066e-04, 1.47980992e-04,
#        5.70280862e-05, 1.53896652e-05, 3.68154578e-06])
# E_inf_1N_lin_p1 = np.array([3.49931014e-03, 2.04681027e-03, 1.51400521e-03, 5.36725395e-04,
#        1.82323154e-04, 5.54336765e-05, 1.42155491e-05])
# t_setup_1N_lin_p1 = np.array([1.57870830e-02, 5.54018560e-02, 2.21699544e-01, 9.59406500e-01,
#        3.72089613e+00, 1.56343989e+01, 6.92983562e+01])
# t_solve_1N_lin_p1 = np.array([8.02597002e-04, 2.41625401e-03, 5.71742000e-03, 1.21013440e-02,
#        3.36142310e-02, 1.92209961e-01, 1.15688159e+00])

# # Aligned-linear mapping, NY=8*NX
# NX_8N_lin_p1 = np.array([  2,   4,   8,  16,  32,  64])
# NY_8N_lin_p1 = np.array([ 16,  32,  64, 128, 256, 512])
# E_2_8N_lin_p1 = np.array([8.20633179e-05, 2.71344517e-04, 2.99954193e-04, 4.82220312e-05,
#        1.20171195e-05, 3.45098543e-06])
# E_inf_8N_lin_p1 = np.array([2.06936166e-04, 8.44461877e-04, 9.83668163e-04, 1.67753004e-04,
#        6.56345446e-05, 2.14405254e-05])
# t_setup_8N_lin_p1 = np.array([3.14992880e-02, 1.05601946e-01, 4.66957492e-01, 1.90034845e+00,
#        8.23071684e+00, 3.57122466e+01])
# t_solve_8N_lin_p1 = np.array([1.18727001e-03, 1.00850400e-02, 2.34359140e-02, 5.46850660e-02,
#        1.86708146e-01, 1.17328632e+00])

# # Aligned-linear mapping, NY=16*NX
# NX_16N_lin_p1 = np.array([   2,    4,    8,   16,   32,   64])
# NY_16N_lin_p1 = np.array([  32,   64,  128,  256,  512, 1024])
# E_2_16N_lin_p1 = np.array([3.17155265e-04, 2.78266882e-04, 1.22758264e-04, 2.34364652e-05,
#        3.44345042e-06, 1.33855021e-06])
# E_inf_16N_lin_p1 = np.array([9.72898656e-04, 7.84054622e-04, 3.54486184e-04, 9.40130799e-05,
#        1.33670255e-05, 5.99419870e-06])
# t_setup_16N_lin_p1 = np.array([5.63502870e-02, 2.22602709e-01, 8.90482683e-01, 3.80712207e+00,
#        1.61416181e+01, 7.28687076e+01])
# t_solve_16N_lin_p1 = np.array([3.54164600e-03, 2.06259210e-02, 4.63028910e-02, 1.30744188e-01,
#        8.16811838e-01, 4.97002833e+00])


# NQX=2, Qord=3

# Straight mapping, NY = NX
NX_1N_str_p1 = np.array([  4,   8,  16,  32,  64, 128, 256])
NY_1N_str_p1 = np.array([  4,   8,  16,  32,  64, 128, 256])
E_2_1N_str_p1 = np.array([3.59749665e-04, 1.40168278e-04, 2.31228587e-04, 2.17554548e-04,
       4.42347817e-05, 1.53188312e-05, 2.97860924e-06])
E_inf_1N_str_p1 = np.array([7.58121514e-04, 4.01120051e-04, 8.38297763e-04, 9.16281714e-04,
       1.67388898e-04, 6.98383153e-05, 1.37830142e-05])
t_setup_1N_str_p1 = np.array([3.19022270e-02, 1.16696040e-01, 4.79157714e-01, 2.34653147e+00,
       1.13808753e+01, 4.54654602e+01, 1.90452777e+02])
t_solve_1N_str_p1 = np.array([7.98669003e-04, 2.38723199e-03, 5.01214400e-03, 1.14242170e-02,
       3.28367690e-02, 1.86688919e-01, 1.26308597e+00])

# Aligned-linear mapping, NY = NX
NX_1N_lin_p1 = np.array([  4,   8,  16,  32,  64, 128, 256])
NY_1N_lin_p1 = np.array([  4,   8,  16,  32,  64, 128, 256])
E_2_1N_lin_p1 = np.array([2.87279886e-04, 1.87668193e-04, 2.74620901e-04, 5.93566458e-05,
       2.25634663e-05, 3.40536885e-06, 9.24697914e-07])
E_inf_1N_lin_p1 = np.array([7.00062539e-04, 4.79149377e-04, 8.81053030e-04, 2.03868773e-04,
       8.08476298e-05, 1.54829464e-05, 4.05191135e-06])
t_setup_1N_lin_p1 = np.array([3.10432300e-02, 1.15957689e-01, 4.71396939e-01, 2.09510442e+00,
       1.16228912e+01, 4.42282288e+01, 1.91961152e+02])
t_solve_1N_lin_p1 = np.array([7.68082013e-04, 2.34664000e-03, 5.29074800e-03, 1.13771820e-02,
       3.27919150e-02, 1.85113280e-01, 1.22448425e+00])

# Aligned-linear mapping, NY=8*NX
NX_8N_lin_p1 = np.array([  2,   4,   8,  16,  32,  64])
NY_8N_lin_p1 = np.array([ 16,  32,  64, 128, 256, 512])
E_2_8N_lin_p1 = np.array([1.15491282e-04, 3.91433311e-04, 1.27458784e-04, 3.38037848e-05,
       8.54005705e-06, 1.63831106e-06])
E_inf_8N_lin_p1 = np.array([3.29129243e-04, 8.87097082e-04, 4.18648169e-04, 1.14134708e-04,
       3.81792987e-05, 8.96939774e-06])
t_setup_8N_lin_p1 = np.array([6.06678310e-02, 2.49329706e-01, 1.28002923e+00, 5.56065522e+00,
       2.54017462e+01, 1.05806246e+02])
t_solve_8N_lin_p1 = np.array([1.06927501e-03, 1.01362790e-02, 2.13501410e-02, 5.16353760e-02,
       1.82588030e-01, 1.19005197e+00])

# Aligned-linear mapping, NY=16*NX
NX_16N_lin_p1 = np.array([   2,    4,    8,   16,   32,   64])
NY_16N_lin_p1 = np.array([  32,   64,  128,  256,  512, 1024])
E_2_16N_lin_p1 = np.array([4.60170673e-05, 1.84583629e-04, 4.16338715e-05, 1.24058750e-05,
       3.09891493e-06, 7.56021163e-07])
E_inf_16N_lin_p1 = np.array([1.43723064e-04, 5.48199233e-04, 1.36628078e-04, 3.95079192e-05,
       1.30308804e-05, 3.52069890e-06])
t_setup_16N_lin_p1 = np.array([1.18422133e-01, 4.62734509e-01, 2.41086919e+00, 1.18944275e+01,
       4.93934678e+01, 2.24193533e+02])
t_solve_16N_lin_p1 = np.array([3.44238999e-03, 2.27093940e-02, 4.32864720e-02, 1.33799224e-01,
       8.68737033e-01, 4.63906908e+00])


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

# clear the current figure, if opened, and set parameters
fig = plt.figure(figsize=(7.75, 3))
fig.subplots_adjust(hspace=0.5, wspace=0.5)
# fig = plt.figure(figsize=(3.875, 3))
# fig.subplots_adjust(left=0.2, right=0.85)

axL1 = plt.subplot(121)
N_1N_str = np.log2(NX_1N_str*NY_1N_str).astype('int')
N_1N_lin = np.log2(NX_1N_lin*NY_1N_lin).astype('int')
N_8N_lin = np.log2(NX_8N_lin*NY_8N_lin).astype('int')
N_16N_lin = np.log2(NX_16N_lin*NY_16N_lin).astype('int')
plt.semilogy(N_1N_str, E_2_1N_str, 'o-', color=blue, label=r'unaligned 1:1')
plt.semilogy(N_1N_lin, E_2_1N_lin, 's-', color=red, label=r'aligned 1:1')
plt.semilogy(N_8N_lin, E_2_8N_lin, '^-', color=orange, label=r'aligned 1:8')
plt.semilogy(N_16N_lin, E_2_16N_lin, 'd-k', label=r'aligned 1:16')
plt.minorticks_off()
plt.xticks(np.arange(4,17,2), np.arange(4,17,2))
plt.xlabel(r'$\log_2(N_xN_y)$')
plt.ylabel(r'$|E_2|$', rotation=0, labelpad=10)
plt.title('Uniform Grid')
axL1.legend()

axR1 = plt.subplot(122)
N_1N_str_p1 = np.log2(NX_1N_str_p1*NY_1N_str_p1).astype('int')
N_1N_lin_p1 = np.log2(NX_1N_lin_p1*NY_1N_lin_p1).astype('int')
N_8N_lin_p1 = np.log2(NX_8N_lin_p1*NY_8N_lin_p1).astype('int')
N_16N_lin_p1 = np.log2(NX_16N_lin_p1*NY_16N_lin_p1).astype('int')
plt.semilogy(N_1N_str_p1, E_2_1N_str_p1, 'o-', color=blue, label=r'unaligned 1:1')
plt.semilogy(N_1N_lin_p1, E_2_1N_lin_p1, 's-', color=red, label=r'aligned 1:1')
plt.semilogy(N_8N_lin_p1, E_2_8N_lin_p1, '^-', color=orange, label=r'aligned 1:8')
plt.semilogy(N_16N_lin_p1, E_2_16N_lin_p1, 'd-k', label=r'aligned 1:16')
plt.minorticks_off()
plt.xticks(np.arange(4,17,2), np.arange(4,17,2))
plt.xlabel(r'$\log_2(N_xN_y)$')
plt.ylabel(r'$|E_2|$', rotation=0, labelpad=10)
plt.title(r'Perturbed Grid ($\leq 10\%$)')
axR1.legend()

# axR1 = plt.subplot(122)
# plt.semilogy(N_1N_str[:-1], eta_1N_str[:-1], 'o-', color=blue, label=r'unaligned 1:1')
# plt.semilogy(N_1N_lin[:-1], eta_1N_lin[:-1], 's-', color=red, label=r'aligned 1:1')
# plt.semilogy(N_8N_lin[:-1], eta_8N_lin[:-1], '^-', color=orange, label=r'aligned 1:8')
# plt.semilogy(N_16N_lin[:-1], eta_16N_lin[:-1], 'd-k', label=r'aligned 1:16')
# plt.minorticks_off()
# plt.xticks(np.arange(4,17,2), np.arange(4,17,2))
# plt.xlabel(r'$\log_2(N_xN_y)$')
# plt.ylabel(r'$\eta$', rotation=0, labelpad=10)
# axR1.legend(loc='lower right')

# plt.margins(0,0)
fig.savefig("Poisson_slant.pdf", bbox_inches = 'tight', pad_inches = 0)
