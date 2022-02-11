# -*- coding: utf-8 -*-
"""
Created on Mon May 17 17:46:29 2021

@author: samal
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# For all of the below, unless otherwise noted
# NY = NX, NQY=NY, NQX=6, Qord=3, quadType='gauss', massLumping=False
NX_array = np.array([  4,   8,  16,  32,  64, 128, 256])
start = int(np.log2(NX_array[0]))
stop = int(np.log2(NX_array[-1]))

##### No constrained nodes!!!!! #####

# Sinusoidal mapping, Uniform spacing
E_2_u_sin = np.array([1.51377118e-04, 1.51345163e-03, 4.27174201e-04, 1.10044134e-04,
        2.75552544e-05, 6.87046871e-06, 1.72329954e-06])
E_inf_u_sin = np.array([3.02754236e-04, 2.54824976e-03, 7.38828574e-04, 2.04472060e-04,
        5.20249246e-05, 1.30128704e-05, 3.31977042e-06])

# Straight mapping, Uniform spacing
E_2_u_str = np.array([2.66565612e-03, 6.47088425e-04, 1.59542252e-04, 3.97356716e-05,
        9.92440295e-06, 2.48050383e-06, 6.20088618e-07])
E_inf_u_str = np.array([5.33131225e-03, 1.29417685e-03, 3.19084503e-04, 7.94713431e-05,
        1.98488059e-05, 4.96100765e-06, 1.24017724e-06])

# Straight mapping, 0.1 perturbation
E_2_p1_str = np.array([0.00285971, 0.00103294, 0.00055518, 0.00027292, 0.00017557,
        0.00014189, 0.00013756])
E_inf_p1_str = np.array([0.00627925, 0.00310992, 0.00187384, 0.00104082, 0.00057817,
        0.0004272 , 0.00031896])


##### Left and bottom borders and centre point constrained #####

# Sinusoidal mapping, 0.1 perturbation, Qord=3
E_2_p1_sin = np.array([1.76380823e-03, 1.63218787e-03, 4.89503662e-04, 1.30604665e-04,
        3.89274495e-05, 1.26881426e-05, 6.15871552e-06])
E_inf_p1_sin = np.array([4.22775291e-03, 3.76831541e-03, 1.05169678e-03, 3.44947421e-04,
        1.26005593e-04, 5.59803653e-05, 3.21214936e-05])
t_setup_p1_sin = np.array([1.20946600e-01, 4.26536300e-01, 1.70480300e+00, 6.39973280e+00,
        2.53950766e+01, 9.80823424e+01, 3.91944770e+02])
t_solve_p1_sin = np.array([2.05040001e-03, 6.14400001e-03, 9.41030000e-03, 2.37355000e-02,
        7.34108000e-02, 2.86898700e-01, 2.22506250e+00])

# Sinusoidal mapping, 0.1 perturbation, Qord=3, ConservativePoint VCI
E_2_p1_sin_CP = np.array([1.72049530e-03, 1.74932219e-03, 4.99569777e-04, 1.25227649e-04,
       3.12106760e-05, 7.83296679e-06, np.nan])
E_inf_p1_sin_CP = np.array([3.94509620e-03, 3.92583363e-03, 1.17370148e-03, 2.89495746e-04,
       7.36176515e-05, 1.76152918e-05, np.nan])
t_setup_p1_sin_CP = np.array([1.14400500e-01, 5.64714000e-01, 3.02665540e+00, 2.56724059e+01,
       2.59681130e+02, np.nan, np.nan])
t_solve_p1_sin_CP = np.array([1.57389999e-03, 4.82690001e-03, 8.23490000e-03, 1.76809000e-02,
       5.92375000e-02, np.nan, np.nan])

# Sinusoidal mapping, 0.1 perturbation, Qord=3, ConservativeCell VCI
E_2_p1_sin_CC = np.array([1.62244282e-03, 1.78174990e-03, 5.05333848e-04, 1.25074630e-04,
       3.11606360e-05, 7.84181311e-06, np.nan])
E_inf_p1_sin_CC = np.array([3.74952526e-03, 4.15017968e-03, 1.22581638e-03, 2.90684750e-04,
       7.37685993e-05, 1.78511294e-05, np.nan])
t_setup_p1_sin_CC = np.array([1.18314700e-01, 5.29194400e-01, 2.39201140e+00, 1.49330466e+01,
       1.26001360e+02, 1.01552816e+03, np.nan])
t_solve_p1_sin_CC = np.array([1.49850000e-03, 4.63169999e-03, 8.30550000e-03, 1.66261000e-02,
       6.22830000e-02, 2.96640500e-01, np.nan])

# Sinusoidal mapping, 0.1 perturbation, Qord=3, ConservativeNode VCI
E_2_p1_sin_CN = np.array([1.19386387e-03, 1.83689078e-03, 5.13330850e-04, 1.26876346e-04,
       3.14171418e-05, 8.74830802e-06, 5.46806186e-06])
E_inf_p1_sin_CN = np.array([2.71166000e-03, 4.11920389e-03, 1.29610762e-03, 3.09127378e-04,
       8.98967456e-05, 4.29242666e-05, 4.33876708e-04])
t_setup_p1_sin_CN = np.array([1.68636100e-01, 6.98860100e-01, 2.58184350e+00, 9.98467950e+00,
       4.09186959e+01, 1.63121791e+02, 6.49763509e+02])
t_solve_p1_sin_CN = np.array([2.021000e-03, 5.128700e-03, 8.931900e-03, 1.908220e-02,
       5.747630e-02, 3.726076e-01, 2.3548181e+00])

# Sinusoidal mapping, 0.1 perturbation, Qord=3, Linear VCI
E_2_p1_sin_L = np.array([1.70905263e-03, 1.71208435e-03, 4.99655257e-04, 1.25161605e-04,
        3.09979365e-05, 8.66766800e-06, 3.38132213e-06])
E_inf_p1_sin_L = np.array([3.96227389e-03, 3.89159769e-03, 1.16773437e-03, 2.91271759e-04,
        7.56499095e-05, 2.37034861e-05, 1.08384079e-05])
t_setup_p1_sin_L = np.array([1.08562400e-01, 4.31015100e-01, 1.73289180e+00, 6.62913230e+00,
        2.67953264e+01, 1.10938673e+02, 4.39067302e+02])
t_solve_p1_sin_L = np.array([2.09550001e-03, 5.45420000e-03, 9.21269998e-03, 1.78824000e-02,
        6.85618000e-02, 7.81468000e-01, 3.26711650e+00])

# Sinusoidal mapping, 0.1 perturbation, Qord=3, Quadratic VCI
E_2_p1_sin_Q = np.array([1.96114501e-03, 1.33686709e-03, 4.60554065e-04, 1.24423715e-04,
        3.13792394e-05, 7.90459913e-06, 2.05436055e-06])
E_inf_p1_sin_Q = np.array([4.29110192e-03, 3.02804777e-03, 1.06999938e-03, 2.92372861e-04,
        7.08789337e-05, 1.81415879e-05, 4.93928290e-06])
t_setup_p1_sin_Q = np.array([3.84015800e-01, 1.47548630e+00, 5.61835990e+00, 2.24718873e+01,
        8.64237513e+01, 3.48133139e+02, 1.41286216e+03])
t_solve_p1_sin_Q = np.array([2.11110001e-03, 7.51730002e-03, 1.23081000e-02, 2.43821000e-02,
        7.63307000e-02, 7.56741700e-01, 3.24162050e+00])

# Sinusoidal mapping, 0.1 perturbation, Qord=10
E_2_p1_sin10 = np.array([1.84392285e-03, 1.67487638e-03, 4.94057917e-04, 1.26032021e-04,
        3.15888578e-05, 8.09440320e-06, 2.19631009e-06])
E_inf_p1_sin10 = np.array([4.40617300e-03, 3.72130389e-03, 1.14300871e-03, 2.82715024e-04,
        7.18992121e-05, 2.27080035e-05, 8.18602074e-06])
t_setup_p1_sin10 = np.array([1.08862270e+00, 4.37006580e+00, 1.71951855e+01, 6.90441187e+01,
        2.72979249e+02, 1.09588131e+03, 6.81203294e+03])
t_solve_p1_sin10 = np.array([1.17229999e-03, 5.89129998e-03, 9.05930001e-03, 1.88243000e-02,
        6.81241000e-02, 9.94588400e-01, 5.55685090e+00])


# Straight mapping, 0.1 perturbation, Qord=10
E_2_p1_str10 = np.array([2.79421197e-03, 7.98841851e-04, 2.24282447e-04, 1.01113862e-04,
        4.50548024e-05, 2.59406008e-05, 1.83385591e-05])
E_inf_p1_str10 = np.array([6.29920113e-03, 1.99154102e-03, 6.31766245e-04, 3.32692194e-04,
        2.06415962e-04, 1.00257387e-04, 6.19472540e-05])

# Straight mapping, 0.1 perturbation, Qord=3, Linear VCI
E_2_p1_str_L = np.array([2.83754386e-03, 9.99162755e-04, 2.72950458e-04, 9.22101598e-05,
        4.59170115e-05, 5.40049628e-05, 5.49855842e-05])
E_inf_p1_str_L = np.array([7.00280166e-03, 2.99459433e-03, 1.15398586e-03, 4.17471144e-04,
        1.52541753e-04, 1.40946823e-04, 1.26561102e-04])


##### Begin Plotting Routines #####

# from matplotlib import rcParams
mpl.rcParams['pdf.fonttype'] = 42
mpl.rc("font", **{"family": "serif", "serif": ["Latin Modern Roman"]})
# mpl.rc("font", **{"family": "serif", "serif": ["Palatino"]})
mpl.rc("text", usetex = True)

# clear the current figure, if opened, and set parameters
fig = plt.gcf()
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
# plt.loglog(NX_array, E_2_u_str, '.-', label=r'Str/Uniform')
plt.loglog(NX_array, E_2_u_sin, '.-', label=r'Uniform')
# plt.loglog(NX_array, E_2_p1_str, '.-', label=r'Str/10% pert')
plt.loglog(NX_array, E_2_p1_sin, '.-', label=r'10\% perturbation')
plt.minorticks_off()
plt.ylim(top=1.5e-2)
plt.xticks(NX_array, NX_array)
plt.xlabel(r'$NX$')
plt.ylabel(r'$|E_2|$', rotation=0, labelpad=10)
plt.title('Uniform vs. Perturbed Grid')

# plot the intra-step order of convergence
axL2 = axL1.twinx()
logN = np.log(NX_array)
intraN = np.logspace(start+0.5, stop-0.5, num=stop-start, base=2.0)
# logE_str = np.log(E_2_u_str)
# order_str = (logE_str[0:-1] - logE_str[1:])/(logN[1:] - logN[0:-1])
# plt.plot(intraN, order_str, '.:', linewidth=1, label=r'Str/Uniform order')
logE_sin = np.log(E_2_u_sin)
order_sin = (logE_sin[0:-1] - logE_sin[1:])/(logN[1:] - logN[0:-1])
plt.plot(intraN, order_sin, '.:', linewidth=1, label=r'Uniform order')
# logE_p1_str = np.log(E_2_p1_str)
# order_p1_str = (logE_p1_str[0:-1] - logE_p1_str[1:])/(logN[1:] - logN[0:-1])
# plt.plot(intraN, order_p1_str, '.:', linewidth=1, label=r'Str/10% pert order')
logE_p1_sin = np.log(E_2_p1_sin)
order_p1_sin = (logE_p1_sin[0:-1] - logE_p1_sin[1:])/(logN[1:] - logN[0:-1])
plt.plot(intraN, order_p1_sin, '.:', linewidth=1, label=r'10\% pert order')

plt.plot(plt.xlim(), [2, 2], 'k:', linewidth=1, label='Expected ord')
ordb = 0
ordt = 2.5
plt.ylim(ordb, ordt)
# plt.yticks(np.linspace(ordb, ordt, ordt - ordb + 1)) # unit spacing
plt.yticks(np.linspace(ordb, ordt, int((ordt - ordb)*2 + 1))) # 0.5 spacing
plt.ylabel(r'Intra-step Order of Convergence')
lines, labels = axL1.get_legend_handles_labels()
lines2, labels2 = axL2.get_legend_handles_labels()
leg = axL2.legend(lines, labels, loc='lower left')
# leg = axL2.legend(lines + lines2, labels + labels2, loc='best')
# leg = axL2.legend(lines + [lines2[-1]], labels + [labels2[-1]], loc='lower left')
plt.margins(0,0)

# plot the error convergence
axR1 = plt.subplot(122)
# plt.loglog(NX_array, E_2_p1_sin, '.-', label=r'Q3')
plt.loglog(NX_array, E_2_p1_sin10, '.-', label=r'Q10')
plt.loglog(NX_array, E_2_p1_sin_L, '.-', label=r'Q3 VC1')
plt.loglog(NX_array, E_2_p1_sin_Q, '.-', label=r'Q3 VC2')
plt.loglog(NX_array, E_2_p1_sin_CN, '.-', label=r'Q3 VC1-C')
plt.minorticks_off()
plt.ylim(top=1.5e-2)
plt.xticks(NX_array, NX_array)
plt.xlabel(r'$NX$')
plt.ylabel(r'$|E_2|$', rotation=0, labelpad=10)
plt.title(r'Improved Quadrature')

# plot the intra-step order of convergence
axR2 = axR1.twinx()
logN = np.log(NX_array)
# logE_p1_sin = np.log(E_2_p1_sin)
# order_p1_sin = (logE_p1_sin[0:-1] - logE_p1_sin[1:])/(logN[1:] - logN[0:-1])
# plt.plot(intraN, order_p1_sin, '.:', linewidth=1, label=r'Q3 order')
logE_sin10 = np.log(E_2_p1_sin10)
order_sin10 = (logE_sin10[0:-1] - logE_sin10[1:])/(logN[1:] - logN[0:-1])
plt.plot(intraN, order_sin10, '.:', linewidth=1, label=r'Q10 order')
logE_p1_sin_L = np.log(E_2_p1_sin_L)
order_p1_sin_L = (logE_p1_sin_L[0:-1] - logE_p1_sin_L[1:])/(logN[1:] - logN[0:-1])
plt.plot(intraN, order_p1_sin_L, '.:', linewidth=1, label=r'Q3 VC1 order')
logE_p1_sin_Q = np.log(E_2_p1_sin_Q)
order_p1_sin_Q = (logE_p1_sin_Q[0:-1] - logE_p1_sin_Q[1:])/(logN[1:] - logN[0:-1])
plt.plot(intraN, order_p1_sin_Q, '.:', linewidth=1, label=r'Q3 VC2 order')
logE_p1_sin_CN = np.log(E_2_p1_sin_CN)
order_p1_sin_CN = (logE_p1_sin_CN[0:-1] - logE_p1_sin_CN[1:])/(logN[1:] - logN[0:-1])
plt.plot(intraN, order_p1_sin_CN, '.:', linewidth=1, label=r'Q3 VC1-C order')

plt.plot(plt.xlim(), [2, 2], 'k:', linewidth=1, label='Expected ord')
ordb = 0
ordt = 2.5
plt.ylim(ordb, ordt)
# plt.yticks(np.linspace(ordb, ordt, ordt - ordb + 1))
plt.yticks(np.linspace(ordb, ordt, int((ordt - ordb)*2 + 1)))
plt.ylabel(r'Intra-step Order of Convergence')
lines, labels = axR1.get_legend_handles_labels()
lines2, labels2 = axR2.get_legend_handles_labels()
leg = axR2.legend(lines, labels, loc='lower left')
# leg = axR2.legend(lines + lines2, labels + labels2, loc='best')
# leg = axR2.legend(lines + [lines2[-1]], labels + [labels2[-1]], loc='lower left')
plt.margins(0,0)

# plt.savefig("Poisson_conv_all_VCI.pdf", bbox_inches = 'tight', pad_inches = 0)