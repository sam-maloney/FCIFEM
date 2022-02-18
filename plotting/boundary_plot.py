# -*- coding: utf-8 -*-
"""
Created on Mon May 17 17:46:29 2021

@author: samal
"""

import numpy as np
import matplotlib.pyplot as plt

# For all of the below, unless otherwise noted
NX_array = np.array([  2,   4,   8,  16,  32,  64, 128])

E_2 = []
E_inf = []
t_setup = []
t_solve = []
labels = []
NX = []
NY = []

# ##### DirichletBoundary
# # n = 20/(2*np.pi), a = 0.02, b = 0
# # NY = 20*NX, NDX = 32, Qord = 1
# E_2.append(np.array([3.19128962e-02, 1.66174040e-02, 7.35715866e-03, 5.05400243e-03,
#         1.37505435e-03, 6.28625444e-04, 1.49246335e-04]))
# E_inf.append(np.array([6.01267696e-02, 3.91689608e-02, 2.59411522e-02, 2.42221312e-02,
#         8.78515282e-03, 6.51887251e-03, 3.40038087e-03]))
# t_setup.append(np.array([4.92162494e-01, 2.07696175e+00, 8.58752604e+00, 3.48628305e+01,
#         1.41986197e+02, 5.63175996e+02, 2.29591821e+03]))
# t_solve.append(np.array([6.79264893e-03, 2.44334240e-02, 4.95543720e-02, 1.81298157e-01,
#         1.14694606e+00, 8.82098613e+00, 6.88338451e+01]))
# labels.append('')
# NX.append(np.array([  2,   4,   8,  16,  32,  64, 128]))
# NY.append(20)

# ##### DirichletBoundary
# # n = 20/(2*np.pi), a = 0.02, b = 0
# # NY = 20*NX, NDX = 32, Qord = 2, px = py = 0
# E_2.append(np.array([3.20631362e-02, 1.38357481e-02, 4.66860478e-03, 2.56289445e-03,
#         9.61139994e-03, 1.37399231e-03]))
# E_inf.append(np.array([9.04214930e-02, 5.24540041e-02, 1.64607034e-02, 1.43771064e-02,
#         9.37708754e-02, 1.07042116e-02]))
# t_setup.append(np.array([1.93709404e+00, 8.34575381e+00, 3.42601427e+01, 1.41035787e+02,
#         5.65862695e+02, 2.32414412e+03]))
# t_solve.append(np.array([7.75411818e-03, 2.43537831e-02, 4.95607180e-02, 1.78673380e-01,
#         1.11589643e+00, 9.6233015e+00]))
# labels.append('')
# NX.append(np.array([ 2,  4,  8, 16, 32, 64]))
# NY.append(20)

# ##### DirichletBoundary
# # xmax = 2*np.pi, n = 1, b = 0.05, a = (1 - b*xmax)/xmax**2
# # NY = 6*NX, NQX = NDX = 11, Qord = 1, px = py = 0
# E_2.append(np.array([8.40270876e-02, 3.40794125e-02, 1.36117243e-02, 5.56202646e-03,
#         1.58087841e-03, 4.49679970e-04, 1.38338765e-04]))
# E_inf.append(np.array([1.71750701e-01, 8.09312973e-02, 5.17261148e-02, 2.18594446e-02,
#         1.29669730e-02, 6.80857945e-03, 3.54536785e-03]))
# t_setup.append(np.array([7.64369000e-02, 3.38252900e-01, 1.60485300e+00, 5.59511730e+00,
#         2.16920194e+01, 8.66802058e+01, 3.43453451e+02]))
# t_solve.append(np.array([1.42950000e-03, 1.35460000e-02, 2.68953000e-02, 6.57390000e-02,
#         2.87634900e-01, 1.60251920e+00, 1.33176866e+01]))
# labels.append('')
# NX.append(np.array([  2,   4,   8,  16,  32,  64, 128]))
# NY.append(6)

# ##### DirichletXPeriodicYBoundary
# # xmax = 2*np.pi, n = 1, b = 0.05, a = (1 - b*xmax)/xmax**2
# # NY = 6*NX, NQX = NDX = 11, Qord = 1, px = py = 0
# E_2.append(np.array([4.44963164e-02, 1.24041757e-02, 6.71724548e-03, 3.68103559e-03,
#         7.03944140e-04, 1.62280865e-04, 3.69943998e-05]))
# E_inf.append(np.array([6.23595076e-02, 2.85233499e-02, 1.56946389e-02, 1.40071560e-02,
#         2.33786277e-03, 5.42100080e-04, 1.23390303e-04]))
# t_setup.append(np.array([6.65398000e-02, 2.46649600e-01, 1.07145090e+00, 4.08933360e+00,
#         1.70970712e+01, 6.73706934e+01, 2.73834063e+02]))
# t_solve.append(np.array([8.81799962e-04, 1.28149998e-03, 4.12850000e-03, 1.87049000e-02,
#         9.95074000e-02, 1.16650010e+00, 9.00375790e+00]))
# labels.append('')
# NX.append(np.array([  2,   4,   8,  16,  32,  64, 128]))
# NY.append(6)

# ##### DirichletXPeriodicYBoundary
# # xmax = 2*np.pi, n = 2, b = 0.05, a = (1 - b*xmax)/xmax**2
# # NY = 12*NX, NQX = NDX = 21, Qord = 1, px = py = 0
# E_2.append(np.array([1.86588223e-02, 3.05002326e-02, 8.20772237e-03, 2.09824270e-03,
#         1.08068762e-03, 2.42351627e-04, 5.99447534e-05]))
# E_inf.append(np.array([2.63846173e-02, 5.06978962e-02, 2.27667358e-02, 5.64787150e-03,
#         4.60911635e-03, 1.19323014e-03, 3.18186002e-04]))
# t_setup.append(np.array([1.98775000e-01, 9.12010200e-01, 3.65064190e+00, 1.66881422e+01,
#         6.86359620e+01, 2.58258212e+02, 9.96503545e+02]))
# t_solve.append(np.array([1.50160003e-03, 1.19340001e-03, 5.79810003e-03, 2.95100000e-02,
#         1.29468000e-01, 1.69622240e+00, 1.64209980e+01]))
# labels.append('')
# NX.append(np.array([  2,   4,   8,  16,  32,  64, 128]))
# NY.append(12)

# ##### DirichletXPeriodicYBoundary
# # xmax = 2*np.pi, n = 2, b = 0.05, a = (1 - b*xmax)/xmax**2
# # NY = 12*NX, NQX = NDX = 21, Qord = 1, px = py = 0.1
# E_2.append(np.array([6.07151990e-02, 3.19257550e-02, 1.79889363e-02, 8.95151686e-03,
#        1.05009337e-01, 8.87828389e-03, 6.64412102e-02]))
# E_inf.append(np.array([1.08287752e-01, 1.22302049e-01, 7.73200531e-02, 4.07025206e-02,
#        8.02995556e-01, 9.11946225e-02, 5.79365054e-01]))
# t_setup.append(np.array([1.99418800e-01, 9.80456400e-01, 3.97551440e+00, 1.58233391e+01,
#        6.51082619e+01, 2.49638775e+02, 1.00581255e+03]))
# t_solve.append(np.array([3.38469999e-03, 6.49430000e-02, 1.38907400e-01, 4.09027600e-01,
#        1.55531940e+00, 1.72606012e+01, 1.45715785e+02]))
# labels.append('')
# NX.append(np.array([  2,   4,   8,  16,  32,  64, 128]))
# NY.append(12)

# ##### DirichletXPeriodicYBoundary with VCI-C
# # xmax = 2*np.pi, n = 2, b = 0.05, a = (1 - b*xmax)/xmax**2
# # NY = 12*NX, NQX = NDX = 21, Qord = 1, px = py = 0.1
# E_2.append(np.array([3.65927318e+00, 3.11706802e-01, 2.11781068e-01, 5.77578695e-02,
#        5.98882897e-03, 9.10699940e-04, 8.89662909e-04]))
# E_inf.append(np.array([3.71732987e+00, 6.04125720e-01, 3.86132420e-01, 1.00654121e-01,
#        2.78458966e-02, 5.73213804e-03, 5.46466299e-03]))
# t_setup.append(np.array([4.37193700e-01, 1.70339670e+00, 7.51503080e+00, 3.39897945e+01,
#        1.40137065e+02, 6.17224359e+02, 2.99862325e+03]))
# t_solve.append(np.array([3.2311000e-03, 8.1359900e-02, 1.7258050e-01, 4.9226420e-01,
#        2.4544921e+00, 1.8101165e+01, 1.26678911e+02]))
# labels.append('')
# NX.append(np.array([  2,   4,   8,  16,  32,  64, 128]))
# NY.append(12)

# ##### DirichletXPeriodicYBoundary
# # xmax = 2*np.pi, n = 2, b = 0.05, a = (1 - b*xmax)/xmax**2
# # NY = 12*NX, NQX = NDX = 21, Qord = 2, px = py = 0.1
# E_2.append(np.array([4.62734961e-02, 1.59876082e-02, 6.37444733e-03, 4.14246114e-03,
#        1.50470528e-02, 1.63164576e-03]))
# E_inf.append(np.array([8.88986539e-02, 4.45647259e-02, 3.07720409e-02, 2.52250026e-02,
#        1.24668247e-01, 1.26571724e-02]))
# t_setup.append(np.array([8.04077100e-01, 3.50453490e+00, 1.47012774e+01, 5.93644995e+01,
#        2.41609634e+02, 9.91597784e+02]))
# t_solve.append(np.array([1.0633800e-02, 6.4964800e-02, 1.5490770e-01, 3.4264040e-01,
#        1.8249460e+00, 1.1010249e+01]))
# labels.append('')
# NX.append(np.array([  2,   4,   8,  16,  32,  64]))
# NY.append(12)


# ##### DirichletBoundary with VCI-C, straight mapping (kamitaki)
# # xmax = 2*np.pi, n = 3, b = 0.05, a = (1 - b*xmax)/xmax**2
# # NY = NX, NQX = NDX = 1, Qord = 2, px = py = 0.1
# E_2.append(np.array([2.21468882e+00, 2.46752741e+00, 1.43327150e+00, 3.39350087e-01,
#        5.24456391e-02, 1.63241775e-02, 3.66845204e-03]))
# E_inf.append(np.array([3.67428033e+00, 5.64904985e+00, 5.54238195e+00, 1.71133213e+00,
#        2.75654660e-01, 1.07981381e-01, 2.88058023e-02]))
# t_setup.append(np.array([2.62224140e-02, 9.03887290e-02, 3.58446303e-01, 1.48171695e+00,
#        5.74055028e+00, 2.29911983e+01, 9.22516124e+01]))
# t_solve.append(np.array([7.77927009e-04, 2.45868599e-03, 5.80557300e-03, 1.27522680e-02,
#        3.64734520e-02, 2.48287155e-01, 1.41428905e+00]))
# labels.append('')
# NX.append(np.array([  2,   4,   8,  16,  32,  64, 128]))
# NY.append(1)


##### DirichletBoundary with VCI-C, straight mapping (kamitaki)
# xmax = 2*np.pi, n = 3, b = 0.05, a = (1 - b*xmax)/xmax**2
# NY = NX, NQX = NDX = 1, Qord = 3, px = py = 0.1
E_2.append(np.array([4.59111639e+00, 2.27775633e+00, 1.35899629e+00, 3.35380006e-01,
        5.22449475e-02, 1.62938097e-02, 3.66296508e-03]))
E_inf.append(np.array([9.06507440e+00, 5.03612804e+00, 5.23512234e+00, 1.67960408e+00,
        2.74025358e-01, 1.07751122e-01, 2.86513354e-02]))
t_setup.append(np.array([5.21622690e-02, 2.00154594e-01, 7.91540935e-01, 3.16729736e+00,
        1.29176043e+01, 5.31730653e+01, 2.14524380e+02]))
t_solve.append(np.array([8.20793997e-04, 2.67740000e-03, 5.86735200e-03, 1.28006490e-02,
        3.72800250e-02, 4.43992867e-01, 1.44460044e+00]))
labels.append('Straight N1')
NX.append(np.array([  2,   4,   8,  16,  32,  64, 128]))
NY.append(1)

# ##### DirichletBoundary with VCI-C, quadratic mapping (kamitaki)
# # xmax = 2*np.pi, n = 3, b = 0.05, a = (1 - b*xmax)/xmax**2
# # NY = NX, NQX = NDX = 1, Qord = 2, px = py = 0.1
# E_2.append(np.array([1.07692117e+01, 3.82997947e+00, 1.22650441e+00, 3.17500861e-01,
#        5.58312728e-02, 1.92023969e-02, 3.22494058e-03]))
# E_inf.append(np.array([2.09156741e+01, 1.23978059e+01, 5.04848585e+00, 1.40562156e+00,
#        3.13471726e-01, 1.08831582e-01, 1.96461321e-02]))
# t_setup.append(np.array([2.67289990e-02, 9.93531940e-02, 3.93674432e-01, 1.55100647e+00,
#        6.10932172e+00, 2.48617006e+01, 1.04077419e+02]))
# t_solve.append(np.array([7.86705001e-04, 2.38527900e-03, 5.57318500e-03, 1.32977180e-02,
#        4.09489160e-02, 3.68675166e-01, 1.90805496e+00]))
# labels.append('')
# NX.append(np.array([  2,   4,   8,  16,  32,  64, 128]))
# NY.append(1)

# ##### DirichletBoundary with VCI-C, quadratic mapping (kamitaki)
# xmax = 2*np.pi, n = 3, b = 0.05, a = (1 - b*xmax)/xmax**2
# NY = NX, NQX = NDX = 1, Qord = 3, px = py = 0.1
E_2.append(np.array([2.08400256e+00, 1.52222905e+00, 7.73961250e-01, 1.76273656e-01,
       3.62339621e-02, 1.10422192e-02, 2.15416166e-03]))
E_inf.append(np.array([3.42499275e+00, 3.46375593e+00, 2.62423516e+00, 7.71263127e-01,
       1.82519927e-01, 7.09939174e-02, 1.30439917e-02]))
t_setup.append(np.array([5.73019570e-02, 2.15856163e-01, 8.45253371e-01, 3.37325748e+00,
       1.36826676e+01, 5.70101088e+01, 2.30978274e+02]))
t_solve.append(np.array([7.46758000e-04, 2.70379000e-03, 5.61551200e-03, 1.23089160e-02,
       3.92049870e-02, 2.58457199e-01, 1.80637419e+00]))
labels.append('Quadratic N1')
NX.append(np.array([  2,   4,   8,  16,  32,  64, 128]))
NY.append(1)

# ##### DirichletBoundary with VCI-C, quadratic mapping (kamitaki)
# # xmax = 2*np.pi, n = 3, b = 0.05, a = (1 - b*xmax)/xmax**2
# # NY = 16*NX, NQX = NDX = 1, Qord = 2, px = py = 0.1
# E_2.append(np.array([7.05252693e+00, 9.45572543e-01, 5.87346082e-01, 8.48034913e-02,
#        3.39513754e-01]))
# E_inf.append(np.array([1.95747289e+01, 3.03739512e+00, 1.00249014e+01, 1.10738998e+00,
#        8.02687735e+00]))
# t_setup.append(np.array([3.69821040e-01, 1.93508854e+00, 5.95127578e+00, 2.40198217e+01,
#        9.82133340e+01]))
# t_solve.append(np.array([1.70987083e+00, 1.14231555e+00, 3.31586668e+00, 1.32440029e+01,
#        5.82842493e+01]))
# labels.append('')
# NX.append(np.array([ 4,  8, 16, 32, 64]))
# NY.append(16)

##### DirichletBoundary with VCI-C, quadratic mapping (kamitaki)
# xmax = 2*np.pi, n = 3, b = 0.05, a = (1 - b*xmax)/xmax**2
# NY = 16*NX, NQX = NDX = 1, Qord = 3, px = py = 0.1
# NX_array = np.array([ 2,  4,  8, 16, 32, 64])
E_2.append(np.array([9.04406032e-01, 1.10654007e+00, 8.46734396e-01, 3.00802309e-01,
       8.33548295e-02, 2.10124921e-02]))
E_inf.append(np.array([2.29383407e+00, 3.09260827e+00, 3.06488059e+00, 1.46121208e+00,
       7.23046191e-01, 1.70706879e-01]))
t_setup.append(np.array([1.99390257e-01, 8.02651423e-01, 3.28052474e+00, 1.34559532e+01,
       5.57998046e+01, 2.29369243e+02]))
t_solve.append(np.array([3.66800200e-03, 1.95079540e-02, 4.51270550e-02, 1.52930891e-01,
       7.09913679e+00, 3.64055848e+01]))
labels.append('Quadratic N16')
NX.append(np.array([ 2,  4,  8, 16, 32, 64]))
NY.append(16)

# ##### DirichletBoundary with VCI-C, quadratic mapping (kamitaki)
# xmax = 2*np.pi, n = 3, b = 0.05, a = (1 - b*xmax)/xmax**2
# NY = 16*NX, NQX = 27, NDX = 0, Qord = 2, px = py = 0.1
# NX_array = np.array([ 2,  4,  8, 16, 32])
E_2.append(np.array([1.47867499e-01, 5.53629445e-02, 1.38097889e-02, 2.61809664e-03,
       7.74379144e-04]))
E_inf.append(np.array([2.47643518e-01, 1.16046688e-01, 3.79628098e-02, 1.05232613e-02,
       4.77595148e-03]))
t_setup.append(np.array([2.39220765e+00, 1.03637169e+01, 3.96004280e+01, 1.70590838e+02,
       1.13104444e+03]))
t_solve.append(np.array([3.40144201e-03, 1.99636080e-02, 4.18134610e-02, 1.37417401e-01,
       1.26429747e+00]))
labels.append('Quadratic N16 NQX27')
NX.append(np.array([ 2,  4,  8, 16, 32]))
NY.append(16)


##### Begin Plotting Routines #####

solid_linewidth = 1.25
dashed_linewidth = 1.0

plt.rcParams['markers.fillstyle'] = 'full'
plt.rcParams['lines.markersize'] = 5.0
plt.rcParams['lines.linewidth'] = solid_linewidth
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['text.usetex'] = True
# fontsize : int or {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}
plt.rcParams['legend.fontsize'] = 'small'
# plt.rcParams['font.size'] = 'small'
# plt.rcParams['axes.titlesize'] = 'medium'
# plt.rcParams['axes.labelsize'] = 'medium'
# plt.rcParams['xtick.labelsize'] = 'small'
# plt.rcParams['ytick.labelsize'] = 'small'
# plt.rcParams['figure.titlesize'] = 'large'

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
blue = colors[0]
orange = colors[1]
green = colors[2]
red = colors[3]
purple = colors[4]
brown = colors[5]
pink = colors[6]
grey = colors[7]
yellow = colors[8]
cyan = colors[9]
black = '#000000'

if len(E_2) < 4:
    colors = [blue, red, black]
else:
    colors = [blue, red, orange, black, green] + colors[4:]
    
markers = ['o', 's', '^', 'd', 'v', '<', '>', 'x', '+']

# clear the current figure, if opened, and set parameters
fig = plt.figure()
fig.clf()
fig.set_size_inches(7.75,3)
fig.subplots_adjust(hspace = 0.5, wspace = 0.5)

axL1, axR1 = fig.subplots(1,2)
N = []
for i, error in enumerate(E_2):
    N.append(np.log2(NY[i]*NX[i]**2).astype('int'))
    axL1.semilogy(N[i], error, markers[i] + '-', color=colors[i], label=labels[i])
axL1.minorticks_off()
N_ticks = np.unique(np.concatenate(N))
axL1.set_xticks(N_ticks)
axL1.set_xlabel(r'$\log_2(N_xN_y)$')
axL1.set_ylabel(r'$|E_2|$', rotation=0, labelpad=10)
axL1.legend()
xlim = axL1.get_xlim()

# plot the intra-step order of convergence
plt.rcParams['lines.linewidth'] = dashed_linewidth
axR1.axhline(2, linestyle=':', color=black, label='Expected order')
for i, error in enumerate(E_2):
    logE = np.log(error)
    logN = np.log(NX[i])
    order = (logE[:-1] - logE[1:])/(logN[1:] - logN[:-1])
    intraN = 0.5 * (N[i][:-1] + N[i][1:])
    axR1.plot(intraN, order, markers[i] + ':', color=colors[i], label=labels[i] + ' order')
axR1.set_xticks(N_ticks)
axR1.set_xlim(xlim)
axR1.set_xlabel(r'$\log_2(N_xN_y)$')
axR1.set_ylabel(r'Intra-step Order of Convergence')
ordb = 0
ordt = 3
ordstep = 0.5
axR1.set_ylim(ordb, ordt)
axR1.set_yticks(np.linspace(ordb, ordt, (ordt - ordb)/ordstep + 1))

# lines, labels = axL1.get_legend_handles_labels()
# lines2, labels2 = axL2.get_legend_handles_labels()
# # leg = axL2.legend(lines, labels, loc='lower left')
# # leg = axL2.legend(lines + lines2, labels + labels2, loc='best')
# # leg = axL2.legend(lines + [lines2[-1]], labels + [labels2[-1]], loc='lower left')

fig.savefig("boundary_conv.pdf", bbox_inches = 'tight', pad_inches = 0)