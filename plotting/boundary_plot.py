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
# px = py = 0.1, seed = 42
# NQY = NY, quadType = 'gauss', massLumping = False

E_2 = []
E_inf = []
t_setup = []
t_solve = []
labels = []
NX = []
NY = []

##### VCI-C (whole domain) #####

# # StraightMapping()
# # NQX = NDX = 1, Qord = 3
# E_2.append(np.array([4.06673103e+00, 5.87982631e-01, 1.81038252e-01, 1.72990902e-01,
#         3.52738228e-02, 5.80219359e-03, 1.25770637e-03, 3.33958611e-04]))
# E_inf.append(np.array([4.06673103e+00, 1.27645289e+00, 5.48873588e-01, 5.24559361e-01,
#         1.41505101e-01, 3.49764170e-02, 8.78641178e-03, 2.06240798e-03]))
# t_setup.append(np.array([1.47075669e-02, 5.07215180e-02, 1.98696069e-01, 7.83551860e-01,
#         3.19821026e+00, 1.29745868e+01, 5.22686873e+01, 2.11342415e+02]))
# t_solve.append(np.array([4.18831129e-04, 7.01643992e-04, 1.49277295e-03, 2.86228815e-03,
#         7.44230812e-03, 2.11537471e-02, 1.17465902e-01, 6.86309253e-01]))
# labels.append('unaligned 1:1')
# NX.append(np.array([  2,   4,   8,  16,  32,  64, 128, 256]))
# NY.append(1)

# # QuadraticMapping(0.95, 0.05)
# # NQX = NDX = 1, Qord = 3
# E_2.append(np.array([2.75177924e-02, 4.66436584e-01, 1.95301545e-01, 7.54000679e-02,
#         1.13931278e-02, 3.80245215e-03, 9.17835917e-04, 2.95894260e-04]))
# E_inf.append(np.array([2.75177924e-02, 9.07245700e-01, 5.70754468e-01, 2.17962142e-01,
#         3.73445243e-02, 1.68892865e-02, 5.76069809e-03, 2.00070348e-03]))
# t_setup.append(np.array([1.57121969e-02, 5.50243820e-02, 2.21912339e-01, 8.73713755e-01,
#         3.48041777e+00, 1.43380649e+01, 5.71902653e+01, 2.32448775e+02]))
# t_solve.append(np.array([3.90805071e-04, 7.43536977e-04, 2.13289703e-03, 6.59624394e-03,
#         1.71172130e-02, 4.74065030e-02, 2.69226695e-01, 1.95620522e+00]))
# labels.append('aligned 1:1')
# NX.append(np.array([  2,   4,   8,  16,  32,  64, 128, 256]))
# NY.append(1)

# # QuadraticMapping(0.95, 0.05)
# # NQX = 32, NDX = 0, Qord = 2
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


##### VCI-C (slice-by-slice) #####

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

# # scipy.sparse.linalg.lsqr (no diagonal preconditioning)
# # StraightMapping()
# # NQX = NDX = 1, Qord = 3
# E_2.append(np.array([8.31999445e-01, 3.48371596e-01, 1.81387733e-01, 4.66057862e-02,
#        1.03976879e-02, 2.53774979e-03, 6.25369330e-04]))
# E_inf.append(np.array([1.23925189e+00, 8.54070165e-01, 5.77384218e-01, 1.68700372e-01,
#        5.19337601e-02, 1.33352624e-02, 3.29734011e-03]))
# t_setup.append(np.array([2.78253229e-02, 1.05478812e-01, 4.27727219e-01, 1.69693359e+00,
#        6.98176884e+00, 3.32064980e+01, 1.58413017e+02]))
# t_solve.append(np.array([3.83008970e-04, 1.30848400e-03, 3.32668901e-03, 6.43713295e-03,
#        1.57915420e-02, 5.36580930e-02, 4.14891262e-01]))
# labels.append('unaligned 2:1')
# NX.append(np.array([  4,   8,  16,  32,  64, 128, 256]))
# NY.append(0.5)

# scipy.sparse.linalg.lsqr (no diagonal preconditioning)
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

# scipy.sparse.linalg.lsqr (no diagonal preconditioning)
# StraightMapping()
# NQX = 4, NDX = 1, Qord = 3
E_2.append(np.array([5.98419021e+00, 1.84821982e+00, 6.73724817e-01, 3.33756682e-01,
       9.75525685e-02, 2.85139264e-02, 8.16929749e-03]))
E_inf.append(np.array([9.38487821e+00, 4.74091396e+00, 2.51045126e+00, 1.16121631e+00,
       4.12906901e-01, 1.39264881e-01, 4.45868675e-02]))
t_setup.append(np.array([1.73296966e-01, 7.48117318e-01, 3.10649718e+00, 1.30626945e+01,
       5.46568710e+01, 2.36789972e+02, 1.10610214e+03]))
t_solve.append(np.array([6.48439978e-04, 2.56401999e-03, 9.53683341e-02, 2.74535109e-02,
       5.08006059e-02, 2.33789191e-01, 2.42643756e+00]))
labels.append('unaligned 1:4')
NX.append(np.array([  2,   4,   8,  16,  32,  64, 128]))
NY.append(4)

# scipy.sparse.linalg.lsqr (no diagonal preconditioning)
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

# scipy.sparse.linalg.lsqr (no diagonal preconditioning)
# QuadraticMapping(0.95, 0.05)
# NQX = NDX = 4, Qord = 3
E_2.append(np.array([1.81152465e-01, 4.35327926e-02, 1.58975943e-02, 3.48755234e-03,
       9.83954595e-04, 2.23904217e-04, 5.92639492e-05]))
E_inf.append(np.array([3.45412180e-01, 1.26784321e-01, 5.60514953e-02, 1.73414706e-02,
       6.09964978e-03, 1.97036584e-03, 5.58711145e-04]))
t_setup.append(np.array([1.89578297e-01, 8.67852243e-01, 3.72747544e+00, 1.60584798e+01,
       7.26796343e+01, 4.31761496e+02, 3.10445716e+03]))
t_solve.append(np.array([3.95891297e-03, 7.28333299e-03, 1.12244280e-02, 2.16142090e-02,
       7.85614200e-02, 3.61005882e-01, 3.33750172e+00]))
labels.append('aligned 1:4')
NX.append(np.array([  2,   4,   8,  16,  32,  64, 128]))
NY.append(4)

# # scipy.sparse.linalg.lsqr (no diagonal preconditioning)
# # QuadraticMapping(0.95, 0.05)
# # NQX = NDX = 8, Qord = 3
# E_2.append(np.array([6.35214050e-02, 2.06042392e-02, 2.92000851e-03, 8.10746432e-04,
#        7.13332481e-04, 9.41543768e-05]))
# E_inf.append(np.array([1.15446805e-01, 4.58063893e-02, 1.63569478e-02, 4.99434024e-03,
#        4.74514978e-03, 5.35647129e-04]))
# t_setup.append(np.array([7.58639161e-01, 3.63395908e+00, 1.69429163e+01, 8.22264779e+01,
#        5.27741515e+02, 3.81616204e+03]))
# t_solve.append(np.array([7.44596403e-03, 1.04098710e-02, 2.09557670e-02, 5.09337870e-02,
#        2.39531398e-01, 1.38012885e+00]))
# labels.append('aligned 1:8')
# NX.append(np.array([  2,   4,   8,  16,  32, 64]))
# NY.append(8)

# scipy.sparse.linalg.lsqr (no diagonal preconditioning)
# QuadraticMapping(0.95, 0.05)
# NQX = NDX = 16, Qord = 3
E_2.append(np.array([1.57370011e-02, 5.62818328e-03, 2.45058916e-03, 3.44192134e-04,
       5.32157118e-05, 2.36291632e-05]))
E_inf.append(np.array([3.50235356e-02, 2.08278605e-02, 7.35151267e-03, 1.35437986e-03,
       2.38589747e-04, 1.99464055e-04]))
t_setup.append(np.array([3.65825985e+00, 1.89858749e+01, 1.21063870e+02, 7.78372482e+02,
       6.47322991e+03, 4.85751577e+04]))
t_solve.append(np.array([3.43625899e-03, 1.79392210e-02, 8.09949220e-02, 1.18058210e-01,
       6.61784298e-01, 6.19807250e+00]))
labels.append('aligned 1:16')
NX.append(np.array([  2,   4,   8,  16,  32,  64]))
NY.append(16)
# # overall convergence order 1.99722126 from np.polynomial.polynomial.polyfit(logN, logE, 1)

# # scipy.sparse.linalg.lsqr (no diagonal preconditioning)
# # new -NDX scheme, seed 1234
# # QuadraticMapping(0.95, 0.05)
# # NQX = 16, NDX = -16, Qord = 3
# E_2.append(np.array([2.51560258e-02, 6.57245107e-03, 1.32464829e-03, 2.21382963e-04,
#        1.29285632e-04, 7.41931103e-05]))
# E_inf.append(np.array([5.43346417e-02, 1.98107519e-02, 3.87856000e-03, 9.44644837e-04,
#        6.98234322e-04, 4.57470767e-04]))
# t_setup.append(np.array([3.43320250e+00, 1.77918734e+01, 1.09893597e+02, 7.40065297e+02,
#        6.27274500e+03, 4.56978290e+04]))
# t_solve.append(np.array([9.90749907e-03, 1.82944880e-02, 4.13602690e-02, 1.27415899e-01,
#        6.38108355e-01, 6.30948775e+00]))
# labels.append('aligned 1:-16')
# NX.append(np.array([ 2,  4,  8, 16, 32, 64]))
# NY.append(16)

# # scipy.sparse.linalg.lsqr (no diagonal preconditioning)
# # new -NDX scheme, seed 666
# # QuadraticMapping(0.95, 0.05)
# # NQX = 16, NDX = -16, Qord = 3
# E_2.append(np.array([2.16668163e-02, 3.97456529e-03, 1.24350650e-03, 2.53100331e-04,
#        1.25346498e-04]))
# E_inf.append(np.array([4.80192986e-02, 1.19667236e-02, 4.06892815e-03, 1.07334052e-03,
#        9.61738147e-04]))
# t_setup.append(np.array([3.50491863e+00, 1.79596183e+01, 1.11989041e+02, 7.49446407e+02,
#        6.60031976e+03]))
# t_solve.append(np.array([3.28885100e-03, 1.82716630e-02, 3.91252079e-02, 1.13084539e-01,
#        6.38108885e-01]))
# labels.append('aligned 1:-16')
# NX.append(np.array([ 2,  4,  8, 16, 32]))
# NY.append(16)

# # scipy.sparse.linalg.lsqr (no diagonal preconditioning)
# # old -NDX scheme
# # QuadraticMapping(0.95, 0.05)
# # NQX = 16, NDX = -16, Qord = 3
# E_2.append(np.array([1.64596436e-02, 3.93167818e-03, 2.53061429e-03, 3.39771766e-04,
#         5.67189727e-05, 2.35502848e-05]))
# E_inf.append(np.array([3.55982462e-02, 1.45360543e-02, 7.44012771e-03, 1.36111488e-03,
#         2.45620924e-04, 1.99262586e-04]))
# t_setup.append(np.array([3.52214511e+00, 1.82474776e+01, 1.14640456e+02, 7.62595968e+02,
#         6.49653711e+03, 4.78427626e+04]))
# t_solve.append(np.array([9.79975396e-03, 1.78325670e-02, 4.08033870e-02, 1.17797782e-01,
#         6.65555478e-01, 6.28807597e+00]))
# labels.append('aligned 1:-16')
# NX.append(np.array([ 2,  4,  8, 16, 32, 64]))
# NY.append(16)
# # overall convergence order 1.95677783 from np.polynomial.polynomial.polyfit(logN, logE, 1)

# # scipy.sparse.linalg.lsqr (no diagonal preconditioning)
# # QuadraticMapping(0.95, 0.05)
# # NQX = 32, NDX = 0, Qord = 2
# E_2.append(np.array([1.73029870e-02, 3.06178064e-03, 1.34694047e-03, 2.30507442e-04,
#         1.5968434e-02]))
# E_inf.append(np.array([2.68872602e-02, 1.37276762e-02, 3.17286845e-03, 9.78016511e-04,
#         6.76976813e-01]))
# t_setup.append(np.array([3.13443866e+00, 1.58049735e+01, 9.47831106e+01, 6.48815792e+02,
#         4.52048285e+03]))
# t_solve.append(np.array([3.39838200e-03, 1.87457300e-02, 4.23197700e-02, 1.16615102e-01,
#         9.61501099e+00]))
# labels.append('aligned 1:16')
# NX.append(np.array([ 2,  4,  8, 16, 32]))
# NY.append(16)


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

##### for double plot #####
fig = plt.figure(figsize=(7.75, 3))
fig.subplots_adjust(hspace=0.5, wspace=0.5)
axL1, axR1 = fig.subplots(1, 2)

# ##### for single plot #####
# fig = plt.figure(figsize=(3.875, 3))
# # fig.subplots_adjust(left=0.2, right=0.85)
# fig.subplots_adjust(left=0.2)
# axL1 = fig.subplots(1, 1)

# ##### plot solution at right, requires a sim object from test #####
# fig = plt.figure(figsize=(7.75, 3))
# fig.subplots_adjust(hspace=0.5, wspace=0.3)
# axL1, axR1 = fig.subplots(1, 2, gridspec_kw={'width_ratios': [1, 1.1]})
# sim.generatePlottingPoints(nx=2, ny=2)
# sim.computePlottingSolution()
# field = axR1.tripcolor(sim.X, sim.Y, sim.U, shading='gouraud', vmin=-1, vmax=1)
# cbar = fig.colorbar(field, ax=axR1)
# cbar.set_ticks(np.linspace(-1,1,5))
# cbar.set_label(r'$u(x,y)$', rotation=0, labelpad=10)
# x = np.linspace(0, sim.nodeX[-1], 100)
# axR1.plot(x, [sim.mapping(np.array([[0, float(yi)]]), i) for i in x], 'k')
# axR1.margins(0,0)
# axR1.set_xticks(np.linspace(0, f.xmax, 6))
# axR1.set_xlabel(r'$x$')
# axR1.set_ylabel(r'$y$', rotation=0, labelpad=10)


axL1.set_prop_cycle(cycler)
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

axR1.set_prop_cycle(cycler)
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
