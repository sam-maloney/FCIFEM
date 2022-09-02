# -*- coding: utf-8 -*-
"""
@author: Samuel A. Maloney

"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as sp_la

import fcifem

from timeit import default_timer

# mapping = fcifem.mappings.SinusoidalMapping(0.2, -np.pi/2)
mapping = fcifem.mappings.StraightMapping()

class sinXY:
    D = 0.01 # diffusivity constant (anisotropic)
    n = 2
    Lx = 2*np.pi
    Ly = 1.
    coef = -D*(n*np.pi)**2*(1/Lx**2 + 1/Ly**2)
    
    def __call__(self, p, t=0.):
        p.shape = (-1,2)
        return np.sin(p[:,0])*np.sin(2*np.pi*p[:,1])*np.exp(self.coef*t)

u = sinXY()

dt = 0.005

# t_final = 0.1
# nSteps = int(np.rint(t_final/dt))

nSteps = 100
t_final = dt*nSteps

kwargs={
    'mapping' : mapping,
    'dt' : dt,
    'velocity' : np.array([0., 0.]),
    'diffusivity' : u.D,
    'px' : 0.,
    'py' : 0.,
    'seed' : 42 }

tolerance = 1e-10

# allocate arrays for convergence testing
start = 2
stop = 5
nSamples = stop - start + 1
NX_array = np.logspace(start, stop, num=nSamples, base=2, dtype='int32')
E_inf = np.empty(nSamples, dtype='float')*np.nan
E_2 = np.empty(nSamples, dtype='float')*np.nan

# loop over N to test convergence where N is the number of
# grid cells along one dimension, each cell forms 2 triangles
# therefore number of nodes equals (N+1)*(N+1)
for iN, NX in enumerate(NX_array):
    
    start_time = default_timer()
    
    NY = NX

    # allocate arrays and compute grid
    sim = fcifem.FciFemSim(NX, NY, **kwargs)
    sim.setInitialConditions(u, mapped=False)
    
    print(f'NX = {NX},\tNY = {NY},\tnNodes = {sim.nNodes}')
    
    # Assemble the stiffness matrix and itialize time-stepping scheme
    sim.computeSpatialDiscretization(NQX=6, NQY=NY, Qord=3, quadType='g',
                                     massLumping=False)
    # sim.initializeTimeIntegrator('BE', dt)
    sim.initializeTimeIntegrator('CN', dt)
    # sim.initializeTimeIntegrator('RK', dt, betas=4)
    
    print(f'setup time = {default_timer()-start_time} s')
    
    # Solve for the approximate solution
    sim.step(nSteps, tol=tolerance, atol=tolerance)
    
    print(f'solution time = {default_timer()-start_time} s')
    start_time = default_timer()
    
    # compute the analytic solution and error norms
    uExact = u(sim.nodes, sim.integrator.time)
    
    E_inf[iN] = np.linalg.norm(sim.u - uExact, np.inf)
    E_2[iN] = np.linalg.norm(sim.u - uExact)/np.sqrt(sim.nNodes)
    
    print(f'analysis time = {default_timer()-start_time} s')
    print(f'max error = {E_inf[iN]}')
    print(f'L2 error  = {E_2[iN]}\n')
    
##### Begin Plotting Routines #####

sim.generatePlottingPoints(nx=1, ny=1)
sim.computePlottingSolution()

vmin = np.min(sim.U)
vmax = np.max(sim.U)

exactSol = u(np.vstack((sim.X,sim.Y)).T, sim.integrator.time)
error = sim.U - exactSol
# maxAbsErr = np.max(np.abs(error))
maxAbsErr = np.max(np.abs(sim.u - uExact))
vmin = -maxAbsErr
vmax = maxAbsErr

# clear the current figure, if opened, and set parameters
fig = plt.gcf()
fig.clf()
fig.set_size_inches(7.75,3)
plt.subplots_adjust(hspace = 0.3, wspace = 0.3)

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

ax1 = plt.subplot(121)
# field = ax1.tripcolor(sim.nodes[:,0], sim.nodes[:,1], sim.u0, shading='gouraud'
#                      ,cmap='seismic', vmin=-np.max(np.abs(sim.u0)), vmax=np.max(np.abs(sim.u0))
#                      )

# field = ax1.tripcolor(sim.X, sim.Y, error, shading='gouraud'
field = ax1.tripcolor(sim.nodes[:,0], sim.nodes[:,1], sim.u - uExact, shading='gouraud'
                      ,cmap='seismic', vmin=vmin, vmax=vmax
                      )
x = np.linspace(0, sim.nodeX[-1], 100)
for yi in [0.4, 0.5, 0.6]:
    ax1.plot(x, [mapping(np.array([[0, yi]]), i) for i in x], 'k')
# for xi in sim.nodeX:
#     ax1.plot([xi, xi], [0, 1], 'k:')
# ax.plot(sim.X[np.argmax(sim.U)], sim.Y[np.argmax(sim.U)],
#   'g+', markersize=10)
# cbar = plt.colorbar(field, format='%.0e')
cbar = plt.colorbar(field)
cbar.formatter.set_powerlimits((0, 0))
plt.xlabel(r'$x$')
plt.ylabel(r'$y$', rotation=0)
plt.xticks(np.linspace(0, 2*np.pi, 7), 
    ['0',r'$\pi/3$',r'$2\pi/3$',r'$\pi$',r'$4\pi/3$',r'$5\pi/3$',r'$2\pi$'])
plt.margins(0,0)

# plot the error convergence
ax1 = plt.subplot(122)
plt.loglog(NX_array, E_inf, '.-', label=r'$E_\infty$ magnitude')
plt.loglog(NX_array, E_2, '.-', label=r'$E_2$ magnitude')
plt.minorticks_off()
plt.xticks(NX_array, NX_array)
plt.xlabel(r'$NX$')
plt.ylabel(r'Magnitude of Error Norm')

# plot the intra-step order of convergence
ax2 = ax1.twinx()
logN = np.log(NX_array)
logE_inf = np.log(E_inf)
logE_2 = np.log(E_2)
order_inf = (logE_inf[0:-1] - logE_inf[1:])/(logN[1:] - logN[0:-1])
order_2 = (logE_2[0:-1] - logE_2[1:])/(logN[1:] - logN[0:-1])
intraN = np.logspace(start+0.5, stop-0.5, num=nSamples-1, base=2.0)
plt.plot(intraN, order_inf, '.:', linewidth=1, label=r'$E_\infty$ order')
plt.plot(intraN, order_2, '.:', linewidth=1, label=r'$E_2$ order')
plt.plot(plt.xlim(), [2, 2], 'k:', linewidth=1, label='Expected')
plt.ylim(0, 5)
plt.yticks(np.linspace(0,5,6))
# plt.ylim(0, 3)
# plt.yticks([0, 0.5, 1, 1.5, 2, 2.5, 3])
plt.ylabel(r'Intra-step Order of Convergence')
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='best')
plt.margins(0,0)

# plt.savefig(f"CD_{kwargs['px']}px_{kwargs['py']}py_notMassLumped_RK4.pdf",
#     bbox_inches = 'tight', pad_inches = 0)

# plt.savefig("CD_MassLumped_RK4.pdf",
#     bbox_inches = 'tight', pad_inches = 0)

# For all of the below
# NX_array = np.array([  4,   8,  16,  32,  64, 128, 256])
# NY = NX, NQY=NY, NQX=6, Qord=3, quadType='g', massLumping=False
# D = 0.01, n = 2, dt = 0.005, nSteps = 100

# # Straight mapping, Uniform spacing, CN
# E_2 = np.array([1.74572355e-02, 4.30716118e-03, 1.06620253e-03, 2.65830623e-04,
#        6.64313804e-05, 1.66260242e-05, 4.17748043e-06])
# E_inf = np.array([3.49144717e-02, 8.61432242e-03, 2.13240597e-03, 5.31662194e-04,
#        1.32863867e-04, 3.32525489e-05, 8.35496139e-06])

# # Straight mapping, 0.1 perturbation, CN
# E_2 = np.array([0.01873866, 0.01795057, 0.01439723, 0.00723706, 0.00369697,
#        0.00200346, 0.00128445])
# E_inf = np.array([0.04203476, 0.04831195, 0.05343568, 0.02944521, 0.01607812,
#        0.00946864, 0.0049019 ])

# # Sinusoidal mapping, Uniform spacing, CN
# E_2 = np.array([2.00401406e-02, 5.33575655e-03, 1.16170918e-03, 2.72763750e-04,
#        7.38494580e-05, 1.78124482e-05, 4.46183625e-06])
# E_inf = np.array([4.00802813e-02, 1.22987430e-02, 2.75653138e-03, 6.09220904e-04,
#        1.98373607e-04, 4.47375699e-05, 1.42873479e-05])

# # Sinusoidal mapping, 0.1 perturbation, CN
# E_2 = np.array([0.02097072, 0.00730126, 0.0023875 , 0.00101677, 0.00050876,
#        0.00024956, 0.00013334])
# E_inf = np.array([0.0458054 , 0.01557467, 0.01064567, 0.00574506, 0.00302401,
#        0.00162349, 0.00095905])


# # Straight mapping, Uniform spacing, BE
# E_2 = np.array([1.73389991e-02, 4.21558915e-03, 9.80687596e-04, 1.81780054e-04,
#         1.72563068e-05, 6.69711442e-05, 7.93970709e-05])
# E_inf = np.array([3.46779982e-02, 8.43117835e-03, 1.96137632e-03, 3.63560680e-04,
#        3.45127898e-05, 1.33944003e-04, 1.58794142e-04])

# # Straight mapping, 0.1 perturbation, BE
# E_2 = np.array([0.01862176, 0.01787337, 0.01437683, 0.00724176, 0.00371535,
#        0.00203912, 0.00134224])
# E_inf = np.array([0.04176515, 0.04816958, 0.05344933, 0.02945702, 0.01612762,
#        0.00957861, 0.00501286])

# # Sinusoidal mapping, Uniform spacing, BE
# E_2 = np.array([1.99162751e-02, 5.24997110e-03, 1.08338805e-03, 2.01269554e-04,
#         4.82423543e-05, 6.90015631e-05, 7.98216896e-05])
# E_inf = np.array([0.03983255, 0.0121076 , 0.00258398, 0.00044398, 0.00014672,
#        0.00013867, 0.00015976])

# # Sinusoidal mapping, 0.1 perturbation, BE
# E_2 = np.array([0.02084625, 0.00722626, 0.00234869, 0.001001  , 0.00050881,
#        0.00026392, 0.00016579])
# E_inf = np.array([0.04554314, 0.01546252, 0.01050661, 0.00577639, 0.00294628,
#        0.00157045, 0.00099328])


# # Straight mapping, Uniform spacing, RK4
# E_2 = np.array([1.74571870e-02, 4.30712865e-03, 1.06617329e-03, 2.65802154e-04,
#        6.64031024e-05,            np.nan,            np.nan])
# E_inf = np.array([0.03491437, 0.00861426, 0.00213235, 0.0005316 , 0.00013281,
#               np.nan,        np.nan])

# # Straight mapping, 0.1 perturbation, RK4
# E_2 = np.array([0.01873859, 0.01795037, 0.01439714, 0.00723705, 0.00369697,
#               np.nan,        np.nan])
# E_inf = np.array([0.04203459, 0.04831139, 0.05343539, 0.02944513, 0.01607814,
#               np.nan,        np.nan])

# # Sinusoidal mapping, Uniform spacing, RK4
# E_2 = np.array([2.00400884e-02, 5.33572515e-03, 1.16168212e-03, 2.72738074e-04,
#        1.47396008e+02,            np.nan,            np.nan])
# E_inf = np.array([4.00801768e-02, 1.22986722e-02, 2.75647020e-03, 6.09162502e-04,
#        6.23178476e+02,            np.nan,            np.nan])

# # Sinusoidal mapping, 0.1 perturbation, RK4
# E_2 = np.array([2.09706601e-02, 7.30121324e-03, 2.38747933e-03, 1.01675822e-03,
#        3.02202430e+10,            np.nan,            np.nan])
# E_inf = np.array([4.58052716e-02, 1.55745490e-02, 1.06455571e-02, 5.74505662e-03,
#        2.78259224e+11,            np.nan,            np.nan])