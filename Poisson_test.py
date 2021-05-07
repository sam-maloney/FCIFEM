# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 13:47:07 2020

@author: samal
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as sp_la

import fcifem

from timeit import default_timer

# mapping = fcifem.SinusoidalMapping(0.2, -np.pi/2)
mapping = fcifem.StraightMapping()
        
def f(p):
    p.shape = (-1,2)
    return np.sin(p[:,0])*np.sin(2*np.pi*p[:,1])

uExactFunc = lambda p : (1/(1+4*np.pi**2))*f(p)

kwargs={
    'mapping' : mapping,
    'dt' : 1.,
    'u0' : f,
    'velocity' : np.array([0., 0.]),
    'diffusivity' : 1., # Makes diffusivity matrix K into Poisson operator
    'px' : 0.1,
    'py' : 0.1,
    'seed' : 42 }

# allocate arrays for convergence testing
start = 2
stop = 8
nSamples = stop - start + 1
NX_array = np.logspace(start, stop, num=nSamples, base=2, dtype='int32')
E_inf = np.empty(nSamples, dtype='float64')
E_2 = np.empty(nSamples, dtype='float64')

# loop over N to test convergence where N is the number of
# grid cells along one dimension, each cell forms 2 triangles
# therefore number of nodes equals (N+1)*(N+1)
for iN, NX in enumerate(NX_array):
    
    start_time = default_timer()
    
    NY = NX

    # allocate arrays and compute grid
    sim = fcifem.FciFemSim(NX, NY, **kwargs)
    
    print(f'NX = {NX},\tNY = {NY},\tnNodes = {sim.nNodes}')
    
    # Assemble the mass matrix and forcing term
    sim.computeSpatialDiscretization(f, NQX=6, NQY=NY, Qord=3, quadType='g',
                                     massLumping=False)
    
    print(f'setup time = {default_timer()-start_time} s')
    
    # sim.u = sp_la.spsolve(sim.K, sim.b)
    sim.u, info = sp_la.lgmres(sim.K, sim.b, tol=1e-10, atol=1e-10)
    
    # compute the analytic solution and error norms
    uExact = uExactFunc(sim.nodes)
    
    E_inf[iN] = np.linalg.norm(sim.u - uExact, np.inf)
    E_2[iN] = np.linalg.norm(sim.u - uExact)/np.sqrt(sim.nNodes)
    
    print(f'max error = {E_inf[iN]}')
    print(f'L2 error  = {E_2[iN]}\n')
    
##### Begin Plotting Routines #####

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

sim.generatePlottingPoints(1,1)
sim.computePlottingSolution()

vmin = np.min(sim.U)
vmax = np.max(sim.U)

exactSol = uExactFunc(np.vstack((sim.X,sim.Y)).T)
error = sim.U - exactSol
# maxAbsErr = np.max(np.abs(error))
maxAbsErr = np.max(np.abs(sim.u - uExact))
vmin = -maxAbsErr
vmax = maxAbsErr

ax1 = plt.subplot(121)
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

# # Sinusoidal mapping, Uniform spacing
# E_2 = np.array([1.51377118e-04, 1.51345163e-03, 4.27174201e-04, 1.10044134e-04,
#        2.75552544e-05, 6.87046871e-06, 1.72329954e-06])
# E_inf = np.array([3.02754236e-04, 2.54824976e-03, 7.38828574e-04, 2.04472060e-04,
#        5.20249246e-05, 1.30128704e-05, 3.31977042e-06])

# # Sinusoidal mapping, 0.1 perturbation
# E_2 = np.array([1.17859899e-03, 1.54709326e-03, 4.48683841e-04, 1.18334966e-04,
#         3.39448878e-05, 1.19400390e-05, 6.06793906e-06])
# E_inf = np.array([2.35722961e-03, 3.07975031e-03, 8.77070705e-04, 2.78394498e-04,
#        1.11023827e-04, 5.50049087e-05, 3.19412273e-05])

# # Straight mapping, Uniform spacing
# E_2 = np.array([2.66565612e-03, 6.47088425e-04, 1.59542252e-04, 3.97356716e-05,
#        9.92440295e-06, 2.48050383e-06, 6.20088618e-07])
# E_inf = np.array([5.33131225e-03, 1.29417685e-03, 3.19084503e-04, 7.94713431e-05,
#        1.98488059e-05, 4.96100765e-06, 1.24017724e-06])

# # Straight mapping, 0.1 perturbation
# E_2 = np.array([0.00285971, 0.00103294, 0.00055518, 0.00027292, 0.00017557,
#        0.00014189, 0.00013756])
# E_inf = np.array([0.00627925, 0.00310992, 0.00187384, 0.00104082, 0.00057817,
#        0.0004272 , 0.00031896])