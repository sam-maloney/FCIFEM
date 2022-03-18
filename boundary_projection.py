# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 13:47:07 2020

@author: Samuel A. Maloney

"""

print("\n!!!!! NEED TO CHANGE COMPUTATION OF B VECTOR FOR THIS TO WORK !!!!!\n")

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as sp_la

import fcifem

class QuadraticTestProblem:
    xmax = 1.
    ymax = 1.
    n = 3
    N = (2*np.pi/ymax)*n
    b = 0.05
    # define a such that (0, 0) maps to (xmax, 1) for given b and xmax
    a = (1 - b*xmax)/xmax**2
    
    umax = xmax
    dudyMax = N*xmax
    dudxMax = 1 + 2*a*N*xmax**2 + b*N*xmax
    
    def __call__(self, p):
        x = p.reshape(-1,2)[:,0]
        y = p.reshape(-1,2)[:,1]
        N = self.N
        a = self.a
        b = self.b
        return N*(N*x*(4*a**2*x**2 + 4*a*b*x + b**2 + 1)*np.sin(N*(y - a*x**2 - b*x))
                  + 2*(3*a*x + b)*np.cos(N*(y - a*x**2 - b*x)))
    
    def solution(self, p):
        x = p.reshape(-1,2)[:,0]
        y = p.reshape(-1,2)[:,1]
        return x*np.sin(2*np.pi*self.n*(y - self.a*x**2 - self.b*x))
        
f = QuadraticTestProblem()
duRatio = f.dudyMax / f.dudxMax

# mapping = fcifem.mappings.StraightMapping()
mapping = fcifem.mappings.QuadraticMapping(f.a, f.b)

perturbation = 0.1
kwargs={
    'mapping' : mapping,
    'dt' : 1.,
    'velocity' : np.array([0., 0.]),
    'diffusivity' : 0.,
    'px' : perturbation,
    'py' : perturbation,
    'seed' : 42,
    'xmax' : f.xmax }

# allocate arrays for convergence testing
start = 1
stop = 7
nSamples = stop - start + 1
NX_array = np.logspace(start, stop, num=nSamples, base=2, dtype='int32')
E_inf = np.empty(nSamples, dtype='float64')
E_2 = np.empty(nSamples, dtype='float64')
NYratio = 1
# NYratio = np.rint(f.dudyMax / f.xmax).astype('int')

# loop over N to test convergence where N is the number of
# grid cells along one dimension, each cell forms 2 triangles
# therefore number of nodes equals (N+1)*(N+1)
for iN, NX in enumerate(NX_array):
    
    NY = NYratio * NX
    NDX = 1
    # NDX = max(np.rint(f.xmax*NYratio / duRatio).astype('int'), 1)

    # allocate arrays and compute grid
    sim = fcifem.FciFemSim(NX, NY, **kwargs)
    BC = fcifem.boundaries.DirichletBoundary(sim, f.solution, NDX=NDX)
    sim.setInitialConditions(np.zeros(BC.nDoFs), mapped=False, BC=BC)
    
    print(f'NX = {NX}, \tNY = {NY}, \tnDoFs = {sim.nDoFs}')
    
    # Assemble the mass matrix and forcing term
    # if NDX == 1:
    #     Qord = 2
    # else:
    #     Qord = 1
    Qord = 2
    sim.computeSpatialDiscretization(f.solution, NQX=NDX, NQY=NY, Qord=Qord,
                                     quadType='g', massLumping=False)
    
    sim.u = sp_la.spsolve(sim.M, sim.b)
    # tolerance = 1e-10
    # sim.u, info = sp_la.lgmres(sim.M, sim.b, tol=tolerance, atol=tolerance)
    
    # compute the analytic solution and error norms
    u_exact = f.solution(sim.DoFs)
    
    E_inf[iN] = np.linalg.norm(sim.u - u_exact, np.inf)
    E_2[iN] = np.linalg.norm(sim.u - u_exact)/np.sqrt(sim.nDoFs)
    
    print(f'max error = {E_inf[iN]}')
    print(f'L2 error  = {E_2[iN]}\n')

#%% Plotting

# clear the current figure, if opened, and set parameters
fig = plt.figure(figsize=(7.75, 3))
fig.subplots_adjust(hspace=0.3, wspace=0.3)

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

sim.generatePlottingPoints(nx=1, ny=1)
# sim.generatePlottingPoints(nx=int(NY/NX), ny=1)
sim.computePlottingSolution()

u_plot = np.sum(sim.phiPlot * sim.u[sim.indPlot], axis=1)

# maxAbsU = np.max(np.abs(u_plot))
vmin = np.min((np.min(u_plot), np.min(sim.U)))
vmax = np.max((np.max(u_plot), np.max(sim.U)))

exactSol = sim.f(np.vstack((sim.X,sim.Y)).T)
error = sim.U - exactSol
maxAbsErr = np.max(np.abs(error))
vmin = -maxAbsErr
vmax = maxAbsErr

ax1 = plt.subplot(121)
# field = ax1.tripcolor(sim.X, sim.Y, error, shading='gouraud'
#                       ,cmap='seismic', vmin=vmin, vmax=vmax)
field = ax1.tripcolor(sim.X, sim.Y, sim.U, shading='gouraud')
x = np.linspace(0, sim.nodeX[-1], 100)
for yi in [0.0]:
    ax1.plot(x, [sim.mapping(np.array([[0, yi]]), i) for i in x], 'k')
# for xi in sim.nodeX:
#     ax1.plot([xi, xi], [0, 1], 'k:')
# ax.plot(sim.X[np.argmax(sim.U)], sim.Y[np.argmax(sim.U)],
#   'g+', markersize=10)
# cbar = plt.colorbar(field, format='%.0e')
cbar = plt.colorbar(field)
cbar.formatter.set_powerlimits((0, 0))
plt.xlabel(r'$x$')
plt.ylabel(r'$y$', rotation=0)
if abs(f.xmax - 2*np.pi) < 1e-10:
    plt.xticks(np.linspace(0, f.xmax, 5),
        ['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
#  plt.xticks(np.linspace(0, 2*np.pi, 7), 
#      ['0',r'$\pi/3$',r'$2\pi/3$',r'$\pi$',r'$4\pi/3$',r'$5\pi/3$',r'$2\pi$'])
else:
    plt.xticks(np.linspace(0, f.xmax, 6))
    ax1.set_aspect('equal')
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
plt.ylabel(r'Intra-step Order of Convergence')
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='best')
plt.margins(0,0)

# plt.savefig("CD_MassLumped_RK4.pdf", bbox_inches = 'tight', pad_inches = 0)
