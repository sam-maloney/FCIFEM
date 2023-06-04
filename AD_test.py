# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 13:47:07 2020

@author: samal
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from timeit import default_timer

import fcifem
# from integrators import *

class slantedSin:
    xmax = 1.
    ymax = 1.
    umax = 1.
    nx = 1
    ny = 1
    theta = np.arctan(nx/ny)
    xfac = 2*np.pi*nx
    yfac = 2*np.pi*ny
    
    def __call__(self, p):
        x = p.reshape(-1,2)[:,0]
        y = p.reshape(-1,2)[:,1]
        return np.sin(self.xfac*x - self.yfac*y)
    
    def solution(self, p):
        return self.__call__(p)

u0 = slantedSin()

# mapping = fcifem.mappings.SinusoidalMapping(0.2, -np.pi/2)
mapping = fcifem.mappings.LinearMapping(u0.nx/u0.ny)
# mapping = fcifem.mappings.StraightMapping()

velocity = np.array([0., 0.])

D_a = 1.
D_i = 0.
theta = u0.theta
diffusivity = D_a*np.array([[np.cos(theta)**2, np.sin(theta)*np.cos(theta)],
                            [np.sin(theta)*np.cos(theta), np.sin(theta)**2]])
diffusivity += D_i*np.eye(2)

dt = 0.01
t_final = 1
nSteps = int(np.rint(t_final/dt))

#%%
perturbation = 0.1
kwargs={
    'mapping' : mapping,
    'velocity' : velocity,
    'diffusivity' : diffusivity,
    'px' : perturbation,
    'py' : perturbation,
    'seed' : 42,
    'xmax' : u0.xmax}

# precon='ilu'
tolerance = 1e-10

# allocate arrays for convergence testing
start = 2
stop = 8
nSamples = np.rint(stop - start + 1).astype('int')
NX_array = np.logspace(start, stop, num=nSamples, base=2, dtype='int')
E_inf = np.empty(nSamples)
E_2 = np.empty(nSamples)
t_setup = np.empty(nSamples)
t_solve = np.empty(nSamples)

uSum = np.empty((nSamples, nSteps))

# loop over N to test convergence where N is the number of
# grid cells along one dimension, each cell forms 2 triangles
# therefore number of nodes equals (N+1)*(N+1)
for iN, NX in enumerate(NX_array):

    start_time = default_timer()
    
    # allocate arrays and compute grid
    NY = 1*NX
    sim = fcifem.FciFemSim(NX, NY, **kwargs)
    sim.setInitialConditions(u0, mapped=False)
    
    print(f'NX = {NX},\tNY = {NY},\tnDoFs = {sim.nDoFs}')

    # vci = 'VCI-C'
    vci = None
    if (vci == 'VCI'):
        sim.computeSpatialDiscretization = sim.computeSpatialDiscretizationLinearVCI
    elif (vci == 'VCI-C'):
        # sim.computeSpatialDiscretization = sim.computeSpatialDiscretizationConservativeLinearVCI
        sim.computeSpatialDiscretization = sim.computeSpatialDiscretizationConservativeLinearVCIold

    # Assemble the stiffness matrix and itialize time-stepping scheme
    NQX = 2*NY//NX
    # NQX = 2
    NQY = NY
    Qord = 2
    sim.computeSpatialDiscretization(NQX=NQX, NQY=NQY, Qord=Qord, quadType='g',
                                     massLumping = False)
    # sim.initializeTimeIntegrator('BE', dt)
    sim.initializeTimeIntegrator('CN', dt)
    # sim.initializeTimeIntegrator('RK', dt, betas=4)
    
    t_setup[iN] = default_timer() - start_time
    print(f'setup time = {t_setup[iN]:.8e} s')
    start_time = default_timer()
    
    # Solve for the approximate solution
    # sim.step(nSteps, tol=tolerance, atol=tolerance)
    for step in range(nSteps):
        uSum[iN, step] = np.sum(sim.u*sim.u_weights)
        sim.step(1, tol=tolerance, atol=tolerance)
    
    t_solve[iN] = default_timer() - start_time
    print(f'solve time = {t_solve[iN]:.8e} s')
    start_time = default_timer()
    
    # compute the analytic solution and error norms
    u_exact = u0.solution(sim.DoFs)
    
    E_inf[iN] = np.linalg.norm(sim.u - u_exact, np.inf)
    E_2[iN] = np.linalg.norm(sim.u - u_exact)/np.sqrt(sim.nDoFs)
        
    print(f'max error  = {E_inf[iN]:.8e}')
    print(f'L2 error   = {E_2[iN]:.8e}')
    print(f'delta_u    = {np.sum((sim.u-sim.u0)*sim.u_weights)}')
    print('', flush=True)
    
##### End of loop over N #####

# print summary
print(f'xmax = {u0.xmax}, {mapping}')
print(f'dt = {dt}, t_final = {t_final}, theta = {u0.theta}')
print(f'px = {kwargs["px"]}, py = {kwargs["py"]}, seed = {kwargs["seed"]}')
print(f'NQX = {NQX}, NQY = {NQY//NY}*NY, Qord = {Qord}')
print(f'VCI: {sim.vci} using {sim.vci_solver}\n')
with np.printoptions(formatter={'float': lambda x: format(x, '.8e')}):
    print('E_2     =', repr(E_2))
    print('E_inf   =', repr(E_inf))
    print('t_setup =', repr(t_setup))
    print('t_solve =', repr(t_solve))

    
#%% Plotting

# open new figure and set parameters
fig = plt.figure()
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

sim.generatePlottingPoints(nx=1, ny=1)
# sim.generatePlottingPoints(nx=10, ny=1)
# sim.generatePlottingPoints(nx=int(NY/NX), ny=1)
# sim.generatePlottingPoints(nx=int(NY/NX), ny=int(NY/NX))
sim.computePlottingSolution()

# vmin = np.min(sim.U)
# vmax = np.max(sim.U)

exactSol = uExactFunc(np.vstack((sim.X,sim.Y)).T)
error = sim.U - exactSol
maxAbsErr = np.max(abs(error))
# maxAbsErr = np.max(abs(sim.u - uExact))
vmin = -maxAbsErr
vmax = maxAbsErr

ax1 = plt.subplot(121)
# plt.title('Absolute Error')
# field = ax1.tripcolor(sim.X, sim.Y, error, shading='gouraud'
#                       ,cmap='seismic', vmin=vmin, vmax=vmax)
plt.title('Final Solution')
field = ax1.tripcolor(sim.X, sim.Y, sim.U, shading='gouraud')
x = np.linspace(0, sim.nodeX[-1], 100)
for yi in [0.0]:
# for yi in [sim.mapping(np.array((x, 0.5)), 0.) for x in sim.nodeX]:
    ax1.plot(x, [sim.mapping(np.array([[0, float(yi)]]), i) for i in x], 'k')
# for xi in sim.nodeX:
#     ax1.plot([xi, xi], [0, 1], 'k:')
# ax.plot(sim.X[np.argmax(sim.U)], sim.Y[np.argmax(sim.U)],
#   'g+', markersize=10)
plt.ylim((0., 1.))
# cbar = plt.colorbar(field, format='%.0e')
cbar = plt.colorbar(field)
cbar.formatter.set_powerlimits((0, 0))
plt.xlabel(r'$x$')
plt.ylabel(r'$y$', rotation=0)
if abs(u0.xmax - 2*np.pi) < 1e-10:
    plt.xticks(np.linspace(0, u0.xmax, 5),
        ['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
#  plt.xticks(np.linspace(0, 2*np.pi, 7),
#      ['0',r'$\pi/3$',r'$2\pi/3$',r'$\pi$',r'$4\pi/3$',r'$5\pi/3$',r'$2\pi$'])
else:
    plt.xticks(np.linspace(0, u0.xmax, 6))
plt.margins(0,0)

# plot the error convergence
axR1 = plt.subplot(122)
plt.loglog(NX_array, E_inf, '.-', label=r'$E_\infty$ magnitude')
plt.loglog(NX_array, E_2, '.-', label=r'$E_2$ magnitude')
plt.minorticks_off()
plt.xticks(NX_array, NX_array)
plt.xlabel(r'$NX$')
plt.ylabel(r'Magnitude of Error Norm')

# plot the intra-step order of convergence
axR2 = axR1.twinx()
logN = np.log(NX_array)
logE_inf = np.log(E_inf)
logE_2 = np.log(E_2)
order_inf = (logE_inf[0:-1] - logE_inf[1:])/(logN[1:] - logN[0:-1])
order_2 = (logE_2[0:-1] - logE_2[1:])/(logN[1:] - logN[0:-1])
intraN = np.logspace(start+0.5, stop-0.5, num=nSamples-1, base=2.0)
plt.plot(intraN, order_inf, '.:', linewidth=1, label=r'$E_\infty$ order')
plt.plot(intraN, order_2, '.:', linewidth=1, label=r'$E_2$ order')
plt.plot(plt.xlim(), [2, 2], 'k:', linewidth=1)#, label='Expected')
plt.ylim(0, 5)
plt.yticks(np.linspace(0,5,6))
# plt.ylim(0, 3)
# plt.yticks([0, 0.5, 1, 1.5, 2, 2.5, 3])
plt.ylabel(r'Intra-step Order of Convergence')
lines, labels = axR1.get_legend_handles_labels()
lines2, labels2 = axR2.get_legend_handles_labels()
axR2.legend(lines + lines2, labels + labels2, loc='best')
plt.title('Convergence')
plt.margins(0,0)

filename = mapping.name
if (mapping.name != 'straight'):
    filename += f'_{int(NY/NX)}N'
filename += '_' + [f'p{perturbation}', 'uniform'][perturbation == 0.]
if (vci is not None):
    filename += '_' + vci

# plt.sca(ax1)
# plt.savefig(filename + '.pdf', bbox_inches = 'tight', pad_inches = 0)

#%% Animation

# sim.generatePlottingPoints(nx=5, ny=2)

# # maxAbsU = np.max(np.abs(sim.U))
# maxAbsU = 1.

# def init_plot():
#     global field, fig, ax, sim, maxAbsU
#     fig, ax = plt.subplots()
#     field = ax.tripcolor(sim.X, sim.Y, sim.U, shading='gouraud'
#                           ,cmap='seismic', vmin=-maxAbsU, vmax=maxAbsU
#                           )
#     # tri = mpl.tri.Triangulation(sim.X,sim.Y)
#     # ax.triplot(tri, 'r-', lw=1)
#     x = np.linspace(0, sim.nodeX[-1], 100)
#     for yi in [0.]:
#         ax.plot(x, [mapping(np.array([[0, yi]]), i) for i in x], 'k')
#     for xi in sim.nodeX:
#         ax.plot([xi, xi], [0, 1], 'k:')
#     # ax.plot(sim.X[np.argmax(sim.U)], sim.Y[np.argmax(sim.U)],
#     #   'g+', markersize=10)
#     plt.colorbar(field)
#     plt.xlabel(r'$x$')
#     plt.ylabel(r'$y$', rotation=0)
#     # plt.xticks(np.linspace(0, 2*np.pi, 7), 
#     #     ['0',r'$\pi/3$',r'$2\pi/3$',r'$\pi$',r'$4\pi/3$',r'$5\pi/3$',r'$2\pi$'])
#     plt.margins(0,0)
#     return [field]

# init_plot()

# field = ax.tripcolor(sim.X, sim.Y, sim.U, shading='gouraud'
#                           ,cmap='seismic', vmin=-maxAbsU, vmax=maxAbsU
#                           )

# def animate(i):
#     global field, sim
#     sim.step(1, tol=tolerance, atol=tolerance)
#     sim.computePlottingSolution()
#     field.set_array(sim.U)
#     plt.title(f"t = {sim.integrator.timestep}")
#     return [field]

# ani = animation.FuncAnimation(
#     fig, animate, frames=nSteps, interval=15)

# # ani.save('movie.mp4', writer='ffmpeg', dpi=200)

# # Advection only, straight mapping, RK4, no mass-lumping
# # v=[0., 0.1], dt = 0.005, t_final = 1, uniform grid, Nquad=5, NY = 4*NX
# NX_array = np.array([  4,   8,  16,  32,  64, 128, 256])
# E_2 = np.array([3.03300012e-02, 1.58515718e-03, 8.72205133e-05, 3.51958541e-05,
#        9.81484218e-06, 2.51775587e-06, 6.33345658e-07])
# E_inf = np.array([1.32846260e-01, 9.11919350e-03, 4.40463595e-04, 1.89907377e-04,
#        5.38855119e-05, 1.39030698e-05, 3.50039920e-06])
