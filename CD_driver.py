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
from integrators import *

mapping = fcifem.mappings.SinusoidalMapping(0.2, -np.pi/2)
# mapping = fcifem.mappings.StraightMapping()

# velocity = np.array([1., 0.])
# velocity = np.array([0., 0.1])
velocity = np.array([2*np.pi, 0.1])
# velocity = np.array([0., 0.])

D_a = 0.
D_i = 0.
theta = np.pi
diffusivity = D_a*np.array([[np.cos(theta)**2, np.sin(theta)*np.cos(theta)],
                            [np.sin(theta)*np.cos(theta), np.sin(theta)**2]])
diffusivity += D_i*np.eye(2)

dt = 0.005
t_final = 0.005
# nSteps = int(np.rint(np.sum(dx[0:3])/dt))
nSteps = int(np.rint(t_final/dt))

##### Initial conditions #####

# Sinusoid in X and Gaussian in Y
# max(du/dy) = A*exp(-1/2)/sigmay
# max(du/dx) = A/2
# ratio = 2*exp(-1/2)/sigmay
class sinXgaussY:
    amplitude = 1.0
    ry = 0.4
    sigmay = 0.05
    pi_2 = 0.5*np.pi
    
    def __call__(self, p):
        p.shape = (-1,2)
        return 0.5 * self.amplitude * (np.sin(p[:,0] + self.pi_2) + 1) \
            * np.exp(-0.5*((p[:,1] - self.ry)/self.sigmay)**2)

# Gaussian in X and Y
class gaussXY:
    amplitude = 1.0
    rx = np.pi
    ry = 0.4
    sigmax = 1.
    sigmay = 0.1
    
    def __call__(self, p):
        return self.amplitude*np.exp(-0.5*(((p[:,0] - self.rx)/self.sigmax)**2
                                      + ((p[:,1] - self.ry)/self.sigmay)**2))
        
u0 = sinXgaussY()
# Exact solution function for convection only simulation
uExactFunc = lambda p: u0( (p -t_final*velocity)
                         % np.array([sim.nodeX[-1], sim.nodeY[-1,-1]]))

# f = lambda x: g(x, True)

kwargs={
    'mapping' : mapping,
    # 'u0' : u0,
    'velocity' : velocity,
    'diffusivity' : diffusivity,
    'px' : 0.0,
    'py' : 0.0,
    'seed' : 42 }

# precon='ilu'
tolerance = 1e-10

# allocate arrays for convergence testing
start = 4
stop = 4
nSamples = stop - start + 1
NX_array = np.logspace(start, stop, num=nSamples, base=2, dtype='int32')
E_inf = np.empty(nSamples, dtype='float64')
E_2 = np.empty(nSamples, dtype='float64')

# loop over N to test convergence where N is the number of
# grid cells along one dimension, each cell forms 2 triangles
# therefore number of nodes equals (N+1)*(N+1)
for iN, NX in enumerate(NX_array):

    start_time = default_timer()

    NY = 4*NX
    
    # allocate arrays and compute grid
    sim = fcifem.FciFemSim(NX, NY, **kwargs)
    sim.setInitialConditions(u0)
    
    print(f'NX = {NX},\tNY = {NY},\tnNodes = {sim.nDoFs}')

    # Assemble the stiffness matrix and itialize time-stepping scheme
    sim.computeSpatialDiscretization(NQX=6, NQY=NY, Qord=3, quadType='g',
                                     massLumping = False)
    # sim.initializeTimeIntegrator('BE', dt)
    sim.initializeTimeIntegrator('CN', dt)
    # sim.initializeTimeIntegrator('RK', dt, betas=4)
    
    print(f'setup time = {default_timer()-start_time} s')
    start_time = default_timer()
    
    # Solve for the approximate solution
    sim.step(nSteps, tol=tolerance, atol=tolerance)
    
    print(f'solution time = {default_timer()-start_time} s')
    start_time = default_timer()
    
    # compute the analytic solution and error norms
    u_exact = uExactFunc(sim.DoFsMapped)
    
    E_inf[iN] = np.linalg.norm(sim.u - u_exact, np.inf)
    E_2[iN] = np.linalg.norm(sim.u - u_exact)/np.sqrt(sim.nDoFs)
    
    # end_time = default_timer()
    
    # print('Condition Number =', sim.cond('fro'))
    
    print(f'analysis time = {default_timer()-start_time} s')
    start_time = default_timer()
    
    print(f'max error = {E_inf[iN]}')
    print(f'L2 error  = {E_2[iN]}\n')
    # print(f'Elapsed time = {end_time-start_time} s\n')
    
##### End of loop over N #####

print(f'min(E_inf) = {np.min(E_inf)}')
print(f'min(E_2)   = {np.min(E_2)}')

# minMax = np.empty((nSteps+1, 2))
# minMax[0] = [0., 1.]
# U_sum = []
# error = []

#     for i in range(nSteps):
#         sim.step(1)
#         # minMax[i+1] = [np.min(u), np.max(u)]
#         U_sum.append(np.sum(u*u_weights))
#         error.append(np.linalg.norm(u - exact_solution))
    
# ##### Begin Plotting Routines #####

# # clear the current figure, if opened, and set parameters
# fig = plt.gcf()
# fig.clf()
# fig.set_size_inches(7.75,3)
# plt.subplots_adjust(hspace = 0.3, wspace = 0.3)

# # SMALL_SIZE = 7
# # MEDIUM_SIZE = 8
# # BIGGER_SIZE = 10
# # plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
# # plt.rc('axes', titlesize=MEDIUM_SIZE)    # fontsize of the axes title
# # plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
# # plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# # plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# # plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
# # plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# # # plot the result
# # plt.subplot(121)
# # plt.tripcolor(sim.nodes[:,0], sim.nodes[:,1], sim.u, shading='gouraud')
# # plt.colorbar()
# # plt.xlabel(r'$x$')
# # plt.ylabel(r'$y$', rotation=0)
# # # plt.title('Final MLS solution')
# # plt.margins(0,0)

# # # plot analytic solution
# # plt.subplot(222)
# # plt.tripcolor(sim.nodes[:,0], sim.nodes[:,1], u_exact, shading='gouraud')
# # plt.colorbar()
# # plt.xlabel(r'$x$')
# # plt.ylabel(r'$y$')
# # # plt.title('Analytic solution')
# # plt.margins(0,0)

# # # plot error
# difference = sim.u - u_exact
# plt.subplot(121)
# plt.tripcolor(sim.nodes[:,0], sim.nodes[:,1], difference,
#               shading='gouraud',
#               cmap='seismic',
#               vmin=-np.max(np.abs(difference)),
#               vmax=np.max(np.abs(difference)))
# plt.xlim(sim.nodeX[0], sim.nodeX[-1])
# plt.ylim(sim.nodeY[0,0], sim.nodeY[-1,-1])
# plt.colorbar()
# plt.xlabel(r'$x$')
# plt.ylabel(r'$y$', rotation=0)
# plt.xticks(np.linspace(0, 2*np.pi, 7), 
#         ['0',r'$\pi/3$',r'$2\pi/3$',r'$\pi$',r'$4\pi/3$',r'$5\pi/3$',r'$2\pi$'])
# # plt.title('Error')
# plt.margins(0,0)

# # plot the error convergence
# ax1 = plt.subplot(122)
# plt.loglog(NX_array, E_inf, '.-', label=r'$E_\infty$ magnitude')
# plt.loglog(NX_array, E_2, '.-', label=r'$E_2$ magnitude')
# plt.minorticks_off()
# plt.xticks(NX_array, NX_array)
# plt.xlabel(r'$NX$')
# plt.ylabel(r'Magnitude of Error Norm')

# # plot the intra-step order of convergence
# ax2 = ax1.twinx()
# logN = np.log(NX_array)
# logE_inf = np.log(E_inf)
# logE_2 = np.log(E_2)
# order_inf = (logE_inf[0:-1] - logE_inf[1:])/(logN[1:] - logN[0:-1])
# order_2 = (logE_2[0:-1] - logE_2[1:])/(logN[1:] - logN[0:-1])
# intraN = np.logspace(start+0.5, stop-0.5, num=nSamples-1, base=2.0)
# plt.plot(intraN, order_inf, '.:', linewidth=1, label=r'$E_\infty$ order')
# plt.plot(intraN, order_2, '.:', linewidth=1, label=r'$E_2$ order')
# plt.plot(plt.xlim(), [2, 2], 'k:', linewidth=1, label='Expected')
# plt.ylim(0, 5)
# plt.yticks(np.linspace(0,5,6))
# # plt.ylim(0, 3)
# # plt.yticks([0, 0.5, 1, 1.5, 2, 2.5, 3])
# plt.ylabel(r'Intra-step Order of Convergence')
# lines, labels = ax1.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# ax2.legend(lines + lines2, labels + labels2, loc='best')
# plt.margins(0,0)

# plt.savefig(f"CD_{kwargs['px']}px_{kwargs['py']}py_notMassLumped_RK4.pdf",
#     bbox_inches = 'tight', pad_inches = 0)

# # plt.savefig("CD_MassLumped_RK4.pdf",
# #     bbox_inches = 'tight', pad_inches = 0)

##### Animation routines #####

sim.generatePlottingPoints(nx=5, ny=2)

# maxAbsU = np.max(np.abs(sim.U))
maxAbsU = 1.

def init_plot():
    global field, fig, ax, sim, maxAbsU
    fig, ax = plt.subplots()
    field = ax.tripcolor(sim.X, sim.Y, sim.U, shading='gouraud'
                          ,cmap='seismic', vmin=-maxAbsU, vmax=maxAbsU
                          )
    # tri = mpl.tri.Triangulation(sim.X,sim.Y)
    # ax.triplot(tri, 'r-', lw=1)
    x = np.linspace(0, sim.nodeX[-1], 100)
    for yi in [0.4, 0.5, 0.6]:
        ax.plot(x, [mapping(np.array([[0, yi]]), i) for i in x], 'k')
    for xi in sim.nodeX:
        ax.plot([xi, xi], [0, 1], 'k:')
    # ax.plot(sim.X[np.argmax(sim.U)], sim.Y[np.argmax(sim.U)],
    #   'g+', markersize=10)
    plt.colorbar(field)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$', rotation=0)
    plt.xticks(np.linspace(0, 2*np.pi, 7), 
        ['0',r'$\pi/3$',r'$2\pi/3$',r'$\pi$',r'$4\pi/3$',r'$5\pi/3$',r'$2\pi$'])
    plt.margins(0,0)
    return [field]

init_plot()

field = ax.tripcolor(sim.X, sim.Y, sim.U, shading='gouraud'
                          ,cmap='seismic', vmin=-maxAbsU, vmax=maxAbsU
                          )

def animate(i):
    global field, sim
    sim.step(1)
    sim.computePlottingSolution()
    field.set_array(sim.U)
    plt.title(f"t = {sim.integrator.timestep}")
    return [field]

ani = animation.FuncAnimation(
    fig, animate, frames=nSteps, interval=15)

# # ani.save('movie.mp4', writer='ffmpeg', dpi=200)

# # Advection only, straight mapping, RK4, no mass-lumping
# # v=[0., 0.1], dt = 0.005, t_final = 1, uniform grid, Nquad=5, NY = 4*NX
# NX_array = np.array([  4,   8,  16,  32,  64, 128, 256])
# E_2 = np.array([3.03300012e-02, 1.58515718e-03, 8.72205133e-05, 3.51958541e-05,
#        9.81484218e-06, 2.51775587e-06, 6.33345658e-07])
# E_inf = np.array([1.32846260e-01, 9.11919350e-03, 4.40463595e-04, 1.89907377e-04,
#        5.38855119e-05, 1.39030698e-05, 3.50039920e-06])