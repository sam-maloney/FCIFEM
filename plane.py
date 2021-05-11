# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 13:47:07 2020

@author: samal
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import fcifem

mapping = fcifem.SinusoidalMapping(0.2, -np.pi/2)
        
def f(p):
    p.shape = (-1,2)
    return p[:,0] + 2*p[:,1]

kwargs={
    'mapping' : mapping,
    'dt' : 1.,
    'velocity' : np.array([0., 0.]),
    'diffusivity' : 0.,
    'Nquad' : 1,
    'px' : 0.0,
    'py' : 0.0,
    'seed' : 42 }

NX = 4
NY = 4*NX

# allocate arrays and compute grid
sim = fcifem.FciFemSim(NX, NY, **kwargs)
sim.setInitialConditions(f)

print(f'NX = {NX},\tNY = {NY},\tnNodes = {sim.nNodes}')
    
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

sim.u = sim.u0func(sim.nodes)

sim.generatePlottingPoints(nx=20, ny=3)
sim.computePlottingSolution()

vmin = np.min(sim.U)
vmax = np.max(sim.U)

ax1 = plt.subplot(121)
field = ax1.tripcolor(sim.X, sim.Y, sim.U, shading='gouraud'
                     ,cmap='Purples', vmin=vmin, vmax=vmax
                     )
x = np.linspace(0, sim.nodeX[-1], 100)
for yi in [0.4, 0.5, 0.6]:
    ax1.plot(x, [mapping(np.array([[0, yi]]), i) for i in x], 'k')
for xi in sim.nodeX:
    ax1.plot([xi, xi], [0, 1], 'k:')
plt.colorbar(field)
plt.xlabel(r'$x$')
plt.ylabel(r'$y$', rotation=0)
plt.xticks(np.linspace(0, 2*np.pi, 7), 
    ['0',r'$\pi/3$',r'$2\pi/3$',r'$\pi$',r'$4\pi/3$',r'$5\pi/3$',r'$2\pi$'])
plt.margins(0,0)

exact_sol = f(np.vstack((sim.X,sim.Y)).T)
inds = (sim.X <= sim.nodeX[-2]) & (sim.Y <= (sim.nodeY[0,-2] - mapping.A)) & (sim.Y >= mapping.A)
error = sim.U[inds] - exact_sol[inds]
maxAbsErr = np.max(np.abs(error))
vmin = -maxAbsErr
vmax = maxAbsErr

ax2 = plt.subplot(122)
field = ax2.tripcolor(sim.X[inds], sim.Y[inds], error, shading='gouraud'
                     ,cmap='seismic', vmin=vmin, vmax=vmax
                     )
x = np.linspace(0, sim.nodeX[-1], 100)
for yi in [0.4, 0.5, 0.6]:
    ax2.plot(x, [mapping(np.array([[0, yi]]), i) for i in x], 'k')
for xi in sim.nodeX:
    ax2.plot([xi, xi], [0, 1], 'k:')
plt.colorbar(field)
plt.xlabel(r'$x$')
plt.ylabel(r'$y$', rotation=0)
plt.xticks(np.linspace(0, 2*np.pi, 7), 
    ['0',r'$\pi/3$',r'$2\pi/3$',r'$\pi$',r'$4\pi/3$',r'$5\pi/3$',r'$2\pi$'])
plt.margins(0,0)

# plt.savefig("plane.pdf",
#     bbox_inches = 'tight', pad_inches = 0)
