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

# ##### standard isotropic and periodic test problem
# def f(p):
#     p.shape = (-1,2)
#     return np.sin(p[:,0])*np.sin(2*np.pi*p[:,1])

# uExactFunc = lambda p : (1/(1+4*np.pi**2))*f(p)

# class TestProblem:
#     def __call__(self, p):
#         p.shape = (-1,2)
#         return np.ones(len(p))
    
#     def solution(self, p):
#         p.shape = (-1,2)
#         return np.ones(len(p))

# class StraightBoundaryFunction:
#     def __call__(self, p):
#         p.shape = (-1,2)
#         return np.full(len(p), np.nan), np.full(len(p), np.nan)

# B = StraightBoundaryFunction()

class TestProblem:
    n = 20
    A = 0.02
    
    def __call__(self, p):
        p.shape = (-1,2)
        x = p[:,0]
        y = p[:,1]
        n = self.n
        A = self.A
        return 6*A*n*x*np.cos(n*(y - A*x**2)) + \
            (4*A**2*n**2*x**3 + n**2*x)*np.sin(n*(y - A*x**2))
    
    def solution(self, p):
        p.shape = (-1,2)
        x = p[:,0]
        y = p[:,1]
        return x*np.sin(self.n*(y - self.A*x**2))
        
f = TestProblem()
uExactFunc = f.solution

class QaudraticBoundaryFunction:
    
    def __init__(self, A):
        self.A = A
        self.invA = 1/A
    
    def __call__(self, p):
        p.shape = (-1,2)
        x = p[:,0]
        y = p[:,1]
        zetaBottom = np.sqrt(x**2 - self.invA*y)
        zetaTop = np.sqrt(x**2 + self.invA*(1 - y))
        return zetaBottom, zetaTop
    
    def deriv(self, p, dim):
        p.shape = (-1,2)
        x = p[:,0]
        y = p[:,1]
        if dim == 0: # x-derivative
            zetaBottom = x / np.sqrt(x**2 - self.invA*y)
            zetaTop = x / np.sqrt(x**2 + self.invA*(1 - y))
            return zetaBottom, zetaTop
        if dim == 1: # y-derivative
            zetaBottom = -0.5*self.invA / np.sqrt(x**2 - self.invA*y)
            zetaTop = -0.5*self.invA / np.sqrt(x**2 + self.invA*(1 - y))
            return zetaBottom, zetaTop
    
B = QaudraticBoundaryFunction(f.A)

mapping = fcifem.QuadraticMapping(f.A)

# mapping = fcifem.SinusoidalMapping(0.2, -np.pi/2)
# mapping = fcifem.LinearMapping(1/(2*np.pi))
# mapping = fcifem.StraightMapping()

kwargs={
    'mapping' : mapping,
    'dt' : 1.,
    'velocity' : np.array([0., 0.]), # Makes the advection matrix zero
    'diffusivity' : 1., # Makes diffusivity matrix K into Poisson operator
    'px' : 0.,
    'py' : 0.,
    'seed' : 42 }

# allocate arrays for convergence testing
start = 2
stop = 6
nSamples = np.rint(stop - start + 1).astype('int')
NX_array = np.logspace(start, stop, num=nSamples, base=2, dtype='int')
E_inf = np.empty(nSamples)
E_2 = np.empty(nSamples)
t_setup = np.empty(nSamples)
t_solve = np.empty(nSamples)
dxi = []

# loop over N to test convergence where N is the number of
# grid cells along one dimension, each cell forms 2 triangles
# therefore number of nodes equals (N+1)*(N+1)
for iN, NX in enumerate(NX_array):
    
    start_time = default_timer()
    
    NY = NX

    # allocate arrays and compute grid
    sim = fcifem.FciFemSim(NX, NY, **kwargs)
    
    # BC = fcifem.PeriodicBoundaryCondition(sim)
    BC = fcifem.DirichletBoundaryCondition(sim, f.solution, B)
    sim.setInitialConditions(np.zeros(BC.nNodes), mapped=False, BC=BC)
    
    # sim.BC.test(np.array((5.969026041820607, 0.)), sim.BC.nXnodes)
    
    # Assemble the mass matrix and forcing term
    sim.computeSpatialDiscretization(f, NQX=1, NQY=NY, Qord=3, quadType='g',
                                     massLumping=False)
    
    try:
        dxi.append(sim.xi[1:])
    except:
        pass
    
    t_setup[iN] = default_timer()-start_time
    print(f'NX = {NX},\tNY = {NY},\tnNodes = {sim.nNodes}')
    print(f'setup time = {t_setup[iN]} s')
    start_time = default_timer()
    
    # Solve for the approximate solution
    # sim.u = sp_la.spsolve(sim.K, sim.b)
    tolerance = 1e-10
    sim.u, info = sp_la.lgmres(sim.K, sim.b, tol=tolerance, atol=tolerance)
    
    t_solve[iN] = default_timer()-start_time
    print(f'solve time = {t_solve[iN]} s')
    start_time = default_timer()
    
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

sim.generatePlottingPoints(nx=1, ny=1)
sim.computePlottingSolution()

# vmin = np.min(sim.U)
# vmax = np.max(sim.U)

exactSol = uExactFunc(np.vstack((sim.X,sim.Y)).T)
F = f(np.vstack((sim.X,sim.Y)).T)
error = sim.U - exactSol
maxAbsErr = np.max(np.abs(error))
# maxAbsErr = np.max(np.abs(sim.u - uExact))
vmin = -maxAbsErr
vmax = maxAbsErr

ax1 = plt.subplot(121)
field = ax1.tripcolor(sim.X, sim.Y, error, shading='gouraud'
# field = ax1.tripcolor(sim.nodes[:,0], sim.nodes[:,1], sim.u - uExact, shading='gouraud'
                       ,cmap='seismic', vmin=vmin, vmax=vmax
                     )
x = np.linspace(0, sim.nodeX[-1], 100)
for yi in [0.0, 0.1, 0.2]:
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