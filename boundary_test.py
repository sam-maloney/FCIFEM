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
# class sinXsinY:
#     def __call__(self, p):
#         x = p.reshape(-1,2)[:,0]
#         y = p.reshape(-1,2)[:,1]
#         return np.sin(x)*np.sin(2*np.pi*y)
    
#     def solution(self, p):
#         return (1 / (1 + 4*np.pi**2)) * self(p)

# class TestProblem:
#     def __call__(self, p):
#         return np.ones(int(p.size / 2))
    
#     def solution(self, p):
#         return np.ones(int(p.size / 2))

# function for quadratic patch test
class linearPatch():
    A = 0.02
    
    dfdyMax = 2.
    dfdxMax = 1.
    
    def __call__(self, p):
        return np.zeros(int(p.size / 2))
    
    def solution(self, p):
        x = p.reshape(-1,2)[:,0]
        y = p.reshape(-1,2)[:,1]
        return x + 2*y


class TestProblem:
    n = 20
    A = 0.02
    
    dfdyMax = 40*np.pi
    dfdxMax = 160*A*np.pi**2 + 1
    
    def __call__(self, p):
        x = p.reshape(-1,2)[:,0]
        y = p.reshape(-1,2)[:,1]
        n = self.n
        A = self.A
        return 6*A*n*x*np.cos(n*(y - A*x**2)) + \
            (4*A**2*n**2*x**3 + n**2*x)*np.sin(n*(y - A*x**2))
    
    def solution(self, p):
        x = p.reshape(-1,2)[:,0]
        y = p.reshape(-1,2)[:,1]
        return x*np.sin(self.n*(y - self.A*x**2))
        
f = TestProblem()
# f = linearPatch()

dfRatio = f.dfdyMax / f.dfdxMax

class QaudraticBoundaryFunction:
    def __init__(self, A):
        self.A = A
        self.invA = 1/A
    
    def __call__(self, p):
        x = p.reshape(-1,2)[:,0]
        y = p.reshape(-1,2)[:,1]
        zetaBottom = np.sqrt(x**2 - self.invA*y)
        zetaTop = np.sqrt(x**2 + self.invA*(1 - y))
        return zetaBottom, zetaTop
    
    def deriv(self, p, boundary):
        x = p.reshape(-1,2)[:,0]
        y = p.reshape(-1,2)[:,1]
        if boundary == 'bottom':
            dBdx = x / np.sqrt(x**2 - self.invA*y)
            dBdy = -0.5*self.invA / np.sqrt(x**2 - self.invA*y)
        elif boundary == 'top':
            dBdx = x / np.sqrt(x**2 + self.invA*(1 - y))
            dBdy = -0.5*self.invA / np.sqrt(x**2 + self.invA*(1 - y))
        return dBdx, dBdy

class LinearBoundaryFunction:
    def __init__(self, slope):
        self.slope = slope
    
    def __call__(self, p):
        x = p.reshape(-1,2)[:,0]
        y = p.reshape(-1,2)[:,1]
        zetaBottom = x - y/self.slope
        zetaTop = x + (1-y)/self.slope
        return zetaBottom, zetaTop
    
    def deriv(self, p, boundary):
        return (1., -1./self.slope)

class StraightBoundaryFunction:
    def __call__(self, p):
        nPoints = int(p.size / 2)
        return np.full(nPoints, np.nan), np.full(nPoints, np.nan)
    
    def deriv(self, p, boundary):
        # this should never be used, just return dummy values
        return (1., 1.)

# B = StraightBoundaryFunction()
# mapping = fcifem.mappings.StraightMapping()\

# mapping = fcifem.mappings.LinearMapping(1/(2*np.pi))
# B = LinearBoundaryFunction(mapping.slope)

B = QaudraticBoundaryFunction(f.A)
mapping = fcifem.mappings.QuadraticMapping(f.A)

# mapping = fcifem.mappings.SinusoidalMapping(0.2, -np.pi/2)

kwargs={
    'mapping' : mapping,
    'dt' : 1.,
    'velocity' : np.array([0., 0.]), # Makes the advection matrix zero
    'diffusivity' : 1., # Makes diffusivity matrix K into Poisson operator
    'px' : 0.1,
    'py' : 0.1,
    'seed' : 42 }

# allocate arrays for convergence testing
start = 1
stop = 3
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
    
    # NY = NX
    NY = max(int(f.dfdyMax / (2*np.pi)) * NX, NX)
    NDX = max(int(2*np.pi*NY / (NX*dfRatio)), 1)

    # allocate arrays and compute grid
    sim = fcifem.FciFemSim(NX, NY, **kwargs)
    
    # BC = fcifem.boundaries.PeriodicBoundary(sim)
    BC = fcifem.boundaries.DirichletBoundary(sim, f.solution, B, NDX=NDX)
    sim.setInitialConditions(np.zeros(BC.nNodes), mapped=False, BC=BC)
    
    print(f'NX = {NX},\tNY = {NY},\tnNodes = {sim.nNodes}')
        
    # Assemble the mass matrix and forcing term
    # if NDX == 1:
    #     Qord = 2
    # else:
    #     Qord = 1
    Qord = 1
    # sim.computeSpatialDiscretization = sim.computeSpatialDiscretizationLinearVCI
    sim.computeSpatialDiscretization = sim.computeSpatialDiscretizationConservativeLinearVCI
    sim.computeSpatialDiscretization(f, NQX=NDX, NQY=NY, Qord=Qord,
                                     quadType='g', massLumping=False)
    
    try:
        dxi.append(sim.xi[1:])
    except:
        pass
    
    t_setup[iN] = default_timer()-start_time
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
    uExact = f.solution(sim.nodes)
    
    E_inf[iN] = np.linalg.norm(sim.u - uExact, np.inf)
    E_2[iN] = np.linalg.norm(sim.u - uExact)/np.sqrt(sim.nNodes)
    
    print(f'max error = {E_inf[iN]}')
    print(f'L2 error  = {E_2[iN]}')
    # print(f'cond(K) = {np.linalg.cond(sim.K.A)}')
    print('')

#%% Plotting

# clear the current figure, if opened, and set parameters
# fig = plt.gcf()
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

# sim.generatePlottingPoints(nx=1, ny=1)
sim.generatePlottingPoints(nx=1, ny=1)
# sim.generatePlottingPoints(nx=int(NY/NX), ny=1)
# sim.generatePlottingPoints(nx=int(NY/NX), ny=int(NY/NX))
sim.computePlottingSolution()

# vmin = np.min(sim.U)
# vmax = np.max(sim.U)

exactSol = sim.f.solution(np.vstack((sim.X,sim.Y)).T)
F = sim.f(np.vstack((sim.X,sim.Y)).T)
error = sim.U - exactSol
maxAbsErr = np.max(np.abs(error))
# maxAbsErr = np.max(np.abs(sim.u - uExact))
vmin = -maxAbsErr
vmax = maxAbsErr

ax1 = plt.subplot(121)
field = ax1.tripcolor(sim.X, sim.Y, error, shading='gouraud'
                      ,cmap='seismic', vmin=vmin, vmax=vmax)
# field = ax1.tripcolor(sim.X, sim.Y, F, shading='gouraud')
# field = ax1.tripcolor(sim.X, sim.Y, sim.U, shading='gouraud')
x = np.linspace(0, sim.nodeX[-1], 100)
# for yi in [0.0, 0.1, 0.2]:
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
plt.xticks(np.linspace(0, 2*np.pi, 5), 
    ['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
# plt.xticks(np.linspace(0, 2*np.pi, 7), 
#     ['0',r'$\pi/3$',r'$2\pi/3$',r'$\pi$',r'$4\pi/3$',r'$5\pi/3$',r'$2\pi$'])
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


# ##### NY = 20*NX, NDX = 32, Qord = 1
# NX_array = np.array([  2,   4,   8,  16,  32,  64, 128])
# E_2 = np.array([3.19128962e-02, 1.66174040e-02, 7.35715866e-03, 5.05400243e-03,
#        1.37505435e-03, 6.28625444e-04, 1.49246335e-04])
# E_inf = np.array([6.01267696e-02, 3.91689608e-02, 2.59411522e-02, 2.42221312e-02,
#        8.78515282e-03, 6.51887251e-03, 3.40038087e-03])
# t_setup = np.array([4.92162494e-01, 2.07696175e+00, 8.58752604e+00, 3.48628305e+01,
#        1.41986197e+02, 5.63175996e+02, 2.29591821e+03])
# t_solve = np.array([6.79264893e-03, 2.44334240e-02, 4.95543720e-02, 1.81298157e-01,
#        1.14694606e+00, 8.82098613e+00, 6.88338451e+01])

# ##### NY = 20*NX, NDX = 32, Qord = 2
# NX_array = np.array([ 2,  4,  8, 16, 32, 64])
# E_2 = np.array([3.20631362e-02, 1.38357481e-02, 4.66860478e-03, 2.56289445e-03,
#         9.61139994e-03, 1.37399231e-03])
# E_inf = np.array([9.04214930e-02, 5.24540041e-02, 1.64607034e-02, 1.43771064e-02,
#        9.37708754e-02, 1.07042116e-02])
# t_setup = np.array([1.93709404e+00, 8.34575381e+00, 3.42601427e+01, 1.41035787e+02,
#        5.65862695e+02, 2.32414412e+03])
# t_solve = np.array([7.75411818e-03, 2.43537831e-02, 4.95607180e-02, 1.78673380e-01,
#        1.11589643e+00, 9.6233015e+00])