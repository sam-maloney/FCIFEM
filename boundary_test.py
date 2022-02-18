# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 13:47:07 2020

@author: Samuel A. Maloney

"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as sp_la

import fcifem

from timeit import default_timer

##### standard isotropic and periodic test problem
class sinXsinY:
    def __call__(self, p):
        x = p.reshape(-1,2)[:,0]
        y = p.reshape(-1,2)[:,1]
        return np.sin(x)*np.sin(2*np.pi*y)
    
    def solution(self, p):
        return (1 / (1 + 4*np.pi**2)) * self(p)

class TestProblem:
    xmax = 2*np.pi
    # a = 0.02
    b = 0.05
    # define a such that (0, 0) maps to (xmax, 1) for given b and xmax
    a = (1 - b*xmax)/xmax**2
    dfdyMax = 1
    dfdxMax = 1
    
    def __call__(self, p):
        return np.ones(p.size // 2)
    
    def solution(self, p):
        x = p.reshape(-1,2)[:,0]
        y = p.reshape(-1,2)[:,1]
        return ((abs(x - np.pi) < 1e-10) & (abs(y) < 1e-10)).astype('float')

# function for linear patch test
class linearPatch():
    xmax = 2*np.pi
    # a = 0.02
    b = 0.05
    # define a such that (0, 0) maps to (xmax, 1) for given b and xmax
    a = (1 - b*xmax)/xmax**2
    
    dfdyMax = 2.
    dfdxMax = 1.
    
    def __call__(self, p):
        return np.zeros(p.size // 2)
    
    def solution(self, p):
        x = p.reshape(-1,2)[:,0]
        y = p.reshape(-1,2)[:,1]
        return x + 2*y


class QuadraticTestProblem:
    xmax = 2*np.pi
    n = 3
    # a = 0.01
    b = 0.05
    # define a such that (0, 0) maps to (xmax, 1) for given b and xmax
    a = (1 - b*xmax)/xmax**2
    
    dfdyMax = 2*np.pi*n*xmax
    dfdxMax = 1 + 2*a*2*np.pi*n*xmax**2 + b*2*np.pi*n*xmax
    
    def __call__(self, p):
        x = p.reshape(-1,2)[:,0]
        y = p.reshape(-1,2)[:,1]
        n = 2*np.pi*self.n
        a = self.a
        b = self.b
        return n*(n*x*(4*a**2*x**2 + 4*a*b*x + b**2 + 1)*np.sin(n*(y - a*x**2 - b*x))
                  + 2*(3*a*x + b)*np.cos(n*(y - a*x**2 - b*x)))
    
    def solution(self, p):
        x = p.reshape(-1,2)[:,0]
        y = p.reshape(-1,2)[:,1]
        return x*np.sin(2*np.pi*self.n*(y - self.a*x**2 - self.b*x))
        
f = QuadraticTestProblem()
# f = linearPatch()

dfRatio = f.dfdyMax / f.dfdxMax

#%% Boundary Functions and Mappings

class QaudraticBoundaryFunction:
    
    def __init__(self, a, b=0.):
        self.a = a
        self.b = b
        if b == 0.:
            self.inva = 1/a
            self.deriv = self.deriv0
            # # This doesn't work
            # self.__call__ = self.call0
    
    def __call__(self, p):
        x = p.reshape(-1,2)[:,0]
        y = p.reshape(-1,2)[:,1]
        a = self.a
        b = self.b
        zetaBottom = (-b + np.sqrt(b**2 - 4*a*(y - a*x**2 - b*x)))/(2*a)
        zetaTop = (-b + np.sqrt(b**2 - 4*a*(y - a*x**2 - b*x - 1)))/(2*a)
        return zetaBottom, zetaTop
    
    def deriv(self, p, boundary):
        x = p.reshape(-1,2)[:,0]
        y = p.reshape(-1,2)[:,1]
        a = self.a
        b = self.b
        if boundary == 'bottom':
            dBdy = -1 / np.sqrt(b**2 - 4*a*(y - a*x**2 - b*x))
            dBdx = -(2*a*x + b) * dBdy
        elif boundary == 'top':
            dBdy = -1 / np.sqrt(b**2 - 4*a*(y - a*x**2 - b*x - 1))
            dBdx = -(2*a*x + b) * dBdy
        return dBdx, dBdy
    
    def call0(self, p):
        x = p.reshape(-1,2)[:,0]
        y = p.reshape(-1,2)[:,1]
        zetaBottom = np.sqrt(x**2 - self.inva*y)
        zetaTop = np.sqrt(x**2 + self.inva*(1 - y))
        return zetaBottom, zetaTop
    
    def deriv0(self, p, boundary):
        x = p.reshape(-1,2)[:,0]
        y = p.reshape(-1,2)[:,1]
        if boundary == 'bottom':
            dBdx = x / np.sqrt(x**2 - self.inva*y)
            dBdy = -0.5*self.inva / np.sqrt(x**2 - self.inva*y)
        elif boundary == 'top':
            dBdx = x / np.sqrt(x**2 + self.inva*(1 - y))
            dBdy = -0.5*self.inva / np.sqrt(x**2 + self.inva*(1 - y))
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
        nPoints = p.size // 2
        return np.full(nPoints, np.nan), np.full(nPoints, np.nan)
    
    def deriv(self, p, boundary):
        # this should never be used, just return dummy values
        return (1., 1.)

# B = StraightBoundaryFunction()
# mapping = fcifem.mappings.StraightMapping()\

# mapping = fcifem.mappings.LinearMapping(1/(2*np.pi))
# B = LinearBoundaryFunction(mapping.slope)

B = QaudraticBoundaryFunction(f.a, f.b)
mapping = fcifem.mappings.QuadraticMapping(f.a, f.b)


#%%
perturbation = 0.1
kwargs={
    'mapping' : mapping,
    'dt' : 1.,
    'velocity' : np.array([0., 0.]), # Makes the advection matrix zero
    'diffusivity' : 1., # Makes diffusivity matrix K into Poisson operator
    'px' : perturbation,
    'py' : perturbation,
    'seed' : 42 }

# allocate arrays for convergence testing
start = 1
stop = 4
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
    
    # NQX = 1
    # NY = 10*NX
    NY = 16*NX
    # NY = max(int(f.dfdyMax / (2*np.pi)) * NX, NX)
    NQX = max(int(2*np.pi*NY / (NX*dfRatio)), 1)
    NQY = NY

    # initialize simulation class
    sim = fcifem.FciFemSim(NX, NY, **kwargs)
    
    # BC = fcifem.boundaries.PeriodicBoundary(sim)
    # BC = fcifem.boundaries.DirichletXPeriodicYBoundary(sim, f.solution)
    BC = fcifem.boundaries.DirichletBoundary(sim, f.solution, B, NDX=0)
    sim.setInitialConditions(np.zeros(BC.nDoFs), mapped=False, BC=BC)
    
    print(f'NX = {NX},\tNY = {NY},\tnDoFs = {sim.nDoFs}')
        
    # Assemble the mass matrix and forcing term
    # if NQX == 1:
    #     Qord = 2
    # else:
    #     Qord = 1
    Qord = 2
    
    vci = 'VCI-C'
    # vci = None
    if (vci == 'VCI'):
        sim.computeSpatialDiscretization = sim.computeSpatialDiscretizationLinearVCI
    elif (vci == 'VCI-C'):
        sim.computeSpatialDiscretization = sim.computeSpatialDiscretizationConservativeLinearVCI
    
    sim.computeSpatialDiscretization(f, NQX=NQX, NQY=NQY, Qord=Qord, 
        quadType='g', massLumping=False, includeBoundaries=True)
    
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
    uExact = f.solution(sim.DoFs)
    
    E_inf[iN] = np.linalg.norm(sim.u - uExact, np.inf)
    E_2[iN] = np.linalg.norm(sim.u - uExact)/np.sqrt(sim.nDoFs)
    
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

sim.generatePlottingPoints(nx=1, ny=1)
# sim.generatePlottingPoints(nx=10, ny=1)
# sim.generatePlottingPoints(nx=int(NY/NX), ny=1)
# sim.generatePlottingPoints(nx=int(NY/NX), ny=int(NY/NX))
sim.computePlottingSolution()

# vmin = np.min(sim.U)
# vmax = np.max(sim.U)

exactSol = f.solution(np.vstack((sim.X,sim.Y)).T)
F = f(np.vstack((sim.X,sim.Y)).T)
error = sim.U - exactSol
maxAbsErr = np.max(abs(error))
# maxAbsErr = np.max(abs(sim.u - uExact))
vmin = -maxAbsErr
vmax = maxAbsErr

ax1 = plt.subplot(121)
# field = ax1.tripcolor(sim.X, sim.Y, error, shading='gouraud'
#                       ,cmap='seismic', vmin=vmin, vmax=vmax)
# field = ax1.tripcolor(sim.X, sim.Y, F, shading='gouraud')
field = ax1.tripcolor(sim.X, sim.Y, sim.U, shading='gouraud')
# x = np.linspace(0, sim.nodeX[-1], 100)
# # for yi in [0.0, 0.1, 0.2]:
# for yi in [sim.mapping(np.array((x, 0.5)), 0.) for x in sim.nodeX]:
#     ax1.plot(x, [sim.mapping(np.array([[0, float(yi)]]), i) for i in x], 'k')
# for xi in sim.nodeX:
#     ax1.plot([xi, xi], [0, 1], 'k:')
# ax.plot(sim.X[np.argmax(sim.U)], sim.Y[np.argmax(sim.U)],
#   'g+', markersize=10)
# cbar = plt.colorbar(field, format='%.0e')
plt.ylim((0., 1.))
cbar = plt.colorbar(field)
cbar.formatter.set_powerlimits((0, 0))
plt.xlabel(r'$x$')
plt.ylabel(r'$y$', rotation=0)
plt.xticks(np.linspace(0, 2*np.pi, 5), 
    ['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
# plt.xticks(np.linspace(0, 2*np.pi, 7), 
#     ['0',r'$\pi/3$',r'$2\pi/3$',r'$\pi$',r'$4\pi/3$',r'$5\pi/3$',r'$2\pi$'])
plt.title('Absolute Error')
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
filename += '_' + ['DxPy', 'DxDy'][BC.name == 'Dirichlet']
if (vci is not None):
    filename += '_' + vci

# plt.sca(ax1)
# plt.savefig(filename + '.pdf', bbox_inches = 'tight', pad_inches = 0)

# #### Plot the quadrature points #####
# from scipy.special import roots_legendre
# for iPlane in range(NX):
#     dx = sim.dx[iPlane]
#     ##### generate quadrature points
#     if sim.quadType.lower() in ('gauss', 'g', 'gaussian'):
#         offsets, weights = roots_legendre(Qord)
#     elif sim.quadType.lower() in ('uniform', 'u'):
#         offsets = np.linspace(1/Qord - 1, 1 - 1/Qord, Qord)
#         weights = np.repeat(2/Qord, Qord)
#     offsets = (offsets * dx * 0.5 / NQX, offsets * 0.5 / NQY)
#     weights = (weights * dx * 0.5 / NQX, weights * 0.5 / NQY)
#     quads = ( np.indices([NQX, NQY], dtype='float').T.
#               reshape(-1, sim.ndim) + 0.5 ) * [dx/NQX, 1/NQY]
#     quadWeights = np.repeat(1., len(quads))
#     for i in range(sim.ndim):
#         quads = np.concatenate( 
#             [quads + offset*np.eye(sim.ndim)[i] for offset in offsets[i]] )
#         quadWeights = np.concatenate(
#             [quadWeights * weight for weight in weights[i]] )
    
#     quads += [sim.nodeX[iPlane], 0]
#     ax1.plot(quads[:,0], quads[:,1], 'k+')
