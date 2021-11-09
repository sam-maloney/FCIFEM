# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 13:47:07 2020

@author: samal

NEED TO CHANGE COMPUTATION OF B VECTOR FOR THIS TO WORK!!!!!

"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as sp_la

import fcifem

class TestProblem:
    n = 20
    A = 0.02
    
    def __call__(self, p):
        originalShape = p.shape
        p.shape = (-1,2)
        x = p[:,0]
        y = p[:,1]
        n = self.n
        A = self.A
        p.shape = originalShape
        return 6*A*n*x*np.cos(n*(y - A*x**2)) + \
            (4*A**2*n**2*x**3 + n**2*x)*np.sin(n*(y - A*x**2))
    
    def solution(self, p):
        originalShape = p.shape
        p.shape = (-1,2)
        x = p[:,0]
        y = p[:,1]
        p.shape = originalShape
        return x*np.sin(self.n*(y - self.A*x**2))
        
f = TestProblem()

class QaudraticBoundaryFunction:
    
    def __init__(self, A):
        self.A = A
        self.invA = 1/A
    
    def __call__(self, p):
        originalShape = p.shape
        p.shape = (-1,2)
        x = p[:,0]
        y = p[:,1]
        zetaBottom = np.sqrt(x**2 - self.invA*y)
        zetaTop = np.sqrt(x**2 + self.invA*(1 - y))
        p.shape = originalShape
        return zetaBottom, zetaTop
    
    def deriv(self, p, boundary):
        originalShape = p.shape
        p.shape = (-1,2)
        x = p[:,0]
        y = p[:,1]
        if boundary == 'bottom':
            dBdx = x / np.sqrt(x**2 - self.invA*y)
            dBdy = -0.5*self.invA / np.sqrt(x**2 - self.invA*y)
        elif boundary == 'top':
            dBdx = x / np.sqrt(x**2 + self.invA*(1 - y))
            dBdy = -0.5*self.invA / np.sqrt(x**2 + self.invA*(1 - y))
        p.shape = originalShape
        return dBdx, dBdy
    
B = QaudraticBoundaryFunction(f.A)

mapping = fcifem.QuadraticMapping(f.A)

kwargs={
    'mapping' : mapping,
    'dt' : 1.,
    'velocity' : np.array([0., 0.]),
    'diffusivity' : 0.,
    'px' : 0.,
    'py' : 0.,
    'seed' : 42 }

# allocate arrays for convergence testing
start = 2
stop = 6
nSamples = stop - start + 1
NX_array = np.logspace(start, stop, num=nSamples, base=2, dtype='int32')
E_inf = np.empty(nSamples, dtype='float64')
E_2 = np.empty(nSamples, dtype='float64')

# loop over N to test convergence where N is the number of
# grid cells along one dimension, each cell forms 2 triangles
# therefore number of nodes equals (N+1)*(N+1)
for iN, NX in enumerate(NX_array):
    
    NY = NX

    # allocate arrays and compute grid
    sim = fcifem.FciFemSim(NX, NY, **kwargs)
    BC = fcifem.DirichletBoundaryCondition(sim, f.solution, B)
    sim.setInitialConditions(np.zeros(BC.nNodes), mapped=False, BC=BC)
    
    print(f'NX = {NX},\tNY = {NY},\tnNodes = {sim.nNodes}')
    
    # Assemble the mass matrix and forcing term
    quadRatio = int(2*np.pi*NY/NX)
    sim.computeSpatialDiscretization(f.solution, NQX=quadRatio, NQY=NY, Qord=3,
                                     quadType='g', massLumping = False)
    
    # sim.u = sp_la.spsolve(sim.M, sim.b)
    tolerance = 1e-10
    sim.u, info = sp_la.lgmres(sim.M, sim.b, tol=tolerance, atol=tolerance)
    
    # compute the analytic solution and error norms
    u_exact = f.solution(sim.nodes)
    
    E_inf[iN] = np.linalg.norm(sim.u - u_exact, np.inf)
    E_2[iN] = np.linalg.norm(sim.u - u_exact)/np.sqrt(sim.nNodes)
    
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

u_plot = np.sum(sim.phiPlot * sim.u[sim.indPlot], axis=1)

# maxAbsU = np.max(np.abs(u_plot))
vmin = np.min((np.min(u_plot), np.min(sim.U)))
vmax = np.max((np.max(u_plot), np.max(sim.U)))

exact_sol = f.solution(np.vstack((sim.X,sim.Y)).T)
error = sim.U - exact_sol
maxAbsErr = np.max(np.abs(error))
vmin = -maxAbsErr
vmax = maxAbsErr

ax1 = plt.subplot(121)
field = ax1.tripcolor(sim.X, sim.Y, error, shading='gouraud'
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
plt.ylabel(r'Intra-step Order of Convergence')
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='best')
plt.margins(0,0)

# plt.savefig("CD_MassLumped_RK4.pdf",
#     bbox_inches = 'tight', pad_inches = 0)

# For all of the below
# NX_array = np.array([  4,   8,  16,  32,  64, 128, 256])
# NY = NX

# # Uniform spacing, Nquad=5
# E_2 = np.array([1.17817353e-01, 9.85576979e-02, 2.45978802e-02, 6.02352877e-03,
#        1.49687962e-03, 3.73653269e-04, 9.33777137e-05])
# E_inf = np.array([2.35634705e-01, 1.74694642e-01, 4.76311914e-02, 1.16460444e-02,
#        2.89217058e-03, 7.21715652e-04, 1.80346973e-04])

# # 0.1 perturbation, Nquad=5
# E_2 = np.array([1.34140716e-01, 1.02430887e-01, 2.53661338e-02, 6.15948128e-03,
#        1.52570919e-03, 3.80520484e-04, 9.52952878e-05])
# E_inf = np.array([2.42670205e-01, 1.84218885e-01, 5.24279763e-02, 1.34969241e-02,
#        3.40146362e-03, 9.10205975e-04, 2.26928637e-04])

# # 0.5 perturbation, Nquad=5
# E_2 = np.array([3.32304951e-01, 1.58513739e-01, 3.94981765e-02, 9.56499954e-03,
#        2.23580423e-03, 5.46624171e-04, 1.40440403e-04])
# E_inf = np.array([5.36106904e-01, 3.84477693e-01, 1.21872302e-01, 3.14200379e-02,
#        7.67059713e-03, 2.10334214e-03, 5.30771969e-04])