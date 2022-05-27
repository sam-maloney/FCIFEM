# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 13:47:07 2020

@author: Samuel A. Maloney

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as sp_la

import fcifem
# import fcifem_periodic as fcifem

from timeit import default_timer

class slantedTestProblem:
    xmax = 1.
    ymax = 1.
    xfac = 2*np.pi/xmax
    yfac = 2*np.pi/ymax

    n = 8

    n2 = n*n
    yf2 = yfac*yfac
    _2nyf2 = 2*n*yf2
    n2xf2pyf2 = n2*(xfac*xfac + yf2)
    n2xf2pyf2pyf2 = n2xf2pyf2 + yf2
    A = 0.5 / n2xf2pyf2
    B = 0.5 / (n2xf2pyf2pyf2 - _2nyf2*_2nyf2/n2xf2pyf2pyf2)
    C = B*_2nyf2 / n2xf2pyf2pyf2

    aA = abs(A)
    aB = abs(B)
    aC = abs(C)
    umax = aA + aB + aC
    dudxMax = umax*xfac*n
    dudyMax = yfac*(aA*n + (aB+aC)*(1+n))
    dudQMax = yfac*(aB+aC)/np.sqrt(2)

    dfdxMax = xfac*n
    dfdyMax = yfac*(n + 0.5)
    dfdQMax = 0.5*yfac/np.sqrt(2)

    def __call__(self, p):
        x = p.reshape(-1,2)[:,0]
        y = p.reshape(-1,2)[:,1]
        yarg = self.yfac*y
        return 0.5*np.sin(self.n*(yarg - self.xfac*x))*(1 + np.sin(yarg))

    def solution(self, p):
        x = p.reshape(-1,2)[:,0]
        y = p.reshape(-1,2)[:,1]
        yarg = self.yfac*y
        xyarg = self.n*(yarg - self.xfac*x)
        return self.A*np.sin(xyarg) + self.B*np.sin(yarg)*np.sin(xyarg) \
                                    + self.C*np.cos(yarg)*np.cos(xyarg)


class simplifiedSlantProblem:
    xmax = 1.
    ymax = 1.
    n = 2

    xfac = 2*np.pi/xmax
    yfac = 2*np.pi/ymax
    umax = 1/(2*n*n*(yfac*yfac + xfac*xfac))
    dudxMax = umax*xfac*n
    dudyMax = umax*yfac*n
    dudQMax = 0

    def __call__(self, p):
        x = p.reshape(-1,2)[:,0]
        y = p.reshape(-1,2)[:,1]
        return 0.5*np.sin(self.n*(self.yfac*y - self.xfac*x))

    def solution(self, p):
        x = p.reshape(-1,2)[:,0]
        y = p.reshape(-1,2)[:,1]
        return self.umax * np.sin(self.n*(self.yfac*y - self.xfac*x))


class sinXsinY:
    xmax = 1.
    ymax = 1.
    xfac = 2*np.pi/xmax
    yfac = 2*np.pi/ymax
    umax = (1 / (xfac**2 + yfac**2))
    dudxMax = umax*xfac
    dudyMax = umax*yfac

    def __call__(self, p):
        x = p.reshape(-1,2)[:,0]
        y = p.reshape(-1,2)[:,1]
        return np.sin(self.xfac*x)*np.sin(self.yfac*y)

    def solution(self, p):
        return self.umax * self(p)

f = slantedTestProblem()
# f = simplifiedSlantProblem()
# f = sinXsinY()

# mapping = fcifem.mappings.SinusoidalMapping(0.2, -0.25*f.xmax, f.xmax)
mapping = fcifem.mappings.LinearMapping(1/f.xmax)
# mapping = fcifem.mappings.StraightMapping()

perturbation = 0.1
kwargs={
    'mapping' : mapping,
    'dt' : 1.,
    'velocity' : np.array([0., 0.]),
    'diffusivity' : 1., # Makes diffusivity matrix K into Poisson operator
    'px' : perturbation,
    'py' : perturbation,
    'seed' : 42,
    'xmax' : f.xmax }

# allocate arrays for convergence testing
start = 1
stop = 6
nSamples = np.rint(stop - start + 1).astype('int')
NX_array = np.logspace(start, stop, num=nSamples, base=2, dtype='int')
E_inf = np.empty(nSamples)
E_2 = np.empty(nSamples)
t_setup = np.empty(nSamples)
t_solve = np.empty(nSamples)
dxi = []

print('Poisson_test.py\n')

# loop over N to test convergence where N is the number of
# grid cells along one dimension, each cell forms 2 triangles
# therefore number of nodes equals (N+1)*(N+1)
for iN, NX in enumerate(NX_array):

    start_time = default_timer()

    NY = 1*NX
    # NX = 16

    NQX = 1
    # NQX = max(NY//NX, 1)
    NQY = NY
    Qord = 2

    # allocate arrays and compute grid
    sim = fcifem.FciFemSim(NX, NY, **kwargs)
    # sim.computeSpatialDiscretization = sim.computeSpatialDiscretizationLinearVCI
    sim.computeSpatialDiscretization = sim.computeSpatialDiscretizationConservativeLinearVCI
    # sim.computeSpatialDiscretization = sim.computeSpatialDiscretizationConservativeLinearVCIold
    ##### These require the fcifem_periodic version of the module #####
    # sim.computeSpatialDiscretization = sim.computeSpatialDiscretizationQuadraticVCI
    # sim.computeSpatialDiscretization = sim.computeSpatialDiscretizationConservativePointVCI
    # sim.computeSpatialDiscretization = sim.computeSpatialDiscretizationConservativeCellVCI
    # sim.computeSpatialDiscretization = sim.computeSpatialDiscretizationConservativeNodeVCI

    sim.setInitialConditions(f)

    print(f'NX = {NX},\tNY = {NY},\tnDoFs = {sim.nDoFs}')

    # Assemble the mass matrix and forcing term
    sim.computeSpatialDiscretization(f, NQX=NQX, NQY=NQY, Qord=Qord, quadType='g',
                                     massLumping=False)

    try:
        dxi.append(sim.xi[1:])
    except:
        pass

    # sim.K.data[0] = 1.
    # sim.K.data[1:sim.K.indptr[1]] = 0.
    # sim.b[0] = f.solution(sim.DoFs[0])

    ##### Enforce exact solution constraints directly #####

    # sim.K.data[0] = 1.
    # sim.K.data[1:sim.K.indptr[1]] = 0.
    # sim.b[0] = 0.

    # n = int(NY/2)
    # sim.K.data[sim.K.indptr[n]:sim.K.indptr[n+1]] = 0.
    # sim.K[n,n] = 1.
    # sim.b[n] = 0., label='Expected'

    # n = int(NX*NY/2)
    # sim.K.data[sim.K.indptr[n]:sim.K.indptr[n+1]] = 0.
    # sim.K[n,n] = 1.
    # sim.b[n] = 0.

    # # n = int(NX*NY*3/4)
    # # sim.K.data[sim.K.indptr[n]:sim.K.indptr[n+1]] = 0.
    # # sim.K[n,n] = 1.
    # # sim.b[n] = 0.

    # Centre point
    n = int(NX*NY/2 + NY/2)
    sim.K.data[sim.K.indptr[n]:sim.K.indptr[n+1]] = 0.
    sim.K[n,n] = 1.
    sim.b[n] = f.solution(sim.DoFs[n])

    for n, node in enumerate(sim.DoFs):
        if node.prod() == 0.:
            sim.K.data[sim.K.indptr[n]:sim.K.indptr[n+1]] = 0.
            sim.K[n,n] = 1.
            sim.b[n] = f.solution(sim.DoFs[n])

    t_setup[iN] = default_timer()-start_time
    print(f'setup time = {t_setup[iN]:.8e} s')
    start_time = default_timer()

    # Solve for the approximate solution
    # u = sp_la.spsolve(sim.K, sim.b)
    tolerance = 1e-10
    sim.u, info = sp_la.lgmres(sim.K, sim.b, tol=tolerance, atol=tolerance)

    t_solve[iN] = default_timer()-start_time
    print(f'solve time = {t_solve[iN]:.8e} s')
    start_time = default_timer()

    # compute the analytic solution and normalized error norms
    uExact = f.solution(sim.DoFs)
    E_inf[iN] = np.linalg.norm(sim.u - uExact, np.inf) / f.umax
    E_2[iN] = np.linalg.norm(sim.u - uExact)/np.sqrt(sim.nDoFs) / f.umax

    print(f'max error  = {E_inf[iN]:.8e}')
    print(f'L2 error   = {E_2[iN]:.8e}\n', flush=True)

# print summary
print(f'xmax = {f.xmax}, {mapping}')
print(f'px = {kwargs["px"]}, py = {kwargs["py"]}, seed = {kwargs["seed"]}')
print(f'NQX = {NQX}, NQY = {NQY//NY}*NY, Qord = {Qord}')
print(f'VCI: {sim.vci} using {sim.vci_solver}\n')
with np.printoptions(formatter={'float': lambda x: format(x, '.8e')}):
    print('E_2     =', repr(E_2))
    print('E_inf   =', repr(E_inf))
    print('t_setup =', repr(t_setup))
    print('t_solve =', repr(t_solve))


#%% Plotting

plt.rc('pdf', fonttype=42)
plt.rc('text', usetex=True)
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
# fontsize : int or {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}
# plt.rc('font', size='small')
# plt.rc('legend', fontsize='small')
# plt.rc('axes', titlesize='medium', labelsize='medium')
# plt.rc('xtick', labelsize='small')
# plt.rc('ytick', labelsize='small')
# plt.rc('figure', titlesize='large')

# clear the current figure, if opened, and set parameters
fig = plt.figure(figsize=(7.75, 3))
fig.subplots_adjust(hspace=0.3, wspace=0.3)

# sim.generatePlottingPoints(nx=1, ny=1)
sim.generatePlottingPoints(nx=int(max(NY/NX,1)), ny=int(max(NX/NY,1)))
sim.computePlottingSolution()

# vmin = np.min(sim.U)
# vmax = np.max(sim.U)

exactSol = f.solution(np.vstack((sim.X,sim.Y)).T)
error = sim.U - exactSol
maxAbsErr = np.max(np.abs(error))
# maxAbsErr = np.max(np.abs(sim.u - uExact))
vmin = -maxAbsErr
vmax = maxAbsErr

ax1 = plt.subplot(121)
ax1.set_title('Final Solution')
# field = ax1.tripcolor(sim.X, sim.Y, error, shading='gouraud'
#                        ,cmap='seismic', vmin=vmin, vmax=vmax)
# field = ax1.tripcolor(sim.DoFs[:,0], sim.DoFs[:,1], sim.u - uExact
#                     ,shading='gouraud', cmap='seismic', vmin=vmin, vmax=vmax)
field = ax1.tripcolor(sim.X, sim.Y, sim.U, shading='gouraud')
# field = ax1.tripcolor(sim.X, sim.Y, exactSol, shading='gouraud')
# field = ax1.tripcolor(sim.X, sim.Y, f(np.vstack((sim.X,sim.Y)).T), shading='gouraud')
x = np.linspace(0, sim.nodeX[-1], 100)
for yi in [0.4, 0.5, 0.6]:
    try:
        ax1.plot(x, [sim.BC.mapping(np.array([[0, yi]]), i) for i in x], 'k')
    except:
        ax1.plot(x, [sim.mapping(np.array([[0, yi]]), i) % 1 for i in x], 'k')
# for xi in sim.nodeX:
#     ax1.plot([xi, xi], [0, 1], 'k:')
# ax.plot(sim.X[np.argmax(sim.U)], sim.Y[np.argmax(sim.U)],
#   'g+', markersize=10)
# cbar = plt.colorbar(field, format='%.0e')
cbar = plt.colorbar(field)
cbar.formatter.set_powerlimits((0, 0))
ax1.set_xlabel(r'$x$')
ax1.set_ylabel(r'$y$', rotation=0)
if abs(f.xmax - 2*np.pi) < 1e-10:
    ax1.set_xticks(np.linspace(0, f.xmax, 5),
        ['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
#  plt.xticks(np.linspace(0, 2*np.pi, 7),
#      ['0',r'$\pi/3$',r'$2\pi/3$',r'$\pi$',r'$4\pi/3$',r'$5\pi/3$',r'$2\pi$'])
else:
    ax1.set_xticks(np.linspace(0, f.xmax, 6))
ax1.margins(0,0)

# plot the error convergence
ax2 = plt.subplot(122)
logN = np.log(NX_array)
ax2.semilogy(logN, E_inf, '.-', label=r'$E_\infty$')
ax2.semilogy(logN, E_2, '.-', label=r'$E_2$')
# ax2.minorticks_off()
ax2.set_xticks(logN, labels=NX_array)
ax2.set_xlabel(r'$NX$')
ax2.set_ylabel(r'Magnitude of Error Norm')

# plot the intra-step order of convergence
ax2R = ax2.twinx()
logE_inf = np.log(E_inf)
logE_2 = np.log(E_2)
order_inf = (logE_inf[0:-1] - logE_inf[1:])/(logN[1:] - logN[0:-1])
order_2 = (logE_2[0:-1] - logE_2[1:])/(logN[1:] - logN[0:-1])
intraN = 0.5 * (logN[:-1] + logN[1:])
ax2R.plot(intraN, order_inf, '.:', linewidth=1, label=r'$E_\infty$ order')
ax2R.plot(intraN, order_2, '.:', linewidth=1, label=r'$E_2$ order')
ax2R.axhline(2, linestyle=':', color='k', linewidth=1, label='Expected')
ax2R.set_ylim(0, 5)
ax2R.set_yticks(np.linspace(0,5,6))
# ax2R.set_lim(0, 3)
# ax2R.set_yticks([0, 0.5, 1, 1.5, 2, 2.5, 3])
ax2R.set_ylabel(r'Intra-step Order of Convergence')
ax2.legend()
# lines, labels = ax1.get_legend_handles_labels()
