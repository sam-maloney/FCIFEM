# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 13:47:07 2020

@author: Samuel A. Maloney

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as sp_la

import fcifem

from timeit import default_timer

##### standard isotropic and periodic test problem
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


class UnityFunction:
    xmax = 1.
    b = 0.05
    # define a such that (0, 0) maps to (xmax, 1) for given b and xmax
    a = (1 - b*xmax)/xmax**2

    umax = 1.
    dudyMax = 1.
    dudxMax = 1.
    dudQMax = 0

    def __call__(self, p):
        return np.ones(p.size // 2)

    def solution(self, p):
        return np.ones(p.size // 2)
        # x = p.reshape(-1,2)[:,0]
        # y = p.reshape(-1,2)[:,1]
        # return ((abs(x - 0.5*self.xmax) < 1e-10) & (abs(y) < 1e-10)).astype('float')


class linearPatch():
    xmax = 1.
    ymax = 1.
    b = 0.05
    # define a such that (0, 0) maps to (xmax, 1) for given b and xmax
    a = (1 - b*xmax)/xmax**2

    umax = xmax + 2*ymax
    dudyMax = 2.
    dudxMax = 1.

    def __call__(self, p):
        return np.zeros(p.size // 2)

    def solution(self, p):
        x = p.reshape(-1,2)[:,0]
        y = p.reshape(-1,2)[:,1]
        return x + 2*y


class QuadraticTestProblem:
    xmax = 1.
    ymax = 1.
    n = 3
    N = (2*np.pi/ymax)*n
    # a = 0.01
    b = 0.05
    # define a such that (0, 0) maps to (xmax, 1) for given b and xmax
    a = (1 - b*xmax)/xmax**2

    umax = xmax
    dudyMax = N*xmax
    dudxMax = 1 + 2*a*N*xmax**2 + b*N*xmax
    dudQMax = 1 # technically also reduced by slope of mapping

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
        return x*np.sin(self.N*(y - self.a*x**2 - self.b*x))

f = QuadraticTestProblem()
# f = linearPatch()
# f = sinXsinY()
# f = UnityFunction()

duRatio = f.dudyMax / f.dudxMax

# mapping = fcifem.mappings.StraightMapping()
# mapping = fcifem.mappings.LinearMapping(1/f.xmax)
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
    'seed' : 42,
    'xmax' : f.xmax }

# allocate arrays for convergence testing
start = 5
stop = 5
nSamples = np.rint(stop - start + 1).astype('int')
NX_array = np.logspace(start, stop, num=nSamples, base=2, dtype='int')
E_inf = np.empty(nSamples)
E_2 = np.empty(nSamples)
t_setup = np.empty(nSamples)
t_solve = np.empty(nSamples)
dxi = []

print('boundary_test.py\n')

# loop over N to test convergence where N is the number of
# grid cells along one dimension, each cell forms 2 triangles
# therefore number of nodes equals (N+1)*(N+1)
for iN, NX in enumerate(NX_array):

    start_time = default_timer()

    NY = 1*NX
    # NY = NX // 2
    # NY = max(int(f.dudyMax / f.xmax) * NX, NX)

    # NQX = max(int(f.xmax*NY / (NX*duRatio)), 1)
    NQX = 1

    NQY = NY
    NDX = 1

    # initialize simulation class
    sim = fcifem.FciFemSim(NX, NY, **kwargs)

    # BC = fcifem.boundaries.PeriodicBoundary(sim)
    # BC = fcifem.boundaries.DirichletXPeriodicYBoundary(sim, f.solution)
    BC = fcifem.boundaries.DirichletBoundary(sim, f.solution, NDX=NDX)
    sim.setInitialConditions(np.zeros(BC.nDoFs), mapped=False, BC=BC)

    print(f'NX = {NX},\tNY = {NY},\tnDoFs = {sim.nDoFs}')

    # if NQX == 1:
    #     Qord = 2
    # else:
    #     Qord = 1
    Qord = 3

    vci = 'VCI-C'
    # vci = None
    if (vci == 'VCI'):
        sim.computeSpatialDiscretization = sim.computeSpatialDiscretizationLinearVCI
    elif (vci == 'VCI-C'):
        sim.computeSpatialDiscretization = sim.computeSpatialDiscretizationConservativeLinearVCI
        # sim.computeSpatialDiscretization = sim.computeSpatialDiscretizationConservativeLinearVCIold

    # Assemble the mass matrix and forcing term
    sim.computeSpatialDiscretization(f, NQX=NQX, NQY=NQY, Qord=Qord,
        quadType='g', massLumping=False, includeBoundaries=True)

    try:
        dxi.append(sim.xi[1:])
    except:
        pass

    t_setup[iN] = default_timer()-start_time
    print(f'setup time = {t_setup[iN]:.8e} s')
    start_time = default_timer()

    # Solve for the approximate solution
    # sim.u = sp_la.spsolve(sim.K, sim.b)
    tolerance = 1e-10
    sim.u, info = sp_la.lgmres(sim.K, sim.b, tol=tolerance, atol=tolerance)

    t_solve[iN] = default_timer()-start_time
    print(f'solve time = {t_solve[iN]:.8e} s')
    start_time = default_timer()

    # compute the analytic solution and error norms
    uExact = f.solution(sim.DoFs)
    E_inf[iN] = np.linalg.norm(sim.u - uExact, np.inf) / f.umax
    E_2[iN] = np.linalg.norm(sim.u - uExact)/np.sqrt(sim.nDoFs) / f.umax

    print(f'max error  = {E_inf[iN]:.8e}')
    print(f'L2 error   = {E_2[iN]:.8e}')
    # print(f'cond(K) = {np.linalg.cond(sim.K.A)}')
    print('', flush=True)

# print summary
print(f'xmax = {f.xmax}, {mapping}')
print(f'px = {kwargs["px"]}, py = {kwargs["py"]}, seed = {kwargs["seed"]}')
print(f'NDX = {NDX}, NQX = {NQX}, NQY = {NQY//NY}*NY, Qord = {Qord}')
print(f'VCI: {sim.vci} using {sim.vci_solver}\n')
with np.printoptions(formatter={'float': lambda x: format(x, '.8e')}):
    print('E_2     =', repr(E_2))
    print('E_inf   =', repr(E_inf))
    print('t_setup =', repr(t_setup))
    print('t_solve =', repr(t_solve))


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
# plt.title('Absolute Error')
# field = ax1.tripcolor(sim.X, sim.Y, error, shading='gouraud'
#                       ,cmap='seismic', vmin=vmin, vmax=vmax)
# field = ax1.tripcolor(sim.X, sim.Y, F, shading='gouraud')
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
if abs(f.xmax - 2*np.pi) < 1e-10:
    plt.xticks(np.linspace(0, f.xmax, 5),
        ['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
#  plt.xticks(np.linspace(0, 2*np.pi, 7),
#      ['0',r'$\pi/3$',r'$2\pi/3$',r'$\pi$',r'$4\pi/3$',r'$5\pi/3$',r'$2\pi$'])
else:
    plt.xticks(np.linspace(0, f.xmax, 6))
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
