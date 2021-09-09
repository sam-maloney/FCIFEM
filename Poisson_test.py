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

# mapping = fcifem.SinusoidalMapping(0.2, -np.pi/2)
mapping = fcifem.LinearMapping(1/(2*np.pi))
# mapping = fcifem.StraightMapping()

class slantedTestProblem:
    n = 10
    p2 = np.pi**2
    C = 0.5/(8*p2*n-(4*p2+(4*p2+1)*n**2)**2/(8*p2*n))
    # C = 4*n*p2 / ((8*p2*n)**2 - (4*p2+4*p2*n**2+100)**2)
    B = C*(4*np.pi**2+(4*np.pi**2+1)*n**2)/(8*np.pi**2*n)
    A = 1/(2*(-4*p2-1)*n**2)
    
    def __call__(self, p):
        p.shape = (-1,2)
        x = p[:,0]
        y = p[:,1]
        _2py = 2*np.pi*y
        return -np.sin(self.n*(_2py - x))*(1 + np.sin(_2py))*0.5
    
    def solution(self, p):
        p.shape = (-1,2)
        x = p[:,0]
        y = p[:,1]
        _2py = 2*np.pi*y
        n2pyx = self.n*(_2py - x)
        # n = self.n
        return self.A*np.sin(n2pyx) + self.B*np.sin(_2py)*np.sin(n2pyx) \
                                    + self.C*np.cos(_2py)*np.cos(n2pyx)
        
f = slantedTestProblem()
uExactFunc = f.solution

# ##### standard test isotropic test problem
# def f(p):
#     p.shape = (-1,2)
#     return np.sin(p[:,0])*np.sin(2*np.pi*p[:,1])

# uExactFunc = lambda p : (1/(1+4*np.pi**2))*f(p)

kwargs={
    'mapping' : mapping,
    'dt' : 1.,
    'velocity' : np.array([0., 0.]),
    'diffusivity' : 1., # Makes diffusivity matrix K into Poisson operator
    'px' : 0.,
    'py' : 0.,
    'seed' : 42 }

# allocate arrays for convergence testing
start = 1
stop = 5
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
    
    NY = 16*NX

    # allocate arrays and compute grid
    sim = fcifem.FciFemSim(NX, NY, **kwargs)
    sim.computeSpatialDiscretization = sim.computeSpatialDiscretizationLinearVCI
    # sim.computeSpatialDiscretization = sim.computeSpatialDiscretizationQuadraticVCI
    # sim.computeSpatialDiscretization = sim.computeSpatialDiscretizationConservativePointVCI
    # sim.computeSpatialDiscretization = sim.computeSpatialDiscretizationConservativeCellVCI
    # sim.computeSpatialDiscretization = sim.computeSpatialDiscretizationConservativeNodeVCI
    
    print(f'NX = {NX},\tNY = {NY},\tnNodes = {sim.nNodes}')
    
    sim.setInitialConditions(f)
    
    # Assemble the mass matrix and forcing term
    sim.computeSpatialDiscretization(f, NQX=6, NQY=NY, Qord=3, quadType='g',
                                     massLumping=False)
    
    try:
        dxi.append(sim.xi[1:])
    except:
        pass
    
    # sim.K.data[0] = 1.
    # sim.K.data[1:sim.K.indptr[1]] = 0.
    # sim.b[0] = uExactFunc(sim.nodes[0])

    ##### Enforce exact solution constraints directly #####
    
    # sim.K.data[0] = 1.
    # sim.K.data[1:sim.K.indptr[1]] = 0.
    # sim.b[0] = 0.
    
    # n = int(NY/2)
    # sim.K.data[sim.K.indptr[n]:sim.K.indptr[n+1]] = 0.
    # sim.K[n,n] = 1.
    # sim.b[n] = 0.
    
    # # n = int(NX*NY/2)
    # # sim.K.data[sim.K.indptr[n]:sim.K.indptr[n+1]] = 0.
    # # sim.K[n,n] = 1.
    # # sim.b[n] = 0.
    
    # n = int(NX*NY*3/4)
    # sim.K.data[sim.K.indptr[n]:sim.K.indptr[n+1]] = 0.
    # sim.K[n,n] = 1.
    # sim.b[n] = 0.
    
    # Centre point
    n = int(NX*NY/2 + NY/2)
    sim.K.data[sim.K.indptr[n]:sim.K.indptr[n+1]] = 0.
    sim.K[n,n] = 1.
    sim.b[n] = uExactFunc(sim.nodes[n])
    
    for n, node in enumerate(sim.nodes):
        if node.prod() == 0.:
            sim.K.data[sim.K.indptr[n]:sim.K.indptr[n+1]] = 0.
            sim.K[n,n] = 1.
            sim.b[n] = uExactFunc(sim.nodes[n])
    
    t_setup[iN] = default_timer()-start_time
    print(f'setup time = {t_setup[iN]} s')
    start_time = default_timer()
    
    # Solve for the approximate solution
    # u = sp_la.spsolve(sim.K, sim.b)
    tolerance = 1e-10
    sim.u, info = sp_la.lgmres(sim.K, sim.b, tol=tolerance, atol=tolerance)
    # sim.u = u[0:sim.nNodes]
    
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

# For all of the below
# NX_array = np.array([  4,   8,  16,  32,  64, 128, 256])
# NY = NX, NQY=NY, NQX=6, Qord=3, quadType='gauss', massLumping=False

##### No constrained nodes!!!!! #####

# # Sinusoidal mapping, Uniform spacing
# E_2 = np.array([1.51377118e-04, 1.51345163e-03, 4.27174201e-04, 1.10044134e-04,
#        2.75552544e-05, 6.87046871e-06, 1.72329954e-06])
# E_inf = np.array([3.02754236e-04, 2.54824976e-03, 7.38828574e-04, 2.04472060e-04,
#        5.20249246e-05, 1.30128704e-05, 3.31977042e-06])

# # Sinusoidal mapping, 0.1 perturbation
# E_2 = np.array([1.17859899e-03, 1.54709326e-03, 4.48683841e-04, 1.18334966e-04,
#         3.39448878e-05, 1.19400390e-05, 6.06793906e-06])
# E_inf = np.array([2.35722961e-03, 3.07975031e-03, 8.77070705e-04, 2.78394498e-04,
#        1.11023827e-04, 5.50049087e-05, 3.19412273e-05])

# # Straight mapping, Uniform spacing
# E_2 = np.array([2.66565612e-03, 6.47088425e-04, 1.59542252e-04, 3.97356716e-05,
#        9.92440295e-06, 2.48050383e-06, 6.20088618e-07])
# E_inf = np.array([5.33131225e-03, 1.29417685e-03, 3.19084503e-04, 7.94713431e-05,
#        1.98488059e-05, 4.96100765e-06, 1.24017724e-06])

# # Straight mapping, 0.1 perturbation
# E_2 = np.array([0.00285971, 0.00103294, 0.00055518, 0.00027292, 0.00017557,
#        0.00014189, 0.00013756])
# E_inf = np.array([0.00627925, 0.00310992, 0.00187384, 0.00104082, 0.00057817,
#        0.0004272 , 0.00031896])


# # Straight mapping, 0.1 perturbation, Qord=10
# E_2 = np.array([2.81587204e-03, 7.96726762e-04, 2.22848448e-04, 1.00650709e-04,
#         4.48431768e-05, 2.58854838e-05, 1.83308565e-05])
# E_inf = np.array([0.00608851, 0.00175   , 0.00063334, 0.00033576, 0.0002074 ,
#         1.01666506e-04, 6.21796929e-05])

# # Sinusoidal mapping, 0.1 perturbation, Qord=10
# E_2 = np.array([1.21186089e-03, 1.57858140e-03, 4.43886069e-04, 1.11422412e-04,
#         2.80300181e-05, 7.18423873e-06, 1.97381071e-06])
# E_inf = np.array([2.29185251e-03, 2.95301673e-03, 8.31177799e-04, 2.21525988e-04,
#         5.71461137e-05, 1.79750652e-05, 7.54640780e-06])


##### Left and bottom borders and centre point constrained #####

# # Sinusoidal mapping, 0.1 perturbation, Qord=3
# E_2 = np.array([1.76380823e-03, 1.63218787e-03, 4.89503662e-04, 1.30604665e-04,
#        3.89274495e-05, 1.26881426e-05, 6.15871552e-06])
# E_inf = np.array([4.22775291e-03, 3.76831541e-03, 1.05169678e-03, 3.44947421e-04,
#        1.26005593e-04, 5.59803653e-05, 3.21214936e-05])
# t_setup = np.array([1.20946600e-01, 4.26536300e-01, 1.70480300e+00, 6.39973280e+00,
#        2.53950766e+01, 9.80823424e+01, 3.91944770e+02])
# t_solve = np.array([2.05040001e-03, 6.14400001e-03, 9.41030000e-03, 2.37355000e-02,
#        7.34108000e-02, 2.86898700e-01, 2.22506250e+00])

# # Sinusoidal mapping, 0.1 perturbation, Qord=10
# E_2 = np.array([1.84392285e-03, 1.67487638e-03, 4.94057917e-04, 1.26032021e-04,
#        3.15888578e-05, 8.09440320e-06, 2.19631009e-06])
# E_inf = np.array([4.40617300e-03, 3.72130389e-03, 1.14300871e-03, 2.82715024e-04,
#        7.18992121e-05, 2.27080035e-05, 8.18602074e-06])
# t_setup = np.array([1.08862270e+00, 4.37006580e+00, 1.71951855e+01, 6.90441187e+01,
#        2.72979249e+02, 1.09588131e+03, 6.81203294e+03])
# t_solve = np.array([1.17229999e-03, 5.89129998e-03, 9.05930001e-03, 1.88243000e-02,
#        6.81241000e-02, 9.94588400e-01, 5.55685090e+00])

# # Straight mapping, 0.1 perturbation, Qord=10
# E_2 = np.array([2.79421197e-03, 7.98841851e-04, 2.24282447e-04, 1.01113862e-04,
#        4.50548024e-05, 2.59406008e-05, 1.83385591e-05])
# E_inf = np.array([6.29920113e-03, 1.99154102e-03, 6.31766245e-04, 3.32692194e-04,
#        2.06415962e-04, 1.00257387e-04, 6.19472540e-05])

# # Sinusoidal mapping, 0.1 perturbation, Qord=3, Linear VCI
# E_2 = np.array([1.70905263e-03, 1.71208435e-03, 4.99655257e-04, 1.25161605e-04,
#        3.09979365e-05, 8.66766800e-06, 3.38132213e-06])
# E_inf = np.array([3.96227389e-03, 3.89159769e-03, 1.16773437e-03, 2.91271759e-04,
#        7.56499095e-05, 2.37034861e-05, 1.08384079e-05])
# t_setup = np.array([1.08562400e-01, 4.31015100e-01, 1.73289180e+00, 6.62913230e+00,
#        2.67953264e+01, 1.10938673e+02, 4.39067302e+02])
# t_solve = np.array([2.09550001e-03, 5.45420000e-03, 9.21269998e-03, 1.78824000e-02,
#        6.85618000e-02, 7.81468000e-01, 3.26711650e+00])

# # Sinusoidal mapping, 0.1 perturbation, Qord=3, ConservativePoint VCI
# E_2 = np.array([1.72049530e-03, 1.74932219e-03, 4.99569777e-04, 1.25227649e-04,
#        3.12106760e-05, 7.83296679e-06, np.nan])
# E_inf = np.array([3.94509620e-03, 3.92583363e-03, 1.17370148e-03, 2.89495746e-04,
#        7.36176515e-05, 1.76152918e-05, np.nan])
# t_setup = np.array([1.14400500e-01, 5.64714000e-01, 3.02665540e+00, 2.56724059e+01,
#        2.59681130e+02, np.nan, np.nan])
# t_solve = np.array([1.57389999e-03, 4.82690001e-03, 8.23490000e-03, 1.76809000e-02,
#        5.92375000e-02, np.nan, np.nan])

# # Sinusoidal mapping, 0.1 perturbation, Qord=3, ConservativeCell VCI
# E_2 = np.array([1.62244282e-03, 1.78174990e-03, 5.05333848e-04, 1.25074630e-04,
#        3.11606360e-05, 7.84181311e-06, np.nan])
# E_inf = np.array([3.74952526e-03, 4.15017968e-03, 1.22581638e-03, 2.90684750e-04,
#        7.37685993e-05, 1.78511294e-05, np.nan])
# t_setup = np.array([1.18314700e-01, 5.29194400e-01, 2.39201140e+00, 1.49330466e+01,
#        1.26001360e+02, 1.01552816e+03, np.nan])
# t_solve = np.array([1.49850000e-03, 4.63169999e-03, 8.30550000e-03, 1.66261000e-02,
#        6.22830000e-02, 2.96640500e-01, np.nan])

# # Straight mapping, 0.1 perturbation, Qord=3, Linear VCI
# E_2 = np.array([2.83754386e-03, 9.99162755e-04, 2.72950458e-04, 9.22101598e-05,
#        4.59170115e-05, 5.40049628e-05, 5.49855842e-05])
# E_inf = np.array([7.00280166e-03, 2.99459433e-03, 1.15398586e-03, 4.17471144e-04,
#        1.52541753e-04, 1.40946823e-04, 1.26561102e-04])

# # Sinusoidal mapping, 0.1 perturbation, Qord=3, Quadratic VCI
# E_2 = np.array([1.96114501e-03, 1.33686709e-03, 4.60554065e-04, 1.24423715e-04,
#        3.13792394e-05, 7.90459913e-06, 2.05436055e-06])
# E_inf = np.array([4.29110192e-03, 3.02804777e-03, 1.06999938e-03, 2.92372861e-04,
#        7.08789337e-05, 1.81415879e-05, 4.93928290e-06])
# t_setup = np.array([3.84015800e-01, 1.47548630e+00, 5.61835990e+00, 2.24718873e+01,
#        8.64237513e+01, 3.48133139e+02, 1.41286216e+03])
# t_solve = np.array([2.11110001e-03, 7.51730002e-03, 1.23081000e-02, 2.43821000e-02,
#        7.63307000e-02, 7.56741700e-01, 3.24162050e+00])
