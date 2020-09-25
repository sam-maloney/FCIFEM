# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 13:47:07 2020

@author: samal
"""

from scipy.special import roots_legendre
from scipy.interpolate import RectBivariateSpline
from scipy.fft import fft, fftshift, fftfreq
import gc
import sparse
import scipy.integrate
import scipy.sparse as sp
import scipy.sparse.linalg as sp_la
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def mapping(points, x=0., theta=False, deriv=False):
    if deriv or theta:
        return np.repeat(0., len(points))
    else:
        return points[:,1]

##### uniform grid spacing
NX = 256 # number of planes
NY = 128 # number of grid divisions on each plane
sizeX = 10
sizeY = 5
nodeX = sizeX*np.arange(NX+1)/NX

Nquad = 2
ndim = 2

##### non-uniform X spacing
# nodeX = np.array([0., 1, 2.5, 3.5, 5, 2*np.pi])
# NX = len(nodeX) - 1
# NY = 20
nodeYfull = sizeY*np.tile(np.linspace(0, 1, NY+1), NX).reshape(NX,-1)
nodeY = sizeY*np.tile(np.linspace(1/NY, 1 - 1/NY, NY-1), NX).reshape(NX,-1)

nNodes = NX * (NY-1)

dx = nodeX[1:]-nodeX[0:-1]
dy = nodeYfull[:,1:]-nodeYfull[:,:-1]

alpha = 1.0    # Adiabaticity (~conductivity)
kappa = 0.5    # Density gradient drive

arcLengths = np.array([scipy.integrate.quad(lambda x: 
    np.sqrt(1 + mapping(np.array([[x,0]]), deriv=True)**2),
    nodeX[i], nodeX[i+1])[0] for i in range(NX)])

dt = 1
# nSteps = int(np.rint(np.sum(dx[0:3])/dt))
# nSteps = int(np.pi/dt)
nSteps = 127

print('Computing spatial discretization... ', end='')

##### pre-allocate arrays for stiffness matrix triplets
nEntries = (2*ndim)**2
nQuads = NY * Nquad**2
nMaxEntries = nEntries * nQuads * NX
Kdata = np.zeros(nMaxEntries)
Mdata = np.zeros(nMaxEntries)
DDXdata = np.zeros(nMaxEntries)
row_ind = np.zeros(nMaxEntries, dtype='int64')
col_ind = np.zeros(nMaxEntries, dtype='int64')

PBentries = (2*ndim)**3
PBmaxEntries = PBentries * nQuads * NX
PBdata = np.zeros(PBmaxEntries)
PBind0 = np.zeros(PBmaxEntries, dtype='int64')
PBind1 = np.zeros(PBmaxEntries, dtype='int64')
PBind2 = np.zeros(PBmaxEntries, dtype='int64')
PBindex = 0

##### compute spatial discretizaton
index = 0
for iPlane in range(NX):
    ##### generate quadrature points
    offsets, weights = roots_legendre(Nquad)
    offsets = [offsets * dx[iPlane] / 2, offsets * dy[iPlane,0] / 2]
    weights = [weights * dx[iPlane] / 2, weights * dy[iPlane,0] / 2]
    quads = ( np.indices([1, NY], dtype='float').T.reshape(-1, ndim) + 0.5 ) \
          * [dx[iPlane], dy[iPlane,0]]
    quadWeights = np.repeat(1., len(quads))
    for i in range(ndim):
        quads = np.concatenate(
            [quads + offset*np.eye(ndim)[i] for offset in offsets[i]] )
        quadWeights = np.concatenate(
            [quadWeights * weight for weight in weights[i]] )
    phiX = quads[:,0] / dx[iPlane]
    mapL = mapping(quads + [nodeX[iPlane], 0], nodeX[iPlane])
    mapR = mapping(quads + [nodeX[iPlane], 0], nodeX[iPlane+1])
    indL = (np.searchsorted(nodeYfull[iPlane], mapL, side='right') - 1) % NY
    indR = (np.searchsorted(nodeYfull[(iPlane+1) % NX], mapR, side='right') - 1)%NY
    phiLY = (mapL - nodeYfull[iPlane][indL]) / dy[0][indL]
    phiRY = (mapR - nodeYfull[iPlane][indR]) / dy[0][indR]
    
    grad = np.array([-1/arcLengths[iPlane], -1/dy[0][0]])
    for iQ, quad in enumerate(quads):
        BC = np.array([[indL[iQ] != 0], [indL[iQ] != NY-1],
                       [indR[iQ] != 0], [indR[iQ] != NY-1]])
        phis = np.array([[(1-phiLY[iQ]), (1-phiX[iQ])],
                         [  phiLY[iQ]  , (1-phiX[iQ])],
                         [(1-phiRY[iQ]),   phiX[iQ]  ],
                         [  phiRY[iQ]  ,   phiX[iQ]  ]])
        phis *= BC
        gradphis = np.vstack((grad          , grad * [1, -1], 
                              grad * [-1, 1], grad * [-1, -1])) * phis
        phis = np.prod(phis, axis=1)
        indices = np.array([indL[iQ] - 1 + (NY-1)*iPlane,
                            indL[iQ] + (NY-1)*iPlane,
                           (indR[iQ] - 1 + (NY-1)*(iPlane+1)) % nNodes,
                           (indR[iQ] + (NY-1)*(iPlane+1)) % nNodes])
        indices *= BC.ravel()
        Kdata[index:index+nEntries] = quadWeights[iQ] * \
            np.ravel( gradphis @ gradphis.T )
        Mdata[index:index+nEntries] = quadWeights[iQ] * \
            np.outer(phis, phis).ravel()
        DDXdata[index:index+nEntries] = quadWeights[iQ] * \
            np.outer(phis, gradphis[:,0]).ravel()
        row_ind[index:index+nEntries] = np.repeat(indices, 2*ndim)
        col_ind[index:index+nEntries] = np.tile(indices, 2*ndim)
        index += nEntries
        
        PBdata[PBindex:PBindex+PBentries] = quadWeights[iQ] * \
            np.outer(phis, np.outer(gradphis[:,0], gradphis[:,1]) - np.outer(gradphis[:,1], gradphis[:,0])).ravel()
        PBind0[PBindex:PBindex+PBentries] = np.repeat(indices, (2*ndim)**2)
        PBind1[PBindex:PBindex+PBentries] = np.tile(np.repeat(indices, 2*ndim), 2*ndim)
        PBind2[PBindex:PBindex+PBentries] = np.tile(indices, (2*ndim)**2)
        PBindex += PBentries
        
ztol = 14 # number of decimal places at which to round to zero

NZinds = np.flatnonzero(Kdata.round(decimals = ztol)) 
K = sp.csr_matrix( (Kdata[NZinds], (row_ind[NZinds], col_ind[NZinds])),
                   shape=(nNodes, nNodes) )

NZinds = np.flatnonzero(Mdata.round(decimals = ztol)) 
M = sp.csr_matrix( (Mdata[NZinds], (row_ind[NZinds], col_ind[NZinds])),
                   shape=(nNodes, nNodes) )

DDX = sp.csr_matrix( (DDXdata, (row_ind, col_ind)), shape=(nNodes, nNodes) )
np.around(DDX.data, decimals = ztol, out = DDX.data)
DDX.eliminate_zeros()

PB = sparse.COO((PBind0, PBind1, PBind2), PBdata,
                shape=(nNodes, nNodes, nNodes))
NZinds = np.flatnonzero(PB.data.round(decimals = ztol))
PB = sparse.COO(PB.coords[:,NZinds], PB.data[NZinds],
                shape=(nNodes, nNodes, nNodes))

del row_ind, col_ind, PBind0, PBind1, PBind2, Kdata, Mdata, DDXdata, PBdata, NZinds
gc.collect()

# a = np.zeros(1)
# PB = sparse.COO((a, a, a), a,
#                 shape=(nNodes, nNodes, nNodes))

DDX *= kappa
K *= -1.

print('complete\nSetting initial conditions... ', end='')

##### set initial conditions
t = 0
pot = np.zeros((nSteps+1, nNodes))
vort = np.zeros((nSteps+1, nNodes))
n = np.zeros((nSteps+1, nNodes))

nModes = int(NX/2)
scale = 0.1 / nModes
for ix in range(NX):
    for iy in range((NY-1)):
        px = nodeX[ix] / sizeX
        py = mapping(np.array([[px, nodeY[ix][iy]]]), 0) / sizeY
        # vort[0, ix*(NY-1) + iy] = 0.1*np.sin(2*np.pi*px + np.pi/2)
        for i in np.arange(1, nModes+1):
             vort[0, ix*(NY-1) + iy] += scale * np.cos(i * 2 * np.pi * px)
pot[0], info = sp_la.lgmres(K, M @ vort[0], tol=1e-10, atol=1e-10)

# ##### only used for RK4 time-stepping
# dudt = np.zeros(nNodes)
# betas = np.array([0.25, 1/3, 0.5, 1])

print('complete\nBeginning simulation...')

sp_pot = sp.csc_matrix(np.ones((nNodes,1)))

def step(nSteps=1):
    global pot, vort, n, K, M, DDX, PB, sp_pot, t
    for i in range(nSteps):
        t += 1
        print(f'\rt = {t}', end='')
        pot[t], info = sp_la.lgmres(K, M @ vort[t-1], x0=pot[t-1], tol=1e-10, atol=1e-10)
        if (info != 0):
            raise SystemExit(f'pot solution failed with error code: {info}')
        PB2 = sp.csr_matrix( (PB.data * pot[t][PB.coords[2]], (PB.coords[0], PB.coords[1])),  shape=(nNodes, nNodes) )
        ##### Backward-Euler #####
        n[t], info = sp_la.lgmres(M*(alpha + 1/dt) + PB2, M/dt @ n[t-1] + (M*alpha - DDX) @ pot[t], x0=n[t-1], tol=1e-10, atol=1e-10)
        if (info != 0):
            raise SystemExit(f'n solution failed with error code: {info}')
        vort[t], info = sp_la.lgmres(M/dt + PB2, M/dt @ vort[t-1] + M*alpha @ (pot[t] - n[t]), x0=vort[t-1], tol=1e-10, atol=1e-10)
        if (info != 0):
            raise SystemExit(f'vort solution failed with error code: {info}')
        # ##### RK4 #####
        # uTemp = u
        # for beta in betas:
        #     dudt, info = sp_la.cg(M, DDX @ uTemp, x0=dudt, tol=1e-10, atol=1e-10)
        #     # self.dudt = sp_la.spsolve(self.M, self.KA@uTemp)
        #     uTemp = u + beta * dt * dudt
        #     if (info != 0):
        #         print(f'solution failed with error code: {info}')
        # u = uTemp

step(nSteps)

X = np.repeat(nodeX, NY+1).reshape(NX+1, NY+1)
Y = np.vstack((nodeYfull, nodeYfull[0]))

plotType = 'contour'

def init_plot():
    global field, fig, ax, X, Y, U, pot, vort, n, maxAbsU, NX, NY, plotType, t
    fig, ax = plt.subplots()
    U = np.hstack((np.zeros((NX,1)), n[t].reshape(NX, NY-1), np.zeros((NX,1))))
    U = np.vstack((U, U[0]))
    # maxAbsU = np.max(np.abs(U))
    maxAbsU = 2e-3
    if plotType.lower().startswith('t'):
        field = ax.tripcolor(X.ravel(), Y.ravel(), U.ravel(), shading='gouraud'
                             , cmap='seismic', vmin=-maxAbsU, vmax=maxAbsU
                              )
    elif plotType.lower().startswith('c'):
        field = ax.contourf(X, Y, U, levels=np.linspace(-maxAbsU, maxAbsU, 25)
                            # , cmap='seismic'
                            , extend='both'
                            )
    # tri = mpl.tri.Triangulation(X,Y)
    # ax.triplot(tri, 'r-', lw=1)
    # x = np.linspace(0, nodeX[-1], 100)
    # for yi in [0.4, 0.5, 0.6]:
    #     ax.plot(x, [mapping(np.array([[0, yi]]), i) for i in x], 'k')
    # for xi in nodeX:
    #     ax.plot([xi, xi], [0, 1], 'k:')
    # ax.plot(X[np.argmax(U)], Y[np.argmax(U)],  'g+', markersize=10)
    plt.colorbar(field)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$', rotation=0)
    plt.margins(0,0)
    plt.title(f't = {t}')
    return [field]

# init_plot()

# U_interp = RectBivariateSpline(X[:,0], Y[0], U)
# x = np.arange(dx[0]/2, 1, dx[0])
# U_BOUT = U_interp(x, x)
# plt.contourf(x,x,np.flipud(U_BOUT.T))
# plt.contourf(x,x,U_real)
# plt.contourf(x, x, np.flipud(U_BOUT.T) - U_real, levels=np.linspace(-0.1, 0.1, 25), cmap='seismic')
# plt.colorbar()
# plt.show()

# field = ax.tripcolor(X, Y, U, shading='gouraud'
#                           ,cmap='seismic', vmin=-maxAbsU, vmax=maxAbsU
#                           )

# def animate(i):
#     global field, pot, vort, n, NX, NY, U, t
#     step(1)
#     U = np.hstack((np.zeros((NX,1)), n[t].reshape(NX, NY-1), np.zeros((NX,1))))
#     U = np.vstack((U, U[0]))
#     if plotType.lower().startswith('t'):
#         field.set_array(U.ravel())
#     elif plotType.lower().startswith('c'):
#         ax.clear()
#         field = ax.contourf(X, Y, U, levels=np.linspace(-maxAbsU, maxAbsU, 25))
#         plt.title(f't = {t}')
#     return [field]

# ani = animation.FuncAnimation(
#     fig, animate, frames=nSteps, interval=60)

# ani.save('movie.mp4', writer='ffmpeg', dpi=200)

pot.shape = (nSteps+1, NX, NY-1)
n.shape = (nSteps+1, NX, NY-1)
vort.shape = (nSteps+1, NX, NY-1)

pot_slice = pot[:,:,int(NY/2)-1]
# n_slice = n[:,:,int(NY/2)-1]
# vort_slice = vort[:,:,int(NY/2)-1]

kxFreq = fftshift( fftfreq(NX, d=dx[0]) )

# Take the spatial FFT
Pot  = fftshift( fft( pot_slice, axis=1, norm="ortho"), axes=1 )
# N    = fftshift( fft(   n_slice, axis=1, norm="ortho"), axes=1 )
# Vort = fftshift( fft(vort_slice, axis=1, norm="ortho"), axes=1 )

omega = fftshift( fftfreq(nSteps+1, d=dt) )

# Take the temporal FFT
PotOmega  = fftshift( fft( Pot, axis=0, norm="ortho"), axes=0 )
# NOmega    = fftshift( fft(   N, axis=0, norm="ortho"), axes=0 )
# VortOmega = fftshift( fft(Vort, axis=0, norm="ortho"), axes=0 )

a = 10
omegaR = -1/a*(a*kxFreq)/(1 + (a*kxFreq)**2)
gamma = (a*kxFreq)**2*omegaR/(1+(a*kxFreq)**2)

# Normalize to real part for each k_z
for row in PotOmega.T:
    row /= np.max(np.abs(np.real(row)))
    # print(np.max(np.abs(np.real(row))))

plt.subplot(1,2,1)
# plt.contourf(kxFreq, omega, np.real(PotOmega))
plt.imshow(np.real(PotOmega), interpolation='bilinear', origin='lower',
            extent=(kxFreq[0], kxFreq[-1], omega[0], omega[-1]), aspect='auto')
plt.plot(kxFreq, omegaR, 'r')
plt.title(r'real($\omega$)')
plt.ylabel(r'$\omega_R$')
plt.xlabel(r'$k_x$')
plt.xlim(-5, 5)
plt.ylim(-0.1, 0.1)
plt.colorbar(shrink=0.8)

# Normalize to imaginary part for each k_z
for row in PotOmega.T:
    row /= np.max(np.abs(np.imag(row)))
    # print(np.max(np.abs(np.imag(row))))

plt.subplot(1,2,2)
plt.imshow(np.imag(PotOmega), interpolation='bilinear', origin='lower',
            extent=(kxFreq[0], kxFreq[-1], omega[0], omega[-1]), aspect='auto')
plt.plot(kxFreq, gamma, 'r')
plt.title(r'imag($\omega$)')
plt.ylabel(r'$\gamma$')
plt.xlabel(r'$k_x$')
plt.xlim(-5, 5)
plt.ylim(-0.1, 0.1)
plt.colorbar(shrink=0.8)

plt.show()

# temp = np.zeros((nSteps+1, NX+1, NY+1))

# temp[:,0:NX,1:NY] = pot
# temp[:,-1,1:NY] = pot[:,0,:]
# pot = temp

# temp[:,0:NX,1:NY] = n
# temp[:,-1,1:NY] = n[:,0,:]
# n = temp

# temp[:,0:NX,1:NY] = vort
# temp[:,-1,1:NY] = vort[:,0,:]
# vort = temp

# del temp