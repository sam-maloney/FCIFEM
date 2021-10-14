# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 13:47:07 2020

@author: samal
"""
from utils.plotdata import plotdata
from utils.showdata import showdata

from scipy.special import roots_legendre
from scipy.interpolate import RectBivariateSpline
try:
    from scipy.fft import fft, fftshift, fftfreq
except(ModuleNotFoundError):
    from scipy.fftpack import fft, fftshift, fftfreq
import gc
import sparse
import scipy.integrate
import scipy.sparse as sp
import scipy.sparse.linalg as sp_la
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def mapping(points, x=0., theta=False, deriv=False):
    if deriv or theta:
        return np.repeat(0., len(points))
    else:
        return points[:,1]

dt = 1
nSteps = 127

Nquad = 2
ndim = 2

alpha = 1.0    # Adiabaticity (~conductivity)
kappa = 0.0    # Density gradient drive

YBC = 'd' # 'dirichlet' or 'periodic' (or 'd' and 'p')
if not (YBC.lower().startswith('d') or YBC.lower().startswith('p')):
    raise SystemExit(f"Unknown y boundary condition: '{YBC}'")

##### uniform grid spacing
NX = 100 # number of planes
NY = 100 # number of grid divisions on each plane
sizeX = 5
sizeY = 5
nodeX = sizeX*np.arange(NX+1)/NX
nodeY = sizeY*np.tile(np.linspace(0, 1, NY+1), NX+1).reshape(NX+1,-1)

##### non-uniform X spacing
# nodeX = np.array([0., 1, 2.5, 3.5, 5, 2*np.pi])
# NX = len(nodeX) - 1
# nodeY = sizeY*np.tile(np.linspace(0, 1, NY+1), NX).reshape(NX,-1)

if YBC.lower().startswith('d'):
    nYNodes = NY-1
elif YBC.lower().startswith('p'):
    nYNodes = NY

nNodes = NX * nYNodes

dx = nodeX[1:]-nodeX[0:-1]
dy = nodeY[:,1:]-nodeY[:,:-1]

arcLengths = np.array([scipy.integrate.quad(lambda x: 
    np.sqrt(1 + mapping(np.array([[x,0]]), deriv=True)**2),
    nodeX[i], nodeX[i+1])[0] for i in range(NX)])

print('Computing spatial discretization... ', end='')

##### pre-allocate arrays for stiffness matrix triplets
nEntries = (2*ndim)**2
nQuads = NY * Nquad**ndim
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
    indL = (np.searchsorted(nodeY[iPlane], mapL, side='right') - 1) % NY
    indR = (np.searchsorted(nodeY[iPlane+1], mapR, side='right') - 1) % NY
    phiLY = (mapL - nodeY[iPlane,indL]) / dy[iPlane,indL]
    phiRY = (mapR - nodeY[iPlane,indR]) / dy[iPlane,indR]
    
    grad = np.array([-1/arcLengths[iPlane], -1/dy[iPlane][0]])
    for iQ, quad in enumerate(quads):
        phis = np.array([[(1-phiLY[iQ]), (1-phiX[iQ])],
                         [  phiLY[iQ]  , (1-phiX[iQ])],
                         [(1-phiRY[iQ]),   phiX[iQ]  ],
                         [  phiRY[iQ]  ,   phiX[iQ]  ]])
        if YBC.lower().startswith('d'):
            indices = np.array([ indL[iQ] - 1 + (NY-1)*iPlane,
                                 indL[iQ] + (NY-1)*iPlane,
                                (indR[iQ] - 1 + (NY-1)*(iPlane+1)) % nNodes,
                                (indR[iQ] + (NY-1)*(iPlane+1)) % nNodes])
            BC = np.array([[indL[iQ] != 0],
                           [indL[iQ] != NY-1],
                           [indR[iQ] != 0],
                           [indR[iQ] != NY-1]])
            phis *= BC
            indices *= BC.ravel()
        elif YBC.lower().startswith('p'):
            indices = np.array([indL[iQ] + NY*iPlane,
                               (indL[iQ]+1) % NY + NY*iPlane,
                               (indR[iQ] + NY*(iPlane+1)) % nNodes,
                               ((indR[iQ]+1) % NY + NY*(iPlane+1)) % nNodes])
        gradphis = np.vstack((grad          , grad * [ 1, -1], 
                              grad * [-1, 1], grad * [-1, -1])) * phis
        phis = np.prod(phis, axis=1)
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
# PB = sparse.COO((a, a, a), a, shape=(nNodes, nNodes, nNodes))

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
    for iy in range(nYNodes):
        px = 2 * np.pi * nodeX[ix] / sizeX
        if YBC.lower().startswith('d'):
            py = 2 * np.pi * mapping(np.array([[px, nodeY[ix][iy+1]]]), 0)[0] / sizeY
        elif YBC.lower().startswith('p'):
            py = 2 * np.pi * mapping(np.array([[px, nodeY[ix][iy]]]), 0)[0] / sizeY
        vort[0, ix*nYNodes + iy] = 0.1*(np.sin(2*px) + np.sin(3*py))
        # for i in np.arange(1, nModes+1):
        #       vort[0, ix*nYNodes + iy] += scale * np.cos(i * px)
pot[0], info = sp_la.lgmres(K, M @ vort[0], tol=1e-10, atol=1e-10)

# ##### only used for RK4 time-stepping
# dudt = np.zeros(nNodes)
# betas = np.array([0.25, 1/3, 0.5, 1])

print('complete\nBeginning simulation...')

def step(nSteps=1):
    global pot, vort, n, K, M, DDX, PB, t
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
Y = nodeY

pot.shape = (nSteps+1, NX, -1)
n.shape = (nSteps+1, NX, -1)
vort.shape = (nSteps+1, NX, -1)

temp = np.zeros((nSteps+1, NX+1, NY+1))

temp[:,0:NX,-(nYNodes+1):-1] = pot
temp[:,-1,:] = temp[:,0,:] # periodic in X
if YBC.lower().startswith('p'):
    temp[:,:,-1] = temp[:,:,0] # periodic in Y
pot = temp.copy()

temp[:,0:NX,-(nYNodes+1):-1] = n
temp[:,-1,:] = temp[:,0,:] # periodic in X
if YBC.lower().startswith('p'):
    temp[:,:,-1] = temp[:,:,0] # periodic in Y
n = temp.copy()

temp[:,0:NX,-(nYNodes+1):-1] = vort
temp[:,-1,:] = temp[:,0,:] # periodic in X
if YBC.lower().startswith('p'):
    temp[:,:,-1] = temp[:,:,0] # periodic in Y
vort = temp.copy()

del temp

# plotdata(n[-1], x=X, y=Y)

# showdata(pot, x=X, y=Y, titles='pot', hold_aspect=True, movie='pot.mp4')
# showdata(n, x=X, y=Y, titles='n', hold_aspect=True, movie='n.mp4')
# showdata(vort, x=X, y=Y, titles='vort', hold_aspect=True, movie='vort.mp4')

##### Plot dispersion relation #####

normalize = False

pot_slice = pot[:,:-1,int(NY/2)]
# n_slice = n[:,:-1,int(NY/2)]
# vort_slice = vort[:,:-1,int(NY/2)]

kxFreq = fftshift( fftfreq(NX, d=dx[0]) )

# Take the spatial FFT
Pot  = fftshift( fft( pot_slice, axis=1, norm="ortho"), axes=1 )
# N    = fftshift( fft(   n_slice, axis=1, norm="ortho"), axes=1 )
# Vort = fftshift( fft(vort_slice, axis=1, norm="ortho"), axes=1 )

# showdata(np.abs(Pot), x=kxFreq, titles='Pot', hold_aspect=True, movie='PotF.mp4')
# showdata(np.abs(Vort), x=kxFreq, titles='Vort', hold_aspect=True, movie='VortF.mp4')
# showdata(np.abs(N), x=kxFreq, titles='N', hold_aspect=True, movie='NF.mp4')

omega = fftshift( fftfreq(nSteps+1, d=dt) )

# Take the temporal FFT
PotOmega  = fftshift( fft( Pot, axis=0, norm="ortho"), axes=0 )
# NOmega    = fftshift( fft(   N, axis=0, norm="ortho"), axes=0 )
# VortOmega = fftshift( fft(Vort, axis=0, norm="ortho"), axes=0 )

a = 10
omegaR = (-1/a)*(a*kxFreq)/(1 + (a*kxFreq)**2)
gamma = (a*kxFreq)**2*omegaR/(1+(a*kxFreq)**2)

# Normalize to real part for each k_x
if normalize:
    for row in PotOmega.T:
        row /= np.max(np.abs(np.real(row)))

plt.subplot(1,2,1)
plotdata(np.real(PotOmega).T, x=kxFreq, y=omega, aspect='auto',
          title=r'real($\omega$)', xtitle=r'$k_x$', ytitle=r'$\gamma$')
plt.plot(kxFreq, omegaR, 'r')
plt.xlim(-5, 5)
plt.ylim(-0.1, 0.1)

# Normalize to imaginary part for each k_x
if normalize:
    for row in PotOmega.T:
        row /= np.max(np.abs(np.imag(row)))

plt.subplot(1,2,2)
plotdata(np.imag(PotOmega).T, x=kxFreq, y=omega, aspect='auto',
          title=r'imag($\omega$)', xtitle=r'$k_x$', ytitle=r'$\gamma$')
plt.plot(kxFreq, gamma, 'r')
plt.xlim(-5, 5)
plt.ylim(-0.1, 0.1)

plt.show()