# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 13:47:07 2020

@author: samal
"""

from scipy.special import roots_legendre
import scipy.integrate
import scipy.sparse as sp
import scipy.sparse.linalg as sp_la
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def mapping(points, x=0., theta=False, deriv=False):
    A = 0.2
    phase = -np.pi/2
    if deriv:
        return A*np.cos(points[:,0] - phase)
    elif theta:
        return np.arctan(A*np.cos(points[:,0] - phase))
    else:
        offsets = points[:,1] - A*np.sin(points[:,0] - phase)
        return (A*np.sin(x - phase) + offsets) % 1

def f(points):
    return np.repeat(0., len(points))

velocity = np.array([1, 0.])

nSteps = 209
dt = 0.01

NX = 3  # number of planes
NY = 20 # number of grid divisions on plane
nNodes = NX*NY

Nquad = 20
ndim = 2

dx = 2*np.pi/NX
dy = 1/NY

arcLengths = np.array([scipy.integrate.quad(lambda x: 
    np.sqrt(1 + mapping(np.array([[x,0]]), deriv=True)**2),
    i*dx, (i+1)*dx)[0] for i in range(NX)])

nodeX = np.array([i*dx for i in range(NX)])
nodeY = np.linspace(0, 1-dy, NY)
# nodeY = [np.linspace(0, 1, NY) for i in range(NX)]

# generate quadrature points
offsets, weights = roots_legendre(Nquad)
offsets = [offsets * np.pi / NX, offsets / (2*NY)]
weights = [weights * np.pi / NX, weights / (2*NY)]
quads = ( np.indices([1, NY], dtype='float').T.reshape(-1, ndim) + 0.5 ) \
      * [2*np.pi/NX, 1/NY]
quadWeights = np.repeat(1., len(quads))
for i in range(ndim):
    quads = np.concatenate( [quads + 
        offset*np.eye(ndim)[i] for offset in offsets[i]] )
    quadWeights = np.concatenate(
        [quadWeights * weight for weight in weights[i]] )
nQuads = len(quads)

# pre-allocate arrays for mass matrix triplets
nEntries = (2*ndim)**2
nMaxEntriesM = nEntries * nQuads * NX
Mdata = np.zeros(nMaxEntriesM)
row_ind = np.zeros(nMaxEntriesM, dtype='int')
col_ind = np.zeros(nMaxEntriesM, dtype='int')

# compute mass matrix
index = 0
phiX = quads[:,0] / dx
for iPlane in range(NX):
    mapL = mapping(quads + [iPlane*dx, 0], iPlane * dx)
    mapR = mapping(quads + [iPlane*dx, 0], (iPlane+1) * dx)
    indL = (mapL // dy).astype('int')
    indR = (mapR // dy).astype('int')
    phiLY = (mapL - indL*dy) / dy
    phiRY = (mapR - indR*dy) / dy
    for iQ, quad in enumerate(quads):
        phis = np.array([[(1-phiLY[iQ]), (1-phiX[iQ])],
                         [  phiLY[iQ]  , (1-phiX[iQ])],
                         [(1-phiRY[iQ]),   phiX[iQ]  ],
                         [  phiRY[iQ]  ,   phiX[iQ]  ]])
        phis = np.prod(phis, axis=1)
        indices = np.array([indL[iQ] + NY*iPlane,
                            (indL[iQ]+1) % NY + NY*iPlane,
                            (indR[iQ] + NY*(iPlane+1)) % nNodes,
                            ((indR[iQ]+1) % NY + NY*(iPlane+1)) % nNodes])
        Mdata[index:index+nEntries] = quadWeights[iQ] * \
            np.outer(phis, phis).ravel()
        row_ind[index:index+nEntries] = np.repeat(indices, 2*ndim)
        col_ind[index:index+nEntries] = np.tile(indices, 2*ndim)
        index += nEntries
M = sp.csr_matrix( (Mdata, (row_ind, col_ind)), shape=(nNodes, nNodes) )

# pre-allocate arrays for convection matrix triplets
nEntries = ndim + 1
nMaxEntriesA = nEntries * nNodes
Adata = np.zeros(nMaxEntriesA)
row_ind = np.zeros(nMaxEntriesA, dtype='int')
col_ind = np.zeros(nMaxEntriesA, dtype='int')

# compute convection matrix
index = 0
magV = np.linalg.norm(velocity)
nodeWeight = dx*dy
nodeIndices = np.arange(-NY, 0)
for iPlane in range(NX):
    nodes = np.vstack((np.repeat(nodeX[iPlane], NY), nodeY)).T
    nodeIndices += NY
    mapL = mapping(nodes, (iPlane - 1)*dx)
    indL = (mapL // dy).astype('int')
    phiLY = (mapL - indL*dy) / dy
    grad = np.array([[-1/arcLengths[iPlane]], [-1/dy]])
    for iN, nodes in enumerate(nodes):
        phis = np.array([(1-phiLY[iN]), phiLY[iN], -1])
        indices = np.array([NY*(iPlane - 1) + indL[iN],
                            NY*(iPlane - 1) + (indL[iN]+1) % NY,
                            nodeIndices[iN]])
        Adata[index:index+nEntries] = phis * magV * nodeWeight * grad[0,0]
        row_ind[index:index+nEntries] = np.repeat(nodeIndices[iN], 3)
        col_ind[index:index+nEntries] = indices
        index += nEntries
col_ind[col_ind<0] += nNodes
A = sp.csr_matrix( (Adata, (row_ind, col_ind)), shape=(nNodes, nNodes) )

# Backward-Euler
M /= dt
K = M - A

u = np.zeros(nNodes)
# u[15] = 1;
A = 1.0
r0 = 0.5
sigma = 0.1
for ix in range(NX):
    for iy in range(NY):
        u[ix*NY + iy] = A*np.exp( -0.5*( ((ix*dx-r0)/sigma)**2
                                       + ((iy*dy-r0)/sigma)**2 ) ) # Gaussian
        # u[ix*NY + iy] = np.sin(ix*dx - np.pi/2) * np.sin(iy*dy*2*np.pi)

dudt = np.zeros(nNodes)
betas = np.array([0.25, 1/3, 0.5, 1])
for i in range(nSteps):
    # uTemp = u
    # for beta in betas:
    #     dudt, info = sp_la.cg(M, A @ uTemp, x0=dudt, tol=1e-10, atol=1e-10)
    #     # self.dudt = sp_la.spsolve(self.M, self.KA@uTemp)
    #     uTemp = u + beta * dt * dudt
    #     if (info != 0):
    #         print(f'solution failed with error code: {info}')
    # u = uTemp
    u, info = sp_la.lgmres(K, M @ u, u, tol=1e-10, atol=1e-10) # Backward-Euler

periodicIndices = np.concatenate([np.append(np.arange(NY),0) + i*NY
                                  for i in range(NX+1)]) % nNodes
X = np.repeat(np.append(nodeX, 2*np.pi), NY+1)
Y = np.tile(np.append(nodeY, 1), NX+1)
U = u[periodicIndices]

uQuad = np.empty(nQuads * NX)
for iPlane in range(NX):
    mapL = mapping(quads + [iPlane*dx, 0], iPlane * dx)
    mapR = mapping(quads + [iPlane*dx, 0], (iPlane+1) * dx)
    indL = (mapL // dy).astype('int')
    indR = (mapR // dy).astype('int')
    phiLY = (mapL - indL*dy) / dy
    phiRY = (mapR - indR*dy) / dy
    for iQ, quad in enumerate(quads):
        phis = np.array([(1-phiLY[iQ]) * (1-phiX[iQ]), phiLY[iQ] * (1-phiX[iQ]),
                         (1-phiRY[iQ]) * phiX[iQ]    , phiRY[iQ] * phiX[iQ]])
        indices = np.array([indL[iQ] + NY*iPlane,
                           (indL[iQ]+1) % NY + NY*iPlane,
                           (indR[iQ] + NY*(iPlane+1)) % nNodes,
                           ((indR[iQ]+1) % NY + NY*(iPlane+1)) % nNodes])
        uQuad[iPlane*nQuads + iQ] = np.sum(phis * u[indices])

# For plotting only the quad points
# X = np.concatenate([quads[:,0] + i*dx for i in range(NX)])
# Y = np.tile(quads[:,1], NX)
# U = uQuad

# For concatenating the quad points to the node points
X = np.concatenate([X] + [quads[:,0] + i*dx for i in range(NX)])
Y = np.concatenate((Y, np.tile(quads[:,1], NX)))
U = np.concatenate((U, uQuad))

plt.tripcolor(X, Y, U, shading='gouraud'
              ,cmap='seismic'
              ,vmin=-np.max(np.abs(U))
              ,vmax=np.max(np.abs(U))
              )
# tri = mpl.tri.Triangulation(X,Y)
# plt.triplot(tri, 'r-', lw=1)
x = np.linspace(0, 2*np.pi, 100)
plt.plot(x, [mapping(np.array([[0, 0.75]]), i) for i in x], 'k')
for xi in nodeX:
    plt.plot([xi, xi], [0, 1], 'k:')
plt.plot(X[np.argmax(U)], Y[np.argmax(U)],  'g+', markersize=10)
plt.colorbar()
plt.xlabel(r'$x$')
plt.ylabel(r'$y$', rotation=0)
plt.xticks(np.linspace(0, 2*np.pi, 7), 
    ['0',r'$\pi/3$',r'$2\pi/3$',r'$\pi$',r'$4\pi/3$',r'$5\pi/3$',r'$2\pi$'])
plt.margins(0,0)
plt.show()