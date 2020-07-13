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
        return (A*np.sin(x - phase) + offsets) % 1 % 1

# mapping(np.array([[9*np.pi/6, 0.1]]), 4*np.pi/3)

def f(points):
    return np.repeat(0., len(points))

NX = 5  # number of planes
NY = 20 # number of grid divisions on plane
nNodes = NX*NY

Nquad = 20
ndim = 2

# nodeX = 2*np.pi*np.arange(NX+1)/NX
nodeX = np.array([0., 1, 2.5, 3.5, 5, 2*np.pi])
nodeY = np.tile(np.linspace(0, 1, NY+1), NX).reshape(NX,-1)

dx = nodeX[1:]-nodeX[0:NX]
dy = nodeY[:,1:]-nodeY[:,:NY]
# dy = 1/NY

# velocity = np.array([1., 0.])
velocity = np.array([1., 0.03183098861837907])
# velocity = np.array([0., 0.])

D_a = 0.
D_i = 0.
theta = np.pi/4
diffusivity = D_a*np.array([[np.cos(theta)**2, np.sin(theta)*np.cos(theta)],
                            [np.sin(theta)*np.cos(theta), np.sin(theta)**2]])
diffusivity += D_i*np.eye(2)

arcLengths = np.array([scipy.integrate.quad(lambda x: 
    np.sqrt(1 + mapping(np.array([[x,0]]), deriv=True)**2),
    nodeX[i], nodeX[i+1])[0] for i in range(NX)])

dt = 0.01
# nSteps = int(np.rint(np.sum(dx[0:3])/dt))
nSteps = 314

# # generate quadrature points
# offsets, weights = roots_legendre(Nquad)
# offsets = [offsets * np.pi / NX, offsets / (2*NY)]
# weights = [weights * np.pi / NX, weights / (2*NY)]
# quads = ( np.indices([1, NY], dtype='float').T.reshape(-1, ndim) + 0.5 ) \
#       * [2*np.pi/NX, 1/NY]
# quadWeights = np.repeat(1., len(quads))
# for i in range(ndim):
#     quads = np.concatenate( [quads + 
#         offset*np.eye(ndim)[i] for offset in offsets[i]] )
#     quadWeights = np.concatenate(
#         [quadWeights * weight for weight in weights[i]] )

# pre-allocate arrays for stiffness matrix triplets
nEntries = (2*ndim)**2
nQuads = NY*Nquad**2
nMaxEntries = nEntries * nQuads * NX
Kdata = np.zeros(nMaxEntries)
Adata = np.zeros(nMaxEntries)
Mdata = np.zeros(nMaxEntries)
row_ind = np.zeros(nMaxEntries, dtype='int')
col_ind = np.zeros(nMaxEntries, dtype='int')
b = np.zeros(nNodes)

# offsets_old, weights_old = roots_legendre(Nquad)
# offsets_old = [offsets_old * np.pi / NX, offsets_old / (2*NY)]
# weights_old = [weights_old * np.pi / NX, weights_old / (2*NY)]
# quads_old = ( np.indices([1, NY], dtype='float').T.reshape(-1, ndim) + 0.5 ) \
#       * [2*np.pi/NX, 1/NY]
# quadWeights_old = np.repeat(1., len(quads_old))
# for i in range(ndim):
#     quads_old = np.concatenate( [quads_old + 
#         offset*np.eye(ndim)[i] for offset in offsets_old[i]] )
#     quadWeights_old = np.concatenate(
#         [quadWeights_old * weight for weight in weights_old[i]] )

# phiX_old = quads_old[:,0] / (2*np.pi/NX)
# Kdata_old = np.zeros(nMaxEntries)
# Adata_old = np.zeros(nMaxEntries)
# Mdata_old = np.zeros(nMaxEntries)
# row_ind_old = np.zeros(nMaxEntries, dtype='int')
# col_ind_old = np.zeros(nMaxEntries, dtype='int')

# compute spatial discretizaton
index = 0
for iPlane in range(NX):
    # generate quadrature points
    offsets, weights = roots_legendre(Nquad)
    offsets = [offsets * dx[iPlane] / 2, offsets / (2*NY)]
    weights = [weights * dx[iPlane] / 2, weights / (2*NY)]
    quads = ( np.indices([1, NY], dtype='float').T.reshape(-1, ndim) + 0.5 ) \
          * [dx[iPlane], 1/NY]
    quadWeights = np.repeat(1., len(quads))
    for i in range(ndim):
        quads = np.concatenate( [quads + 
            offset*np.eye(ndim)[i] for offset in offsets[i]] )
        quadWeights = np.concatenate(
            [quadWeights * weight for weight in weights[i]] )
    phiX = quads[:,0] / dx[iPlane]
    mapL = mapping(quads + [nodeX[iPlane], 0], nodeX[iPlane])
    mapR = mapping(quads + [nodeX[iPlane], 0], nodeX[iPlane+1])
    indL = (np.searchsorted(nodeY[iPlane], mapL, side='right') - 1) % NY
    indR = (np.searchsorted(nodeY[(iPlane+1) % NX], mapR, side='right') - 1)%NY
    phiLY = (mapL - nodeY[iPlane][indL]) / dy[0][indL]
    phiRY = (mapR - nodeY[iPlane][indR]) / dy[0][indR]
    
    # mapL_old = mapping(quads_old + [iPlane*2*np.pi/NX, 0], iPlane * 2*np.pi/NX)
    # mapR_old = mapping(quads_old + [iPlane*2*np.pi/NX, 0], (iPlane+1) * 2*np.pi/NX)
    # indL_old = (mapL_old // (1/NY)).astype('int')
    # indR_old = (mapR_old // (1/NY)).astype('int')
    # phiLY_old = (mapL_old - indL_old/NY) * NY
    # phiRY_old = (mapR_old - indR_old/NY) * NY
    # assert np.allclose(mapL, mapL_old)
    # assert np.allclose(mapR, mapR_old)
    # assert np.allclose(indL, indL_old)
    # assert np.allclose(indR, indR_old)
    # assert np.allclose(phiLY, phiLY_old)
    # assert np.allclose(phiRY, phiRY_old)
    # assert np.allclose(offsets, offsets_old)
    # assert np.allclose(weights, weights_old)
    # assert np.allclose(quads, quads_old)
    # assert np.allclose(quadWeights, quadWeights_old)
    # grad_old = np.array([[-1/arcLengths[iPlane]], [-NY]])
    # grads = np.tile(grad_old, nQuads).T
    
    grad = np.array([-1/arcLengths[iPlane], -1/dy[0][0]])
    for iQ, quad in enumerate(quads):
        phis = np.array([[(1-phiLY[iQ]), (1-phiX[iQ])],
                         [  phiLY[iQ]  , (1-phiX[iQ])],
                         [(1-phiRY[iQ]),   phiX[iQ]  ],
                         [  phiRY[iQ]  ,   phiX[iQ]  ]])
        
        # phis_old = np.array([[(1-phiLY_old[iQ]), (1-phiX_old[iQ])],
        #                      [  phiLY_old[iQ]  , (1-phiX_old[iQ])],
        #                      [(1-phiRY_old[iQ]),   phiX_old[iQ]  ],
        #                      [  phiRY_old[iQ]  ,   phiX_old[iQ]  ]])
        # assert np.allclose(phis, phis_old)
        
        gradphis = np.vstack((grad          , grad * [1, -1], 
                              grad * [-1, 1], grad * [-1, -1])) * phis
        phis = np.prod(phis, axis=1)
        indices = np.array([indL[iQ] + NY*iPlane,
                           (indL[iQ]+1) % NY + NY*iPlane,
                           (indR[iQ] + NY*(iPlane+1)) % nNodes,
                           ((indR[iQ]+1) % NY + NY*(iPlane+1)) % nNodes])
        Kdata[index:index+nEntries] = quadWeights[iQ] * \
                np.ravel( gradphis @ (diffusivity @ gradphis.T) )
        Adata[index:index+nEntries] = ( quadWeights[iQ] *
            np.outer(np.dot(gradphis, velocity), phis).ravel() )
        Mdata[index:index+nEntries] = quadWeights[iQ] * \
            np.outer(phis, phis).ravel()
        row_ind[index:index+nEntries] = np.repeat(indices, 2*ndim)
        col_ind[index:index+nEntries] = np.tile(indices, 2*ndim)
        
        # gradphis_old = np.vstack((grads[iQ]          , grads[iQ] * [1, -1], 
        #                           grads[iQ] * [-1, 1], grads[iQ] * [-1, -1])) * phis_old
        # phis_old = np.prod(phis_old, axis=1)
        # indices_old = np.array([indL_old[iQ] + NY*iPlane,
        #                        (indL_old[iQ]+1) % NY + NY*iPlane,
        #                        (indR_old[iQ] + NY*(iPlane+1)) % nNodes,
        #                        ((indR_old[iQ]+1) % NY + NY*(iPlane+1)) % nNodes])
        # Kdata_old[index:index+nEntries] = quadWeights_old[iQ] * \
        #         np.ravel( gradphis_old @ (diffusivity @ gradphis_old.T) )
        # Adata_old[index:index+nEntries] = ( quadWeights_old[iQ] *
        #     np.outer(np.dot(gradphis_old, velocity), phis_old).ravel() )
        # Mdata_old[index:index+nEntries] = quadWeights_old[iQ] * \
        #     np.outer(phis_old, phis_old).ravel()
        # row_ind_old[index:index+nEntries] = np.repeat(indices_old, 2*ndim)
        # col_ind_old[index:index+nEntries] = np.tile(indices_old, 2*ndim)
        # assert np.allclose(phis, phis_old)
        # assert np.allclose(gradphis, gradphis_old)
        # assert np.allclose(indices, indices_old)
        # assert np.allclose(Kdata, Kdata_old)
        # assert np.allclose(Adata, Adata_old)
        # assert np.allclose(Mdata, Mdata_old)
        # assert np.allclose(row_ind, row_ind_old)
        # assert np.allclose(col_ind, col_ind_old)
        
        index += nEntries
#         b[indices] += f(quad) * phis * quadWeights[iQ]
K = sp.csr_matrix( (Kdata, (row_ind, col_ind)), shape=(nNodes, nNodes) )
A = sp.csr_matrix( (Adata, (row_ind, col_ind)), shape=(nNodes, nNodes) )
M = sp.csr_matrix( (Mdata, (row_ind, col_ind)), shape=(nNodes, nNodes) )

# Backward-Euler
M /= dt
K = M + K - A

#set initial conditions
u = np.zeros(nNodes)
# u[[8,46,60,70,71,92,93]] = 1
# Amplitude = 1.0
# rx = np.pi
ry = 0.4
# sigmax = 1.
sigmay = 0.1
for ix in range(NX):
    for iy in range(NY):
        px = nodeX[ix]
        py = mapping(np.array([[px, nodeY[ix][iy]]]), 0)
        # u[ix*NY + iy] = Amplitude*np.exp( -0.5*( ((px - rx)/sigmax)**2
        #                     + ((py - ry)/sigmay)**2 ) ) # Gaussian
        u[ix*NY + iy] = (0.5*np.sin(nodeX[ix] + np.pi/2) + 0.5) * np.exp( -0.5*( ((py - ry)/sigmay)**2 ) )

# minMax = np.empty((nSteps+1, 2))
# minMax[0] = [0., 1.]
dudt = np.zeros(nNodes)
betas = np.array([0.25, 1/3, 0.5, 1])

def step(nSteps=1):
    global u, K, M
    for i in range(nSteps):
        # uTemp = u
        # for beta in betas:
        #     dudt, info = sp_la.cg(M, A @ uTemp, x0=dudt, tol=1e-10, atol=1e-10)
        #     # self.dudt = sp_la.spsolve(self.M, self.KA@uTemp)
        #     uTemp = u + beta * dt * dudt
        #     if (info != 0):
        #         print(f'solution failed with error code: {info}')
        # u = uTemp
        u, info = sp_la.cg(K, M @ u, u, tol=1e-10, atol=1e-10) # Backward-Euler
        # minMax[i+1] = [np.min(u), np.max(u)]

# generate plotting points
nx = 20
ny = 3
nPoints = nx*(NY*ny + 1)
phiPlot = np.empty((nPoints*NX + NY*ny + 1, 4))
indPlot = np.empty((nPoints*NX + NY*ny + 1, 4), dtype='int')
X = np.empty(0)
for iPlane in range(NX):
    points = np.indices((nx, NY*ny + 1), dtype='float').reshape(ndim, -1).T \
           * [dx[iPlane]/nx, 1/(NY*ny)]
    X = np.append(X, points[:,0] + nodeX[iPlane])
    phiX = points[:,0] / dx[iPlane]
    mapL = mapping(points + [nodeX[iPlane], 0], nodeX[iPlane])
    mapR = mapping(points + [nodeX[iPlane], 0], nodeX[iPlane+1])
    indL = (np.searchsorted(nodeY[iPlane], mapL, side='right') - 1) % NY
    indR = (np.searchsorted(nodeY[(iPlane+1) % NX], mapR, side='right') - 1)%NY
    phiLY = (mapL - nodeY[iPlane][indL]) / dy[0][indL]
    phiRY = (mapR - nodeY[iPlane][indR]) / dy[0][indR]
    for iP, point in enumerate(points):
        phiPlot[iPlane*nPoints + iP] = [
            (1-phiLY[iP]) * (1-phiX[iP]), phiLY[iP] * (1-phiX[iP]),
            (1-phiRY[iP]) * phiX[iP]    , phiRY[iP] * phiX[iP] ]
        indPlot[iPlane*nPoints + iP] = [
            indL[iP] + NY*iPlane,
            (indL[iP]+1) % NY + NY*iPlane,
            (indR[iP] + NY*(iPlane+1)) % nNodes,
            ((indR[iP]+1) % NY + NY*(iPlane+1)) % nNodes ]

phiPlot[iPlane*nPoints + iP + 1:] = phiPlot[0:NY*ny + 1]
indPlot[iPlane*nPoints + iP + 1:] = indPlot[0:NY*ny + 1]

X = np.append(X, [2*np.pi*np.ones(NY*ny + 1)])
Y = np.concatenate([np.tile(points[:,1], NX), points[0:NY*ny + 1,1]])
U = np.sum(phiPlot * u[indPlot], axis=1)

# maxAbsU = np.max(np.abs(U))
maxAbsU = 1.

def init_plot():
    global field, fig, ax, X, Y, U, maxAbsU
    fig, ax = plt.subplots()
    field = ax.tripcolor(X, Y, U, shading='gouraud'
                         ,cmap='seismic', vmin=-maxAbsU, vmax=maxAbsU
                         )
    # tri = mpl.tri.Triangulation(X,Y)
    # ax.triplot(tri, 'r-', lw=1)
    x = np.linspace(0, 2*np.pi, 100)
    for yi in [0.4, 0.5, 0.6]:
        ax.plot(x, [mapping(np.array([[0, yi]]), i) for i in x], 'k')
    for xi in nodeX:
        ax.plot([xi, xi], [0, 1], 'k:')
    # ax.plot(X[np.argmax(U)], Y[np.argmax(U)],  'g+', markersize=10)
    plt.colorbar(field)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$', rotation=0)
    plt.xticks(np.linspace(0, 2*np.pi, 7), 
        ['0',r'$\pi/3$',r'$2\pi/3$',r'$\pi$',r'$4\pi/3$',r'$5\pi/3$',r'$2\pi$'])
    plt.margins(0,0)
    return [field]

step(nSteps)
U = np.sum(phiPlot * u[indPlot], axis=1)
init_plot()

# field = ax.tripcolor(X, Y, U, shading='gouraud'
#                           ,cmap='seismic', vmin=-maxAbsU, vmax=maxAbsU
#                           )

# def animate(i):
#     global field, U, u, phiPlot, indPlot
#     step(1)
#     U = np.sum(phiPlot * u[indPlot], axis=1)
#     field.set_array(U)
#     return [field]

# ani = animation.FuncAnimation(
#     fig, animate, frames=nSteps, interval=15)

# ani.save('movie.mp4', writer='ffmpeg', dpi=200)

print(f'nSteps = {nSteps}')
print(f'max(u) = {np.max(u)}')
print(f'min(u) = {np.min(u)}')