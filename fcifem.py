# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 13:47:07 2020

@author: samal
"""

from scipy.special import roots_legendre
# import scipy.integrate
import scipy.sparse as sp
import scipy.sparse.linalg as sp_la
import numpy as np

from abc import ABCMeta, abstractmethod

class Mapping(metaclass=ABCMeta):
    @property
    @abstractmethod
    def name(self): 
        raise NotImplementedError

    @abstractmethod
    def __call__(self, points, x=0.):
        """Compute mapped y-coordinates from all points to given x-coordinate

        Parameters
        ----------
        points : numpy.ndarray, shape=(n,ndim)
            (x,y) coordinates of starting points.
        x : float
            x-coordinate value of plane to which points should be mapped.

        Returns
        -------
        numpy.ndarray, shape=(n,)
            Values of y-coordinate for all points mapped to given x-coordinate.

        """
        raise NotImplementedError
    
    @abstractmethod
    def deriv(self, points):
        """Compute dy/dx derivative of mapping function at given points.

        Parameters
        ----------
        points : numpy.ndarray, shape=(n,ndim)
            (x,y) coordinates of evaluation points.

        Returns
        -------
        numpy.ndarray, shape=(n,)
            Values of dy/dx evaluated at all points.

        """
        raise NotImplementedError
    
    @abstractmethod
    def theta(self, points):
        """Compute angle of mapping function from +x-axis at given points.

        Parameters
        ----------
        points : numpy.ndarray, shape=(n,ndim)
            (x,y) coordinates of evaluation points.

        Returns
        -------
        numpy.ndarray, shape=(n,)
            Angles of mapping function from +x-axis evaluated at all points.

        """
        raise NotImplementedError
    
    @abstractmethod
    def __repr__(self):
        return f"{self.__class__.__name__}"

class StraightMapping(Mapping):
    @property
    def name(self): 
        return 'straight'

    def __call__(self, points, x=0.):
        points.shape = (-1, 2)
        return points[:,1]
    
    def deriv(self, points):
        points.shape = (-1, 2)
        return np.repeat(0., len(points))
    
    def theta(self, points):
        points.shape = (-1, 2)
        return np.repeat(0., len(points))

class SinusoidalMapping(Mapping):
    @property
    def name(self): 
        return 'sinusoidal'
    
    def __init__(self, amplitude, phase):
        self.A = amplitude
        self.phase = phase

    def __call__(self, points, x=0.):
        points.shape = (-1, 2)
        offsets = points[:,1] - self.A*np.sin(points[:,0] - self.phase)
        return (self.A*np.sin(x - self.phase) + offsets) % 1 % 1
    
    def deriv(self, points):
        points.shape = (-1, 2)
        return self.A*np.cos(points[:,0] - self.phase)
    
    def theta(self, points):
        points.shape = (-1, 2)
        return np.arctan(self.A*np.cos(points[:,0] - self.phase))
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.A},{self.phase})"

class FciFemSim(metaclass=ABCMeta):
    """Class for flux-coordinate independent FEM (FCIFEM) method.
    
    Attributes
    ----------
    NX : int
        Number of planes along x-dimension. Must be NX >= 2.
    NY : int
        Number of nodes on each plane. Must be NY >= 2.
    Nquad : int
        Number of quadrature points in each grid cell along one dimension.
    nodeX : numpy.ndarray, shape=(NX+1,)
        x-coords of FCI planes (includes right boundary).
    dx : numpy.ndarray, shape=(NX,)
        Spacing between FCI planes
    nodeY : numpy.ndarray, shape=(NX+1, NY+1)
        y-coords of nodes on each FCI plane (includes right/top boundaries).
    dy : numpy.ndarray, shape=(NX+1, NY)
        Spacing between nodes on each FCI plane (includes right boundary).
    nNodes : int
        Number of unique nodal points in the simulation domain (equals NX*NY).
    velocity : np.array([vx,vy,vz], dtype='float64')
        Background velocity of the fluid.
    diffusivity : {numpy.ndarray, float}
        Diffusion coefficient for the quantity of interest.
        If an array, it must have shape (ndim,ndim). If a float, it will
        be converted to diffusivity*np.eye(ndim, dtype='float64').
    f : callable
        Forcing function. Must take 2D array of points and return 1D array.
    dt : float
        Time interval between each successive timestep.
    timestep : int
        Current timestep of the simulation.
    time : float
        Current time of the simulation; equal to timestep*dt.
        
    """
    
    def __init__(self, NX, NY, mapping, dt, u0, velocity, diffusivity=0.,
                 Nquad=5, px=0., py=0., seed=None, f=None, **kwargs):
        """Initialize attributes of FCIFEM simulation class

        Parameters
        ----------
        NX : int
            Number of planes along x-dimension. Must be NX >= 2.
        NY : int
            Number of nodes on each plane. Must be NY >= 2.
        mapping : Mapping
            Mapping function for the FCIFEM method.
            Must be an object derived from fcifem.Mapping.
        dt : float
            Time interval between each successive timestep.
        u0 : {numpy.ndarray, function object}
            Initial conditions for the simulation.
            Must be an array of shape (self.nNodes,) or a function returning
            such an array and taking as input the array of (x,y) node
            coordinates with shape ({self.nNodes}, 2).
        velocity : np.array([vx, vy], dtype='float')
            Background velocity of the fluid.
        diffusivity : {numpy.ndarray, float}, optional
            Diffusion coefficient for the quantity of interest.
            If an array, it must have shape (ndim,ndim). If a float, it will
            be converted to diffusivity*np.eye(ndim, dtype='float').
            The default is 0.
        Nquad : int, optional
            Number of quadrature points in each grid cell along one dimension.
            Must be > 0. The default is 5.
        px : float, optional
            Max amplitude of random perturbations added to FCI plane locations.
            Size is relative to grid spacing (px*2*pi/NX). The default is 0.
        py : float, optional
            Max amplitude of random perturbations added to node y-coords.
            Size is relative to grid spacing (py/NY). The default is 0.
        seed : {None, int, array_like[ints], numpy.random.SeedSequence}, optional
            A seed to initialize the RNG. If None, then fresh, unpredictable
            entropy will be pulled from the OS. Default is None.
        f : function object, optional
            Function defining the forcing term throughout the domain.
            The object must take an nx2 numpy.ndarray of points and return a
            1D numpy.ndarray of size n for the function values at those points.
            The default is f(x) = 0.
        **kwargs
            Keyword arguments
            
        """
        self.ndim = 2
        self.NX = NX
        self.NY = NY
        self.dt = dt
        self.time = 0.0
        self.timestep = 0
        self.Nquad = Nquad
        self.velocity = velocity
        if isinstance(diffusivity, np.ndarray):
            self.diffusivity = diffusivity
        else:
            self.diffusivity = np.array(diffusivity, dtype='float')
            if self.diffusivity.shape != (self.ndim, self.ndim):
                self.diffusivity = diffusivity * np.eye(self.ndim, dtype='float')
        if self.diffusivity.shape != (self.ndim,self.ndim):
            raise SystemExit("diffusivity must be (or be convertible to) a "
                             "numpy.ndarray with shape (2, 2}).")
        rng = np.random.Generator(np.random.PCG64(seed))
        if "nodeX" in kwargs:
            self.nodeX = kwargs["nodeX"]
        else:
            self.nodeX = 2*np.pi*np.arange(NX+1)/NX
            px *= 2*np.pi/NX
            self.nodeX[1:-1] += rng.uniform(-px, px, self.nodeX[1:-1].shape)
        self.nodeY = np.tile(np.linspace(0, 1, NY+1), NX+1).reshape(NX+1,-1)
        py /= NY
        self.nodeY[:-1,1:-1] += rng.uniform(-py, py, self.nodeY[:-1,1:-1].shape)
        self.nodeY[-1] = self.nodeY[0]
        self.nNodes = NX*NY
        self.dx = self.nodeX[1:] - self.nodeX[0:-1]
        self.dy = self.nodeY[:,1:] - self.nodeY[:,:-1]
        if f is None:
            self.f = lambda x: np.repeat(0., len(x.reshape(-1, self.ndim)))
        self.setInitialConditions(u0)
        self.mapping = mapping
    
    def setInitialConditions(self, u0):
        """Initialize the nodal coefficients for the given IC.

        Returns
        -------
        None.

        """
        self.u0 = u0
        
    def computeSpatialDiscretization(self):
        """Assemble the system discretization matrices K, A, M in CSR format.
        K is the stiffness matrix from the diffusion term
        A is the advection matrix
        M is the mass matrix from the time derivative
        KA = K + A

        Returns
        -------
        None.

        """
        # pre-allocate arrays for stiffness matrix triplets
        nEntries = (2*self.ndim)**2
        nQuads = self.NY*self.Nquad**2
        nMaxEntries = nEntries * nQuads * self.NX
        Kdata = np.zeros(nMaxEntries)
        Adata = np.zeros(nMaxEntries)
        Mdata = np.zeros(nMaxEntries)
        row_ind = np.zeros(nMaxEntries, dtype='int')
        col_ind = np.zeros(nMaxEntries, dtype='int')
        self.b = np.zeros(self.nNodes)
        self.u_weights = np.zeros(self.nNodes)
        
        ##### compute spatial discretizaton
        index = 0
        for iPlane in range(self.NX):
            ##### generate quadrature points
            offsets, weights = roots_legendre(self.Nquad)
            offsets = [offsets * self.dx[iPlane] / 2, offsets / (2*self.NY)]
            weights = [weights * self.dx[iPlane] / 2, weights / (2*self.NY)]
            quads = ( np.indices([1, self.NY], dtype='float').T.
                      reshape(-1, self.ndim) + 0.5 ) \
                    * [self.dx[iPlane], 1/self.NY]
            quadWeights = np.repeat(1., len(quads))
            for i in range(self.ndim):
                quads = np.concatenate( 
                    [quads + offset*np.eye(self.ndim)[i] for offset in offsets[i]] )
                quadWeights = np.concatenate(
                    [quadWeights * weight for weight in weights[i]] )
            phiX = quads[:,0] / self.dx[iPlane]
            mapL = self.mapping(quads + [self.nodeX[iPlane], 0], self.nodeX[iPlane])
            mapR = self.mapping(quads + [self.nodeX[iPlane], 0], self.nodeX[iPlane+1])
            indL = (np.searchsorted(self.nodeY[iPlane], mapL, side='right') - 1) % self.NY
            indR = (np.searchsorted(self.nodeY[iPlane + 1], mapR, side='right') - 1) % self.NY
            phiLY = (mapL - self.nodeY[iPlane][indL]) / self.dy[iPlane][indL]
            phiRY = (mapR - self.nodeY[iPlane + 1][indR]) / self.dy[iPlane + 1][indR]
            
            for iQ, quad in enumerate(quads):
                phis = np.array([[(1-phiLY[iQ]), (1-phiX[iQ])],
                                  [  phiLY[iQ]  , (1-phiX[iQ])],
                                  [(1-phiRY[iQ]),   phiX[iQ]  ],
                                  [  phiRY[iQ]  ,   phiX[iQ]  ]])
                
                # # Gradients along the mapping direction
                # gradL = np.array([-1/arcLengths[iPlane], -1/dy[iPlane][indL[iQ]]])
                # gradR = np.array([ 1/arcLengths[iPlane], -1/dy[iPlane + 1][indR[iQ]]])
                
                # Gradients along the coordinate direction
                gradL = np.array([-1/self.dx[iPlane], -1/self.dy[iPlane][indL[iQ]]])
                gradR = np.array([ 1/self.dx[iPlane], -1/self.dy[iPlane + 1][indR[iQ]]])
                
                gradphis = np.vstack((gradL, gradL * [1, -1], 
                                      gradR, gradR * [1, -1])) * phis
                phis = np.prod(phis, axis=1)
                indices = np.array([indL[iQ] + self.NY*iPlane,
                                    (indL[iQ]+1) % self.NY + self.NY*iPlane,
                                    (indR[iQ] + self.NY*(iPlane+1)) % self.nNodes,
                                    ((indR[iQ]+1) % self.NY + self.NY*(iPlane+1)) % self.nNodes])
                Kdata[index:index+nEntries] = ( quadWeights[iQ] * 
                    np.ravel( gradphis @ (self.diffusivity @ gradphis.T) ) )
                Adata[index:index+nEntries] = ( quadWeights[iQ] *
                    np.outer(np.dot(gradphis, self.velocity), phis).ravel() )
                Mdata[index:index+nEntries] = ( quadWeights[iQ] * 
                    np.outer(phis, phis).ravel() )
                row_ind[index:index+nEntries] = np.repeat(indices, 2*self.ndim)
                col_ind[index:index+nEntries] = np.tile(indices, 2*self.ndim)
                
                self.u_weights[indices] += quadWeights[iQ] * phis
                
                index += nEntries
                self.b[indices] += self.f(quad) * phis * quadWeights[iQ]
        
        self.K = sp.csr_matrix( (Kdata, (row_ind, col_ind)),
                                shape=(self.nNodes, self.nNodes) )
        self.A = sp.csr_matrix( (Adata, (row_ind, col_ind)),
                                shape=(self.nNodes, self.nNodes) )
        self.M = sp.csr_matrix( (Mdata, (row_ind, col_ind)),
                                shape=(self.nNodes, self.nNodes) )
    
    # def step(self, nSteps = 1, **kwargs):
    #     """Advance the simulation a given number of timesteps.

    #     Parameters
    #     ----------
    #     nSteps : int, optional
    #         Number of timesteps to compute. The default is 1.
    #     **kwargs
    #         Used to specify optional arguments passed to the linear solver.
    #         Note that kwargs["M"] will be overwritten, use self.precon(...)
    #         instead to generate or specify a preconditioner.

    #     Returns
    #     -------
    #     None.

    #     """
    #     kwargs["M"] = self.P
    #     info = 0
    #     betas = np.array([0.25, 1/3, 0.5, 1]) ## RK4 ##
    #     # betas = np.array([1.]) ## Forward Euler ##
    #     for i in range(nSteps):
    #         uTemp = self.uI
    #         for beta in betas:
    #             self.dudt, info = sp_la.cg(self.M, self.KA@uTemp,
    #                                        x0=self.dudt, **kwargs)
    #             # self.dudt = sp_la.spsolve(self.M, self.KA@uTemp)
    #             uTemp = self.uI + beta*self.dt*self.dudt
    #             if (info != 0):
    #                 print(f'solution failed with error code: {info}')
    #         self.uI = uTemp
    #         self.timestep += 1
    #     self.time = self.timestep * self.dt


# # # Mass-lumped matrix
# # M = sp.diags(u_weights, format='csr')

# # # For debugging
# # KA = K.A
# # AA = A.A
# # MA = M.A

# # Ad = A.diagonal()
# # A = 0.5*(A-A.T)
# # A.setdiag(Ad)

# # u_weights = np.sum(M, axis=1)
# # u_weights = M.diagonal()

# # Backward-Euler
# M /= dt
# K = M + K - A

# # Set initial conditions
# u = np.zeros(nNodes)
# exact_solution = np.zeros(nNodes)
# # u[[8,46,60,70,71,92,93]] = 1
# # Amplitude = 1.0
# # rx = np.pi
# ry = 0.4
# # sigmax = 1.
# sigmay = 0.1
# pi_2 = 0.5*np.pi
# for ix in range(NX):
#     for iy in range(NY):
#         px = nodeX[ix]
#         py = mapping(np.array([[px, nodeY[ix][iy]]]), 0)
#         # u[ix*NY + iy] = Amplitude*np.exp( -0.5*( ((px - rx)/sigmax)**2
#         #                     + ((py - ry)/sigmay)**2 ) ) # Gaussian
#         u[ix*NY + iy] = 0.5*(np.sin(px + pi_2) + 1) * np.exp(-0.5*((py - ry)/sigmay)**2)
#         exact_solution[ix*NY + iy] = 0.5*(np.sin(px + pi_2) + 1) * np.exp(-0.5*((py - (ry+0.1))/sigmay)**2)

# # minMax = np.empty((nSteps+1, 2))
# # minMax[0] = [0., 1.]
# dudt = np.zeros(nNodes)
# betas = np.array([0.25, 1/3, 0.5, 1])

# U_sum = []
# error = []

# def step(nSteps=1):
#     global u, K, M, U_sum, u_weights, exact_solution
#     for i in range(nSteps):
#         # uTemp = u
#         # for beta in betas:
#         #     dudt, info = sp_la.lgmres(M, A @ uTemp, x0=dudt, tol=1e-10, atol=1e-10)
#         #     # self.dudt = sp_la.spsolve(self.M, self.KA@uTemp)
#         #     uTemp = u + beta * dt * dudt
#         #     if (info != 0):
#         #         print(f'solution failed with error code: {info}')
#         # u = uTemp
#         u, info = sp_la.lgmres(K, M @ u, u, tol=1e-10, atol=1e-10) # Backward-Euler
#         # minMax[i+1] = [np.min(u), np.max(u)]
#         U_sum.append(np.sum(u*u_weights))
#         error.append(np.linalg.norm(u - exact_solution))

# # generate plotting points
# nx = 20
# ny = 3
# nPoints = nx*(NY*ny + 1)
# phiPlot = np.empty((nPoints*NX + NY*ny + 1, 4))
# indPlot = np.empty((nPoints*NX + NY*ny + 1, 4), dtype='int')
# X = np.empty(0)
# for iPlane in range(NX):
#     points = np.indices((nx, NY*ny + 1), dtype='float').reshape(ndim, -1).T \
#            * [dx[iPlane]/nx, 1/(NY*ny)]
#     X = np.append(X, points[:,0] + nodeX[iPlane])
#     phiX = points[:,0] / dx[iPlane]
#     mapL = mapping(points + [nodeX[iPlane], 0], nodeX[iPlane])
#     mapR = mapping(points + [nodeX[iPlane], 0], nodeX[iPlane+1])
#     indL = (np.searchsorted(nodeY[iPlane], mapL, side='right') - 1) % NY
#     indR = (np.searchsorted(nodeY[(iPlane+1) % NX], mapR, side='right') - 1)%NY
#     phiLY = (mapL - nodeY[iPlane][indL]) / dy[iPlane][indL]
#     phiRY = (mapR - nodeY[iPlane + 1][indR]) / dy[iPlane + 1][indR]
#     for iP, point in enumerate(points):
#         phiPlot[iPlane*nPoints + iP] = [
#             (1-phiLY[iP]) * (1-phiX[iP]), phiLY[iP] * (1-phiX[iP]),
#             (1-phiRY[iP]) * phiX[iP]    , phiRY[iP] * phiX[iP] ]
#         indPlot[iPlane*nPoints + iP] = [
#             indL[iP] + NY*iPlane,
#             (indL[iP]+1) % NY + NY*iPlane,
#             (indR[iP] + NY*(iPlane+1)) % nNodes,
#             ((indR[iP]+1) % NY + NY*(iPlane+1)) % nNodes ]

# phiPlot[iPlane*nPoints + iP + 1:] = phiPlot[0:NY*ny + 1]
# indPlot[iPlane*nPoints + iP + 1:] = indPlot[0:NY*ny + 1]

# X = np.append(X, [2*np.pi*np.ones(NY*ny + 1)])
# Y = np.concatenate([np.tile(points[:,1], NX), points[0:NY*ny + 1,1]])
# U = np.sum(phiPlot * u[indPlot], axis=1)

# # maxAbsU = np.max(np.abs(U))
# maxAbsU = 1.

# def init_plot():
#     global field, fig, ax, X, Y, U, maxAbsU
#     fig, ax = plt.subplots()
#     field = ax.tripcolor(X, Y, U, shading='gouraud'
#                          ,cmap='seismic', vmin=-maxAbsU, vmax=maxAbsU
#                          )
#     # tri = mpl.tri.Triangulation(X,Y)
#     # ax.triplot(tri, 'r-', lw=1)
#     x = np.linspace(0, 2*np.pi, 100)
#     for yi in [0.4, 0.5, 0.6]:
#         ax.plot(x, [mapping(np.array([[0, yi]]), i) for i in x], 'k')
#     for xi in nodeX:
#         ax.plot([xi, xi], [0, 1], 'k:')
#     # ax.plot(X[np.argmax(U)], Y[np.argmax(U)],  'g+', markersize=10)
#     plt.colorbar(field)
#     plt.xlabel(r'$x$')
#     plt.ylabel(r'$y$', rotation=0)
#     plt.xticks(np.linspace(0, 2*np.pi, 7), 
#         ['0',r'$\pi/3$',r'$2\pi/3$',r'$\pi$',r'$4\pi/3$',r'$5\pi/3$',r'$2\pi$'])
#     plt.margins(0,0)
#     return [field]

# step(nSteps)
# U = np.sum(phiPlot * u[indPlot], axis=1)
# init_plot()

# # field = ax.tripcolor(X, Y, U, shading='gouraud'
# #                           ,cmap='seismic', vmin=-maxAbsU, vmax=maxAbsU
# #                           )

# # def animate(i):
# #     global field, U, u, phiPlot, indPlot
# #     step(1)
# #     U = np.sum(phiPlot * u[indPlot], axis=1)
# #     field.set_array(U)
# #     return [field]

# # ani = animation.FuncAnimation(
# #     fig, animate, frames=nSteps, interval=15)

# # ani.save('movie.mp4', writer='ffmpeg', dpi=200)

# print(f'nSteps = {nSteps}')
# print(f'max(u) = {np.max(u)}')
# print(f'min(u) = {np.min(u)}')