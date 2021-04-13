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

from integrators import *

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
    
    # def __repr__(self):
    #     return f"{self.__class__.__name__}"

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
    velocity : np.array([vx,vy], dtype='float64')
        Background velocity of the fluid.
    diffusivity : {numpy.ndarray, float}
        Diffusion coefficient for the quantity of interest.
        If an array, it must have shape (ndim,ndim). If a float, it will
        be converted to diffusivity*np.eye(ndim, dtype='float64').
    f : callable
        Forcing function. Must take 2D array of points and return 1D array.
    integrator : Integrator
        Object defining time-integration scheme to be used.
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
        u0 : {numpy.ndarray, callable}
            Initial conditions for the simulation.
            Must be an array of shape (self.nNodes,) or a callable object
            returning such an array and taking as input the array of node
            coordinates with shape (self.nNodes, self.ndim).
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
        self.mapping = mapping
        self.velocity = velocity
        if isinstance(diffusivity, np.ndarray):
            self.diffusivity = diffusivity
        else:
            self.diffusivity = np.array(diffusivity, dtype='float')
            if self.diffusivity.shape != (self.ndim, self.ndim):
                self.diffusivity = diffusivity * np.eye(self.ndim, dtype='float')
        if self.diffusivity.shape != (self.ndim,self.ndim):
            raise SystemExit(f"diffusivity must be (or be convertible to) a "
                f"numpy.ndarray with shape ({self.ndim}, {self.ndim}).")
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
    
    def setInitialConditions(self, u0):
        """Initialize the nodal coefficients for the given IC.
        
        Parameters
        ----------
        u0 : {numpy.ndarray, callable}
            Initial conditions for the simulation.
            Must be an array of shape (self.nNodes,) or a callable object
            returning such an array and taking as input the array of node
            coordinates with shape (self.nNodes, self.ndim).

        Returns
        -------
        None.

        """
        if isinstance(u0, np.ndarray) and u0.shape == (self.nNodes,):
            self.u = u0.copy()
            self.u0 = u0
            self.u0func = None
        elif callable(u0):
            self.u0func = u0
            self.nodes = np.vstack( (np.repeat(self.nodeX[:-1], self.NY),
                                self.nodeY[:-1,:-1].ravel()) ).T
            self.nodesMapped = self.nodes.copy()
            self.nodesMapped[:,1] = self.mapping(self.nodes, 0)
            self.u = u0(self.nodesMapped)
            self.u0 = self.u.copy()
        else:
            SystemExit(f"u0 must be an array of shape ({self.nNodes},) or a "
                f"callable object returning such an array and taking as input "
                f"the array of node coordinates with shape "
                f"({self.nNodes}, {self.ndim}).")
        
    def computeSpatialDiscretization(self, massLumping=False):
        """Assemble the system discretization matrices K, A, M in CSR format.
        K is the stiffness matrix from the diffusion term
        A is the advection matrix
        M is the mass matrix from the time derivative
        KA = K + A

        Returns
        -------
        None.

        """
        ndim = self.ndim
        nNodes = self.nNodes
        NX = self.NX
        NY = self.NY
        # pre-allocate arrays for stiffness matrix triplets
        nEntries = (2*ndim)**2
        nQuads = NY*self.Nquad**2
        nMaxEntries = nEntries * nQuads * NX
        Kdata = np.zeros(nMaxEntries)
        Adata = np.zeros(nMaxEntries)
        if not massLumping:
            Mdata = np.zeros(nMaxEntries)
        row_ind = np.zeros(nMaxEntries, dtype='int')
        col_ind = np.zeros(nMaxEntries, dtype='int')
        self.b = np.zeros(nNodes)
        self.u_weights = np.zeros(nNodes)
        
        # # For gradients along the mapping direction, needs scipy.integrate
        # arcLengths = np.array([scipy.integrate.quad(lambda x: 
        #     np.sqrt(1 + self.mapping.deriv(np.array([[x,0]]))**2),
        #     self.nodeX[i], self.nodeX[i+1])[0] for i in range(NX)])
        
        ##### compute spatial discretizaton
        index = 0
        for iPlane in range(NX):
            dx = self.dx[iPlane]
            nodeX = self.nodeX[iPlane]
            ##### generate quadrature points
            offsets, weights = roots_legendre(self.Nquad)
            offsets = [offsets * dx / 2, offsets / (2*NY)]
            weights = [weights * dx / 2, weights / (2*NY)]
            quads = ( np.indices([1, NY], dtype='float').T.
                      reshape(-1, ndim) + 0.5 ) \
                    * [dx, 1/NY]
            quadWeights = np.repeat(1., len(quads))
            for i in range(ndim):
                quads = np.concatenate( 
                    [quads + offset*np.eye(ndim)[i] for offset in offsets[i]] )
                quadWeights = np.concatenate(
                    [quadWeights * weight for weight in weights[i]] )
            phiX = quads[:,0] / dx
            mapL = self.mapping(quads + [nodeX, 0], nodeX)
            mapR = self.mapping(quads + [nodeX, 0], self.nodeX[iPlane+1])
            indL = (np.searchsorted(self.nodeY[iPlane], mapL, side='right') - 1) % NY
            indR = (np.searchsorted(self.nodeY[iPlane + 1], mapR, side='right') - 1) % NY
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
                gradL = np.array([-1/dx, -1/self.dy[iPlane][indL[iQ]]])
                gradR = np.array([ 1/dx, -1/self.dy[iPlane + 1][indR[iQ]]])
                
                gradphis = np.vstack((gradL, gradL * [1, -1], 
                                      gradR, gradR * [1, -1])) * phis
                phis = np.prod(phis, axis=1)
                indices = np.array([indL[iQ] + NY*iPlane,
                                    (indL[iQ]+1) % NY + NY*iPlane,
                                    (indR[iQ] + NY*(iPlane+1)) % nNodes,
                                    ((indR[iQ]+1) % NY + NY*(iPlane+1)) % nNodes])
                Kdata[index:index+nEntries] = ( quadWeights[iQ] * 
                    np.ravel( gradphis @ (self.diffusivity @ gradphis.T) ) )
                Adata[index:index+nEntries] = ( quadWeights[iQ] *
                    np.outer(np.dot(gradphis, self.velocity), phis).ravel() )
                if not massLumping:
                    Mdata[index:index+nEntries] = ( quadWeights[iQ] * 
                        np.outer(phis, phis).ravel() )
                row_ind[index:index+nEntries] = np.repeat(indices, 2*ndim)
                col_ind[index:index+nEntries] = np.tile(indices, 2*ndim)
                
                self.u_weights[indices] += quadWeights[iQ] * phis
                
                index += nEntries
                self.b[indices] += self.f(quad) * phis * quadWeights[iQ]
        
        self.K = sp.csr_matrix( (Kdata, (row_ind, col_ind)),
                                shape=(nNodes, nNodes) )
        self.A = sp.csr_matrix( (Adata, (row_ind, col_ind)),
                                shape=(nNodes, nNodes) )
        if massLumping:
            self.M = sp.diags(self.u_weights, format='csr')
        else:
            self.M = sp.csr_matrix( (Mdata, (row_ind, col_ind)),
                                shape=(nNodes, nNodes) )
    
    def initializeTimeIntegrator(self, integrator, P='ilu', **kwargs):
        """Initialize and register the time integration scheme to be used.

        Parameters
        ----------
        integrator : {Integrator (object or subclass type), string}
            Integrator object or string specifiying which scheme is to be used.
            If a string, must be one of 'RK' or 'BackwardEuler' ('BE').
        P : {string, scipy.sparse.linalg.LinearOperator, None}, optional
            Which preconditioning method to use. P can be a LinearOperator to
            directly specifiy the preconditioner to be used. Otherwise it must
            be one of 'jacobi', 'ilu', or None. The default is 'ilu'.
        **kwargs
            Used to specify optional arguments for scipy.sparse.linalg.spilu.
            Only relevant if P is 'ilu', otherwise unsused.

        Returns
        -------
        None.

        """
        if isinstance(integrator, Integrator):
            self.integrator = Integrator
            return
        if isinstance(integrator, str):
            if integrator.lower() in ['backwardeuler', 'be']:
                Type = BackwardEuler
            elif integrator.lower() in ['rk']:
                Type = RK
        else: # if integrator not an Integrator object or string, assume it's a type
            Type = integrator
        self.integrator = Type(self, self.A - self.K, self.M, self.dt, P, **kwargs)
    
    def step(self, nSteps = 1, **kwargs):
        """Advance the simulation a given number of timesteps.

        Parameters
        ----------
        nSteps : int, optional
            Number of timesteps to compute. The default is 1.
        **kwargs
            Used to specify optional arguments passed to the linear solver.
            Note that kwargs["M"] will be overwritten, instead use
            self.precondition(...) to generate or specify a preconditioner.

        Returns
        -------
        None.

        """
        self.integrator.step(nSteps, **kwargs)

    def generatePlottingPoints(self, nx=20, ny=3):
        """Generate set of interpolation points to use for plotting.

        Parameters
        ----------
        nx : int, optional
            Number of points per grid division in the x-direction.
            The default is 20.
        ny : int, optional
            Number of points per grid division in the y-direction.
            The default is 3.

        Returns
        -------
        None.

        """
        NX = self.NX
        NY = self.NY
        nNodes = self.nNodes
        nPointsPerPlane = nx*(NY*ny + 1)
        nPointsTotal = nPointsPerPlane*NX + NY*ny + 1
        self.phiPlot = np.empty((nPointsTotal, 4))
        self.indPlot = np.empty((nPointsTotal, 4), dtype='int')
        self.X = np.empty(0)
        for iPlane in range(NX):
            dx = self.dx[iPlane]
            nodeX = self.nodeX[iPlane]
            points = np.indices((nx, NY*ny + 1), dtype='float') \
                .reshape(self.ndim, -1).T * [dx/nx, 1/(NY*ny)]
            self.X = np.append(self.X, points[:,0] + nodeX)
            phiX = points[:,0] / dx
            mapL = self.mapping(points + [nodeX, 0], nodeX)
            mapR = self.mapping(points + [nodeX, 0], self.nodeX[iPlane+1])
            indL = (np.searchsorted(self.nodeY[iPlane], mapL, side='right') - 1) % NY
            indR = (np.searchsorted(self.nodeY[(iPlane+1) % NX], mapR, side='right') - 1)%NY
            phiLY = (mapL - self.nodeY[iPlane][indL]) / self.dy[iPlane][indL]
            phiRY = (mapR - self.nodeY[iPlane + 1][indR]) / self.dy[iPlane + 1][indR]
            for iP, point in enumerate(points):
                self.phiPlot[iPlane*nPointsPerPlane + iP] = [
                    (1-phiLY[iP]) * (1-phiX[iP]), phiLY[iP] * (1-phiX[iP]),
                    (1-phiRY[iP]) * phiX[iP]    , phiRY[iP] * phiX[iP] ]
                self.indPlot[iPlane*nPointsPerPlane + iP] = [
                    indL[iP] + NY*iPlane,
                    (indL[iP]+1) % NY + NY*iPlane,
                    (indR[iP] + NY*(iPlane+1)) % nNodes,
                    ((indR[iP]+1) % NY + NY*(iPlane+1)) % nNodes ]
        
        self.phiPlot[iPlane*nPointsPerPlane + iP + 1:] = self.phiPlot[0:NY*ny + 1]
        self.indPlot[iPlane*nPointsPerPlane + iP + 1:] = self.indPlot[0:NY*ny + 1]
        
        self.X = np.append(self.X, [2*np.pi*np.ones(NY*ny + 1)])
        self.Y = np.concatenate([np.tile(points[:,1], NX), points[0:NY*ny + 1,1]])
        self.U = np.sum(self.phiPlot * self.u[self.indPlot], axis=1)
    
    def computePlottingSolution(self):
        """Compute interpolated solution at the plotting points.

        Returns
        -------
        None.

        """
        self.U = np.sum(self.phiPlot * self.u[self.indPlot], axis=1)