# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 13:47:07 2020

@author: samal
"""

from scipy.special import roots_legendre
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sp_la
import numpy as np

from abc import ABCMeta, abstractmethod

import integrators

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
    Implements the convection-diffusion equation on a rectangular domain
    [x, y] = [0...2*pi, 0...1] with doubly-periodic boundary conditions.
    
    Attributes
    ----------
    NX : int
        Number of planes along x-dimension. Must be NX >= 2.
    NY : int
        Number of nodes on each plane. Must be NY >= 2.
    nodeX : numpy.ndarray, shape=(NX+1,)
        x-coords of FCI planes (includes right boundary).
    dx : numpy.ndarray, shape=(NX,)
        Spacing between FCI planes
    nodeY : numpy.ndarray, shape=(NX+1, NY+1)
        y-coords of nodes on each FCI plane (includes right/top boundaries).
    idy : numpy.ndarray, shape=(NX+1, NY)
        1/spacing between nodes on each FCI plane (includes right boundary).
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
    NQX : int
        Number of quadrature cell divisions between FCI planes.
    NQY : int
        Number of quadrature cell divisions in y-direction.
    Qord : int
        Number of quadrature points in each grid cell along one dimension.
    quadType : string, optional
        Type of quadrature to be used. Must be either 'gauss' or 'uniform'.
        Produces Gauss-Legendre or Newton-Cotes points/weights respectively.
    massLumping : bool, optional
        Determines whether mass-lumping was used to calculate M matrix.
    K : scipy.sparse.csr_matrix
        The stiffness matrix from the diffusion term
    A : scipy.sparse.csr_matrix
        The advection matrix
    M : scipy.sparse.csr_matrix
        The mass matrix from the time derivative
    b : numpy.ndarray, shape=(nNodes,)
        RHS forcing vector generated from forcing function f.
    integrator : Integrator
        Object defining time-integration scheme to be used. 
    """
    
    def __init__(self, NX, NY, mapping, velocity, diffusivity=0.,
                 px=0., py=0., seed=None, **kwargs):
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
        velocity : np.array([vx, vy], dtype='float')
            Background velocity of the fluid.
        diffusivity : {numpy.ndarray, float}, optional
            Diffusion coefficient for the quantity of interest.
            If an array, it must have shape (ndim,ndim). If a float, it will
            be converted to diffusivity*np.eye(ndim, dtype='float').
            The default is 0.
        px : float, optional
            Max amplitude of random perturbations added to FCI plane locations.
            Size is relative to grid spacing (px*2*pi/NX). The default is 0.
        py : float, optional
            Max amplitude of random perturbations added to node y-coords.
            Size is relative to grid spacing (py/NY). The default is 0.
        seed : {None, int, array_like[ints], numpy.random.SeedSequence}, optional
            A seed to initialize the RNG. If None, then fresh, unpredictable
            entropy will be pulled from the OS. The default is None.
        **kwargs
            Keyword arguments
            
        """
        self.ndim = 2
        self.NX = NX
        self.NY = NY
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
        self.idy = 1. / (self.nodeY[:,1:] - self.nodeY[:,:-1])
    
    def setInitialConditions(self, u0, mapped=True):
        """Initialize the nodal coefficients for the given IC.
        
        Parameters
        ----------
        u0 : {numpy.ndarray, callable}
            Initial conditions for the simulation.
            Must be an array of shape (self.nNodes,) or a callable object
            returning such an array and taking as input the array of node
            coordinates with shape (self.nNodes, self.ndim).
        mapped : bool, optional
            Whether mapping is applied to node positions before applying ICs.
            The default is True.

        Returns
        -------
        None.

        """
        if isinstance(u0, np.ndarray) and u0.shape == (self.nNodes,):
            self.u0 = u0
            self.u = u0.copy()
            self.u0func = None
        elif callable(u0):
            self.u0func = u0
            self.nodes = np.vstack( (np.repeat(self.nodeX[:-1], self.NY),
                                self.nodeY[:-1,:-1].ravel()) ).T
            self.nodesMapped = self.nodes.copy()
            self.nodesMapped[:,1] = self.mapping(self.nodes, 0)
            if mapped:
                self.u = u0(self.nodesMapped)
            else:
                self.u = u0(self.nodes)
            self.u0 = self.u.copy()
        else:
            raise SystemExit(f"u0 must be an array of shape ({self.nNodes},) "
                f"or a callable object returning such an array and taking as "
                f"input the array of node coordinates with shape "
                f"({self.nNodes}, {self.ndim}).")
   
    def computeSpatialDiscretization(self, f=None, NQX=1, NQY=None, Qord=2,
                                      quadType='gauss', massLumping=False):
        """Assemble the system discretization matrices K, A, M in CSR format.
        
        K is the stiffness matrix from the diffusion term
        A is the advection matrix
        M is the mass matrix from the time derivative
        
        Parameters
        ----------
        f : {callable, None}, optional
            Forcing function. Must take 2D array of points and return 1D array.
            The default is None.
        NQX : int, optional
            Number of quadrature cell divisions between FCI planes.
            The default is 1.
        NQY : {int, None}, optional
            Number of quadrature cell divisions in y-direction.
            The default is None, which sets NQY = NY.
        Qord : int, optional
            Number of quadrature points in each grid cell along one dimension.
            The default is 2.
        quadType : string, optional
            Type of quadrature to be used. Must be either 'gauss' or 'uniform'.
            Produces either Gauss-Legendre or Newton-Cotes type points/weights.
            The default is 'gauss'.
        massLumping : bool, optional
            Determines whether mass-lumping is used to calculate M matrix.
            The default is False.

        Returns
        -------
        None.

        """
        ndim = self.ndim
        nNodes = self.nNodes
        NX = self.NX
        NY = self.NY
        if NQY is None:
            NQY = NY
        self.f = f
        self.NQX = NQX
        self.NQY = NQY
        self.Qord = Qord
        self.quadType = quadType
        self.massLumping = massLumping
        # pre-allocate arrays for stiffness matrix triplets
        nEntries = (2*ndim)**2
        nQuads = NQX * NQY * Qord**2
        nMaxEntries = nEntries * nQuads * NX
        Kdata = np.zeros(nMaxEntries)
        Adata = np.zeros(nMaxEntries)
        if not massLumping:
            Mdata = np.zeros(nMaxEntries)
        row_ind = np.zeros(nMaxEntries, dtype='int')
        col_ind = np.zeros(nMaxEntries, dtype='int')
        self.b = np.zeros(nNodes)
        self.u_weights = np.zeros(nNodes)
        
        ##### compute spatial discretizaton
        index = 0
        for iPlane in range(NX):
            dx = self.dx[iPlane]
            nodeX = self.nodeX[iPlane]
            ##### generate quadrature points
            if quadType.lower() in ('gauss', 'g', 'gaussian'):
                offsets, weights = roots_legendre(Qord)
            elif quadType.lower() in ('uniform', 'u'):
                offsets = np.linspace(-1+1/Qord, 1-1/Qord, Qord)
                weights = np.repeat(1., Qord)
            
            offsets = (offsets * dx * 0.5 / NQX, offsets * 0.5 / NQY)
            weights = (weights * dx * 0.5 / NQX, weights * 0.5 / NQY)
            quads = ( np.indices([NQX, NQY], dtype='float').T.
                      reshape(-1, ndim) + 0.5 ) * [dx/NQX, 1/NQY]
            quadWeights = np.repeat(1., len(quads))
            
            for i in range(ndim):
                quads = np.concatenate( 
                    [quads + offset*np.eye(ndim)[i] for offset in offsets[i]] )
                quadWeights = np.concatenate(
                    [quadWeights * weight for weight in weights[i]] )
            phiX = quads[:,0] / dx
            quads += [nodeX, 0]
            mapL = self.mapping(quads, nodeX)
            mapR = self.mapping(quads, self.nodeX[iPlane+1])
            indL = (np.searchsorted(self.nodeY[iPlane], mapL, side='right') - 1) % NY
            indR = (np.searchsorted(self.nodeY[iPlane + 1], mapR, side='right') - 1) % NY
            phiLY = (mapL - self.nodeY[iPlane][indL]) * self.idy[iPlane][indL]
            phiRY = (mapR - self.nodeY[iPlane + 1][indR]) * self.idy[iPlane + 1][indR]
            
            gradX = 1/dx
            gradTemplate = np.array(((-gradX, 0.),(-gradX, 0.),
                                      ( gradX, 0.),( gradX, 0.)))
            
            for iQ, quad in enumerate(quads):
                phis = np.array((((1-phiLY[iQ]) , (1-phiX[iQ])),
                                  (  phiLY[iQ]  , (1-phiX[iQ])),
                                  ((1-phiRY[iQ]),   phiX[iQ]  ),
                                  (  phiRY[iQ]  ,   phiX[iQ]  )))
                
                gradTemplate[:,1] = (-self.idy[iPlane][indL[iQ]],
                                      self.idy[iPlane][indL[iQ]],
                                      -self.idy[iPlane + 1][indR[iQ]],
                                      self.idy[iPlane + 1][indR[iQ]])
                gradphis = gradTemplate * phis
                
                # Gradients along the coordinate direction
                gradphis[:,0] -= self.mapping.deriv(quad)*gradphis[:,1]
            
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
                if f is not None:
                    self.b[indices] += f(quad) * phis * quadWeights[iQ]
        
        self.K = sp.csr_matrix( (Kdata, (row_ind, col_ind)),
                                shape=(nNodes, nNodes) )
        self.A = sp.csr_matrix( (Adata, (row_ind, col_ind)),
                                shape=(nNodes, nNodes) )
        if massLumping:
            self.M = sp.diags(self.u_weights, format='csr')
        else:
            self.M = sp.csr_matrix( (Mdata, (row_ind, col_ind)),
                                shape=(nNodes, nNodes) )
    
    def computeSpatialDiscretizationLinearVCI(self, f=None, NQX=1, NQY=None, Qord=2,
                                      quadType='gauss', massLumping=False):
        """Assemble the system discretization matrices K, A, M in CSR format.
        
        K is the stiffness matrix from the diffusion term
        A is the advection matrix
        M is the mass matrix from the time derivative
        
        Parameters
        ----------
        f : {callable, None}, optional
            Forcing function. Must take 2D array of points and return 1D array.
            The default is None.
        NQX : int, optional
            Number of quadrature cell divisions between FCI planes.
            The default is 1.
        NQY : {int, None}, optional
            Number of quadrature cell divisions in y-direction.
            The default is None, which sets NQY = NY.
        Qord : int, optional
            Number of quadrature points in each grid cell along one dimension.
            The default is 2.
        quadType : string, optional
            Type of quadrature to be used. Must be either 'gauss' or 'uniform'.
            Produces either Gauss-Legendre or Newton-Cotes type points/weights.
            The default is 'gauss'.
        massLumping : bool, optional
            Determines whether mass-lumping is used to calculate M matrix.
            The default is False.

        Returns
        -------
        None.

        """
        ndim = self.ndim
        nNodes = self.nNodes
        NX = self.NX
        NY = self.NY
        if NQY is None:
            NQY = NY
        self.f = f
        self.NQX = NQX
        self.NQY = NQY
        self.Qord = Qord
        self.quadType = quadType
        self.massLumping = massLumping
        # pre-allocate arrays for stiffness matrix triplets
        nEntries = (2*ndim)**2
        nQuads = NQX * NQY * Qord**2
        nMaxEntries = nEntries * nQuads * NX
        Kdata = np.zeros(nMaxEntries)
        Adata = np.zeros(nMaxEntries)
        if not massLumping:
            Mdata = np.zeros(nMaxEntries)
        row_ind = np.zeros(nMaxEntries, dtype='int')
        col_ind = np.zeros(nMaxEntries, dtype='int')
        self.b = np.zeros(nNodes)
        self.u_weights = np.zeros(nNodes)
        
        self.store = []
        self.areas = np.zeros(nNodes)
        self.xis = np.zeros((self.nNodes, self.ndim))
        
        ##### compute spatial discretizaton
        index = 0
        for iPlane in range(NX):
            dx = self.dx[iPlane]
            nodeX = self.nodeX[iPlane]
            ##### generate quadrature points
            if quadType.lower() in ('gauss', 'g'):
                offsets, weights = roots_legendre(Qord)
            elif quadType.lower() in ('uniform', 'u'):
                offsets = np.linspace(-1+1/Qord, 1-1/Qord, Qord)
                weights = np.repeat(1., Qord)
            
            offsets = (offsets * dx * 0.5 / NQX, offsets * 0.5 / NQY)
            weights = (weights * dx * 0.5 / NQX, weights * 0.5 / NQY)
            quads = ( np.indices([NQX, NQY], dtype='float').T.
                      reshape(-1, ndim) + 0.5 ) * [dx/NQX, 1/NQY]
            quadWeights = np.repeat(1., len(quads))
            
            for i in range(ndim):
                quads = np.concatenate( 
                    [quads + offset*np.eye(ndim)[i] for offset in offsets[i]] )
                quadWeights = np.concatenate(
                    [quadWeights * weight for weight in weights[i]] )
            phiX = quads[:,0] / dx
            quads += [nodeX, 0]
            mapL = self.mapping(quads, nodeX)
            mapR = self.mapping(quads, self.nodeX[iPlane+1])
            indL = (np.searchsorted(self.nodeY[iPlane], mapL, side='right') - 1) % NY
            indR = (np.searchsorted(self.nodeY[iPlane + 1], mapR, side='right') - 1) % NY
            phiLY = (mapL - self.nodeY[iPlane][indL]) * self.idy[iPlane][indL]
            phiRY = (mapR - self.nodeY[iPlane + 1][indR]) * self.idy[iPlane + 1][indR]
            
            gradX = 1/dx
            gradTemplate = np.array(((-gradX, 0.),(-gradX, 0.),
                                      ( gradX, 0.),( gradX, 0.)))
            
            for iQ, quad in enumerate(quads):
                phis = np.array((((1-phiLY[iQ]) , (1-phiX[iQ])),
                                  (  phiLY[iQ]  , (1-phiX[iQ])),
                                  ((1-phiRY[iQ]),   phiX[iQ]  ),
                                  (  phiRY[iQ]  ,   phiX[iQ]  )))
                
                gradTemplate[:,1] = (-self.idy[iPlane][indL[iQ]],
                                      self.idy[iPlane][indL[iQ]],
                                      -self.idy[iPlane + 1][indR[iQ]],
                                      self.idy[iPlane + 1][indR[iQ]])
                gradphis = gradTemplate * phis
                
                # Gradients along the coordinate direction
                gradphis[:,0] -= self.mapping.deriv(quad)*gradphis[:,1]
            
                phis = np.prod(phis, axis=1)
                indices = np.array([indL[iQ] + NY*iPlane,
                                    (indL[iQ]+1) % NY + NY*iPlane,
                                    (indR[iQ] + NY*(iPlane+1)) % nNodes,
                                    ((indR[iQ]+1) % NY + NY*(iPlane+1)) % nNodes])
                quadWeight = quadWeights[iQ]
                self.store.append((indices, phis, gradphis, quadWeight))
                self.areas[indices] += quadWeight
                self.xis[indices] -= gradphis * quadWeight                
                self.u_weights[indices] += quadWeight * phis
                
                if f is not None:
                    self.b[indices] += f(quad) * phis * quadWeight
        
        self.xis /= self.areas.reshape(-1,1)
        
        for iQ, (indices, phis, gradphis, quadWeight) in enumerate(self.store):
            testgrads = gradphis + self.xis[indices]
            # Kdata[index:index+nEntries] = ( quadWeight * 
            #     np.ravel( gradphis @ (self.diffusivity @ testgrads.T) ) )
            Kdata[index:index+nEntries] = ( quadWeight * 
                np.ravel( testgrads @ (self.diffusivity @ gradphis.T) ) )
            Adata[index:index+nEntries] = ( quadWeight *
                np.outer(np.dot(testgrads, self.velocity), phis).ravel() )
            if not massLumping:
                Mdata[index:index+nEntries] = ( quadWeight * 
                    np.outer(phis, phis).ravel() )
            row_ind[index:index+nEntries] = np.repeat(indices, 2*ndim)
            col_ind[index:index+nEntries] = np.tile(indices, 2*ndim)
            index += nEntries
        
        self.K = sp.csr_matrix( (Kdata, (row_ind, col_ind)),
                                shape=(nNodes, nNodes) )
        self.A = sp.csr_matrix( (Adata, (row_ind, col_ind)),
                                shape=(nNodes, nNodes) )
        if massLumping:
            self.M = sp.diags(self.u_weights, format='csr')
        else:
            self.M = sp.csr_matrix( (Mdata, (row_ind, col_ind)),
                                shape=(nNodes, nNodes) )
    
    def computeSpatialDiscretizationQuadraticVCI(self, f=None, NQX=1, NQY=None, Qord=2,
                                      quadType='gauss', massLumping=False):
        """Assemble the system discretization matrices K, A, M in CSR format.
        
        K is the stiffness matrix from the diffusion term
        A is the advection matrix
        M is the mass matrix from the time derivative
        
        Parameters
        ----------
        f : {callable, None}, optional
            Forcing function. Must take 2D array of points and return 1D array.
            The default is None.
        NQX : int, optional
            Number of quadrature cell divisions between FCI planes.
            The default is 1.
        NQY : {int, None}, optional
            Number of quadrature cell divisions in y-direction.
            The default is None, which sets NQY = NY.
        Qord : int, optional
            Number of quadrature points in each grid cell along one dimension.
            The default is 2.
        quadType : string, optional
            Type of quadrature to be used. Must be either 'gauss' or 'uniform'.
            Produces either Gauss-Legendre or Newton-Cotes type points/weights.
            The default is 'gauss'.
        massLumping : bool, optional
            Determines whether mass-lumping is used to calculate M matrix.
            The default is False.

        Returns
        -------
        None.

        """
        ndim = self.ndim
        nNodes = self.nNodes
        NX = self.NX
        NY = self.NY
        if NQY is None:
            NQY = NY
        self.f = f
        self.NQX = NQX
        self.NQY = NQY
        self.Qord = Qord
        self.quadType = quadType
        self.massLumping = massLumping
        # pre-allocate arrays for stiffness matrix triplets
        nEntries = (2*ndim)**2
        nQuads = NQX * NQY * Qord**2
        nMaxEntries = nEntries * nQuads * NX
        Kdata = np.zeros(nMaxEntries)
        Adata = np.zeros(nMaxEntries)
        if not massLumping:
            Mdata = np.zeros(nMaxEntries)
        row_ind = np.zeros(nMaxEntries, dtype='int')
        col_ind = np.zeros(nMaxEntries, dtype='int')
        self.b = np.zeros(nNodes)
        self.u_weights = np.zeros(nNodes)
        
        self.store = []
        A = np.zeros((self.nNodes, 3, 3))
        r = np.zeros((self.nNodes, self.ndim, 3))
        self.xi = np.empty((self.nNodes, self.ndim, 3))
        # self.areas = np.zeros(nNodes)
        # self.xis = np.zeros((self.nNodes, self.ndim))
        
        ##### compute spatial discretizaton
        index = 0
        for iPlane in range(NX):
            dx = self.dx[iPlane]
            nodeX = self.nodeX[iPlane]
            nodeX1 = self.nodeX[iPlane+1]
            ##### generate quadrature points
            if quadType.lower() in ('gauss', 'g'):
                offsets, weights = roots_legendre(Qord)
            elif quadType.lower() in ('uniform', 'u'):
                offsets = np.linspace(-1+1/Qord, 1-1/Qord, Qord)
                weights = np.repeat(1., Qord)
            
            offsets = (offsets * dx * 0.5 / NQX, offsets * 0.5 / NQY)
            weights = (weights * dx * 0.5 / NQX, weights * 0.5 / NQY)
            quads = ( np.indices([NQX, NQY], dtype='float').T.
                      reshape(-1, ndim) + 0.5 ) * [dx/NQX, 1/NQY]
            quadWeights = np.repeat(1., len(quads))
            
            for i in range(ndim):
                quads = np.concatenate( 
                    [quads + offset*np.eye(ndim)[i] for offset in offsets[i]] )
                quadWeights = np.concatenate(
                    [quadWeights * weight for weight in weights[i]] )
            phiX = quads[:,0] / dx
            quads += [nodeX, 0]
            mapL = self.mapping(quads, nodeX)
            mapR = self.mapping(quads, nodeX1)
            indL = (np.searchsorted(self.nodeY[iPlane], mapL, side='right') - 1) % NY
            indR = (np.searchsorted(self.nodeY[iPlane + 1], mapR, side='right') - 1) % NY
            phiLY = (mapL - self.nodeY[iPlane][indL]) * self.idy[iPlane][indL]
            phiRY = (mapR - self.nodeY[iPlane + 1][indR]) * self.idy[iPlane + 1][indR]
            
            gradX = 1/dx
            gradTemplate = np.array(((-gradX, 0.),(-gradX, 0.),
                                      ( gradX, 0.),( gradX, 0.)))
            
            for iQ, quad in enumerate(quads):
                phis = np.array((((1-phiLY[iQ]) , (1-phiX[iQ])),
                                  (  phiLY[iQ]  , (1-phiX[iQ])),
                                  ((1-phiRY[iQ]),   phiX[iQ]  ),
                                  (  phiRY[iQ]  ,   phiX[iQ]  )))
                
                gradTemplate[:,1] = (-self.idy[iPlane][indL[iQ]],
                                      self.idy[iPlane][indL[iQ]],
                                      -self.idy[iPlane + 1][indR[iQ]],
                                      self.idy[iPlane + 1][indR[iQ]])
                gradphis = gradTemplate * phis
                
                # Gradients along the coordinate direction
                gradphis[:,0] -= self.mapping.deriv(quad)*gradphis[:,1]
            
                phis = np.prod(phis, axis=1)
                indices = np.array([indL[iQ] + NY*iPlane,
                                    (indL[iQ]+1) % NY + NY*iPlane,
                                    (indR[iQ] + NY*(iPlane+1)) % nNodes,
                                    ((indR[iQ]+1) % NY + NY*(iPlane+1)) % nNodes])
                quadWeight = quadWeights[iQ]
                # self.store.append((indices, phis, gradphis, quadWeight))
                # self.areas[indices] += quadWeight
                # self.xis[indices] -= gradphis * quadWeight                
                self.u_weights[indices] += quadWeight * phis
                
                disps = np.array(((quad[0][0] - nodeX,  quad[0][1] - self.nodeY[iPlane][indL[iQ]]),
                                  (quad[0][0] - nodeX,  quad[0][1] - self.nodeY[iPlane][indL[iQ]+1]),
                                  (quad[0][0] - nodeX1, quad[0][1] - self.nodeY[iPlane+1][indR[iQ]]),
                                  (quad[0][0] - nodeX1, quad[0][1] - self.nodeY[iPlane+1][indR[iQ]+1])))
                
                self.store.append((indices, gradphis, disps, quadWeight))
                P = np.hstack((np.ones((len(indices), 1)), disps))
                A[indices] += quadWeight * \
                    np.apply_along_axis(lambda x: np.outer(x,x), 1, P)
                r[indices,:,0] -= gradphis * quadWeight
                r[indices,0,1] -= phis * quadWeight
                r[indices,1,2] -= phis * quadWeight
                r[indices,:,1:3] -= np.apply_along_axis(lambda x: np.outer(x[0:2],
                    x[2:4]), 1, np.hstack((gradphis, disps))) * quadWeight
                
                if f is not None:
                    self.b[indices] += f(quad) * phis * quadWeight
        
        # self.xis /= self.areas.reshape(-1,1)
        
        for i, row in enumerate(A):
            lu, piv = la.lu_factor(A[i], True, False)
            for j in range(self.ndim):
                self.xi[i,j] = la.lu_solve((lu, piv), r[i,j], 0, True, False)
        
        for iQ, (indices, gradphis, disps, quadWeight) in enumerate(self.store):
            # Kdata[index:index+nEntries] = quadWeight * \
            #     np.ravel((gradphis + self.xi[indices,:,0] +
            #         self.xi[indices,:,1] * disps[:,0:1] +
            #         self.xi[indices,:,2] * disps[:,1:2]) @ (self.diffusivity @ gradphis.T))
            # Adata[index:index+nEntries] = ( quadWeight *
            #     np.outer(np.dot(gradphis + self.xi[indices,:,0] +
            #         self.xi[indices,:,1] * disps[:,0:1] +
            #         self.xi[indices,:,2] * disps[:,1:2], self.velocity), phis).ravel() )
            
            testgrads = (gradphis + self.xi[indices,:,0] +
                    self.xi[indices,:,1] * disps[:,0:1] +
                    self.xi[indices,:,2] * disps[:,1:2])
            # testgrads -= 0.25*np.sum(testgrads, axis=0)
            
            Kdata[index:index+nEntries] = quadWeight * \
                np.ravel(testgrads @ (self.diffusivity @ gradphis.T))
            Adata[index:index+nEntries] = ( quadWeight *
                np.outer(np.dot(testgrads, self.velocity), phis).ravel() )
            
            # if np.sum((gradphis + self.xi[indices,:,0] +
            #         self.xi[indices,:,1] * disps[:,0:1] +
            #         self.xi[indices,:,2] * disps[:,1:2]) @
            #         (self.diffusivity @ gradphis.T)) > 1e-10:
            #     print('non-zero K sum detected.')
            
            # Kdata[index:index+nEntries] = ( quadWeight * 
            #     np.ravel( (gradphis + self.xis[indices]) @ (self.diffusivity @ gradphis.T) ) )
            # Adata[index:index+nEntries] = ( quadWeight *
            #     np.outer(np.dot(gradphis + self.xis[indices], self.velocity), phis).ravel() )
            if not massLumping:
                Mdata[index:index+nEntries] = ( quadWeight * 
                    np.outer(phis, phis).ravel() )
            row_ind[index:index+nEntries] = np.repeat(indices, 2*ndim)
            col_ind[index:index+nEntries] = np.tile(indices, 2*ndim)
            index += nEntries
        
        self.K = sp.csr_matrix( (Kdata, (row_ind, col_ind)),
                                shape=(nNodes, nNodes) )
        self.A = sp.csr_matrix( (Adata, (row_ind, col_ind)),
                                shape=(nNodes, nNodes) )
        if massLumping:
            self.M = sp.diags(self.u_weights, format='csr')
        else:
            self.M = sp.csr_matrix( (Mdata, (row_ind, col_ind)),
                                shape=(nNodes, nNodes) )
            
    def computeSpatialDiscretizationConservativeLinearVCI(self, f=None, NQX=1, NQY=None, Qord=2,
                                      quadType='gauss', massLumping=False):
        """Assemble the system discretization matrices K, A, M in CSR format.
        
        K is the stiffness matrix from the diffusion term
        A is the advection matrix
        M is the mass matrix from the time derivative
        
        Parameters
        ----------
        f : {callable, None}, optional
            Forcing function. Must take 2D array of points and return 1D array.
            The default is None.
        NQX : int, optional
            Number of quadrature cell divisions between FCI planes.
            The default is 1.
        NQY : {int, None}, optional
            Number of quadrature cell divisions in y-direction.
            The default is None, which sets NQY = NY.
        Qord : int, optional
            Number of quadrature points in each grid cell along one dimension.
            The default is 2.
        quadType : string, optional
            Type of quadrature to be used. Must be either 'gauss' or 'uniform'.
            Produces either Gauss-Legendre or Newton-Cotes type points/weights.
            The default is 'gauss'.
        massLumping : bool, optional
            Determines whether mass-lumping is used to calculate M matrix.
            The default is False.

        Returns
        -------
        None.

        """
        ndim = self.ndim
        nNodes = self.nNodes
        NX = self.NX
        NY = self.NY
        if NQY is None:
            NQY = NY
        self.f = f
        self.NQX = NQX
        self.NQY = NQY
        self.Qord = Qord
        self.quadType = quadType
        self.massLumping = massLumping
        # pre-allocate arrays for stiffness matrix triplets
        nEntries = (2*ndim)**2
        nQuads = NQX * NQY * Qord**2
        nMaxEntries = nEntries * nQuads * NX
        Kdata = np.zeros(nMaxEntries)
        Adata = np.zeros(nMaxEntries)
        if not massLumping:
            Mdata = np.zeros(nMaxEntries)
        row_ind = np.zeros(nMaxEntries, dtype='int')
        col_ind = np.zeros(nMaxEntries, dtype='int')
        self.b = np.zeros(nNodes)
        self.u_weights = np.zeros(nNodes)
        
        self.store = []
        # self.areas = np.zeros(nNodes)
        # self.xis = np.zeros((self.nNodes, self.ndim))
        self.gradphiSums = np.zeros((self.nNodes, self.ndim))
        self.gradphiSumsNew = np.zeros((self.nNodes, self.ndim))
        
        wg = np.zeros(8 * nQuads * NX)
        ri = np.zeros(8 * nQuads * NX, dtype='int')
        ci = np.zeros(8 * nQuads * NX, dtype='int')
        
        ##### compute spatial discretizaton
        index = 0
        for iPlane in range(NX):
            dx = self.dx[iPlane]
            nodeX = self.nodeX[iPlane]
            ##### generate quadrature points
            if quadType.lower() in ('gauss', 'g'):
                offsets, weights = roots_legendre(Qord)
            elif quadType.lower() in ('uniform', 'u'):
                offsets = np.linspace(-1+1/Qord, 1-1/Qord, Qord)
                weights = np.repeat(1., Qord)
            
            offsets = (offsets * dx * 0.5 / NQX, offsets * 0.5 / NQY)
            weights = (weights * dx * 0.5 / NQX, weights * 0.5 / NQY)
            quads = ( np.indices([NQX, NQY], dtype='float').T.
                      reshape(-1, ndim) + 0.5 ) * [dx/NQX, 1/NQY]
            quadWeights = np.repeat(1., len(quads))
            
            for i in range(ndim):
                quads = np.concatenate( 
                    [quads + offset*np.eye(ndim)[i] for offset in offsets[i]] )
                quadWeights = np.concatenate(
                    [quadWeights * weight for weight in weights[i]] )
            phiX = quads[:,0] / dx
            quads += [nodeX, 0]
            mapL = self.mapping(quads, nodeX)
            mapR = self.mapping(quads, self.nodeX[iPlane+1])
            indL = (np.searchsorted(self.nodeY[iPlane], mapL, side='right') - 1) % NY
            indR = (np.searchsorted(self.nodeY[iPlane + 1], mapR, side='right') - 1) % NY
            phiLY = (mapL - self.nodeY[iPlane][indL]) * self.idy[iPlane][indL]
            phiRY = (mapR - self.nodeY[iPlane + 1][indR]) * self.idy[iPlane + 1][indR]
            
            gradX = 1/dx
            gradTemplate = np.array(((-gradX, 0.),(-gradX, 0.),
                                      ( gradX, 0.),( gradX, 0.)))
            
            for iQ, quad in enumerate(quads):
                phis = np.array((((1-phiLY[iQ]) , (1-phiX[iQ])),
                                  (  phiLY[iQ]  , (1-phiX[iQ])),
                                  ((1-phiRY[iQ]),   phiX[iQ]  ),
                                  (  phiRY[iQ]  ,   phiX[iQ]  )))
                
                gradTemplate[:,1] = (-self.idy[iPlane][indL[iQ]],
                                      self.idy[iPlane][indL[iQ]],
                                      -self.idy[iPlane + 1][indR[iQ]],
                                      self.idy[iPlane + 1][indR[iQ]])
                gradphis = gradTemplate * phis
                
                # Gradients along the coordinate direction
                gradphis[:,0] -= self.mapping.deriv(quad)*gradphis[:,1]
            
                phis = np.prod(phis, axis=1)
                indices = np.array([indL[iQ] + NY*iPlane,
                                    (indL[iQ]+1) % NY + NY*iPlane,
                                    (indR[iQ] + NY*(iPlane+1)) % nNodes,
                                    ((indR[iQ]+1) % NY + NY*(iPlane+1)) % nNodes])
                quadWeight = quadWeights[iQ]
                self.store.append((indices, phis, gradphis, quadWeight))
                # self.areas[indices] += quadWeight
                # self.xis[indices] -= gradphis * quadWeight
                self.u_weights[indices] += quadWeight * phis
                
                wg[index:index+8] = quadWeight * gradphis.T.ravel()
                ri[index:index+4] = indices
                ri[index+4:index+8] = indices + self.nNodes
                ci[index:index+8] = iQ + iPlane*nQuads
                index += 8
                
                self.gradphiSums[indices] += gradphis * quadWeight
                
                if f is not None:
                    self.b[indices] += f(quad) * phis * quadWeight
        
        self.WG = sp.csr_matrix((wg, (ri, ci)), shape=(2*nNodes, nQuads * NX))
        # tolerance = 1e-10
        # self.xi, info = sp_la.lgmres(self.WG, np.zeros(nQuads * NX), x0=np.ones(nQuads * NX),
        #                              tol=tolerance, atol=tolerance)
        
        # _1 = np.ones(nQuads * NX)
        tol = np.finfo(float).eps
        maxit = nQuads * NX
        # self.xi = (_1 - self.WG.T @ sp_la.inv(self.WG@self.WG.T) @ self.WG @ _1)
        # self.xi = sp_la.lsqr(self.WG, np.zeros(2*self.nNodes), x0=np.ones(nQuads * NX))[0]
        # self.dxi = sp_la.lsqr(self.WG, -self.WG@np.ones(nQuads * NX), x0=np.zeros(nQuads * NX), iter_lim = 1000)
        # self.dxi = sp_la.lsqr(self.WG, -self.WG@np.ones(nQuads * NX), x0=np.zeros(nQuads * NX), atol=tol, btol=tol, iter_lim=maxit)
        self.dxi = sp_la.lsmr(self.WG, -self.WG@np.ones(nQuads * NX), x0=np.zeros(nQuads * NX), atol=tol, btol=tol, maxiter=maxit)
        self.xi = 1 + self.dxi[0]
        
        # self.xis /= self.areas.reshape(-1,1)
        
        index = 0
        for iQ, (indices, phis, gradphis, quadWeight) in enumerate(self.store):
            testgrads = gradphis * self.xi[iQ]
            Kdata[index:index+nEntries] = ( quadWeight * 
                np.ravel( testgrads @ (self.diffusivity @ gradphis.T) ) )
            Adata[index:index+nEntries] = ( quadWeight *
                np.outer(np.dot(testgrads, self.velocity), phis).ravel() )
            if not massLumping:
                Mdata[index:index+nEntries] = ( quadWeight * 
                    np.outer(phis, phis).ravel() )
            row_ind[index:index+nEntries] = np.repeat(indices, 2*ndim)
            col_ind[index:index+nEntries] = np.tile(indices, 2*ndim)
            index += nEntries
            
            self.gradphiSumsNew[indices] += testgrads * quadWeight
        
        self.K = sp.csr_matrix( (Kdata, (row_ind, col_ind)),
                                shape=(nNodes, nNodes) )
        self.A = sp.csr_matrix( (Adata, (row_ind, col_ind)),
                                shape=(nNodes, nNodes) )
        if massLumping:
            self.M = sp.diags(self.u_weights, format='csr')
        else:
            self.M = sp.csr_matrix( (Mdata, (row_ind, col_ind)),
                                shape=(nNodes, nNodes) )
    
    def initializeTimeIntegrator(self, integrator, dt, P='ilu', **kwargs):
        """Initialize and register the time integration scheme to be used.

        Parameters
        ----------
        integrator : {Integrator (object or subclass type), string}
            Integrator object or string specifiying which scheme is to be used.
            If a string, must be one of 'LowStorageRK' ('RK' or 'LSRK'),
            'BackwardEuler' ('BE'), or 'CrankNicolson' ('CN').
        dt : float
            Time interval between each successive timestep.
        P : {string, scipy.sparse.linalg.LinearOperator, None}, optional
            Which preconditioning method to use. P can be a LinearOperator to
            directly specifiy the preconditioner to be used. Otherwise it must
            be one of 'jacobi', 'ilu', or None. The default is 'ilu'.
        **kwargs
            Used to specify optional arguments for the time integrator.
            Will be passed to scipy.sparse.linalg.spilu if 'ilu' is used, or
            can be used to specify betas for LowStorageRK schemes.

        Returns
        -------
        None.

        """
        if isinstance(integrator, integrators.Integrator):
            self.integrator = integrator
            return
        if isinstance(integrator, str):
            if integrator.lower() in ('backwardeuler', 'be'):
                Type = integrators.BackwardEuler
            elif integrator.lower() in ('cranknicolson', 'cn'):
                Type = integrators.CrankNicolson
            elif integrator.lower() in ('lowstoragerk', 'rk', 'lsrk'):
                Type = integrators.LowStorageRK
        else: # if integrator not an Integrator object or string, assume it's a type
            Type = integrator
        # Instantiate and store the integrator object
        try:
            self.integrator = Type(self, self.A - self.K, self.M, dt, P, **kwargs)
        except:
            raise SystemExit("Unable to instantiate integrator of type "
                f"{repr(Type)}. Should be a string containing one of "
                "'LowStorageRK' ('RK' or 'LSRK') or 'BackwardEuler' ('BE'), a "
                "type derived from integrators.Integrator, or an object of "
                "such a type.")
    
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

    def generatePlottingPoints(self, nx=1, ny=1):
        """Generate set of interpolation points to use for plotting.

        Parameters
        ----------
        nx : int, optional
            Number of points per grid division in the x-direction.
            The default is 1.
        ny : int, optional
            Number of points per grid division in the y-direction.
            The default is 1.

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
            points += [nodeX, 0]
            mapL = self.mapping(points, nodeX)
            mapR = self.mapping(points, self.nodeX[iPlane+1])
            indL = (np.searchsorted(self.nodeY[iPlane], mapL, side='right') - 1) % NY
            indR = (np.searchsorted(self.nodeY[iPlane + 1], mapR, side='right') - 1) % NY
            phiLY = (mapL - self.nodeY[iPlane][indL]) * self.idy[iPlane][indL]
            phiRY = (mapR - self.nodeY[iPlane + 1][indR]) * self.idy[iPlane + 1][indR]
            for iP, point in enumerate(points):
                self.phiPlot[iPlane*nPointsPerPlane + iP] = (
                    (1-phiLY[iP]) * (1-phiX[iP]), phiLY[iP] * (1-phiX[iP]),
                    (1-phiRY[iP]) * phiX[iP]    , phiRY[iP] * phiX[iP] )
                self.indPlot[iPlane*nPointsPerPlane + iP] = (
                    indL[iP] + NY*iPlane,
                    (indL[iP]+1) % NY + NY*iPlane,
                    (indR[iP] + NY*(iPlane+1)) % nNodes,
                    ((indR[iP]+1) % NY + NY*(iPlane+1)) % nNodes )
        
        self.phiPlot[iPlane*nPointsPerPlane + iP + 1:] = self.phiPlot[0:NY*ny + 1]
        self.indPlot[iPlane*nPointsPerPlane + iP + 1:] = self.indPlot[0:NY*ny + 1]
        
        self.X = np.append(self.X, (2*np.pi*np.ones(NY*ny + 1)))
        self.Y = np.concatenate((np.tile(points[:,1], NX), points[0:NY*ny + 1,1]))
        self.U = np.sum(self.phiPlot * self.u[self.indPlot], axis=1)
    
    def computePlottingSolution(self):
        """Compute interpolated solution at the plotting points.

        Returns
        -------
        None.

        """
        self.U = np.sum(self.phiPlot * self.u[self.indPlot], axis=1)