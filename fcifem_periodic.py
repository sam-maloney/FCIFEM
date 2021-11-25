# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 13:47:07 2020

@author: Samuel A. Maloney

"""

from scipy.special import roots_legendre
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sp_la
import numpy as np
# import scipy
import ssqr

import integrators
import mappings
import boundaries


class FciFemSim:
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
    nDoFs : int
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
    b : numpy.ndarray, shape=(nDoFs,)
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
        NX = int(NX) # 'numpy.int**' classes can cause problems with SuiteSparse
        NY = int(NY)
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
        self.nDoFs = NX*NY
        self.nNodes = self.nDoFs
        self.dx = self.nodeX[1:] - self.nodeX[0:-1]
        self.idy = 1. / (self.nodeY[:,1:] - self.nodeY[:,:-1])
    
    def setInitialConditions(self, u0, mapped=True):
        """Initialize the nodal coefficients for the given IC.
        
        Parameters
        ----------
        u0 : {numpy.ndarray, callable}
            Initial conditions for the simulation.
            Must be an array of shape (self.nDoFs,) or a callable object
            returning such an array and taking as input the array of node
            coordinates with shape (self.nDoFs, self.ndim).
        mapped : bool, optional
            Whether mapping is applied to node positions before applying ICs.
            The default is True.

        Returns
        -------
        None.

        """
        if isinstance(u0, np.ndarray) and u0.shape == (self.nDoFs,):
            self.u0 = u0
            self.u = u0.copy()
            self.u0func = None
        elif callable(u0):
            self.u0func = u0
            self.DoFs = np.vstack( (np.repeat(self.nodeX[:-1], self.NY),
                                self.nodeY[:-1,:-1].ravel()) ).T
            self.nodes = self.DoFs
            self.DoFsMapped = self.DoFs.copy()
            self.DoFsMapped[:,1] = self.mapping(self.DoFs, 0) % 1
            if mapped:
                self.u = u0(self.DoFsMapped)
            else:
                self.u = u0(self.DoFs)
            self.u0 = self.u.copy()
        else:
            raise SystemExit(f"u0 must be an array of shape ({self.nDoFs},) "
                f"or a callable object returning such an array and taking as "
                f"input the array of node coordinates with shape "
                f"({self.nDoFs}, {self.ndim}).")
   
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
        nDoFs = self.nDoFs
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
        self.b = np.zeros(nDoFs)
        self.u_weights = np.zeros(nDoFs)
        
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
            mapL = self.mapping(quads, nodeX) % 1
            mapR = self.mapping(quads, self.nodeX[iPlane+1]) % 1
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
                                    (indR[iQ] + NY*(iPlane+1)) % nDoFs,
                                    ((indR[iQ]+1) % NY + NY*(iPlane+1)) % nDoFs])
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
                                shape=(nDoFs, nDoFs) )
        self.A = sp.csr_matrix( (Adata, (row_ind, col_ind)),
                                shape=(nDoFs, nDoFs) )
        if massLumping:
            self.M = sp.diags(self.u_weights, format='csr')
        else:
            self.M = sp.csr_matrix( (Mdata, (row_ind, col_ind)),
                                shape=(nDoFs, nDoFs) )
    
    def computeSpatialDiscretizationLinearVCI(self, f=None, NQX=1, NQY=None, Qord=2,
                                      quadType='gauss', massLumping=False):
        """Assemble the system discretization matrices K, A, M in CSR format.
        Implements linear variationally consistent integration using assumed
        strain method of Chen2013 https://doi.org/10.1002/nme.4512
        
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
        nDoFs = self.nDoFs
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
        self.b = np.zeros(nDoFs)
        self.u_weights = np.zeros(nDoFs)
        
        self.store = []
        self.areas = np.zeros(nDoFs)
        self.xis = np.zeros((self.nDoFs, self.ndim))
        
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
            mapL = self.mapping(quads, nodeX) % 1
            mapR = self.mapping(quads, self.nodeX[iPlane+1]) % 1
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
                                    (indR[iQ] + NY*(iPlane+1)) % nDoFs,
                                    ((indR[iQ]+1) % NY + NY*(iPlane+1)) % nDoFs])
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
                                shape=(nDoFs, nDoFs) )
        self.A = sp.csr_matrix( (Adata, (row_ind, col_ind)),
                                shape=(nDoFs, nDoFs) )
        if massLumping:
            self.M = sp.diags(self.u_weights, format='csr')
        else:
            self.M = sp.csr_matrix( (Mdata, (row_ind, col_ind)),
                                shape=(nDoFs, nDoFs) )
    
    def computeSpatialDiscretizationQuadraticVCI(self, f=None, NQX=1, NQY=None, Qord=2,
                                      quadType='gauss', massLumping=False):
        """Assemble the system discretization matrices K, A, M in CSR format.
        Implements quadratic variationally consistent integration using assumed
        strain method of Chen2013 https://doi.org/10.1002/nme.4512
        
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
        nDoFs = self.nDoFs
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
        self.b = np.zeros(nDoFs)
        self.u_weights = np.zeros(nDoFs)
        
        self.store = []
        A = np.zeros((self.nDoFs, 3, 3))
        r = np.zeros((self.nDoFs, self.ndim, 3))
        self.xi = np.empty((self.nDoFs, self.ndim, 3))
        
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
            mapL = self.mapping(quads, nodeX) % 1
            mapR = self.mapping(quads, nodeX1) % 1
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
                                    (indR[iQ] + NY*(iPlane+1)) % nDoFs,
                                    ((indR[iQ]+1) % NY + NY*(iPlane+1)) % nDoFs])
                quadWeight = quadWeights[iQ]               
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
                
        for i, row in enumerate(A):
            lu, piv = la.lu_factor(A[i], True, False)
            for j in range(self.ndim):
                self.xi[i,j] = la.lu_solve((lu, piv), r[i,j], 0, True, False)
        
        for iQ, (indices, gradphis, disps, quadWeight) in enumerate(self.store):
            testgrads = (gradphis + self.xi[indices,:,0] +
                    self.xi[indices,:,1] * disps[:,0:1] +
                    self.xi[indices,:,2] * disps[:,1:2])
            
            Kdata[index:index+nEntries] = quadWeight * \
                np.ravel(testgrads @ (self.diffusivity @ gradphis.T))
            Adata[index:index+nEntries] = ( quadWeight *
                np.outer(np.dot(testgrads, self.velocity), phis).ravel() )
            
            if not massLumping:
                Mdata[index:index+nEntries] = ( quadWeight * 
                    np.outer(phis, phis).ravel() )
            row_ind[index:index+nEntries] = np.repeat(indices, 2*ndim)
            col_ind[index:index+nEntries] = np.tile(indices, 2*ndim)
            index += nEntries
        
        self.K = sp.csr_matrix( (Kdata, (row_ind, col_ind)),
                                shape=(nDoFs, nDoFs) )
        self.A = sp.csr_matrix( (Adata, (row_ind, col_ind)),
                                shape=(nDoFs, nDoFs) )
        if massLumping:
            self.M = sp.diags(self.u_weights, format='csr')
        else:
            self.M = sp.csr_matrix( (Mdata, (row_ind, col_ind)),
                                shape=(nDoFs, nDoFs) )
            
    def computeSpatialDiscretizationConservativeNodeVCI(self, f=None, NQX=1, NQY=None, Qord=2,
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
        nDoFs = self.nDoFs
        NX = self.NX
        NY = self.NY
        if NQY is None:
            NQY = NY
        self.f = f
        self.NQX = NQX
        self.NQY = NQY
        self.Qord = Qord
        self.quadType = quadType
        self.massLumping = massLumping            # Kdata[index:index+nEntries] = ( quadWeight * 
            #     np.ravel( gradphis @ (self.diffusivity @ testgrads.T) ) )
        nQuads = NQX * NQY * Qord**2
        
        self.store = []
        self.gradphiSums = np.zeros((nDoFs, ndim))
        self.gradphiSumsNew = np.zeros((nDoFs, ndim))
        
        self.nCells = 2*nDoFs
        nCells = self.nCells
        gd = np.empty(16*nQuads*NX)
        ri = np.empty(16*nQuads*NX, dtype='int64')
        ci = np.empty(16*nQuads*NX, dtype='int64')
        
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
            mapL = self.mapping(quads, nodeX) % 1
            mapR = self.mapping(quads, self.nodeX[iPlane+1]) % 1
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
                indices = np.array((indL[iQ] + NY*iPlane,
                                    (indL[iQ]+1) % NY + NY*iPlane,
                                    (indR[iQ] + NY*(iPlane+1)) % nDoFs,
                                    ((indR[iQ]+1) % NY + NY*(iPlane+1)) % nDoFs))
                quadWeight = quadWeights[iQ]
                yCell = int(quad[0,1]*NQY)
                cellCentre = np.array((nodeX + 0.5*dx, (yCell + 0.5)/NQY))
                disp = quad[0] - cellCentre
                cellId = iPlane*NY + yCell
                self.store.append((indices, phis, gradphis, quadWeight, quad, cellId, disp))
                
                # idx = (indices%NY).astype('bool')
                # ni = 2*sum(idx)
                # gd[index:index+2*ni] = np.repeat(gradphis[idx].T,2) * np.tile(disp, ni)
                # ri[index:index+ni] = np.repeat(indices[idx], 2)
                # ri[index+ni:index+2*ni] = np.repeat(indices[idx], 2) + nDoFs
                # ci[index:index+2*ni] = np.tile((cellId, cellId + nDoFs), ni)
                # index += 2*ni
                
                gd[index:index+16] = np.repeat(gradphis.T,2) * np.tile(disp, 8)
                ri[index:index+8] = np.repeat(indices, 2)
                ri[index+8:index+16] = np.repeat(indices, 2) + nDoFs
                ci[index:index+16] = np.tile((cellId, cellId + nDoFs), 8)
                index += 16
                
                self.gradphiSums[indices] -= gradphis * quadWeight
        
        del quads, quadWeights, mapL, mapR, indL, indR, phiLY, phiRY, phiX
        
        # gd[index:index+nCells] = 1.0
        # ri[index:index+nCells] = 2*nDoFs
        # ci[index:index+nCells] = np.arange(nCells)
        
        ##### Using SuiteSparse #####
        # idx = (np.arange(nCells)%NY).astype('bool')
        # self.G = sp.csc_matrix((gd[:index], (ci[:index], ri[:index])),
        #                         shape=(np.iinfo('int32').max + 1, nCells))
        self.G = sp.csc_matrix((gd, (ci, ri)),
                                shape=(np.iinfo('int32').max + 1, nCells))
        self.G._shape = (nCells, nCells)
        # self.G.indptr = self.G.indptr[np.hstack((idx, True))]
        # self.G._shape = (nCells, nCells-2*NX)
        del gd, ci, ri
        QR, r = ssqr.QR_C(self.G)
        # rhs = self.gradphiSums.T.ravel()[(np.arange(nCells)%NY).astype('bool')]
        try:
            QR.E[0][0]
            E = np.frombuffer(QR.E[0], dtype=np.int64, count=nCells)
            # rhs = rhs[E[:r]]
            rhs = self.gradphiSums.T.ravel()[E[:r]]
        except:
            # rhs = rhs[:r]
            rhs = self.gradphiSums.T.ravel()[:r]
        R = ssqr.cholmodSparseToScipyCsc(QR.R)
        x = np.empty(nCells)
        x[:r] = sp_la.spsolve_triangular(R.T[:r,:r], rhs, lower=True,
                                          overwrite_A=True, overwrite_b=True)
        x[r:] = 0.
        self.xi = (ssqr.qmult(QR, x), r)
        
        # self.xi = (np.zeros(nCells), 0)
        
        # ri -= np.ceil(ri/NY).astype('int')
        # self.G = sp.csr_matrix((gd, (ri, ci)), shape=(nCells-2*NX, nCells))
        # rhs = self.gradphiSums.T.ravel()[(np.arange(nCells)%NY).astype('bool')]
        # v0 = np.zeros(nCells)
        # # maxit = 2*nDoFs
        # maxit = nQuads * NX
        # # tol = np.finfo(float).eps
        # tol = 1e-10
        # # self.xi = sp_la.lsqr(self.G, rhs, x0=v0, atol=tol, btol=tol, iter_lim=maxit)
        # # self.xi = (self.xi[0], self.xi[1], self.xi[2], self.xi[3], self.xi[4])
        # D = sp.diags(1/sp_la.norm(self.G, axis=0), format='csc')
        # # self.xi = sp_la.lsmr(self.G @ D, rhs, x0=v0, atol=tol, btol=tol, maxiter=maxit)
        # self.xi = sp_la.lsqr(self.G @ D, rhs, x0=v0, atol=tol, btol=tol, iter_lim=maxit)
        # self.xi = (D @ self.xi[0], self.xi[1], self.xi[2], self.xi[3], self.xi[4])
        # # self.xi = sp_la.bicgstab(self.G, rhs, M=D, x0=v0, tol=tol, atol=tol)
        # # self.xi = sp_la.gcrotmk(self.G, rhs, M=D, x0=v0, tol=tol, atol=tol)
        # # ilu = sp_la.spilu(self.G)
        # # P = sp_la.LinearOperator(self.G.shape, lambda x: ilu.solve(x))
        # # self.xi = sp_la.lgmres(self.G, rhs, M=P, x0=v0, tol=tol, atol=tol)
        
        # gd[index:index+nCells] = 1.0
        # ri[index:index+nCells] = 2*nDoFs
        # ci[index:index+nCells] = np.arange(nCells)
        # self.G = sp.csr_matrix((gd, (ri, ci)), shape=(2*nDoFs+1, nCells+1))
        # self.G[:,-1] = self.G.sum(axis=1)
        # self.G.data[-1] = nCells
        # rhs = np.concatenate((self.gradphiSums.T.ravel(), np.zeros(1)))
        # v0 = np.zeros(nCells+1)
        # # maxit = 2*nDoFs
        # maxit = nQuads * NX
        # # tol = np.finfo(float).eps
        # tol = 1e-10
        # D = sp.diags(1/sp_la.norm(self.G, axis=0), format='csr')
        # # self.xi = sp_la.lsmr(self.G @ D, rhs, x0=v0, atol=tol, btol=tol, maxiter=maxit)
        # self.xi = sp_la.lsqr(self.G @ D, rhs, x0=v0, atol=tol, btol=tol, iter_lim=maxit)
        # self.xi = (D @ self.xi[0], self.xi[1], self.xi[2], self.xi[3], self.xi[4])
        # # ilu = sp_la.spilu(self.G, fill_factor=1.)
        # # P = sp_la.LinearOperator(self.G.shape, lambda x: ilu.solve(x))
        # # self.xi = sp_la.lgmres(self.G, rhs, x0=v0, maxiter=maxit, tol=tol, atol=tol)
        
        # pre-allocate arrays for stiffness matrix triplets
        nEntries = (2*ndim)**2
        nMaxEntries = nEntries * nQuads * NX
        Kdata = np.empty(nMaxEntries)
        Adata = np.empty(nMaxEntries)
        if not massLumping:
            Mdata = np.empty(nMaxEntries)
        row_ind = np.empty(nMaxEntries, dtype='int')
        col_ind = np.empty(nMaxEntries, dtype='int')
        self.b = np.zeros(nDoFs)
        self.u_weights = np.zeros(nDoFs)
        index = 0
        for iQ, (indices, phis, gradphis, quadWeight, quad, cellId, disp) in enumerate(self.store):
            quadWeight += np.sum(self.xi[0][[cellId, cellId + nDoFs]] * disp)
            Kdata[index:index+nEntries] = ( quadWeight * 
                np.ravel( gradphis @ (self.diffusivity @ gradphis.T) ) )
            Adata[index:index+nEntries] = ( quadWeight *
                np.outer(np.dot(gradphis, self.velocity), phis).ravel() )
            if not massLumping:
                Mdata[index:index+nEntries] = ( quadWeight * 
                    np.outer(phis, phis).ravel() )
            row_ind[index:index+nEntries] = np.repeat(indices, 2*ndim)
            col_ind[index:index+nEntries] = np.tile(indices, 2*ndim)
            index += nEntries
            
            self.gradphiSumsNew[indices] += gradphis * quadWeight
            self.u_weights[indices] += quadWeight * phis
            if f is not None:
                self.b[indices] += f(quad) * phis * quadWeight
        
        self.K = sp.csr_matrix( (Kdata, (row_ind, col_ind)),
                                shape=(nDoFs, nDoFs) )
        self.A = sp.csr_matrix( (Adata, (row_ind, col_ind)),
                                shape=(nDoFs, nDoFs) )
        if massLumping:
            self.M = sp.diags(self.u_weights, format='csr')
        else:
            self.M = sp.csr_matrix( (Mdata, (row_ind, col_ind)),
                                shape=(nDoFs, nDoFs) )
    
    def computeSpatialDiscretizationConservativeCellVCI(self, f=None, NQX=1, NQY=None, Qord=2,
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
        nDoFs = self.nDoFs
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
        self.b = np.zeros(nDoFs)
        self.u_weights = np.zeros(nDoFs)
        
        self.store = []
        self.gradphiSums = np.zeros((nDoFs, ndim))
        self.gradphiSumsNew = np.zeros((nDoFs, ndim))
        
        self.nCells = NQX*NQY*NX
        nCells = self.nCells
        gd = np.empty(8*nQuads*NX + nCells)
        ri = np.empty(8*nQuads*NX + nCells, dtype='int')
        ci = np.empty(8*nQuads*NX + nCells, dtype='int')
        
        ##### compute spatial discretizaton
        index = 0
        for iPlane in range(NX):
            dx = self.dx[iPlane]
            nodeX = self.nodeX[iPlane]
            iqdx = NQX/dx
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
            mapL = self.mapping(quads, nodeX) % 1
            mapR = self.mapping(quads, self.nodeX[iPlane+1]) % 1
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
                                    (indR[iQ] + NY*(iPlane+1)) % nDoFs,
                                    ((indR[iQ]+1) % NY + NY*(iPlane+1)) % nDoFs])
                quadWeight = quadWeights[iQ]
                cellId = iPlane*NQX*NQY + NQX*int(quad[0,1]*NQY) + int((quad[0,0]-nodeX)*iqdx)
                self.store.append((indices, phis, gradphis, quadWeight, quad, cellId))
                
                gd[index:index+8] = gradphis.T.ravel()
                ri[index:index+4] = indices
                ri[index+4:index+8] = indices + self.nDoFs
                ci[index:index+8] = cellId
                index += 8
                
                self.gradphiSums[indices] -= gradphis * quadWeight
        
        gd[index:index+nCells] = 1.0
        ri[index:index+nCells] = 2*nDoFs
        ci[index:index+nCells] = np.arange(nCells)
        
        self.G = sp.csr_matrix((gd, (ri, ci)), shape=(2*nDoFs+1, nCells))
        rhs = np.concatenate((self.gradphiSums.T.ravel(), np.zeros(1)))
        v0 = np.zeros(nCells)
        # maxit = 2*nDoFs
        maxit = nQuads * NX
        # tol = np.finfo(float).eps
        tol = 1e-10
        D = sp.diags(1/sp_la.norm(self.G, axis=0), format='csr')
        # self.xi = sp_la.lsmr(self.G @ D, rhs, x0=v0, atol=tol, btol=tol, maxiter=maxit)
        self.xi = sp_la.lsqr(self.G @ D, rhs, x0=v0, atol=tol, btol=tol, iter_lim=maxit)
        self.xi = (D @ self.xi[0], self.xi[1], self.xi[2], self.xi[3])
        
        index = 0
        for iQ, (indices, phis, gradphis, quadWeight, quad, cellId) in enumerate(self.store):
            quadWeight += self.xi[0][cellId] # Every quad cell
            Kdata[index:index+nEntries] = ( quadWeight * 
                np.ravel( gradphis @ (self.diffusivity @ gradphis.T) ) )
            Adata[index:index+nEntries] = ( quadWeight *
                np.outer(np.dot(gradphis, self.velocity), phis).ravel() )
            if not massLumping:
                Mdata[index:index+nEntries] = ( quadWeight * 
                    np.outer(phis, phis).ravel() )
            row_ind[index:index+nEntries] = np.repeat(indices, 2*ndim)
            col_ind[index:index+nEntries] = np.tile(indices, 2*ndim)
            index += nEntries
            
            self.gradphiSumsNew[indices] += gradphis * quadWeight
            self.u_weights[indices] += quadWeight * phis
            if f is not None:
                self.b[indices] += f(quad) * phis * quadWeight
        
        self.K = sp.csr_matrix( (Kdata, (row_ind, col_ind)),
                                shape=(nDoFs, nDoFs) )
        self.A = sp.csr_matrix( (Adata, (row_ind, col_ind)),
                                shape=(nDoFs, nDoFs) )
        if massLumping:
            self.M = sp.diags(self.u_weights, format='csr')
        else:
            self.M = sp.csr_matrix( (Mdata, (row_ind, col_ind)),
                                shape=(nDoFs, nDoFs) )
            
    def computeSpatialDiscretizationConservativePointVCI(self, f=None, NQX=1, NQY=None, Qord=2,
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
        nDoFs = self.nDoFs
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
        self.b = np.zeros(nDoFs)
        self.u_weights = np.zeros(nDoFs)
        
        self.store = []
        self.gradphiSums = np.zeros((nDoFs, ndim))
        self.gradphiSumsNew = np.zeros((nDoFs, ndim))
        
        gd = np.empty(9 * nQuads * NX)
        ri = np.empty(9 * nQuads * NX, dtype='int')
        ci = np.empty(9 * nQuads * NX, dtype='int')
        
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
            mapL = self.mapping(quads, nodeX) % 1
            mapR = self.mapping(quads, self.nodeX[iPlane+1]) % 1
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
                                    (indR[iQ] + NY*(iPlane+1)) % nDoFs,
                                    ((indR[iQ]+1) % NY + NY*(iPlane+1)) % nDoFs])
                quadWeight = quadWeights[iQ]
                self.store.append((indices, phis, gradphis, quadWeight, quad))
                
                gd[index:index+8] = gradphis.T.ravel()
                ri[index:index+4] = indices
                ri[index+4:index+8] = indices + self.nDoFs
                ci[index:index+8] = iQ + iPlane*nQuads
                index += 8
                
                self.gradphiSums[indices] -= gradphis * quadWeight
        
        gd[index:index+(nQuads * NX)] = 1.0
        ri[index:index+(nQuads * NX)] = 2*nDoFs
        ci[index:index+(nQuads * NX)] = np.arange(nQuads * NX)
    
        self.G = sp.csr_matrix((gd, (ri, ci)), shape=(2*nDoFs+1, nQuads * NX))
        rhs = np.concatenate((self.gradphiSums.T.ravel(), np.zeros(1)))
        v0 = np.zeros(nQuads * NX)
        maxit = nQuads * NX
        # tol = np.finfo(float).eps
        tol = 1e-10
        # self.xi = sp_la.lsmr(self.G, rhs, x0=v0, atol=tol, btol=tol, maxiter=maxit)
        self.xi = sp_la.lsqr(self.G, rhs, x0=v0, atol=tol, btol=tol, iter_lim=maxit)
                
        index = 0
        for iQ, (indices, phis, gradphis, quadWeight, quad) in enumerate(self.store):
            quadWeight += self.xi[0][iQ]
            Kdata[index:index+nEntries] = ( quadWeight * 
                np.ravel( gradphis @ (self.diffusivity @ gradphis.T) ) )
            Adata[index:index+nEntries] = ( quadWeight *
                np.outer(np.dot(gradphis, self.velocity), phis).ravel() )
            if not massLumping:
                Mdata[index:index+nEntries] = ( quadWeight * 
                    np.outer(phis, phis).ravel() )
            row_ind[index:index+nEntries] = np.repeat(indices, 2*ndim)
            col_ind[index:index+nEntries] = np.tile(indices, 2*ndim)
            index += nEntries
            
            self.gradphiSumsNew[indices] += gradphis * quadWeight
            self.u_weights[indices] += quadWeight * phis
            if f is not None:
                self.b[indices] += f(quad) * phis * quadWeight
        
        self.K = sp.csr_matrix( (Kdata, (row_ind, col_ind)),
                                shape=(nDoFs, nDoFs) )
        self.A = sp.csr_matrix( (Adata, (row_ind, col_ind)),
                                shape=(nDoFs, nDoFs) )
        if massLumping:
            self.M = sp.diags(self.u_weights, format='csr')
        else:
            self.M = sp.csr_matrix( (Mdata, (row_ind, col_ind)),
                                shape=(nDoFs, nDoFs) )
    
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
        nDoFs = self.nDoFs
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
            # Note: negative numbers very close to zero (about -3.5e-10) may be
            # rounded to 1.0 after the 1st modulo, hence why the 2nd is needed.
            mapL = self.mapping(points, nodeX) % 1 % 1
            mapR = self.mapping(points, self.nodeX[iPlane+1]) % 1 % 1
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
                    (indR[iP] + NY*(iPlane+1)) % nDoFs,
                    ((indR[iP]+1) % NY + NY*(iPlane+1)) % nDoFs )
        
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