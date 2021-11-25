# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 13:47:07 2020

@author: Samuel A. Maloney

"""

from scipy.special import roots_legendre
import scipy.sparse as sp
import scipy.sparse.linalg as sp_la
import numpy as np
import ssqr

import integrators
import mappings
import boundaries

from timeit import default_timer


class FciFemSim:
    """Class for flux-coordinate independent FEM (FCIFEM) method.
    Implements the convection-diffusion equation on a rectangular domain
    [x, y] = [0...2*pi, 0...1].
    
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
        RHS forcing vector generated from source/sink function f.
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
        self.dx = self.nodeX[1:] - self.nodeX[0:-1]
        self.idy = 1. / (self.nodeY[:,1:] - self.nodeY[:,:-1])
    
    def setInitialConditions(self, u0, mapped=True, BC='periodic'):
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
        BC : {boundaries.Boundary, string}, optionalquads, iPlane)
            Either an object of type boundaries.Boundary, or string 'periodic'.
            The default is 'periodic'.

        Returns
        -------
        None.

        """
        if isinstance(BC, boundaries.Boundary):
             self.BC = BC
        elif BC.lower() in ('periodic', 'p'):
            self.BC = boundaries.PeriodicBoundary(self)
        else:
            raise SystemExit(f"Unkown boundary condition: {BC}")
        self.nDoFs = self.BC.nDoFs
        self.nNodes = self.BC.nNodes
        self.nodes = self.BC.computeNodes()
        self.DoFs = self.nodes[:self.nDoFs]
        self.DoFsMapped = self.DoFs.copy()
        self.DoFsMapped[:,1] = self.BC.mapping(self.DoFs, 0)
        if isinstance(u0, np.ndarray) and u0.shape == (self.nDoFs,):
            self.u0 = u0
            self.u = u0.copy()
            self.u0func = None
        elif callable(u0):
            self.u0func = u0
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
            ##### generate quadrature points
            if quadType.lower() in ('gauss', 'g', 'gaussian'):
                offsets, weights = roots_legendre(Qord)
            elif quadType.lower() in ('uniform', 'u'):
                offsets = np.linspace(1/Qord - 1, 1 - 1/Qord, Qord)
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
            
            quads += [self.nodeX[iPlane], 0]
            
            for iQ, quad in enumerate(quads):
                if f is not None:
                    fq = f(quad)
                phis, gradphis, inds = self.BC(quad, iPlane)
                for alpha, i in enumerate(inds):
                    if i < 0:
                        continue # move to next i if boundary node
                    for beta, j in enumerate(inds):
                        if j < 0: # j is boundary node
                            ##### Not sure if this can always be uncommmented? #####
                            ##### Needed for projection; but does it affect Poisson/CD #####
                            # self.b[i] -= quadWeights[iQ] * (
                            #     phis[alpha] * phis[beta] )
                            self.b[i] -= quadWeights[iQ] * (
                                (gradphis[alpha] @ self.velocity) * phis[beta] +
                                (gradphis[alpha] @ (self.diffusivity @ gradphis[beta])) )
                        else: # i and j are both interior
                            if not massLumping:
                                Mdata[index] = quadWeights[iQ] * phis[alpha] * phis[beta]
                            Adata[index] = quadWeights[iQ] * (gradphis[alpha] @ self.velocity) * phis[beta]
                            Kdata[index] = quadWeights[iQ] * (gradphis[alpha] @ (self.diffusivity @ gradphis[beta]))
                            row_ind[index] = i
                            col_ind[index] = j
                            index += 1
                    self.u_weights[i] += quadWeights[iQ] * phis[alpha]
                    if f is not None:
                        self.b[i] += quadWeights[iQ] * fq * phis[alpha]
        
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
        self.areas = np.zeros(nDoFs + 1)
        self.xis = np.zeros((self.nDoFs + 1, self.ndim))
        
        ##### compute spatial discretizaton
        index = 0
        for iPlane in range(NX):
            dx = self.dx[iPlane]
            ##### generate quadrature points
            if quadType.lower() in ('gauss', 'g', 'gaussian'):
                offsets, weights = roots_legendre(Qord)
            elif quadType.lower() in ('uniform', 'u'):
                offsets = np.linspace(1/Qord - 1, 1 - 1/Qord, Qord)
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
            
            quads += [self.nodeX[iPlane], 0]
            
            for iQ, quad in enumerate(quads):
                phis, gradphis, inds = self.BC(quad, iPlane)
                quadWeight = quadWeights[iQ]
                self.store.append((inds, phis, gradphis, quadWeight, quad))
                inds[inds < 0] = -1
                self.areas[inds] += quadWeight
                self.xis[inds] -= gradphis * quadWeight   
        
        # self.gradphiSumsOld = -self.xis[0:-1]
        # self.gradphiSumsNew = np.zeros((nDoFs, 2))
        self.xis /= self.areas.reshape(-1,1)
                    
        for iQ, (inds, phis, gradphis, quadWeight, quad) in enumerate(self.store):
            if f is not None:
                fq = f(quad)
            for alpha, i in enumerate(inds):
                if i < 0:
                    continue # move to next i if boundary node
                testgrad = gradphis[alpha] + self.xis[i]
                # self.gradphiSumsNew[i] += testgrad * quadWeight
                self.u_weights[i] += quadWeight * phis[alpha]
                if f is not None:
                    self.b[i] += quadWeight * fq * phis[alpha]
                for beta, j in enumerate(inds):
                    if j < 0: # j is boundary node
                        ##### Not sure if this can always be uncommmented? #####
                        ##### Needed for projection; but does it affect Poisson/CD #####
                        # self.b[i] -= quadWeight * (
                        #     phis[alpha] * phis[beta] )
                        self.b[i] -= quadWeight * (
                            (testgrad @ self.velocity) * phis[beta] +
                            (testgrad @ (self.diffusivity @ gradphis[beta])) )
                    else: # i and j are both interior
                        if not massLumping:
                            Mdata[index] = quadWeight * phis[alpha] * phis[beta]
                        Adata[index] = quadWeight * (testgrad @ self.velocity) * phis[beta]
                        Kdata[index] = quadWeight * (testgrad @ (self.diffusivity @ gradphis[beta]))
                        row_ind[index] = i
                        col_ind[index] = j
                        index += 1
        
        self.K = sp.csr_matrix( (Kdata, (row_ind, col_ind)),
                                shape=(nDoFs, nDoFs) )
        self.A = sp.csr_matrix( (Adata, (row_ind, col_ind)),
                                shape=(nDoFs, nDoFs) )
        if massLumping:
            self.M = sp.diags(self.u_weights, format='csr')
        else:
            self.M = sp.csr_matrix( (Mdata, (row_ind, col_ind)),
                                shape=(nDoFs, nDoFs) )
    
    def computeSpatialDiscretizationConservativeLinearVCI(self, f=None, NQX=1,
            NQY=None, Qord=2, quadType='gauss', massLumping=False):
        """Assemble the system discretization matrices K, A, M in CSR format.
        Implements linear variationally consistent integration by re-weighting
        the quadrature points.
        
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
        self.b = np.zeros(nDoFs)
        self.u_weights = np.zeros(nNodes)
        
        self.store = []
        
        gd = np.empty(9 * nQuads * NX)
        ri = np.empty(9 * nQuads * NX, dtype='int')
        ci = np.empty(9 * nQuads * NX, dtype='int')
        
        self.rOld = np.zeros((nNodes, self.ndim, 3))
        
        ##### compute spatial discretizaton
        index = 0
        for iPlane in range(NX):
            dx = self.dx[iPlane]
            ##### generate quadrature points
            if quadType.lower() in ('gauss', 'g', 'gaussian'):
                offsets, weights = roots_legendre(Qord)
            elif quadType.lower() in ('uniform', 'u'):
                offsets = np.linspace(1/Qord - 1, 1 - 1/Qord, Qord)
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
            
            quads += [self.nodeX[iPlane], 0]
            
            for iQ, quad in enumerate(quads):
                phis, gradphis, inds = self.BC(quad, iPlane)
                quadWeight = quadWeights[iQ]
                self.store.append((inds, phis, gradphis, quadWeight, quad))
                
                for alpha, i in enumerate(inds):
                    disp = quad - self.nodes[i]
                    self.rOld[i,:,0] -= gradphis[alpha] * quadWeight
                    self.rOld[i,0,1] -= phis[alpha] * quadWeight
                    self.rOld[i,1,2] -= phis[alpha] * quadWeight
                    self.rOld[i,:,1:3] -= np.outer(gradphis[alpha], disp) * quadWeight
                    if i < 0:
                        continue # move to next i if boundary node
                    gd[index:index+2] = gradphis[alpha]
                    ri[index:index+2] = (i, i+nDoFs)
                    ci[index:index+2] = iQ + iPlane*nQuads
                    index += 2
        
        gd[index:index + nQuads*NX] = 1.0
        ri[index:index + nQuads*NX] = 2*nDoFs
        ci[index:index + nQuads*NX] = np.arange(nQuads * NX)
        index += nQuads * NX
        
        if self.BC.name == 'Dirichlet':
            nYnodes = self.BC.nYnodes
            NDX = self.BC.NDX
            g = self.BC.g
            # left boundary
            self.rOld[-nYnodes:,0,0] -= g(self.nodes[-nYnodes:]) \
                * 0.5 * np.flip(self.nodeY[0,2:] - self.nodeY[0,:-2])
            # right boundary
            self.rOld[-2*nYnodes:-nYnodes,0,0] += g(self.nodes[-2*nYnodes:-nYnodes]) \
                * 0.5 * np.flip(self.nodeY[1,2:] - self.nodeY[1,:-2])
            # bottom boundary
            self.rOld[-2*nYnodes - NX*NDX:-2*nYnodes - 1,1,0] -= \
                g(self.nodes[-2*nYnodes - NX*NDX:-2*nYnodes - 1]) \
                * 0.5*(self.nodes[-2*nYnodes - NX*NDX - 1:-2*nYnodes-2,0] - 
                       self.nodes[-2*nYnodes - NX*NDX + 1:-2*nYnodes,0])
            # top boundary
            self.rOld[nDoFs+1:-2*nYnodes - NX*NDX - 2,1,0] += \
                g(self.nodes[nDoFs+1:-2*nYnodes - NX*NDX - 2]) \
                * 0.5*(self.nodes[nDoFs:-2*nYnodes - NX*NDX - 3,0] -
                       self.nodes[nDoFs+2:-2*nYnodes - NX*NDX - 1,0])
            # [0., 0.]
            self.rOld[-2*nYnodes - 1,:,0] -= g(self.nodes[-2*nYnodes - 1]) \
                * 0.5*[1/self.idy[0][0], self.nodes[-2*nYnodes-2,0]]
            # [6.28318531, 0.        ]
            self.rOld[nDoFs + NX*NDX + 1,:,0] += g(self.nodes[nDoFs + NX*NDX + 1]) \
                * 0.5*[1/self.idy[-1][0], self.nodes[nDoFs + NX*NDX + 2,0] - 2*np.pi]
            ##### TODO
            # [0., 1.]
            g(self.nodes[nDoFs + NX*NDX])
            # [6.28318531, 1.        ]
            g(self.nodes[nDoFs])
            
        
        self.gradphiSums = self.rOld[:nDoFs,:,0]
        
        ##### Using SuiteSparse QR decomposition #####
        # Form the transpose of G (i.e. ri and ci intentionally swapped)
        # n.b. using np.iinfo('int32').max + 1 forces indices to be int64
        self.G = sp.csc_matrix((gd[:index], (ci[:index], ri[:index])),
                      shape=(np.iinfo('int32').max + 1, 2*nDoFs + 1))
        self.G._shape = (nQuads * NX, 2*nDoFs + 1)
        del gd, ci, ri, offsets, weights, quads, quadWeights
        start_time = default_timer()
        QR, r = ssqr.QR_C(self.G)
        if r == -1:
            raise SystemExit("Error in QR decomposition")
        try:
            QR.E[0][0]
            E = np.frombuffer(QR.E[0], dtype=np.int64, count=r)
            rhs = np.concatenate((self.gradphiSums.T.ravel(), np.zeros(1)))[E]
        except:
            rhs = np.concatenate((self.gradphiSums.T.ravel(), np.zeros(1)))[:r]
        R = ssqr.cholmodSparseToScipyCsc(QR.R)
        x = np.empty(nQuads * NX)
        x[:r] = sp_la.spsolve_triangular(R.T[:r,:r], rhs, lower=True,
                                          overwrite_A=True, overwrite_b=True)
        x[r:] = 0.
        self.xi = (ssqr.qmult(QR, x), r)
        print(f'xi solve time = {default_timer()-start_time} s')

        # ##### Using scipy.sparse.linalg, much slower, uses less memory #####
        # self.G = sp.csr_matrix((gd[:index], (ri[:index], ci[:index])),
        #                         shape=(2*nDoFs+1, nQuads * NX))
        # rhs = np.concatenate((self.gradphiSums.T.ravel(), np.zeros(1)))
        # v0 = np.zeros(nQuads * NX)
        # maxit = nQuads * NX
        # # tol = np.finfo(float).eps
        # tol = 1e-10
        # start_time = default_timer()
        # # self.xi = sp_la.lsmr(self.G, rhs, x0=v0, atol=tol, btol=tol, maxiter=maxit)
        # self.xi = sp_la.lsqr(self.G, rhs, x0=v0, atol=tol, btol=tol, iter_lim=maxit)
        # print(f'xi solve time = {default_timer()-start_time} s')
        
        self.rNew = np.zeros((nNodes, self.ndim, 3))
        
        index = 0
        for iQ, (inds, phis, gradphis, quadWeight, quad) in enumerate(self.store):
            quadWeight += self.xi[0][iQ]
            if f is not None:
                fq = f(quad)
            for alpha, i in enumerate(inds):
                disp = quad - self.nodes[i]
                self.rNew[i,:,0] -= gradphis[alpha] * quadWeight
                self.rNew[i,0,1] -= phis[alpha] * quadWeight
                self.rNew[i,1,2] -= phis[alpha] * quadWeight
                self.rNew[i,:,1:3] -= np.outer(gradphis[alpha], disp) * quadWeight
                self.u_weights[i] += quadWeight * np.abs(phis[alpha])
                if i < 0:
                    continue # move to next i if boundary node
                
                if f is not None:
                    self.b[i] += quadWeight * fq * phis[alpha]
                for beta, j in enumerate(inds):
                    if j < 0: # j is boundary node
                        ##### Not sure if this can always be uncommmented? #####
                        ##### Needed for projection; but does it affect Poisson/CD #####
                        # self.b[i] -= quadWeight * (
                        #     phis[alpha] * phis[beta] )
                        self.b[i] -= quadWeight * (
                            (gradphis[alpha] @ self.velocity) * phis[beta] +
                            (gradphis[alpha] @ (self.diffusivity @ gradphis[beta])) )
                    else: # i and j are both interior
                        if not massLumping:
                            Mdata[index] = quadWeight * phis[alpha] * phis[beta]
                        Adata[index] = quadWeight * (gradphis[alpha] @ self.velocity) * phis[beta]
                        Kdata[index] = quadWeight * (gradphis[alpha] @ (self.diffusivity @ gradphis[beta]))
                        row_ind[index] = i
                        col_ind[index] = j
                        index += 1
        
        self.gradphiSumsNew = self.rNew[:nDoFs,:,0]
        
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
        nPointsPerPlane = nx*(NY*ny + 1)
        nPointsTotal = nPointsPerPlane*NX + NY*ny + 1
        self.phiPlot = np.empty((nPointsTotal, 4))
        self.indPlot = np.empty((nPointsTotal, 4), dtype='int')
        self.X = np.empty(0)       
        
        if self.BC.name == 'Dirichlet':
            self.uPlot = np.concatenate((self.u, [1.]))
        else:
            self.uPlot = self.u    
            
        for iPlane in range(NX):
            # dx = self.dx[iPlane]
            points = np.indices((nx, NY*ny + 1), dtype='float') \
                .reshape(self.ndim, -1).T * [self.dx[iPlane]/nx, 1/(NY*ny)]
            points[:,0] += self.nodeX[iPlane]
            self.X = np.append(self.X, points[:,0])
            for iP, point in enumerate(points):
                phis, _, inds = self.BC(point, iPlane)
                self.phiPlot[iPlane*nPointsPerPlane + iP] = phis
                inds[inds < 0] = -1
                self.indPlot[iPlane*nPointsPerPlane + iP] = inds
        # Deal with right boundary
        points = np.hstack((np.full((NY*ny + 1, 1), 2*np.pi), points[0:NY*ny + 1,1:2]))
        for iP, point in enumerate(points):
                phis, _, inds = self.BC(point, iPlane)
                self.phiPlot[NX*nPointsPerPlane + iP] = phis
                inds[inds < 0] = -1
                self.indPlot[NX*nPointsPerPlane + iP] = inds
        
        self.X = np.append(self.X, np.full(NY*ny+1, 2*np.pi))
        self.Y = np.tile(points[0:NY*ny + 1,1], NX*nx + 1)
        self.U = np.sum(self.phiPlot * self.uPlot[self.indPlot], axis=1)
    
    def computePlottingSolution(self):
        """Compute interpolated solution at the plotting points.

        Returns
        -------
        None.

        """
        self.uPlot[0:self.nDoFs] = self.u
        self.U = np.sum(self.phiPlot * self.uPlot[self.indPlot], axis=1)