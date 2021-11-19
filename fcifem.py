# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 13:47:07 2020

@author: samal
"""

from scipy.special import roots_legendre
import scipy.sparse as sp
import numpy as np

import integrators
import mappings
import boundaries


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
            Must be an array of shape (self.nNodes,) or a callable object
            returning such an array and taking as input the array of node
            coordinates with shape (self.nNodes, self.ndim).
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
        self.nNodes = self.BC.nNodes
        self.nodes = self.BC.computeNodes()
        self.nodesMapped = self.nodes.copy()
        self.nodesMapped[:,1] = self.BC.mapping(self.nodes, 0)
        if isinstance(u0, np.ndarray) and u0.shape == (self.nNodes,):
            self.u0 = u0
            self.u = u0.copy()
            self.u0func = None
        elif callable(u0):
            self.u0func = u0
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
        self.areas = np.zeros(nNodes + 1)
        self.xis = np.zeros((self.nNodes + 1, self.ndim))
        
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
                self.areas[inds] += quadWeight
                self.xis[inds] -= gradphis * quadWeight   
        
        # self.gradphiSumsOld = -self.xis[0:-1]
        # self.gradphiSumsNew = np.zeros((nNodes, 2))
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
                self.indPlot[iPlane*nPointsPerPlane + iP] = inds
        # Deal with right boundary
        points = np.hstack((np.full((NY*ny + 1, 1), 2*np.pi), points[0:NY*ny + 1,1:2]))
        for iP, point in enumerate(points):
                phis, _, inds = self.BC(point, iPlane)
                self.phiPlot[NX*nPointsPerPlane + iP] = phis
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
        self.uPlot[0:self.nNodes] = self.u
        self.U = np.sum(self.phiPlot * self.uPlot[self.indPlot], axis=1)