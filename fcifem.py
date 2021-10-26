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
# import scipy
import ssqr
import warnings

from abc import ABCMeta, abstractmethod

import integrators

class Mapping(metaclass=ABCMeta):
    @property
    @abstractmethod
    def name(self): 
        raise NotImplementedError

    @abstractmethod
    def __call__(self, points, zeta=0.):
        """Compute mapped y-coordinates from all points to given x-coordinate

        Parameters
        ----------
        points : numpy.ndarray, shape=(n,ndim)
            (x,y) coordinates of starting points.
        zeta : float
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
        return np.arctan(self.deriv(points))
    
    def __repr__(self):
        return f"{self.__class__.__name__}()"

class StraightMapping(Mapping):
    @property
    def name(self): 
        return 'straight'

    def __call__(self, points, zeta=0.):
        points.shape = (-1, 2)
        return points[:,1] % 1
    
    def deriv(self, points):
        points.shape = (-1, 2)
        return np.repeat(0., len(points))
    
    def theta(self, points):
        points.shape = (-1, 2)
        return np.repeat(0., len(points))

class LinearMapping(Mapping):
    @property
    def name(self): 
        return 'linear'
    
    def __init__(self, slope):
        self.slope = slope

    def __call__(self, points, zeta=0.):
        points.shape = (-1, 2)
        return (points[:,1] + self.slope*(zeta - points[:,0]))
    
    def deriv(self, points):
        points.shape = (-1, 2)
        return np.repeat(self.slope, len(points))
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.slope})"
    
class QuadraticMapping(Mapping):
    @property
    def name(self): 
        return 'quadratic'
    
    def __init__(self, a, b=0.):
        self.a = a
        self.b = b

    def __call__(self, points, zeta=0.):
        points.shape = (-1, 2)
        x = points[:,0]
        return (points[:,1] + self.a*(zeta**2 - x**2) + self.b*(zeta - x))
    
    def deriv(self, points):
        points.shape = (-1, 2)
        return 2*self.a*points[:,0] + self.b
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.a}, {self.b})"

class SinusoidalMapping(Mapping):
    @property
    def name(self): 
        return 'sinusoidal'
    
    def __init__(self, amplitude, phase):
        self.A = amplitude
        self.phase = phase

    def __call__(self, points, zeta=0.):
        points.shape = (-1, 2)
        offsets = points[:,1] - self.A*np.sin(points[:,0] - self.phase)
        return (self.A*np.sin(zeta - self.phase) + offsets)
    
    def deriv(self, points):
        points.shape = (-1, 2)
        return self.A*np.cos(points[:,0] - self.phase)
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.A}, {self.phase})"


class BoundaryCondition(metaclass=ABCMeta):
    @property
    @abstractmethod
    def name(self): 
        raise NotImplementedError
    
    def __init__(self, sim):
        self.sim = sim
        self.nNodes = self.nXnodes * self.nYnodes
        
    @abstractmethod
    def __call__(self, points, iPlane):
        raise NotImplementedError
    
    @abstractmethod
    def computeNodes(self):
        raise NotImplementedError
        
    def mapping(self, points, zeta=0.):
        return self.sim.mapping(points, zeta)
    
class PeriodicBoundaryCondition(BoundaryCondition):
    @property
    def name(self): 
        return 'periodic'
    
    def __init__(self, sim):
        self.nXnodes = sim.NX
        self.nYnodes = sim.NY
        super().__init__(sim)
    
    def __call__(self, points, iPlane):
        isBoundary = np.full(len(points), False)
        mapL = self.sim.mapping(points, self.sim.nodeX[iPlane]) % 1
        mapR = self.sim.mapping(points, self.sim.nodeX[iPlane+1]) % 1
        return (isBoundary, mapL, isBoundary, mapR)
    
    def computeNodes(self):
        self.DoFs = np.vstack( (np.repeat(self.sim.nodeX[:-1], self.sim.NY),
                                self.sim.nodeY[:-1,:-1].ravel()) ).T
        return self.DoFs
    
    def mapping(self, points, zeta=0.):
        return self.sim.mapping(points, zeta) % 1
    
class DirichletBoundaryCondition(BoundaryCondition):
    @property
    def name(self): 
        return 'Dirichlet'
    
    def __init__(self, sim, g, B):
        self.nXnodes = sim.NX - 1
        self.nYnodes = sim.NY - 1
        super().__init__(sim)
        self.B = B
        self.g = g
        self.inds = np.empty(4, dtype='int')
        self.phis = np.empty(4)
        self.gradphis = np.empty((4,2))
    
    def __call__(self, points, iPlane):
        nodeX = self.sim.nodeX[iPlane]
        nodeXp1 = self.sim.nodeX[iPlane+1]
        # ignore warnings about nan's where points don't map to any boundaries
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'invalid value encountered in')
            maps = self.B(points)
            isBoundaryL = (maps[0] > nodeX) * (maps[0] < nodeXp1)
            isBoundaryR = (maps[1] > nodeX) * (maps[1] < nodeXp1)
        maps[0][~isBoundaryL] = self.sim.mapping(points[~isBoundaryL], nodeX)
        maps[1][~isBoundaryR] = self.sim.mapping(points[~isBoundaryR], nodeXp1)
        return (isBoundaryL, maps[0], isBoundaryR, maps[1])

    def computeNodes(self):
        self.DoFs = np.vstack( (np.repeat(self.sim.nodeX[1:-1], self.sim.NY-1),
                                self.sim.nodeY[1:-1,1:-1].ravel()) ).T
        self.DirichletNodes = np.vstack((self.sim.nodeX, self.sim.nodeX))
        return self.DoFs
    
    def test(self, p, iPlane):
        nodeX = self.sim.nodeX[iPlane]
        nodeXp1 = self.sim.nodeX[iPlane + 1]
        isBoundaryMinus = isBoundaryPlus = False
        # ignore warnings about nan's where p doesn't map to any boundaries
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'invalid value encountered in')
            zetaBottom, zetaTop = self.B(p)
            p.shape = (2,)
            isBoundaryBottom = (zetaBottom > nodeX) * (zetaBottom < nodeXp1)
            isBoundaryTop = (zetaTop > nodeX) * (zetaTop < nodeXp1)
            if isBoundaryBottom and (zetaBottom < p[0]):
                zetaMinus = zetaBottom
                isBoundaryMinus = True
            if isBoundaryTop and (zetaTop < p[0]):
                zetaMinus = zetaTop
                isBoundaryMinus = True
            if isBoundaryBottom and (zetaBottom > p[0]):
                zetaPlus = zetaBottom
                isBoundaryPlus = True
            if isBoundaryTop and (zetaTop > p[0]):
                zetaPlus = zetaTop
                isBoundaryPlus = True
        self.inds.fill(-1)
        if not isBoundaryMinus:
            zetaMinus = nodeX
            mapL = self.sim.mapping(p, nodeX)
            self.inds[1] = np.searchsorted(self.sim.nodeY[iPlane], mapL, side='right')
            ##### if inds[0] is not on lower boundary #####
            if self.inds[1] > 1:
                self.inds[0] = self.inds[1] - 1 + iPlane*self.nYnodes
            ##### otherwise inds[0] is on lower boundary #####
            else:
                pass
            ##### if inds[1] is not on upper boundary #####
            if self.inds[1] < self.sim.NY:
                self.inds[1] += iPlane*self.nYnodes
            ##### otherwise inds[1] is on upper boundary #####
            else:
                self.inds[1] = -1
                # self.vals[1] = self.g()
        if not isBoundaryPlus:
            zetaPlus = nodeXp1
            mapR = self.sim.mapping(p, nodeXp1)
            self.inds[3] = np.searchsorted(self.sim.nodeY[iPlane + 1], mapR, side='right')
            ##### if inds[2] is not on lower boundary #####
            if self.inds[3] > 1:
                self.inds[2] = self.inds[3] - 1 + iPlane*self.nYnodes
            ##### otherwise inds[0] is on lower boundary #####
            else:
                pass
            ##### if inds[3] is not on upper boundary #####
            if self.inds[3] < self.sim.NY:
                self.inds[3] += iPlane*self.nYnodes
             ##### otherwise inds[3] is on upper boundary #####
            else:
                self.inds[3] = -1
        rho = (p[0] - nodeX) / (zetaPlus - zetaMinus)
        self.phis *= rho
        return self.phis, self.gradphis, self.inds


class FciFemSim(metaclass=ABCMeta):
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
        BC : {BoundaryCondition, string}, optionalquads, iPlane)
            Either an object of type BoundaryCondition, or string 'periodic'.
            The default is 'periodic'.

        Returns
        -------
        None.

        """
        if isinstance(BC, BoundaryCondition):
             self.BC = BC
        elif BC.lower() in ('periodic', 'p'):
            self.BC = PeriodicBoundaryCondition(self)
        else:
            raise SystemExit(f"Unkown boundary condition: {BC}")
        self.nNodes = self.BC.nNodes
        if isinstance(u0, np.ndarray) and u0.shape == (self.nNodes,):
            self.u0 = u0
            self.u = u0.copy()
            self.u0func = None
        elif callable(u0):
            self.u0func = u0
            self.nodes = self.BC.computeNodes()
            self.nodesMapped = self.nodes.copy()
            self.nodesMapped[:,1] = self.BC.mapping(self.nodes, 0)
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
            nodeXp1 = self.nodeX[iPlane + 1]
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
            
            quads += [self.nodeX[iPlane], 0]
            
            if self.BC.name == 'periodic':
                (isBoundaryL, mapL, isBoundaryR, mapR) = self.BC(quads, iPlane)
                
                zetaMinus = np.repeat(nodeX  , nQuads)
                zetaPlus  = np.repeat(nodeXp1, nQuads)
                zetaMinus[isBoundaryL] = mapL[isBoundaryL]
                zetaPlus [isBoundaryR] = mapR[isBoundaryR]
                rho = (quads[:,0] - zetaMinus) / (zetaPlus - zetaMinus)
                
                indL = (np.searchsorted(self.nodeY[iPlane], mapL, side='right') - 1)
                indR = (np.searchsorted(self.nodeY[iPlane + 1], mapR, side='right') - 1)
                phiL = np.empty(nQuads)
                phiR = np.empty(nQuads)
                phiL[~isBoundaryL] = (mapL[~isBoundaryL] - self.nodeY[iPlane][indL[~isBoundaryL]]) * self.idy[iPlane][indL[~isBoundaryL]]
                phiR[~isBoundaryR] = (mapR[~isBoundaryR] - self.nodeY[iPlane + 1][indR[~isBoundaryR]]) * self.idy[iPlane + 1][indR[~isBoundaryR]]
                
                phiL[isBoundaryL] = (mapL[isBoundaryL] - nodeX) / dx
                phiR[isBoundaryR] = (mapR[isBoundaryR] - nodeX) / dx
                
                gradRho = 1 / (zetaPlus - zetaMinus)
            
            for iQ, quad in enumerate(quads):
                if self.BC.name == 'Dirichlet':
                    phis, gradphis, inds = self.BC.test(quad, iPlane)
                    for i in inds:
                        if i == -1:
                            continue # move to next i if boundary node
                        for j in inds:
                            if not massLumping:
                                Mdata[index] = quadWeights[iQ]*phis[i]*phis[j]
                            row_ind[index] = i
                            col_ind[index] = j
                            index += 1
                                                               
                
                if self.BC.name == 'periodic':
                    phis = np.array((((1-phiL[iQ]), (1-rho[iQ])),
                                     (   phiL[iQ] , (1-rho[iQ])),
                                     ((1-phiR[iQ]),    rho[iQ] ),
                                     (   phiR[iQ] ,    rho[iQ] )))
                    
                    gradphis = phis * np.array((
                        (-gradRho[iQ], -self.idy[iPlane][indL[iQ]]),
                        (-gradRho[iQ],  self.idy[iPlane][indL[iQ]]),
                        ( gradRho[iQ], -self.idy[iPlane + 1][indR[iQ]]),
                        ( gradRho[iQ],  self.idy[iPlane + 1][indR[iQ]])))
                    
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
    
    # def computeSpatialDiscretization(self, f=None, NQX=1, NQY=None, Qord=2,
    #                                   quadType='gauss', massLumping=False):
    #     """Assemble the system discretization matrices K, A, M in CSR format.
        
    #     K is the stiffness matrix from the diffusion term
    #     A is the advection matrix
    #     M is the mass matrix from the time derivative
        
    #     Parameters
    #     ----------
    #     f : {callable, None}, optional
    #         Forcing function. Must take 2D array of points and return 1D array.
    #         The default is None.
    #     NQX : int, optional
    #         Number of quadrature cell divisions between FCI planes.
    #         The default is 1.
    #     NQY : {int, None}, optional
    #         Number of quadrature cell divisions in y-direction.
    #         The default is None, which sets NQY = NY.
    #     Qord : int, optional
    #         Number of quadrature points in each grid cell along one dimension.
    #         The default is 2.
    #     quadType : string, optional
    #         Type of quadrature to be used. Must be either 'gauss' or 'uniform'.
    #         Produces either Gauss-Legendre or Newton-Cotes type points/weights.
    #         The default is 'gauss'.
    #     massLumping : bool, optional
    #         Determines whether mass-lumping is used to calculate M matrix.
    #         The default is False.

    #     Returns
    #     -------
    #     None.

    #     """
    #     ndim = self.ndim
    #     nNodes = self.nNodes
    #     NX = self.NX
    #     NY = self.NY
    #     if NQY is None:
    #         NQY = NY
    #     self.f = f
    #     self.NQX = NQX
    #     self.NQY = NQY
    #     self.Qord = Qord
    #     self.quadType = quadType
    #     self.massLumping = massLumping
    #     # pre-allocate arrays for stiffness matrix triplets
    #     nEntries = (2*ndim)**2
    #     nQuads = NQX * NQY * Qord**2
    #     nMaxEntries = nEntries * nQuads * NX
    #     Kdata = np.zeros(nMaxEntries)
    #     Adata = np.zeros(nMaxEntries)
    #     if not massLumping:
    #         Mdata = np.zeros(nMaxEntries)
    #     row_ind = np.zeros(nMaxEntries, dtype='int')
    #     col_ind = np.zeros(nMaxEntries, dtype='int')
    #     self.b = np.zeros(nNodes)
    #     self.u_weights = np.zeros(nNodes)
        
    #     ##### compute spatial discretizaton
    #     index = 0
    #     for iPlane in range(NX):
    #         dx = self.dx[iPlane]
    #         nodeX = self.nodeX[iPlane]
    #         nodeXp1 = self.nodeX[iPlane + 1]
    #         ##### generate quadrature points
    #         if quadType.lower() in ('gauss', 'g', 'gaussian'):
    #             offsets, weights = roots_legendre(Qord)
    #         elif quadType.lower() in ('uniform', 'u'):
    #             offsets = np.linspace(-1+1/Qord, 1-1/Qord, Qord)
    #             weights = np.repeat(1., Qord)
    #         offsets = (offsets * dx * 0.5 / NQX, offsets * 0.5 / NQY)
    #         weights = (weights * dx * 0.5 / NQX, weights * 0.5 / NQY)
    #         quads = ( np.indices([NQX, NQY], dtype='float').T.
    #                   reshape(-1, ndim) + 0.5 ) * [dx/NQX, 1/NQY]
    #         quadWeights = np.repeat(1., len(quads))
    #         for i in range(ndim):
    #             quads = np.concatenate( 
    #                 [quads + offset*np.eye(ndim)[i] for offset in offsets[i]] )
    #             quadWeights = np.concatenate(
    #                 [quadWeights * weight for weight in weights[i]] )
            
    #         quads += [self.nodeX[iPlane], 0]
    #         (isBoundaryL, mapL, isBoundaryR, mapR) = self.BC(quads, iPlane)
            
    #         zetaMinus = np.repeat(nodeX  , nQuads)
    #         zetaPlus  = np.repeat(nodeXp1, nQuads)
    #         zetaMinus[isBoundaryL] = mapL[isBoundaryL]
    #         zetaPlus [isBoundaryR] = mapR[isBoundaryR]
    #         rho = (quads[:,0] - zetaMinus) / (zetaPlus - zetaMinus)
            
    #         indL = (np.searchsorted(self.nodeY[iPlane], mapL, side='right') - 1)
    #         indR = (np.searchsorted(self.nodeY[iPlane + 1], mapR, side='right') - 1)
    #         phiL = np.empty(nQuads)
    #         phiR = np.empty(nQuads)
    #         phiL[~isBoundaryL] = (mapL[~isBoundaryL] - self.nodeY[iPlane][indL[~isBoundaryL]]) * self.idy[iPlane][indL[~isBoundaryL]]
    #         phiR[~isBoundaryR] = (mapR[~isBoundaryR] - self.nodeY[iPlane + 1][indR[~isBoundaryR]]) * self.idy[iPlane + 1][indR[~isBoundaryR]]
            
    #         phiL[isBoundaryL] = (mapL[isBoundaryL] - nodeX) / dx
    #         phiR[isBoundaryR] = (mapR[isBoundaryR] - nodeX) / dx
            
    #         gradRho = 1 / (zetaPlus - zetaMinus)
            
    #         for iQ, quad in enumerate(quads):
    #             phis = np.array((((1-phiL[iQ]), (1-rho[iQ])),
    #                              (   phiL[iQ] , (1-rho[iQ])),
    #                              ((1-phiR[iQ]),    rho[iQ] ),
    #                              (   phiR[iQ] ,    rho[iQ] )))
                
    #             gradphis = phis * np.array((
    #                 (-gradRho[iQ], -self.idy[iPlane][indL[iQ]]),
    #                 (-gradRho[iQ],  self.idy[iPlane][indL[iQ]]),
    #                 ( gradRho[iQ], -self.idy[iPlane + 1][indR[iQ]]),
    #                 ( gradRho[iQ],  self.idy[iPlane + 1][indR[iQ]])))
                
    #             # Gradients along the coordinate direction
    #             gradphis[:,0] -= self.mapping.deriv(quad)*gradphis[:,1]
            
    #             phis = np.prod(phis, axis=1)
    #             indices = np.array([indL[iQ] + NY*iPlane,
    #                                 (indL[iQ]+1) % NY + NY*iPlane,
    #                                 (indR[iQ] + NY*(iPlane+1)) % nNodes,
    #                                 ((indR[iQ]+1) % NY + NY*(iPlane+1)) % nNodes])
    #             Kdata[index:index+nEntries] = ( quadWeights[iQ] * 
    #                 np.ravel( gradphis @ (self.diffusivity @ gradphis.T) ) )
    #             Adata[index:index+nEntries] = ( quadWeights[iQ] *
    #                 np.outer(np.dot(gradphis, self.velocity), phis).ravel() )
    #             if not massLumping:
    #                 Mdata[index:index+nEntries] = ( quadWeights[iQ] * 
    #                     np.outer(phis, phis).ravel() )
    #             row_ind[index:index+nEntries] = np.repeat(indices, 2*ndim)
    #             col_ind[index:index+nEntries] = np.tile(indices, 2*ndim)
                
    #             self.u_weights[indices] += quadWeights[iQ] * phis
                
    #             index += nEntries
    #             if f is not None:
    #                 self.b[indices] += f(quad) * phis * quadWeights[iQ]
        
    #     self.K = sp.csr_matrix( (Kdata, (row_ind, col_ind)),
    #                             shape=(nNodes, nNodes) )
    #     self.A = sp.csr_matrix( (Adata, (row_ind, col_ind)),
    #                             shape=(nNodes, nNodes) )
    #     if massLumping:
    #         self.M = sp.diags(self.u_weights, format='csr')
    #     else:
    #         self.M = sp.csr_matrix( (Mdata, (row_ind, col_ind)),
    #                             shape=(nNodes, nNodes) )
    
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
            rho = points[:,0] / dx
            points += [nodeX, 0]
            mapL = self.BC.mapping(points, nodeX)
            mapR = self.BC.mapping(points, self.nodeX[iPlane+1])
            indL = (np.searchsorted(self.nodeY[iPlane], mapL, side='right') - 1) % NY
            indR = (np.searchsorted(self.nodeY[iPlane + 1], mapR, side='right') - 1) % NY
            phiL = (mapL - self.nodeY[iPlane][indL]) * self.idy[iPlane][indL]
            phiR = (mapR - self.nodeY[iPlane + 1][indR]) * self.idy[iPlane + 1][indR]
            for iP, point in enumerate(points):
                self.phiPlot[iPlane*nPointsPerPlane + iP] = (
                    (1-phiL[iP]) * (1-rho[iP]), phiL[iP] * (1-rho[iP]),
                    (1-phiR[iP]) * rho[iP]    , phiR[iP] * rho[iP] )
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