#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 15:46:03 2021

@author: phrhpf
"""

from abc import ABCMeta, abstractmethod
import numpy as np
import warnings


class Boundary(metaclass=ABCMeta):
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
    
class PeriodicBoundary(Boundary):
    @property
    def name(self): 
        return 'periodic'
    
    def __init__(self, sim):
        self.nXnodes = sim.NX
        self.nYnodes = sim.NY
        super().__init__(sim)
        self.inds = np.empty(4, dtype='int')
        self.phis = np.empty(4)
        self.gradphis = np.empty((4,2))
    
    def __call__(self, p, iPlane):
        originalShape = p.shape
        p.shape = (2,)
        nNodes = self.nNodes
        NY = self.sim.NY
        nodeX = self.sim.nodeX[iPlane]
        nodeXp1 = self.sim.nodeX[iPlane + 1]
        nodeY = self.sim.nodeY[iPlane]
        nodeYp1 = self.sim.nodeY[iPlane + 1]
        idy = self.sim.idy[iPlane]
        idyp1 = self.sim.idy[iPlane + 1]
            
        ##### left #####
        mapL = float(self.sim.mapping(p, nodeX)) % 1
        indL = (np.searchsorted(nodeY, mapL, side='right') - 1) % NY
        self.phis[1] = (mapL - nodeY[indL]) * idy[indL]
        self.phis[0] = 1 - self.phis[1]
        self.gradphis[1,1] = idy[indL]
        self.gradphis[0,1] = -self.gradphis[1,1]
        self.inds[:2] = (indL + iPlane*NY, (indL + 1) % NY + iPlane*NY)        
            
        ##### right #####
        mapR = float(self.sim.mapping(p, nodeXp1)) % 1
        indR = (np.searchsorted(nodeYp1, mapR, side='right') - 1) % NY
        self.phis[3] = (mapR - nodeYp1[indR]) * idyp1[indR]
        self.phis[2] = 1 - self.phis[3]
        self.gradphis[3,1] = idyp1[indR]
        self.gradphis[2,1] = -self.gradphis[3,1]
        self.inds[2:] = ((indR + NY*(iPlane+1)) % nNodes,
                         ((indR+1) % NY + NY*(iPlane+1)) % nNodes)

        gradRho = 1.0 / self.sim.dx[iPlane]
        self.gradphis[:,0] = np.array((-gradRho, -gradRho, gradRho, gradRho))
        rho = (p[0] - nodeX) * gradRho
        rhos = np.array((1-rho, 1-rho, rho, rho))
        self.gradphis[:,0] *= self.phis
        self.gradphis[:,1] *= rhos
        self.gradphis[:,0] -= self.sim.mapping.deriv(p)*self.gradphis[:,1]
        # At this point self.phis = phi_FEM, so we multiply by ramp
        self.phis *= rhos
        p.shape = originalShape
        return self.phis, self.gradphis, self.inds
    
    def computeNodes(self):
        self.DoFs = np.vstack( (np.repeat(self.sim.nodeX[:-1], self.sim.NY),
                                self.sim.nodeY[:-1,:-1].ravel()) ).T
        return self.DoFs
    
    def mapping(self, points, zeta=0.):
        return self.sim.mapping(points, zeta) % 1


class DirichletBoundary(Boundary):
    @property
    def name(self): 
        return 'Dirichlet'
    
    def __init__(self, sim, g, B, NDX=1):
        NX = sim.NX
        nodeX = sim.nodeX
        self.nXnodes = NX - 1
        self.nYnodes = sim.NY - 1
        super().__init__(sim)
        self.g = g
        self.B = B
        self.inds = np.empty(4, dtype='int')
        self.phis = np.empty(4)
        self.gradphis = np.empty((4,2,2))
        # TODO: Allow user to specify self.DirichletNodes for top and bottom
        self.DirichletNodes = np.tile(np.vstack(
            [np.linspace(nodeX[i], nodeX[i+1], NDX+1) for i in range(NX)]),
            (2,1)).reshape(2,-1,NDX+1)

    def computeNodes(self):
        self.DoFs = np.vstack( (np.repeat(self.sim.nodeX[1:-1], self.sim.NY-1),
                                self.sim.nodeY[1:-1,1:-1].ravel()) ).T
        return self.DoFs
    
    def __call__(self, p, iPlane):
        originalShape = p.shape
        p.shape = (2,)
        nodeX = self.sim.nodeX[iPlane]
        nodeXp1 = self.sim.nodeX[iPlane + 1]
        isBoundaryMinus = isBoundaryPlus = False
        i0 = i1 = i2 = i3 = -1
        
        self.gradphis.fill(0.0)
        gradRho = 1.0 / (nodeXp1 - nodeX)
        self.gradphis[:,0,1] = np.array((-gradRho, -gradRho, gradRho, gradRho))
        
        # ignore warnings about nan's where p doesn't map to any boundaries
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'invalid value encountered in')
            zetaBottom, zetaTop = self.B(p)
            zetaBottom = float(zetaBottom)
            zetaTop = float(zetaTop)
            isBoundaryBottom = (zetaBottom > nodeX) * (zetaBottom < nodeXp1)
            isBoundaryTop = (zetaTop > nodeX) * (zetaTop < nodeXp1)
            
            if isBoundaryBottom and (zetaBottom <= p[0]):
                zetaMinus = zetaBottom
                isBoundaryMinus = True
                dBdx, dBdy = self.B.deriv(p, 'bottom')
            if isBoundaryTop and (zetaTop <= p[0]):
                zetaMinus = zetaTop
                isBoundaryMinus = True
                dBdx, dBdy = self.B.deriv(p, 'top')
            if isBoundaryMinus:
                DirichletNodes = self.DirichletNodes[int(isBoundaryTop)][iPlane]
                iR = np.searchsorted(DirichletNodes, zetaMinus, side='right')
                zeta0 = DirichletNodes[iR]
                zeta1 = DirichletNodes[iR-1]
                idx = 1.0 / (zeta0 - zeta1)
                self.gradphis[0,0,0] = dBdx *idx
                self.gradphis[1,0,0] = -self.gradphis[0,0,0]
                self.gradphis[2:,0,1] = \
                    (nodeXp1 - zetaMinus + dBdx*(p[0] - nodeXp1)) / \
                    (zetaMinus - nodeXp1)**2
                self.gradphis[:2,0,1] = -self.gradphis[2:,0,1]
                
                self.gradphis[0,1,0] = dBdy * idx
                self.gradphis[1,1,0] = -self.gradphis[0,1,0]
                self.gradphis[2:,1,1] = dBdy*(p[0] - nodeXp1) \
                                      / (zetaMinus - nodeXp1)**2
                self.gradphis[:2,1,1] = -self.gradphis[2:,1,1]
                self.phis[0] = (zetaMinus - zeta1) * idx
                self.phis[1] = 1 - self.phis[0]
                g0 = self.g(np.array((zeta0, float(isBoundaryTop))))
                self.phis[0] *= g0
                self.gradphis[0,:,0] *= g0
                g1 = self.g(np.array((zeta1, float(isBoundaryTop))))
                self.phis[1] *= g1
                self.gradphis[1,:,0] *= g1
            
            if isBoundaryBottom and (zetaBottom > p[0]):
                zetaPlus = zetaBottom
                isBoundaryPlus = True
                dBdx, dBdy = self.B.deriv(p, 'bottom')
            if isBoundaryTop and (zetaTop > p[0]):
                zetaPlus = zetaTop
                isBoundaryPlus = True
                dBdx, dBdy = self.B.deriv(p, 'top')
            if isBoundaryPlus:
                DirichletNodes = self.DirichletNodes[int(isBoundaryTop)][iPlane]
                iR = np.searchsorted(DirichletNodes, zetaPlus, side='right')
                zeta2 = DirichletNodes[iR]
                zeta3 = DirichletNodes[iR-1]
                idx = 1.0 / (zeta2 - zeta3)
                self.gradphis[2,0,0] = dBdx * idx
                self.gradphis[3,0,0] = -self.gradphis[2,0,0]
                self.gradphis[:2,0,1] = \
                    (nodeX - zetaPlus + dBdx*(p[0] - nodeX)) / \
                    (zetaPlus - nodeX)**2
                self.gradphis[2:,0,1] = -self.gradphis[:2,0,1]
                
                self.gradphis[2,1,0] = dBdy * idx
                self.gradphis[3,1,0] = -self.gradphis[2,1,0]
                self.gradphis[:2,1,1] = dBdy*(p[0] - nodeX) \
                                      / (zetaPlus - nodeX)**2
                self.gradphis[2:,1,1] = -self.gradphis[:2,1,1]
                
                self.phis[2] = (zetaPlus - zeta3) * idx
                self.phis[3] = 1 - self.phis[2]
                g2 = self.g(np.array((zeta2, float(isBoundaryTop))))
                self.phis[2] *= g2
                self.gradphis[2,:,0] *= g2
                g3 = self.g(np.array((zeta3, float(isBoundaryTop))))
                self.phis[3] *= g3
                self.gradphis[3,:,0] *= g3
            
        if not isBoundaryMinus:
            zetaMinus = nodeX
            mapL = float(self.sim.mapping(p, nodeX))
            i1 = np.searchsorted(self.sim.nodeY[iPlane], mapL, side='right') - 1
            if i1 > self.nYnodes: # for points right at upper boundary nodes
                i1 -= 1
            i0 = i1 - 1
            self.phis[1] = (mapL - self.sim.nodeY[iPlane][i1]) * self.sim.idy[iPlane][i1]
            self.phis[0] = 1 - self.phis[1]
            self.gradphis[1,1,0] = self.sim.idy[iPlane][i1]
            self.gradphis[0,1,0] = -self.gradphis[1,1,0]
            self.gradphis[:2,0,0] = -self.sim.mapping.deriv(p)*self.gradphis[:2,1,0]
            ##### if inds[0] on the left or lower boundary #####
            if (iPlane == 0) or (i0 < 0):
                g0 = self.g(np.array((nodeX, self.sim.nodeY[iPlane][i1])))
                self.phis[0] *= g0
                self.gradphis[0,:,0] *= g0
                i0 = -1 # necessary for left boundary
            else: # inds[0] is interior
                i0 += (iPlane - 1)*self.nYnodes
            ##### if inds[1] on the left or upper boundary #####
            if (iPlane == 0) or (i1 >= self.nYnodes):
                g1 = self.g(np.array((nodeX, self.sim.nodeY[iPlane][i1+1])))
                self.phis[1] *= g1
                self.gradphis[1,:,0] *= g1
                i1 = -1 # necessary for right boundary
            else: # inds[1] is interior
                i1 += (iPlane - 1)*self.nYnodes
        if not isBoundaryPlus:
            zetaPlus = nodeXp1
            mapR = float(self.sim.mapping(p, nodeXp1))
            i3 = np.searchsorted(self.sim.nodeY[iPlane+1], mapR, side='right') - 1
            if i3 > self.nYnodes: # for points right at upper boundary nodes
                i3 -= 1
            i2 = i3 - 1
            self.phis[3] = (mapR - self.sim.nodeY[iPlane+1][i3]) * self.sim.idy[iPlane+1][i3]
            self.phis[2] = 1 - self.phis[3]
            self.gradphis[3,1,0] = self.sim.idy[iPlane+1][i3]
            self.gradphis[2,1,0] = -self.gradphis[3,1,0]
            self.gradphis[2:,0,0] = -self.sim.mapping.deriv(p)*self.gradphis[2:,1,0]
            ##### if inds[2] on the right or lower boundary #####
            if (iPlane == self.nXnodes) or (i2 < 0):
                g2 = self.g(np.array((nodeXp1, self.sim.nodeY[iPlane+1][i3])))
                self.phis[2] *= g2
                self.gradphis[2,:,0] *= g2
                i2 = -1
            else:
                i2 += iPlane*self.nYnodes
            ##### if inds[3] on the right or upper boundary #####
            if (iPlane == self.nXnodes) or (i3 >= self.nYnodes):
                g3 = self.g(np.array((nodeXp1, self.sim.nodeY[iPlane+1][i3+1])))
                self.phis[3] *= g3
                self.gradphis[3,:,0] *= g3
                i3 = -1
            else:
                i3 += iPlane*self.nYnodes
        rho = (p[0] - zetaMinus) / (zetaPlus - zetaMinus)
        self.inds[:] = (i0, i1, i2, i3)
        
        self.gradphis[:,:,1] *= self.phis.reshape(4,1)
        self.gradphis[:,:,0] *= np.array((1-rho, 1-rho, rho, rho)).reshape(4,1)
        
        # At this point self.phis = phi_FEM, so we then multiply by ramp
        self.phis[0:2] *= (1 - rho)
        self.phis[2:4] *= rho
        p.shape = originalShape
        return self.phis, self.gradphis.sum(axis=-1), self.inds