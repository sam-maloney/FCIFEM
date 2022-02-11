#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 09:27:41 2021

@author: Samuel A. Maloney
"""

import numpy as np

import fcifem


class TestProblem:
    xmax = 2*np.pi
    # a = 0.01
    b = 0.05
    # define a such that (0, 0) maps to (xmax, 1) for given b and xmax
    a = (1 - b*xmax)/xmax**2
    
    dfdyMax = 1
    dfdxMax = 1
    
    def __call__(self, p):
        return np.ones(p.size // 2)
    
    def solution(self, p):
        return np.ones(p.size // 2)
        # x = p.reshape(-1,2)[:,0]
        # y = p.reshape(-1,2)[:,1]
        # return ((abs(x - np.pi) < 1e-10) & (abs(y) < 1e-10)).astype('float')

class QuadraticTestProblem:
    xmax = 2*np.pi
    n = 20
    # a = 0.01
    b = 0.05
    # define a such that (0, 0) maps to (xmax, 1) for given b and xmax
    a = (1 - b*xmax)/xmax**2
    
    dfdyMax = n*xmax
    dfdxMax = 1 + 2*a*n*xmax**2 + b*n*xmax
    
    def __call__(self, p):
        x = p.reshape(-1,2)[:,0]
        y = p.reshape(-1,2)[:,1]
        n = self.n
        a = self.a
        b = self.b
        return 2*n*(3*a*x - b)*np.cos(n*(y - a*x**2 - b*x)) + \
            n**2*x*(4*a**2*x**2 + 4*a*b*x + b**2 + 1) * \
            np.sin(n*(y - a*x**2 - b*x))
        # return (6*a*n*x - 2*b*n)*np.cos(n*(y - a*x**2 - b*x)) + \
        #     (4*a**2*n**2*x**3 + 4*a*b*n**2*x**2 + b**2*n**2*x + n**2*x) * \
        #     np.sin(n*(y - a*x**2 - b*x))
    
    def solution(self, p):
        x = p.reshape(-1,2)[:,0]
        y = p.reshape(-1,2)[:,1]
        return x*np.sin(self.n*(y - self.a*x**2 - self.b*x))
        
f = TestProblem()
# f = QuadraticTestProblem()

class QaudraticBoundaryFunction:
    
    def __init__(self, a, b=0.):
        self.a = a
        self.b = b
        if b == 0.:
            self.inva = 1/a
            self.deriv = self.deriv0
            # # This doesn't work
            # self.__call__ = self.call0
    
    def __call__(self, p):
        x = p.reshape(-1,2)[:,0]
        y = p.reshape(-1,2)[:,1]
        a = self.a
        b = self.b
        zetaBottom = (-b + np.sqrt(b**2 - 4*a*(y - a*x**2 - b*x)))/(2*a)
        zetaTop = (-b + np.sqrt(b**2 - 4*a*(y - a*x**2 - b*x - 1)))/(2*a)
        return zetaBottom, zetaTop
    
    def deriv(self, p, boundary):
        x = p.reshape(-1,2)[:,0]
        y = p.reshape(-1,2)[:,1]
        a = self.a
        b = self.b
        if boundary == 'bottom':
            dBdy = -1 / np.sqrt(b**2 - 4*a*(y - a*x**2 - b*x))
            dBdx = -(2*a*x + b) * dBdy
        elif boundary == 'top':
            dBdy = -1 / np.sqrt(b**2 - 4*a*(y - a*x**2 - b*x - 1))
            dBdx = -(2*a*x + b) * dBdy
        return dBdx, dBdy
    
    def call0(self, p):
        x = p.reshape(-1,2)[:,0]
        y = p.reshape(-1,2)[:,1]
        zetaBottom = np.sqrt(x**2 - self.inva*y)
        zetaTop = np.sqrt(x**2 + self.inva*(1 - y))
        return zetaBottom, zetaTop
    
    def deriv0(self, p, boundary):
        x = p.reshape(-1,2)[:,0]
        y = p.reshape(-1,2)[:,1]
        if boundary == 'bottom':
            dBdx = x / np.sqrt(x**2 - self.inva*y)
            dBdy = -0.5*self.inva / np.sqrt(x**2 - self.inva*y)
        elif boundary == 'top':
            dBdx = x / np.sqrt(x**2 + self.inva*(1 - y))
            dBdy = -0.5*self.inva / np.sqrt(x**2 + self.inva*(1 - y))
        return dBdx, dBdy

B = QaudraticBoundaryFunction(f.a, f.b)
mapping = fcifem.mappings.QuadraticMapping(f.a, f.b)

kwargs={
    'mapping' : mapping,
    'dt' : 1.,
    'velocity' : np.array([0., 0.]), # Makes the advection matrix zero
    'diffusivity' : 1., # Makes diffusivity matrix K into Poisson operator
    'px' : 0.,
    'py' : 0.,
    'seed' : 42 }

NX = 8
# NY = 20
NY = NX
# NY = max(int(f.dfdyMax / (2*np.pi)) * NX, NX)

# allocate arrays and compute grid
sim = fcifem.FciFemSim(NX, NY, **kwargs)

# BC = fcifem.boundaries.PeriodicBoundary(sim)
BC = fcifem.boundaries.DirichletXPeriodicYBoundary(sim, f.solution)
# BC = fcifem.boundaries.DirichletBoundary(sim, f.solution, B, NDX=None)
sim.setInitialConditions(np.zeros(BC.nDoFs), mapped=False, BC=BC)


nPoints = 1000
seed = 42
rtol = 1e-4
atol = 1e-5
phitol = 1e-10

dfRatio = 2*np.pi*(NY//NX)
# dx = 2*np.pi/(NX*100)
dy = 1/(NY*5000)
dx = dy*dfRatio


prng = np.random.default_rng(seed=seed)
points = prng.random((nPoints, 2)) * (2*np.pi, 1.)
phis = np.empty((nPoints, 4))
gradphis = np.empty((nPoints, 4, 2))
gradphisNum = np.empty((nPoints, 4, 2))
inds = np.empty((nPoints, 4), dtype='int')

for i, point in enumerate(points):
    phis[i], gradphis[i], inds[i] \
        = BC(point, np.searchsorted(sim.nodeX[1:], point[0]))
    if np.abs(phis[i].sum() - 1) > phitol:
        print(f'No partition of unity for point {i} = {point}')
    try:
        phiR, _, tmp_inds = BC(point + (dx, 0.))
        np.testing.assert_array_equal(inds[i], tmp_inds)
        phiL, _, tmp_inds = BC(point - (dx, 0.))
        np.testing.assert_array_equal(inds[i], tmp_inds)
        phiU, _, tmp_inds = BC(point + (0., dy))
        np.testing.assert_array_equal(inds[i], tmp_inds)
        phiD, _, tmp_inds = BC(point - (0., dy))
        np.testing.assert_array_equal(inds[i], tmp_inds)
    except(AssertionError):
        gradphisNum[i] = np.nan
        print(f'index mismatch for point {i} = {point}')
        continue
    gradphisNum[i,:,0] = (phiR - phiL) / (2*dx)
    gradphisNum[i,:,1] = (phiU - phiD) / (2*dy)
    try:
        np.testing.assert_allclose(gradphis[i], gradphisNum[i],
                                   rtol=rtol, atol = atol)
    except(AssertionError):
        print(f'Gradient mismatch for point {i} = {point}, '
              f'max error = {np.max(abs(gradphis[i]-gradphisNum[i]))}')
        