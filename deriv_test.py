#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 09:27:41 2021

@author: Samuel A. Maloney
"""

import numpy as np

import fcifem
import boundaryFunctions

class UnityFunction:
    xmax = 1.
    
    def __call__(self, p):
        return np.ones(p.size // 2)
    
    def solution(self, p):
        return np.ones(p.size // 2)
        
f = UnityFunction()

a = 0.95
b = 0.05

B = boundaryFunctions.QaudraticBoundaryFunction(a, b)
mapping = fcifem.mappings.QuadraticMapping(a, b)

kwargs={
    'mapping' : mapping,
    'dt' : 1.,
    'velocity' : np.array([0., 0.]), # Makes the advection matrix zero
    'diffusivity' : 1., # Makes diffusivity matrix K into Poisson operator
    'px' : 0.,
    'py' : 0.,
    'seed' : 42,
    'xmax' : f.xmax }

NX = 2
NY = NX

# allocate arrays and compute grid
sim = fcifem.FciFemSim(NX, NY, **kwargs)

# BC = fcifem.boundaries.PeriodicBoundary(sim)
# BC = fcifem.boundaries.DirichletXPeriodicYBoundary(sim, f.solution)
BC = fcifem.boundaries.DirichletBoundary(sim, f.solution, B, NDX=None)
sim.setInitialConditions(np.zeros(BC.nDoFs), mapped=False, BC=BC)


nPoints = 1000
seed = 42
rtol = 1e-6
atol = 1e-7
phitol = 1e-15

dx = 1e-5
dy = dx


prng = np.random.default_rng(seed=seed)
points = prng.random((nPoints, 2))
phis = np.empty((nPoints, 4))
gradphis = np.empty((nPoints, 4, 2))
gradphisNum = np.empty((nPoints, 4, 2))
inds = np.empty((nPoints, 4), dtype='int')

for i, point in enumerate(points):
    phis[i], gradphis[i], inds[i] = BC(point)
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
        
