#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 09:27:41 2021

@author: Samuel A. Maloney
"""

import numpy as np

import fcifem
import boundaryFunctions
import warnings
import time

a = 0.95
b = 0.05
B = boundaryFunctions.QaudraticBoundaryFunction(a, b)
mapping = fcifem.mappings.QuadraticMapping(a, b)


nPoints = 1000
# seed = 42
seed = int(time.time())
maptol = 1e-15
dBtol = 1e-5

dx = 1e-5
dy = dx

prng = np.random.default_rng(seed=seed)
points = prng.random((nPoints, 2))
intersects = np.empty((nPoints, 2))
maps = np.zeros((nPoints, 2))
dBtop = np.zeros((nPoints, 2))
dBbottom = np.zeros((nPoints, 2))
dBtopNum = np.zeros((nPoints, 2))
dBbottomNum = np.zeros((nPoints, 2))

for i, point in enumerate(points):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'invalid value encountered in')
        intersects[i,0], intersects[i,1] = B(point)
    if not np.isnan(intersects[i,0]):
        maps[i,0] = mapping(point, intersects[i,0])
        dBbottom[i,0], dBbottom[i,1] = B.deriv(point, 'bottom')
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'invalid value encountered in')
            zetaXp, _ = B(point + (dx, 0 ))
            zetaXm, _ = B(point - (dx, 0 ))
            zetaYp, _ = B(point + (0 , dy))
            zetaYm, _ = B(point - (0 , dy))
        dBbottomNum[i] = (float(zetaXp - zetaXm)/(2*dx),
                          float(zetaYp - zetaYm)/(2*dy))
        try:
            np.testing.assert_allclose(dBbottom[i], dBbottomNum[i],
                                       rtol=dBtol, atol=dBtol)
        except(AssertionError):
            print(f'bottom derivative error for point {i} = {point}\n'
                  f'{dBbottom[i]} vs. {dBbottomNum[i]}')
    if not np.isnan(intersects[i,1]):
        maps[i,1] = mapping(point, intersects[i,1])
        dBtop[i,0], dBtop[i,1] = B.deriv(point, 'top')
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'invalid value encountered in')
            _, zetaXp = B(point + (dx, 0 ))
            _, zetaXm = B(point - (dx, 0 ))
            _, zetaYp = B(point + (0 , dy))
            _, zetaYm = B(point - (0 , dy))
        dBtopNum[i] = (float(zetaXp - zetaXm)/(2*dx),
                       float(zetaYp - zetaYm)/(2*dy))
        try:
            np.testing.assert_allclose(dBtop[i], dBtopNum[i],
                                       rtol=dBtol, atol=dBtol)
        except(AssertionError):
            print(f'top derivative error for point {i} = {point}\n'
                  f'{dBtop[i]} vs. {dBtopNum[i]}')
    try:
        np.testing.assert_allclose(maps[i], (0,1), rtol=maptol, atol=maptol)
    except(AssertionError):
        print(f'mapping error for point {i} = {point}')
        
