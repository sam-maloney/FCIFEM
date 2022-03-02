#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 15:33:58 2021

@author: Samuel A. Maloney

"""

import numpy as np

from abc import ABCMeta, abstractmethod

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
    
    def __repr__(self):
        return f"{self.__class__.__name__}()"

class StraightMapping(Mapping):
    @property
    def name(self): 
        return 'straight'

    def __call__(self, points, zeta=0.):
        y = points.reshape(-1,2)[:,1]
        return y
    
    def deriv(self, points):
        nPoints = int(points.size / 2)
        return np.repeat(0., nPoints)

class LinearMapping(Mapping):
    @property
    def name(self): 
        return 'linear'
    
    def __init__(self, slope):
        self.slope = slope

    def __call__(self, points, zeta=0.):
        x = points.reshape(-1,2)[:,0]
        y = points.reshape(-1,2)[:,1]
        return y + self.slope*(zeta - x)
    
    def deriv(self, points):
        nPoints = int(points.size / 2)
        return np.repeat(self.slope, nPoints)
    
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
        x = points.reshape(-1,2)[:,0]
        y = points.reshape(-1,2)[:,1]
        return y + self.a*(zeta**2 - x**2) + self.b*(zeta - x)
    
    def deriv(self, points):
        x = points.reshape(-1,2)[:,0]
        return 2*self.a*x + self.b
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.a}, {self.b})"

class SinusoidalMapping(Mapping):
    @property
    def name(self): 
        return 'sinusoidal'
    
    def __init__(self, amplitude, phase, xmax=2*np.pi):
        self.A = amplitude
        self.phase = phase
        self.xmax = xmax
        self.xfac = 2*np.pi/xmax

    def __call__(self, points, zeta=0.):
        x = points.reshape(-1,2)[:,0]
        y = points.reshape(-1,2)[:,1]
        offsets = y - self.A*np.sin(self.xfac*(x - self.phase))
        return (self.A*np.sin(self.xfac*(zeta - self.phase)) + offsets)
    
    def deriv(self, points):
        x = points.reshape(-1,2)[:,0]
        return self.A*self.xfac*np.cos(self.xfac*(x - self.phase))
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.A}, {self.phase}, {self.xmax})"