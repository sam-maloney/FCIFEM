# -*- coding: utf-8 -*-
"""
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
    
    def B(self, points):
        """Compute x-coordinates of boundary intersections from given points.
        Only required if Dirichlet BC used in y-direction.

        Parameters
        ----------
        points : numpy.ndarray, shape=(n,ndim)
            (x,y) coordinates of evaluation points.

        Returns
        -------
        zetaBottom : numpy.ndarray, shape=(n,)
            x-coordinates of mapping intersections with bottom boundary.
            If no such intersection exists then np.nan is returned.
        zetaTop : numpy.ndarray, shape=(n,)
            x-coordinates of mapping intersections with top boundary.
            If no such intersection exists then numpy.nan is returned.
            
        """
        raise NotImplementedError
    
    def dBtop(self, points):
        """Compute gradient of top boundary function at given points.
        Only required if Dirichlet BC used in y-direction.

        Parameters
        ----------
        points : numpy.ndarray, shape=(n,ndim)
            (x,y) coordinates of evaluation points.

        Returns
        -------
        dBdx : numpy.ndarray, shape=(n,)
            x-derivatives of top boundary function at given points.
        dBdy : numpy.ndarray, shape=(n,)
            y-derivatives of top boundary function at given points.

        """
        raise NotImplementedError
    
    def dBbottom(self, points):
        """Compute gradient of bottom boundary function at given points.
        Only required if Dirichlet BC used in y-direction.

        Parameters
        ----------
        points : numpy.ndarray, shape=(n,ndim)
            (x,y) coordinates of evaluation points.

        Returns
        -------
        dBdx : numpy.ndarray, shape=(n,)
            x-derivatives of bottom boundary function at given points.
        dBdy : numpy.ndarray, shape=(n,)
            y-derivatives of bottom boundary function at given points.

        """
        raise NotImplementedError
    
    def __repr__(self):
        return f"{self.__class__.__name__}()"

class StraightMapping(Mapping):
    @property
    def name(self): 
        return 'straight'
    
    def __init__(self):
        self.dBtop = self.dB
        self.dBbottom = self.dB

    def __call__(self, points, zeta=0.):
        y = points.reshape(-1,2)[:,1]
        return y
    
    def deriv(self, points):
        nPoints = int(points.size / 2)
        return np.zeros(nPoints)
    
    def B(self, points):
        nPoints = points.size // 2
        return np.full(nPoints, np.nan), np.full(nPoints, np.nan)
    
    def dB(self, points):
        print('Error: Straight mapping should never intersect y-boundary.')
        # this should never be used, just return dummy values
        nPoints = points.size // 2
        return np.ones(nPoints), np.ones(nPoints)


class LinearMapping(Mapping):
    @property
    def name(self): 
        return 'linear'
    
    def __init__(self, slope):
        self.slope = slope
        self.dBtop = self.dB
        self.dBbottom = self.dB

    def __call__(self, points, zeta=0.):
        x = points.reshape(-1,2)[:,0]
        y = points.reshape(-1,2)[:,1]
        return y + self.slope*(zeta - x)
    
    def deriv(self, points):
        nPoints = int(points.size / 2)
        return np.repeat(self.slope, nPoints)
    
    def B(self, points):
        x = points.reshape(-1,2)[:,0]
        y = points.reshape(-1,2)[:,1]
        zetaBottom = x - y/self.slope
        zetaTop = x + (1-y)/self.slope
        return zetaBottom, zetaTop
    
    def dB(self, points):
        nPoints = points.size // 2
        return np.ones(nPoints), np.full(nPoints, -1./self.slope)
    
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
    
    def B(self, points):
        x = points.reshape(-1,2)[:,0]
        y = points.reshape(-1,2)[:,1]
        a = self.a
        b = self.b
        zetaBottom = (-b + np.sqrt(b**2 - 4*a*(y - a*x**2 - b*x)))/(2*a)
        zetaTop = (-b + np.sqrt(b**2 - 4*a*(y - a*x**2 - b*x - 1)))/(2*a)
        return zetaBottom, zetaTop
    
    def dBtop(self, points):
        x = points.reshape(-1,2)[:,0]
        y = points.reshape(-1,2)[:,1]
        a = self.a
        b = self.b
        dBdy = -1 / np.sqrt(b**2 - 4*a*(y - a*x**2 - b*x - 1))
        dBdx = -(2*a*x + b) * dBdy
        return dBdx, dBdy
    
    def dBbottom(self, points):
        x = points.reshape(-1,2)[:,0]
        y = points.reshape(-1,2)[:,1]
        a = self.a
        b = self.b
        dBdy = -1 / np.sqrt(b**2 - 4*a*(y - a*x**2 - b*x))
        dBdx = -(2*a*x + b) * dBdy
        return dBdx, dBdy
    
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