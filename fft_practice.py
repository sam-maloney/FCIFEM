# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 09:51:49 2020

@author: samal
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift, fftfreq

NX = 1001
mode = 10

x = np.linspace(0, 1, NX)
y = np.sin(2*np.pi * mode * x) * np.exp(-4*x)

kx = fftshift(fftfreq(NX, d= x[1] - x[0]))
Y = fftshift(fft(y, norm='ortho'))

fig, axs = plt.subplots(1, 2)

axs[0].plot(x, y)
axs[1].plot(kx, Y)
axs[1].set_xlim(-2*mode, 2*mode)