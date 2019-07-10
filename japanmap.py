# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 09:56:40 2019

@author: yu
"""


import os
import pickle
import glob
import pdb

import numpy as np
import pandas as pd

from PIL import Image

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from mpl_toolkits.basemap import Basemap
import mpl_toolkits.axisartist.floating_axes as floating_axes
from matplotlib.transforms import Affine2D
import matplotlib.transforms as mtransforms
from matplotlib.path import Path
from matplotlib.patches import PathPatch


from scipy.interpolate import griddata
"""
# Create some example data
x = np.array([1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6])
y = np.array([1,2,3,4,2,3,4,5,3,4,5,6,4,5,6,7,5,6,7,8,6,7,8,9])
z = np.linspace(0,100,len(y))

# Grid and interpolate between points
yi, xi = np.mgrid[int(y.min()):int(y.max()),int(x.min()):int(x.max())]
zi = griddata((x, y), z, (xi, yi), method='nearest')

path = Path([[1, 1], [1, 4], [6, 9], [6, 6], [1, 1]])
patch = PathPatch(path, facecolor='none')

# Plot the figure
im = plt.imshow(
  zi, extent=[x.min(), x.max(), y.min(), y.max()],
  origin="lower", interpolation='bicubic', aspect='auto',
  clip_path=patch, clip_on=True)

plt.colorbar()


plt.gca().add_patch(patch)
im.set_clip_path(patch)
plt.show()
"""
df = pd.DataFrame({'x':[0, 0, 1, 1, 3, 3, 3, 4, 4, 4], 
                   'y':[0, 1, 0, 1, 0.2, 0.7, 1.4, 0.2, 1.4, 2], 
                   'z':[50, 40, 40, 30, 30, 30, 20, 20, 20, 10]})

x = df['x']
y = df['y']
z = df['z']

xi = np.linspace(x.min(), x.max(), 100)
yi = np.linspace(y.min(), y.max(), 100)
z_grid = griddata(x, y, z, xi, yi, interp='linear')

clipindex = [ [0,2,4,7,8,9,6,3,1,0],
              [0,2,4,7,5,8,9,6,3,1,0],
              [0,2,4,7,8,9,6,5,3,1,0]]

fig, axes = plt.subplots(ncols=3, sharey=True)
for i, ax in enumerate(axes):
    cont = ax.contourf(xi, yi, z_grid, 15)
    ax.scatter(x, y, color='k') # The original data points
    ax.plot(x[clipindex[i]], y[clipindex[i]], color="crimson")

    clippath = Path(np.c_[x[clipindex[i]], y[clipindex[i]]])
    patch = PathPatch(clippath, facecolor='none')
    ax.add_patch(patch)
    for c in cont.collections:
        c.set_clip_path(patch)

plt.show()
#import cv2

"""
img = cv2.imread("clock.png",0)
plt.imshow(img)
plt.show()
"""
"""
fig = plt.figure(figsize=(8,4))
plt.figtext(0.9,0.9,"$x$")
plt.show()
"""
def sample(X,n):
    if (int(X/n)):
        
        return sample(int(X/n),n)+str(X%n)
    return str(X%n)

x1 = 0.6
x2 = sample(x1,12)
print(x2)

"""
west = 129
south = 30
east = 141
north = 38

fig = plt.figure(figsize=(8,4))
# 日本地図作成
japanMap = Basemap(projection='merc',resolution='h',llcrnrlon=west,llcrnrlat=south,urcrnrlon=east,urcrnrlat=north)
japanMap.drawcoastlines(color='green')
japanMap.drawcountries(color='palegreen')
japanMap.fillcontinents(color='palegreen', lake_color="lightblue")
japanMap.drawmapboundary(fill_color='lightcyan')


plt.show()
"""