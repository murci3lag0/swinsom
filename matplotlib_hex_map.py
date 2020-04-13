#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 14:45:41 2019

@author: amaya
"""
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import numpy as np

cpalette = ['#e6194B', '#3cb44b', '#4363d8', '#f58231', '#42d4f4', '#f032e6', '#469990', '#e6beff', '#9A6324', '#800000', '#000075']
linscale = 4

def add_matplotlib_hexagon(ax, xc, yc, width, color=None, lcolor='black', alpha=1.0, cmap=None, cmin=0, cmax=1, size=1.0, r=1.0, scale=1.0):

    assert(len(width)==6)

    if cmap==None:
        cmap = plt.cm.get_cmap('jet')
                
    norm = matplotlib.colors.Normalize(vmin=cmin, vmax=cmax)
    if np.isscalar(color):
        fcolor = cmap(norm(color))
    else:
        fcolor = tuple(norm(color))
    
    if len(fcolor)<=2:
        fcolor = fcolor + (1.0,)
    if len(fcolor)==1:
        fcolor = fcolor + (1.0,)
    
    x0=xc+np.array([0,    r,    r,    0,   -r,   -r])
    y0=yc+np.array([r,    r/2, -r/2, -r,   -r/2,  r/2])
    x1=xc+np.array([r,    r,    0,   -r,   -r,    0])
    y1=yc+np.array([r/2, -r/2, -r,   -r/2,  r/2,  r])
   
    for i in range(6):
        clr = cpalette[int(lcolor[i])] if type(lcolor)!=str else lcolor
        line = lines.Line2D([x0[i],x1[i]], [y0[i],y1[i]],
                            lw=linscale*width[i]**scale, color=clr, axes=ax, alpha=width[i])
        ax.add_line(line)
    
    assert(size<=1.0 and size>=0)
    x0=xc+size*np.array([0,    r,    r,    0,   -r,   -r])
    y0=yc+size*np.array([r,    r/2, -r/2, -r,   -r/2,  r/2])
    polygon = patches.Polygon(
        xy=list(zip(x0,y0)),
        ls=None,
        lw=0.0,
        closed=True,
        color=fcolor,
        alpha=alpha)
    ax.add_patch(polygon) 
        
    return ax


def som_hexmesh(x, y, r=0.5):
    xx, yy = np.meshgrid(x, y)
    xx = 2*r*xx.astype(float)
    yy = 1.5*r*yy.astype(float)
    xx[::2] -= r
    return xx, yy

def matplotlib_hex_map(ax, dist, color, som_m, som_n, usezero=False, size=None, lcolor='white', alpha=None, r=0.5, scale=1.0, cmap=None, title=None, colorbar=False, axecolor='w', cbmin=0.0, cbmax=1.0, savefig=False, filename='fig.png'):
    xx, yy = som_hexmesh(range(som_m), range(som_n), r=0.5)
    matplotlib.rcParams.update({'font.size': 20})
    x0, y0, W, H = ax.get_position().bounds
    # fig, ax = plt.subplots()
    dmin = dist[dist.nonzero()].min()
    dmax = dist.max()
    d = np.zeros_like(dist)
    d[dist==0] = dmin
    if dmax!=dmin:
        d = (dist - dmin)/(dmax-dmin)
    
    if usezero:
        d = 2*dist/linscale
    
    if cmap is None:
        cmap = plt.cm.get_cmap('jet')
    else:
        cmap = plt.cm.get_cmap(cmap)
    if alpha is None:
        alpha = np.ones((som_m,som_n))

    for i in range(som_m):
        for j in range(som_n):
            lc = lcolor if type(lcolor)==str else lcolor[i,j]
            ax = add_matplotlib_hexagon(ax,
                                             xx[j,i],
                                             yy[j,i],
                                             d[i,j],
                                             color=color[i,j],
                                             lcolor=lc,
                                             alpha=alpha[i,j],
                                             cmap=cmap,
                                             size=size[i,j],
                                             r=r,
                                             scale=scale)
    # if colorbar:
    #     norm = matplotlib.colors.Normalize(vmin=cbmin,vmax=cbmax)
    #     sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    #     sm.set_array([])
    #     plt.colorbar(sm)
    ax.set_aspect('equal')
    ax.set_xlim(-1, som_m-0.5)
    ax.set_ylim(-0.5, som_n*0.75-0.25)
    ax.axis('off')
    # ax.set_aspect('equal')
    # ax.set_facecolor(axecolor)
    ax.set_title(title)
    # plt.autoscale()
    # if savefig:
    #     plt.savefig(filename, bbox_inches='tight', transparent=True)
    # plt.show()
    return ax
    
def hex_map_test(x, y, color=[[0.7]], cmin=0, cmax=1, size=1.0, r=0.5, axecolor='w'):
    fig, ax = plt.subplots(2,2)
    ax[1][0].set_facecolor(axecolor)
    d = np.array([0.1,0.3,0.5,0.7,0.9,1.0])
    dmin = d[d.nonzero()].min()
    dmax = d.max()
    d[d==0] = dmin
    d = (d - dmin)/(dmax-dmin)
    
    lc = [1, 2, 3, 4, 5, 6]
    
    cmap = plt.cm.get_cmap()
    for i in range(len(x)):
        ax[1][0] = add_matplotlib_hexagon(ax[1][0], y[i], x[i],
                                         d,
                                         color=color[i],
                                         lcolor=lc,
                                         cmap=cmap,
                                         cmin=cmin,
                                         cmax=cmax,
                                         size=size[i],
                                         r=r,
                                         scale=2)
    norm = matplotlib.colors.Normalize(vmin=np.array(color).min(),vmax=np.array(color).max())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    ax[1][0].set_aspect('equal')
    plt.autoscale()
    plt.show()
    
if __name__ == "__main__":
    if True:
        x=[0,0,0.75]
        y=[0,1,0.5]
        color=[[100, 30, 110],[20, 50, 10],[100, 100, 100]]
        cmin=0
        cmax=255
        size=[1, 0.5, 0.25]
        
        hex_map_test(x, y, color=color, size=size, cmin=cmin, cmax=cmax, r=0.5, axecolor=(0.5,0.5,0.5))