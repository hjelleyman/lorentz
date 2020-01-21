"""
doppler
----------
helper functions for the doppler notebook.

Functions:
----------
    
    findnearest(array, value)
        
    
"""

#--------------------------------------------- Importing relevant modules --------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

#-------------------------------------------------- Implemented functions --------------------------------------------------


def animate_soundwaves(N=100):
    
    
    theta = np.linspace(0,2*np.pi,100)
    
    def init():
        r = 0
        for circle in circles:
            circle.set_data(r*np.cos(theta), r*np.sin(theta))
            r+=1
        return circles

    def run(t):
        r = 0
        T = t/N
        for circle in circles:
            circle.set_data((r+T)*np.cos(theta), (r+T)*np.sin(theta))
            r+=1
        return circles

    fig, ax = plt.subplots(figsize = (10,10))
    # axis stuff
    plt.xticks([],[])
    plt.yticks([],[])
    plt.xlim([-5,5])
    plt.ylim([-5,5])
    
    plt.plot([0],[0], 'o', color = 'black', markersize = 10)
    
    circles = []
    for r in range(10):
        circle, = plt.plot([],[],'black')
        circles += [circle]


    ani = animation.FuncAnimation(fig, run, frames = N, blit=True, interval=20,
                              repeat=True, init_func=init)
    return HTML(ani.to_jshtml())

def animate_soundwaves_moving_source(N=200):
    
    
    theta = np.linspace(0,2*np.pi,100)
    time = np.linspace(0,1,N)
    v = 20

    def run(t,xstart, tstart, circles):
        source.set_data(x0[t], y0[t])
        if t%20 == 0:
            xstart += [x0[t]]
            tstart += [time[t]]
            circles += plt.plot([],[],'black')
            
        for i in range(len(circles)):
            
            R = v*(time[t] - tstart[i])
            
            circles[i].set_data(xstart[i] + R*np.cos(theta),R*np.sin(theta))

    fig, ax = plt.subplots(figsize = (10,10))
    # axis stuff
    plt.xticks([],[])
    plt.yticks([],[])
    plt.xlim([-10,10])
    plt.ylim([-10,10])
    
    source, = plt.plot([0],[0], 'o', color = 'black', markersize = 10)
    x0 = np.linspace(-5,10,N)
    y0 = np.zeros(N)
    circles = []
    xstart = []
    tstart = []
    
    ani = animation.FuncAnimation(fig, run, frames = range(N), blit=False, interval=20, repeat=True, fargs = [xstart, tstart, circles])
    return HTML(ani.to_jshtml())

def animate_transverse_moving_source(N=200):
    theta = np.linspace(0,2*np.pi,100)
    time = np.linspace(0,1,N)
    v = 30

    def run(t,xstart, tstart, circles, transverse):
        source.set_data(x0[t], y0[t])
        if t%20 == 0:
            xstart += [x0[t]]
            tstart += [time[t]]
            circles += plt.plot([],[],'black', alpha=0.2)
            
        for i in range(len(circles)):
            
            R = v*(time[t] - tstart[i])
            
            circles[i].set_data(xstart[i] + R*np.cos(theta),R*np.sin(theta))
            
        if len(xstart) > 0:
            transverse[0][:500] = np.linspace((xstart[0] - v*(time[t] - tstart[0])),x0[t],500)
            transverse[0][500:] = np.linspace(x0[t],(xstart[0] + v*(time[t] - tstart[0])),500)
            
            transverse[1][:500] = 1 * np.sin(np.linspace(0,2*np.pi*(t/20),500))
            transverse[1][500:] = 1 * np.sin(np.linspace(2*np.pi*(t/20),0,500))
            
        line.set_data(*transverse)

    fig, ax = plt.subplots(figsize = (10,10))
    # axis stuff
    plt.xticks([],[])
    plt.yticks([],[])
    plt.xlim([-10,10])
    plt.ylim([-10,10])
    
    source, = plt.plot([0],[0], 'o', color = 'black', markersize = 10)
    x0 = np.linspace(-5,10,N)
    y0 = np.zeros(N)
    circles = []
    xstart = []
    tstart = []
    transverse = [np.zeros(1000)-5,np.zeros(1000)]
    line, = plt.plot([],[],'-')
    
    ani = animation.FuncAnimation(fig, run, frames = range(N), blit=False, interval=100, repeat=True, fargs = [xstart, tstart, circles, transverse])
    return HTML(ani.to_jshtml())

#-------------------------------------------------- WIP --------------------------------------------------

