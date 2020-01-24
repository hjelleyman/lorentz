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

from modules.lorentz import lorentz

#-------------------------------------------------- Implemented functions --------------------------------------------------


def animate_soundwaves(N=200):
    theta = np.linspace(0,2*np.pi,100)
    time = np.linspace(0,1,N)
    v = 25 
    
    
    def run(t, tstart, circles):
        if t%30 == 0:
            tstart += [time[t]]
            circles += plt.plot([],[],'black')
            
        for i in range(len(circles)):
            
            R = v*(time[t] - tstart[i])
            
            circles[i].set_data(R*np.cos(theta),R*np.sin(theta))

    fig, ax = plt.subplots(figsize = (10,10))
    # axis stuff
    plt.xticks([],[])
    plt.yticks([],[])
    plt.xlim([-20,20])
    plt.ylim([-20,20])
    
    plt.plot([0],[0], 'o', color = 'black', markersize = 10)
    tstart = []
    circles = []


    ani = animation.FuncAnimation(fig, run, frames = range(N), blit=False, interval=50, repeat=True, fargs = [tstart, circles])
    return HTML(ani.to_jshtml())

def animate_soundwaves_moving_source(N=200, source_history=False):
    theta = np.linspace(0,2*np.pi,100)
    time = np.linspace(0,1,N)
    v = 25

    def run(t,xstart, tstart, circles):
        source.set_data(x0[t], y0[t])
        if t%30 == 0:
            xstart += [x0[t]]
            tstart += [time[t]]
            circles += plt.plot([],[],'black')
            
        for i in range(len(circles)):
            
            R = v*(time[t] - tstart[i])
            
            circles[i].set_data(xstart[i] + R*np.cos(theta),R*np.sin(theta))
        
        if source_history:
            sources = plt.plot(xstart, np.zeros(len(xstart)),'o', alpha= 0.2, color = 'black')

    fig, ax = plt.subplots(figsize = (10,10))
    # axis stuff
    plt.xticks([],[])
    plt.yticks([],[])
    plt.xlim([-20,20])
    plt.ylim([-20,20])
    
    source, = plt.plot([0],[0], 'o', color = 'black', markersize = 10)
    x0 = np.linspace(-8,10,N)
    y0 = np.zeros(N)
    circles = []
    xstart = []
    tstart = []
    
    ani = animation.FuncAnimation(fig, run, frames = range(N), blit=False, interval=50, repeat=True, fargs = [xstart, tstart, circles])
    return HTML(ani.to_jshtml())

def animate_transverse_moving_source(N=200):
    theta = np.linspace(0,2*np.pi,100)
    time = np.linspace(0,1,N)
    v = 25

    def run(t,xstart, tstart, circles, transverse):
        source.set_data(x0[t], y0[t])
        if t%30 == 0:
            xstart += [x0[t]]
            tstart += [time[t]]
            circles += plt.plot([],[],'black', alpha=0.2)
            
        for i in range(len(circles)):
            
            R = v*(time[t] - tstart[i])
            
            circles[i].set_data(xstart[i] + R*np.cos(theta),R*np.sin(theta))
            
        if len(xstart) > 0:
            transverse[0][:500] = np.linspace((xstart[0] - v*(time[t] - tstart[0])),x0[t],500)
            transverse[0][500:] = np.linspace(x0[t],(xstart[0] + v*(time[t] - tstart[0])),500)
            
            transverse[1][:500] = 1 * np.cos(np.linspace(0,2*np.pi*(t/30),500))
            transverse[1][500:] = 1 * np.cos(np.linspace(2*np.pi*(t/30),0,500))
            
        line.set_data(*transverse)

    fig, ax = plt.subplots(figsize = (10,10))
    # axis stuff
    plt.xticks([],[])
    plt.yticks([],[])
    plt.xlim([-20,20])
    plt.ylim([-20,20])
    
    source, = plt.plot([0],[0], 'o', color = 'black', markersize = 10)
    x0 = np.linspace(-8,10,N)
    y0 = np.zeros(N)
    circles = []
    xstart = []
    tstart = []
    transverse = [np.zeros(1000)-5,np.zeros(1000)]
    line, = plt.plot([],[],'-')
    
    ani = animation.FuncAnimation(fig, run, frames = range(N), blit=False, interval=50, repeat=True, fargs = [xstart, tstart, circles, transverse])
    return HTML(ani.to_jshtml())

#-------------------------------------------------- WIP --------------------------------------------------

def full_doppler(v=0, c = 3e8, N=200, v_wave = 3e8, freq = 20, relativistic=True,classical=False):
    
    theta = np.linspace(0,2*np.pi,100)
    time = np.arange(N)

    fig, ax = plt.subplots(figsize = (10,10))
    # axis stuff
    plt.xticks([],[])
    plt.yticks([],[])
    plt.xlim([-v_wave*N*1.5,v_wave*N*1.5])
    plt.ylim([-v_wave*N*1.5,v_wave*N*1.5])
    
    source, = plt.plot([0],[0], 'o', color = 'black', markersize = 10)
    x0 = np.linspace(-v*N*1/2,v*N*1/2,N)
    y0 = np.zeros(N)
    rel_circles = []
    clas_circles = []
    tstart = []
    xstart = []
    
    
    def run(t, tstart, xstart, rel_circles, clas_circles):
        source.set_data(x0[t],y0[t])
        if t % freq == 0:
            tstart += [t]
            xstart += [x0[t]]
            if relativistic:
                rel_circles += plt.plot([],[],'black')
            if classical:
                clas_circles += plt.plot([],[],'--',color = 'red')
                
                
        if relativistic:    
            for i in range(len(rel_circles)):
                circle = rel_circles[i]
                
                # Beta will vary depending on the angle between the motion of the wave leaving the source and the motion of the source.
                beta = v/c*np.cos(theta)

                t_source = (t - tstart[i])  # For the source, each wave leaves at an even time period.
                d_source = v_wave * t_source # Then the wave propergates outwards at a constant velocity (circle).
                
                # We can use a lorentz transform to work out the time this takes for a stationary observer.
                t_observer = t_source / np.sqrt(1-beta**2) 
                # Because the time dilation is different for each angle we need to scale our results so the distance a wavefront travels is the distance it does in a set time period.
                d_observer = t_observer * d_source / t_source 
                
                # Parameterising the expansion of the wave and moving the source of each wavefront.
                x = d_observer * np.cos(theta) + xstart[i] 
                y = d_observer * np.sin(theta)

                circle.set_data(x,y)
        
        if classical:
            for i in range(len(clas_circles)):
                circle = clas_circles[i]

                t_observed = (t - tstart[i])
                d = v_wave * t_observed

                x = d * np.cos(theta) + xstart[i]
                y = d * np.sin(theta)

                circle.set_data(x,y)
            
            
    
    ani = animation.FuncAnimation(fig, run, frames = time, blit=False, interval=10000/N, repeat=True, fargs = [tstart, xstart, rel_circles, clas_circles])
    return HTML(ani.to_jshtml())
    