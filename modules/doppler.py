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

def plot_relativistic_observer(v=0, c = 3e8, N=200, v_wave = 3e8):
    
    
    theta = np.linspace(0,2*np.pi,100)
    time = np.arange(N)

    def run(t, tstart, circles):
        if t % (N//10) == 0:
            tstart += [t]
            circles += plt.plot([],[],'black')
            
        for i in range(len(circles)):
            
            R = v_wave*(t - tstart[i])
            
            betax = v*np.cos(theta)/c
            xvec = (R * np.cos(theta)) * np.sqrt((1+betax)/(1-betax))
            
            yvec = R*np.sin(theta)
            
            circles[i].set_data(xvec,yvec)

    fig, ax = plt.subplots(figsize = (10,10))
    # axis stuff
    plt.xticks([],[])
    plt.yticks([],[])
    plt.xlim([-v_wave*N*1.5,v_wave*N*1.5])
    plt.ylim([-v_wave*N*1.5,v_wave*N*1.5])
    
    source, = plt.plot([0],[0], 'o', color = 'black', markersize = 10)
    circles = []
    tstart = []
    
    ani = animation.FuncAnimation(fig, run, frames = time, blit=False, interval=10000/N, repeat=True, fargs = [tstart, circles])
    return HTML(ani.to_jshtml())