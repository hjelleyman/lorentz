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

def full_doppler(v=0, c = 3e8, N=200, v_wave = 3e8, freq = 20, relativistic=True,classical=False):
    
    theta = np.linspace(0,2*np.pi,100)
    time = np.arange(N)
    xkcd = list(mcd.XKCD_COLORS.values())

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
    
    
    tstart = np.arange(0,N, freq)
    xstart = np.zeros(len(tstart))
    
    adjusted_tstart = np.zeros([len(tstart), len(theta)])
    adjusted_xstart = np.zeros([len(tstart), len(theta)])
    
    for i in range(len(tstart)):
        if relativistic:
            rel_circles += plt.plot([],[],color = 'blue')
        if classical:
            clas_circles += plt.plot([],[],'--', color = 'red')
        for jj in range(len(theta)):
            adjusted_tstart[i,jj], adjusted_xstart[i,jj] = np.einsum('jk,k->j', 
                                                                     lorentz(v/c),
                                                                     np.array([tstart.copy()[i],xstart.copy()[i]]))
    x_now = v_wave*(0-adjusted_tstart)+adjusted_xstart
    
    def run(t, tstart, xstart, rel_circles, clas_circles):
        if relativistic:
            x_now = v_wave*(t-adjusted_tstart)+adjusted_xstart
            
            for i in range(len(rel_circles)):
                circle = rel_circles[i]
                
                x_circle = x_now[i]
                
                x = x_circle * np.cos(theta) + adjusted_tstart[i]*v
                y = x_circle * np.sin(theta)
                
                
                x = x[t >= adjusted_tstart[i]] 
                y = y[t >= adjusted_tstart[i]]
                circle.set_data(x,y)
        source.set_data(t*v,0)
                
        if classical:
            for i in range(len(clas_circles)):
                circle = clas_circles[i]

                t_observed = (t - tstart[i])
                d = v_wave * t_observed

                x = d * np.cos(theta) + tstart[i]*v
                y = d * np.sin(theta)

                circle.set_data(x[d>0],y[d>0])
            
            
    
    ani = animation.FuncAnimation(fig, run, frames = time, blit=False, interval=10000/N, repeat=True, fargs = [tstart, xstart, rel_circles, clas_circles])
    return HTML(ani.to_jshtml())
    
def lorentz(v):
        """De=fines the Lorentz transformation as a 2x2 matrix."""
        gamma=1.0/np.sqrt(1-v*v)
        return np.array([[gamma,-gamma*v],[-gamma*v,gamma]])
        

def spacetime_plot(c = 3e8, N=120, v_wave = 3e8, freq = 20, relativistic=True, classical=True):
    xkcd = list(mcd.XKCD_COLORS.values())
    time=np.linspace(-6,20,100)
    space=np.linspace(-20,20,100)
    line1=np.linspace(-20,20,100)
    line2=np.linspace(20,-20,100)
    line3=np.zeros(11)
    line4=np.linspace(0,10,11)
    n = int(relativistic)+ int(classical)
    fig, AX = plt.subplots(1, n,figsize =(5*n,3.5))
    
    if type(AX) != np.ndarray:
        AX = [AX]
    
    if relativistic:
        ax = AX[0]
        ax.set_xlabel('distance')
        ax.set_ylabel('time')
        ax.set_title('Relativistic Dopppler Effect')
        l1, = ax.plot([], [], lw=1,color='red')
        l2, = ax.plot([], [], lw=1,color='red')

        velocities=np.linspace(-0.999,0.999,2001)
        lines = [np.zeros((len(velocities),2))] * 3
        for j in range(len(lines)):
            for ii in range(len(velocities)):
                vel=velocities[ii]
                gamma=1.0/np.sqrt(1.0-vel*vel)
                lines[j][ii] = np.dot(lorentz(vel),np.array([range(0,11,3)[j+1],0]))
            ax.plot(lines[j][:,1], lines[j][:,0],linewidth=1,color='black',alpha=0.5)
        l1.set_data(space,line1)
        l2.set_data(space,line2)
        ax.set_xlim(-20,20)
        ax.set_ylim(-2,20)
        
    if classical:
        ax = AX[-1]
        ax.set_xlabel('distance')
        ax.set_ylabel('time')
        ax.set_title('Classical Dopppler Effect')
        l1, = ax.plot([], [], lw=1,color='red')
        l2, = ax.plot([], [], lw=1,color='red')

        for i in range(0,11,3):
            ax.plot(np.linspace(-i,i), np.linspace(i,i),linewidth=1,color='black',alpha=0.5)
        l1.set_data(space,line1)
        l2.set_data(space,line2)
        ax.set_xlim(-20,20)
        ax.set_ylim(-2,20)
    
    text = AX[-1].text(10,3,'$u$ = {:.2f}c'.format(0), size = 10)
    v_wave = v_wave/c
    wavefronts = []
    initial_data = []
    classical_wavefronts = []
    for i in range(0,11,3):
        initial_data += [[np.linspace(0,1000,2),np.linspace(i,i+1000/v_wave,2)], [np.linspace(0,-1000),np.linspace(i,i+1000/v_wave)]]
    
    v = 0
    i = 0
    for data in initial_data:
        if relativistic:
            wavefronts += AX[0].plot(*data, color = 'red')
        if classical:
            classical_wavefronts += AX[-1].plot(*data, color = 'blue')
        i += 1
        
    if relativistic:
        ax = AX[0]
        for ii in range(len(initial_data)):
            xdata,ydata = initial_data[ii]
            xdata = xdata.copy()
            ydata = ydata.copy()
            wavefronts[ii].set_data(xdata,ydata)

    def run(v):
        if relativistic:
            ax = AX[0]
            for ii in range(len(initial_data)):
                xdata,ydata = initial_data[ii]
                xdata = xdata.copy()
                ydata = ydata.copy()
                for jj in range(len(xdata)):
                    point2=np.array([ydata[jj],xdata[jj]])  #remember that time is the first element.
                    point2=np.dot(lorentz(v),point2)   #dot does matrix multiplication
                    xdata[jj]=point2[1]
                    ydata[jj]=point2[0]
                wavefronts[ii].set_data(xdata,ydata)
            
        text.set_text('$u$ = {:.2f}c'.format(v))
        
        if classical:
            ax = AX[-1]
            for ii in range(len(initial_data)):
                xdata,ydata = initial_data[ii]
                xdata = xdata.copy()
                ydata = ydata.copy()
                angle = np.arctan(v)
                xdata = xdata - ii//2 * v*3
                ydata = ydata 
                classical_wavefronts[ii].set_data(xdata,ydata)
            
    ani = animation.FuncAnimation(fig, run, frames = np.linspace(1,-1,(N+2)/n)[1:-1], blit=False, interval=10000/N, repeat=True)
    
    
    return HTML(ani.to_jshtml())

#-------------------------------------------------- WIP --------------------------------------------------

import mpl_toolkits.mplot3d.axes3d as p3
import pandas as pd

import matplotlib._color_data as mcd
import matplotlib.patches as mpatch

def _transition_plot(v=0, c = 3e8, N=200, v_wave = 3e8, freq = 20, relativistic=False,classical=True):
    theta = np.linspace(0,2*np.pi,100)
    time = np.arange(N)

    
    fig = plt.figure(figsize = (10,10))
    ax = p3.Axes3D(fig)
    # axis stuff
    plt.xticks([],[])
    plt.yticks([],[])
    ax.set_xlim3d([-v_wave*N*1.5,v_wave*N*1.5])
    ax.set_ylim3d([-v_wave*N*1.5,v_wave*N*1.5])
    ax.set_zlim3d([-N*0.2, N*1.2])
    ax.set_zlabel('time')
    
    ax.plot(time*v,0*time,time, '-', color = 'black')
    
#     source, = ax.plot([0],[0],[0], 'o', color = 'black', markersize = 10)
    
    x0 = np.linspace(-v*N*1/2,v*N*1/2,N)
    y0 = np.zeros(N)
    rel_circles = []
    clas_circles = []
    tstart = []
    xstart = []
    
    clas_spacetime = []
    rel_spacetime = []
    
    
    tstart = np.arange(0,N, freq)
    xstart = np.zeros(len(tstart))
    
    adjusted_tstart = np.zeros([len(tstart), len(theta)])
    adjusted_xstart = np.zeros([len(tstart), len(theta)])
    
    for i in range(len(tstart)):
        for jj in range(len(theta)):
            adjusted_tstart[i,jj], adjusted_xstart[i,jj] = np.einsum('jk,k->j', 
                                                                     lorentz(v/c),
                                                                     np.array([tstart.copy()[i],xstart.copy()[i]]))
    x_now = v_wave*(0-adjusted_tstart)+adjusted_xstart
    
    circle_dicts = {}
    
    xkcd = list(mcd.XKCD_COLORS.values())
    
    for t in time[::1]:
        for i in range(len(tstart)):
            t_observed = (t - tstart[i])
            if t_observed > 0:
                
                d = v_wave * t_observed

                x = d * np.cos(theta) + tstart[i]*v
                y = d * np.sin(theta)

                z = np.array([t])

                ax.plot(x,y,z, color = xkcd[i], alpha = 0.5)

                xref = np.linspace(d, 0)
                yref = np.linspace(0,-d)*1.2
                xref = xref[d>0]
                yref = yref[d>0]

#                 ax.plot(xref+tstart[i]*v,yref,[t])
#                 ax.plot(-xref+tstart[i]*v,yref,[t])
    plt.show()