"""lorentz
----------
helper functions for the lorentz notebook.

Functions:
----------
    
    findnearest(array, value)
        Returns the nearest element of an array to a number.
        
    plot_empty_space()
        Plots an empty plot to represent empty space.
    
    plot_light_cones()
        Plots light cones with labels for different regions of spacetime.
        
    plot_event_at_origin()
        PLots an event at the origin of a set of light cones.
        
    plot_flashing_lighthouse()
        Plots the sequence of lights flashing at a lighthouse.
        
    lorentz(v)
        Defines the Lorentz transformation as a 2x2 matrix.
        
    plot_lighthouse_transform()
        Plots a transformed persepective of a lighthouse.
    
    animation_lorentz_1()
        Creates an animation showing how regularly spaced events move through space for a moving observer.
        
    animation_with_hyperbolae()
        Creates an animation showing how regularly spaced events move through space for a moving observer with hyperbolae.
    
    
"""

#--------------------------------------------- Importing relevant modules --------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interactive, FloatSlider
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import pandas as pd

from numpy import genfromtxt

#-------------------------------------------------- Implemented functions --------------------------------------------------


def findnearest(array, value):
    """Returns the nearest element of an array to a number."""
    idx = np.abs(array - value).argmin()
    return array[idx]

def plot_empty_space():
    """Plots an empty plot to represent empty space."""
    time = genfromtxt('data/lz_time.csv', delimiter=',')
    space = genfromtxt('data/lz_space.csv', delimiter=',')
    
    fig, ax = plt.subplots(figsize =(10,7))
    plt.plot(space,time,linewidth=0,label='Playground')
    plt.legend()
    plt.show()
    
    
def plot_light_cones():
    """Plots light cones with labels for different regions of spacetime."""
    time = genfromtxt('data/lz_time.csv', delimiter=',')
    space = genfromtxt('data/lz_space.csv', delimiter=',')
    line1 = genfromtxt('data/lz_line1.csv', delimiter=',')
    line2 = genfromtxt('data/lz_line2.csv', delimiter=',')
    fig, ax = plt.subplots(figsize =(10,7))
    plt.plot(space,line1,linewidth=1,color='red')
    plt.plot(space,line2,linewidth=1,color='red')
    plt.xlim(-20,20)
    plt.ylim(-10,20)
    plt.annotate(' Causal Future',(-5,10),
                xytext=(0.5, 0.9), textcoords='axes fraction',
                fontsize=16,
                horizontalalignment='center', verticalalignment='top')
    plt.annotate('Causal Past',(-5,10),
                xytext=(0.5, 0.1), textcoords='axes fraction',
                fontsize=16,
                horizontalalignment='center', verticalalignment='top')
    plt.annotate('Acausal region',(0,10),
                xytext=(0.8, 0.4), textcoords='axes fraction',
                fontsize=16,
                horizontalalignment='center', verticalalignment='top')
    plt.annotate('Acausal region',(0,10),
                xytext=(0.2, 0.4), textcoords='axes fraction',
                fontsize=16,
                horizontalalignment='center', verticalalignment='top')


    plt.title('Light Cones')
    plt.show()
    
    
def plot_event_at_origin():
    """PLots an event at the origin of a set of light cones."""
    time = genfromtxt('data/lz_time.csv', delimiter=',')
    space = genfromtxt('data/lz_space.csv', delimiter=',')
    line1 = genfromtxt('data/lz_line1.csv', delimiter=',')
    line2 = genfromtxt('data/lz_line2.csv', delimiter=',')
    fig, ax = plt.subplots(figsize =(10,7))
    plt.plot(space,line1,linewidth=1,color='red')
    plt.plot(space,line2,linewidth=1,color='red')
    plt.xlim(-20,20)
    plt.ylim(-2,20)
    plt.plot([0], [0], 'o')

    plt.title('Transform of an event at the origin')
    plt.show()
    
    
def plot_flashing_lighthouse():
    """Plots the sequence of lights flashing at a lighthouse."""
    time = genfromtxt('data/lz_time.csv', delimiter=',')
    space = genfromtxt('data/lz_space.csv', delimiter=',')
    line1 = genfromtxt('data/lz_line1.csv', delimiter=',')
    line2 = genfromtxt('data/lz_line2.csv', delimiter=',')
    line3 = genfromtxt('data/lz_line3.csv', delimiter=',')
    line4 = genfromtxt('data/lz_line4.csv', delimiter=',')
    
    fig, ax = plt.subplots(figsize =(10,7))
    plt.plot(space,line1,linewidth=1,color='red')
    plt.plot(space,line2,linewidth=1,color='red')
    plt.xlim(-20,20)
    plt.ylim(-2,20)
    plt.plot(line3, line4, 'o')

    plt.title('Flashing lighthouse at the origin')
    plt.show()
    
def lorentz(v):
    """De=fines the Lorentz transformation as a 2x2 matrix."""
    gamma=1.0/np.sqrt(1-v*v)
    return np.array([[gamma,-gamma*v],[-gamma*v,gamma]])

def plot_lighthouse_transform():
    """Plots a transformed persepective of a lighthouse."""
    time = genfromtxt('data/lz_time.csv', delimiter=',')
    space = genfromtxt('data/lz_space.csv', delimiter=',')
    line1 = genfromtxt('data/lz_line1.csv', delimiter=',')
    line2 = genfromtxt('data/lz_line2.csv', delimiter=',')
    line3 = genfromtxt('data/lz_line3.csv', delimiter=',')
    line4 = genfromtxt('data/lz_line4.csv', delimiter=',')
    line5 = pd.read_hdf('data/lz_line5.hdf', 'line5')
    line6 = pd.read_hdf('data/lz_line6.hdf', 'line6')
    line5 = line5[findnearest(line5.columns,0.8)]
    line6 = line6[findnearest(line6.columns,0.8)]
    
    fig, ax = plt.subplots(figsize =(10,7))
    plt.plot(space,line1,linewidth=1,color='red')
    plt.plot(space,line2,linewidth=1,color='red')
    plt.xlim(-20,20)
    plt.ylim(-2,20)
    plt.plot(line6, line5, 'o')
    plt.plot(line3, line4, 'o',color='green')

    plt.title('Flashing lighthouse at the origin - moving observer')
    plt.show()
    

def animation_lorentz_1():
    """Creates an animation showing how regularly spaced events move through space for a moving observer."""
    time=np.linspace(-6,20,100)
    space=np.linspace(-20,20,100)
    line1=np.linspace(-20,20,100)
    line2=np.linspace(20,-20,100)
    line3=np.zeros(11)
    line4=np.linspace(0,10,11)
    line5 = pd.read_hdf('data/lz_line5.hdf', 'line5')
    line6 = pd.read_hdf('data/lz_line6.hdf', 'line6')
    
    def datagen(u=1.05):
        while u > -1:
            u -= 0.05
            yield u
    
    def init():
        l1.set_data(space,line1)
        l2.set_data(space,line2)
        l4.set_data(line3, line4)
        ax.set_xlim(-20,20)
        ax.set_ylim(-2,20)
        
    def run(u):
        l3.set_data(line6[findnearest(line6.columns, u)], line5[findnearest(line5.columns, u)])
        text.set_text('$u$ = {:.2f}c'.format(u))
        return l3
    
    
    fig, ax = plt.subplots(figsize =(10,7))
    ax.set_xlabel('distance')
    ax.set_ylabel('time')
    l1, = ax.plot([], [], lw=1,color='red')
    l2, = ax.plot([], [], lw=1,color='red')
    l3, = ax.plot([], [], 'o', color = 'blue')
    l4, = ax.plot([], [], 'o', color = 'green')
    text = plt.text(10,3,'$u$ = {:.2f}'.format(0.1), size = 20)
    
    ani = animation.FuncAnimation(fig, run, datagen, blit=False, interval=100,
                              repeat=True, init_func=init)
    return HTML(ani.to_jshtml())

def animation_with_hyperbolae():
    """Creates an animation showing how regularly spaced events move through space for a moving observer with hyperbolae."""
    time=np.linspace(-6,20,100)
    space=np.linspace(-20,20,100)
    line1=np.linspace(-20,20,100)
    line2=np.linspace(20,-20,100)
    line3=np.zeros(11)
    line4=np.linspace(0,10,11)
    line5 = pd.read_hdf('data/lz_line5.hdf', 'line5')
    line6 = pd.read_hdf('data/lz_line6.hdf', 'line6')
    
    
    def run(u):
        l3.set_data(line6[findnearest(line6.columns, u)], line5[findnearest(line5.columns, u)])
        text.set_text('$u$ = {:.2f}c\n$T$ = {:.2f}$T_0$'.format(u,1/np.sqrt(1-u*u)))
        reference_delta_T.set_data([7,7], [5, 5-1/np.sqrt(1-u*u)])
        return l3
    
    
    fig, ax = plt.subplots(figsize =(10,7))
    ax.set_xlabel('distance')
    ax.set_ylabel('time')
    l1, = ax.plot([], [], lw=1,color='red')
    l2, = ax.plot([], [], lw=1,color='red')
    l3, = ax.plot([], [], 'o', color = 'blue')
    l4, = ax.plot([], [], 'o', color = 'green')
    
    
    reference_T_0, = ax.plot([6,6],[4,5],'-o', color = 'green')
    reference_delta_T, = ax.plot([],[],'-o', color = 'blue')
    
    l1.set_data(space,line1)
    l2.set_data(space,line2)
    l4.set_data(line3, line4)
    ax.set_xlim(-20,20)
    ax.set_ylim(-2,20)
    
    velocities=np.linspace(-0.999,0.999,2001)

    ln1=np.zeros((len(velocities),2))
    ln2=np.zeros((len(velocities),2))
    ln3=np.zeros((len(velocities),2))
    ln4=np.zeros((len(velocities),2))
    ln5=np.zeros((len(velocities),2))
    ln6=np.zeros((len(velocities),2))
    ln7=np.zeros((len(velocities),2))
    ln8=np.zeros((len(velocities),2))
    ln9=np.zeros((len(velocities),2))
    ln10=np.zeros((len(velocities),2))
    

    for ii in range(len(velocities)):
        vel=velocities[ii]
        gamma=1.0/np.sqrt(1.0-vel*vel)
        ln1[ii]=np.dot(lorentz(vel),np.array([1,0]))
        ln2[ii]=np.dot(lorentz(vel),np.array([2,0]))
        ln3[ii]=np.dot(lorentz(vel),np.array([3,0]))
        ln4[ii]=np.dot(lorentz(vel),np.array([4,0]))
        ln5[ii]=np.dot(lorentz(vel),np.array([5,0]))
        ln6[ii]=np.dot(lorentz(vel),np.array([6,0]))
        ln7[ii]=np.dot(lorentz(vel),np.array([7,0]))
        ln8[ii]=np.dot(lorentz(vel),np.array([8,0]))
        ln9[ii]=np.dot(lorentz(vel),np.array([9,0]))
        ln10[ii]=np.dot(lorentz(vel),np.array([10,0]))
    plt.plot(ln1[:,1],ln1[:,0],linewidth=1,color='black')
    plt.plot(ln2[:,1],ln2[:,0],linewidth=1,color='black')
    plt.plot(ln3[:,1],ln3[:,0],linewidth=1,color='black')
    plt.plot(ln4[:,1],ln4[:,0],linewidth=1,color='black')
    plt.plot(ln5[:,1],ln5[:,0],linewidth=1,color='black')
    plt.plot(ln6[:,1],ln6[:,0],linewidth=1,color='black')
    plt.plot(ln7[:,1],ln7[:,0],linewidth=1,color='black')
    plt.plot(ln8[:,1],ln8[:,0],linewidth=1,color='black')
    plt.plot(ln9[:,1],ln9[:,0],linewidth=1,color='black')
    plt.plot(ln10[:,1],ln10[:,0],linewidth=1,color='black')
    text = plt.text(10,3,'$u$ = {:.2f}'.format(0.1), size = 20)
    
    ani = animation.FuncAnimation(fig, run, frames = np.linspace(1,-1,200)[1:], blit=False, interval=50, repeat=True)
    return HTML(ani.to_jshtml())

def lighthouse():
    time = genfromtxt('data/lz_time.csv', delimiter=',')
    space = genfromtxt('data/lz_space.csv', delimiter=',')
    line1=np.linspace(-20,20,100)
    line2=np.linspace(20,-20,100)
    fig, ax = plt.subplots(figsize =(10,7))
    plt.plot(space,line1,linewidth=1,color='red')
    plt.plot(space,line2,linewidth=1,color='red')
    plt.xlim(-15,15)
    plt.ylim(-2,20)
    line3=np.zeros(11)
    line4=np.linspace(0,10,11)
    plt.plot(line3, line4, 'o',color='green')
    plt.plot(line3+1, line4, 'o',color='red')

    plt.title('Flashing lighthouses measured by an observer in their reference frame')
    plt.show()
    
    
def animated_lighthouse():
    time=np.linspace(-6,20,100)
    space=np.linspace(-20,20,100)
    line1=np.linspace(-20,20,100)
    line2=np.linspace(20,-20,100)
    line3=np.zeros(11)
    line4=np.linspace(0,10,11)
    line5 = pd.read_hdf('data/lz_line5.hdf', 'line5')
    line6 = pd.read_hdf('data/lz_line6.hdf', 'line6')
    
    def init():
        l1.set_data(space,line1)
        l2.set_data(space,line2)
        l4.set_data(line3, line4)
        l5.set_data(line3+2, line4)
        l6.set_data(line3+2, line4)
        ax.set_xlim(-20,20)
        ax.set_ylim(-2,20)
        
    def run(u):
        l3.set_data(line6[findnearest(line6.columns, u)], line5[findnearest(line5.columns, u)])
        line3=np.zeros(11)
        line4=np.linspace(0,10,11)
        newx = np.zeros(11)
        newy = np.zeros(11)
        for ii in range(len(line3)):
            point2=np.array([line4[ii],line3[ii]+2])  #remember that time is the first element.
            point2=np.dot(lorentz(u),point2)   #dot does matrix multiplication
            newy[ii]=point2[0]
            newx[ii]=point2[1]
        l6.set_data(newx,newy)
        changed_d.set_data([10,10 + 2*np.sqrt(1-u*u)],[1,1])
        
        text.set_text('$u$ = {:.2f}c\n$L$ = {:.2f}$L_0$'.format(u,np.sqrt(1-u*u)))
        return l3,l6
    
    
    fig, ax = plt.subplots(figsize =(10,7))
    ax.set_xlabel('distance')
    ax.set_ylabel('time')
    l1, = ax.plot([], [], lw=1,color='red')
    l2, = ax.plot([], [], lw=1,color='red')
    l3, = ax.plot([], [], 'o-', color = 'blue')
    l4, = ax.plot([], [], 'o-', color = 'blue', alpha = 0.3)
    l5, = ax.plot([], [],'o-', color = 'red', alpha = 0.3)
    l6, = ax.plot([], [],'o-', color = 'red')
    
    refernce_d, = ax.plot([10,12],[2,2],'o-', color = 'blue')
    changed_d, = ax.plot([10,12],[1,1],'o-', color = 'green')
    
    velocities=np.linspace(-0.999,0.999,2001)
    lines = [np.zeros((len(velocities),2))] * 11
    for j in range(len(lines)):
        for ii in range(len(velocities)):
            vel=velocities[ii]
            gamma=1.0/np.sqrt(1.0-vel*vel)
            lines[j][ii] = np.dot(lorentz(vel),np.array([j,0]))
        plt.plot(lines[j][:,1], lines[j][:,0],linewidth=1,color='black',alpha=0.5)
    text = plt.text(10,3,'$u$ = {:.2f}'.format(0.1), size = 20, fontname = 'computer modern')
    
    ani = animation.FuncAnimation(fig, run, np.linspace(1,-1,100), blit=False, interval=50,
                              repeat=True, init_func=init)
    return HTML(ani.to_jshtml())

#------------------------------------------------------------ WIP --------------------------------------------------

    
#------------------------------------------------------------ Currently Unused ------------------------------------------------
    
# def interactive_lorentz_1():
#     """CUrrently unused function which allows for an interactive plot of the lorentz treansform being applied to a flashing light with different observer speeds."""
#     time = genfromtxt('data/lz_time.csv', delimiter=',')
#     space = genfromtxt('data/lz_space.csv', delimiter=',')
#     time = genfromtxt('data/lz_time.csv', delimiter=',')
#     space = genfromtxt('data/lz_space.csv', delimiter=',')
#     line1 = genfromtxt('data/lz_line1.csv', delimiter=',')
#     line2 = genfromtxt('data/lz_line2.csv', delimiter=',')
#     line3 = genfromtxt('data/lz_line3.csv', delimiter=',')
#     line4 = genfromtxt('data/lz_line4.csv', delimiter=',')
#     line5 = pd.read_hdf('data/lz_line5.hdf', 'line5')
#     line6 = pd.read_hdf('data/lz_line6.hdf', 'line6')
    
#     def f(u):
#         plt.figure(6,figsize=[12.0, 9.0])
#         plt.plot(space,line1,linewidth=1,color='red')
#         plt.plot(space,line2,linewidth=1,color='red')
#         plt.xlim(-20,20)
#         plt.ylim(-2,20)
#         plt.plot(line6[findnearest(line6.columns, u)], line5[findnearest(line5.columns, u)], 'o')
#         plt.plot(line3, line4, 'o',color='green')
#         plt.title('Flashing lighthouse at the origin - moving observer')

#     interactive_plot = interactive(f, u=FloatSlider(min=-0.999, max=0.999, step=1e-4, continuous_update=False))
#     output = interactive_plot.children[-1]
#     output.layout.height = '650px'
#     return interactive_plot


# def ineractive_with_hyperbolae():
#     """ Currently unused function for plotting an interactive plot with hyperbole."""
#     time=np.linspace(-6,20,100)
#     space=np.linspace(-20,20,100)
#     line1=np.linspace(-20,20,100)
#     line2=np.linspace(20,-20,100)
#     line3=np.zeros(11)
#     line4=np.linspace(0,10,11)
#     line5=np.zeros(len(line3))
#     line6=np.zeros(len(line3))
    
    
#     velocities=np.linspace(-0.999,0.999,2001)

#     ln1=np.zeros((len(velocities),2))
#     ln2=np.zeros((len(velocities),2))
#     ln3=np.zeros((len(velocities),2))
#     ln4=np.zeros((len(velocities),2))
#     ln5=np.zeros((len(velocities),2))
#     ln6=np.zeros((len(velocities),2))
#     ln7=np.zeros((len(velocities),2))
#     ln8=np.zeros((len(velocities),2))
#     ln9=np.zeros((len(velocities),2))
#     ln10=np.zeros((len(velocities),2))
    

#     for ii in range(len(velocities)):
#         vel=velocities[ii]
#         gamma=1.0/np.sqrt(1.0-vel*vel)
#         ln1[ii]=np.dot(lorentz(vel),np.array([1,0]))
#         ln2[ii]=np.dot(lorentz(vel),np.array([2,0]))
#         ln3[ii]=np.dot(lorentz(vel),np.array([3,0]))
#         ln4[ii]=np.dot(lorentz(vel),np.array([4,0]))
#         ln5[ii]=np.dot(lorentz(vel),np.array([5,0]))
#         ln6[ii]=np.dot(lorentz(vel),np.array([6,0]))
#         ln7[ii]=np.dot(lorentz(vel),np.array([7,0]))
#         ln8[ii]=np.dot(lorentz(vel),np.array([8,0]))
#         ln9[ii]=np.dot(lorentz(vel),np.array([9,0]))
#         ln10[ii]=np.dot(lorentz(vel),np.array([10,0]))


#     def f2(u):
#         plt.figure(7,figsize=[12.0, 9.0])
#         plt.plot(space,line1,linewidth=1,color='red')
#         plt.plot(space,line2,linewidth=1,color='red')
#         plt.plot(ln1[:,1],ln1[:,0],linewidth=1,color='black')
#         plt.plot(ln2[:,1],ln2[:,0],linewidth=1,color='black')
#         plt.plot(ln3[:,1],ln3[:,0],linewidth=1,color='black')
#         plt.plot(ln4[:,1],ln4[:,0],linewidth=1,color='black')
#         plt.plot(ln5[:,1],ln5[:,0],linewidth=1,color='black')
#         plt.plot(ln6[:,1],ln6[:,0],linewidth=1,color='black')
#         plt.plot(ln7[:,1],ln7[:,0],linewidth=1,color='black')
#         plt.plot(ln8[:,1],ln8[:,0],linewidth=1,color='black')
#         plt.plot(ln9[:,1],ln9[:,0],linewidth=1,color='black')
#         plt.plot(ln10[:,1],ln10[:,0],linewidth=1,color='black')
#         plt.xlim(-20,20)
#         plt.ylim(-2,20)

#         for ii in range(len(line3)):
#             point=np.array([line4[ii],line3[ii]])  #remember that time is the first element.
#             point=np.dot(lorentz(u),point)   #dot does matrix multiplication
#             line5[ii]=point[0]
#             line6[ii]=point[1]
#         plt.plot(line6, line5, 'o')
#         plt.plot(line3, line4, 'o',color='green')
#         plt.title('Flashing lighthouse at the origin - moving observer')
#         plt.show()

#     interactive_plot = interactive(f2, u=FloatSlider(min=-0.999, max=0.999, step=1e-4, continuous_update=False))
#     output = interactive_plot.children[-1]
#     output.layout.height = '650px'
#     return interactive_plot    