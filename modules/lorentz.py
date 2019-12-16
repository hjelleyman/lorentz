"""lorentz
----------
helper functions for the lorentz notebook.
"""
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interactive, FloatSlider

from numpy import genfromtxt
my_data = genfromtxt('my_file.csv', delimiter=',')

def findnearest(array, value):
    idx = np.abs(array - value).argmin()
    return array[idx]

def plot_empty_space():
    """Plots an empty plot to represent empty space."""
    time = genfromtxt('lz_time.csv', delimiter=',')
    space = genfromtxt('lz_space.csv', delimiter=',')
    
    plt.figure()
    plt.plot(space,time,linewidth=0,label='Playground')
    plt.legend()
    plt.show()
    
    
def plot_light_cones():
    """Plots light cones with labels for different regions of spacetime."""
    line1 = genfromtxt('lz_line1.csv', delimiter=',')
    line2 = genfromtxt('lz_line2.csv', delimiter=',')
    plt.figure()
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
    line1 = genfromtxt('lz_line1.csv', delimiter=',')
    line2 = genfromtxt('lz_line2.csv', delimiter=',')
    plt.figure(3)
    plt.plot(space,line1,linewidth=1,color='red')
    plt.plot(space,line2,linewidth=1,color='red')
    plt.xlim(-20,20)
    plt.ylim(-2,20)
    plt.plot([0], [0], 'o')

    plt.title('Transform of an event at the origin')
    plt.show()
    
    
def plot_flashing_lighthouse():
    """Plots the sequence of lights flashing at a lighthouse."""
    line1 = genfromtxt('lz_line1.csv', delimiter=',')
    line2 = genfromtxt('lz_line2.csv', delimiter=',')
    line3 = genfromtxt('lz_line3.csv', delimiter=',')
    line4 = genfromtxt('lz_line4.csv', delimiter=',')
    
    plt.figure(4)
    plt.plot(space,line1,linewidth=1,color='red')
    plt.plot(space,line2,linewidth=1,color='red')
    plt.xlim(-20,20)
    plt.ylim(-2,20)
    plt.plot(line3, line4, 'o')

    plt.title('Flashing lighthouse at the origin')
    plt.show()
    
def lorentz(v):
    """Defines the Lorentz transformation as a 2x2 matrix"""
    gamma=1.0/np.sqrt(1-v*v)
    return np.array([[gamma,-gamma*v],[-gamma*v,gamma]])

def plot_lighthouse_transform():
    """plots a transformed persepective of a lighthouse"""
    line1 = genfromtxt('lz_line1.csv', delimiter=',')
    line2 = genfromtxt('lz_line2.csv', delimiter=',')
    line3 = genfromtxt('lz_line3.csv', delimiter=',')
    line4 = genfromtxt('lz_line4.csv', delimiter=',')
    line5 = pd.read_hdf('data/lz_line5.hdf', 'line5')
    line6 = pd.read_hdf('data/lz_line6.hdf', 'line6')
    line5 = line5[findnearest(line5.columns,0.8)]
    line6 = line6[findnearest(line6.columns,0.8)]
    
    plt.figure()
    plt.plot(space,line1,linewidth=1,color='red')
    plt.plot(space,line2,linewidth=1,color='red')
    plt.xlim(-20,20)
    plt.ylim(-2,20)
    plt.plot(line6, line5, 'o')
    plt.plot(line3, line4, 'o',color='green')

    plt.title('Flashing lighthouse at the origin - moving observer')
    plt.show()
    
    
def interactive_lorentz_1():
    time = genfromtxt('lz_time.csv', delimiter=',')
    space = genfromtxt('lz_space.csv', delimiter=',')
    line1 = genfromtxt('lz_line1.csv', delimiter=',')
    line2 = genfromtxt('lz_line2.csv', delimiter=',')
    line3 = genfromtxt('lz_line3.csv', delimiter=',')
    line4 = genfromtxt('lz_line4.csv', delimiter=',')
    line5 = pd.read_hdf('data/lz_line5.hdf', 'line5')
    line6 = pd.read_hdf('data/lz_line6.hdf', 'line6')
    
    def f(u):
        plt.figure(6,figsize=[12.0, 9.0])
        plt.plot(space,line1,linewidth=1,color='red')
        plt.plot(space,line2,linewidth=1,color='red')
        plt.xlim(-20,20)
        plt.ylim(-2,20)
        plt.plot(line6[findnearest(line6.columns, u)], line5[ufindnearest(line5.columns, u)], 'o')
        plt.plot(line3, line4, 'o',color='green')
        plt.title('Flashing lighthouse at the origin - moving observer')

    interactive_plot = interactive(f, u=FloatSlider(min=-0.999, max=0.999, step=1e-4, continuous_update=False))
    output = interactive_plot.children[-1]
    output.layout.height = '650px'
    return interactive_plot

def ineractive_with_hyperbolae():
    time=np.linspace(-6,20,100)
    space=np.linspace(-20,20,100)
    line1=np.linspace(-20,20,100)
    line2=np.linspace(20,-20,100)
    line3=np.zeros(11)
    line4=np.linspace(0,10,11)
    line5=np.zeros(len(line3))
    line6=np.zeros(len(line3))
    
    
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


    def f2(u):
        plt.figure(7,figsize=[12.0, 9.0])
        plt.plot(space,line1,linewidth=1,color='red')
        plt.plot(space,line2,linewidth=1,color='red')
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
        plt.xlim(-20,20)
        plt.ylim(-2,20)

        for ii in range(len(line3)):
            point=np.array([line4[ii],line3[ii]])  #remember that time is the first element.
            point=np.dot(lorentz(u),point)   #dot does matrix multiplication
            line5[ii]=point[0]
            line6[ii]=point[1]
        plt.plot(line6, line5, 'o')
        plt.plot(line3, line4, 'o',color='green')
        plt.title('Flashing lighthouse at the origin - moving observer')
        plt.show()

    interactive_plot = interactive(f2, u=FloatSlider(min=-0.999, max=0.999, step=1e-4, continuous_update=False))
    output = interactive_plot.children[-1]
    output.layout.height = '650px'
    return interactive_plot

def lighthouse():
    line1=np.linspace(-20,20,100)
    line2=np.linspace(20,-20,100)
    plt.figure(8)
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
    
    
def interactive_lighthouse():
    time=np.linspace(-6,20,100)
    space=np.linspace(-20,20,100)
    line1=np.linspace(-20,20,100)
    line2=np.linspace(20,-20,100)
    line3=np.zeros(11)
    line4=np.linspace(0,10,11)
    line5=np.zeros(11)
    line6=np.zeros(11)
    line7=np.zeros(11)
    line8=np.zeros(11)
    
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

    def f3(u):
        plt.figure(9,figsize=[12.0, 9.0])
        plt.plot(space,line1,linewidth=1,color='red')
        plt.plot(space,line2,linewidth=1,color='red')
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
        plt.xlim(-15,15)
        plt.ylim(-2,20)

        for ii in range(len(line3)):
            point=np.array([line4[ii],line3[ii]])  #remember that time is the first element.
            point=np.dot(lorentz(u),point)   #dot does matrix multiplication
            point2=np.array([line4[ii],line3[ii]+1])  #remember that time is the first element.
            point2=np.dot(lorentz(u),point2)   #dot does matrix multiplication
            line5[ii]=point[0]
            line6[ii]=point[1]
            line7[ii]=point2[0]
            line8[ii]=point2[1]

        plt.plot(line6, line5, 'o-')
        plt.plot(line8, line7, 'o-',color='black')
        plt.plot(line3, line4, 'o-',color='green')
        plt.plot(line3+1, line4, 'o-',color='red')
        plt.title('Flashing lighthouse at the origin - moving observer')
        plt.show()

    interactive_plot = interactive(f3, u=FloatSlider(min=-0.999, max=0.999, step=1e-4, continuous_update=False))
    output = interactive_plot.children[-1]
    output.layout.height = '650px'
    return interactive_plot
    