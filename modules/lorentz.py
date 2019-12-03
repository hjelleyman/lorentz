"""lorentz
----------
helper functions for the lorentz notebook.
"""
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('modules/matplotlibrc')


time=np.linspace(-6,20,100)
space=np.linspace(-20,20,100)

def plot_empty_space():
    """Plots an empty plot to represent empty space."""
    plt.figure(1)
    plt.plot(space,time,linewidth=0,label='Playground')
    plt.legend()
    plt.show()
    
    
def plot_light_cones():
    """Plots light cones with labels for different regions of spacetime."""
    line1=np.linspace(-20,20,100)
    line2=np.linspace(20,-20,100)
    plt.figure(2)
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
    line1=np.linspace(-20,20,100)
    line2=np.linspace(20,-20,100)
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
    line1=np.linspace(-20,20,100)
    line2=np.linspace(20,-20,100)
    plt.figure(4)
    plt.plot(space,line1,linewidth=1,color='red')
    plt.plot(space,line2,linewidth=1,color='red')
    plt.xlim(-20,20)
    plt.ylim(-2,20)
    line3=np.zeros(11)
    line4=np.linspace(0,10,11)
    plt.plot(line3, line4, 'o')

    plt.title('Flashing lighthouse at the origin')
    plt.show()
    
def lorentz(v):
    """Defines the Lorentz transformation as a 2x2 matrix"""
    gamma=1.0/np.sqrt(1-v*v)
    return np.array([[gamma,-gamma*v],[-gamma*v,gamma]])

def plot_lighthouse_transform():
    """plots a transformed persepective of a lighthouse"""
    line1=np.linspace(-20,20,100)
    line2=np.linspace(20,-20,100)
    line3=np.zeros(11)
    line4=np.linspace(0,10,11)
    line5=np.zeros(len(line3))
    line6=np.zeros(len(line3))
    for ii in range(len(line3)):
        point=np.array([line4[ii],line3[ii]])  #remember that time is the first element.
        point=np.dot(lorentz(0.8),point)   #dot does matrix multiplication
        line5[ii]=point[0]
        line6[ii]=point[1]
        #print(point)

        plt.figure(5)
    plt.plot(space,line1,linewidth=1,color='red')
    plt.plot(space,line2,linewidth=1,color='red')
    plt.xlim(-20,20)
    plt.ylim(-2,20)
    plt.plot(line6, line5, 'o')
    plt.plot(line3, line4, 'o',color='green')

    plt.title('Flashing lighthouse at the origin - moving observer')
    plt.show()