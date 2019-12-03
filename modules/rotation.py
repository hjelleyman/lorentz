"""rotation
----------
This script contains helper functions for the rotation notebook.

"""



import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interactive, FloatSlider
plt.style.use('modules/matplotlibrc')

time=np.linspace(-6,20,100)
space=np.linspace(-20,20,100)
line1=np.linspace(-20,20,100)
line2=np.linspace(20,-20,100)
line3=np.zeros(11)
line4=np.linspace(0,10,11)
line5=np.zeros(len(line3))
line6=np.zeros(len(line3))
from modules import lorentz as lz


def plot_euclidian_vector():
    
    def f(theta):
        thetavec = np.linspace(0,theta)
        x = np.cos(theta)
        y = np.sin(theta)
        
        fig, ax = plt.subplots(figsize=(10,10))
        plt.plot([-2,2],[0,0],'k', alpha = 0.1)
        plt.plot([0,0],[-2,2],'k', alpha = 0.1)
        plt.plot([0,x], [0,y],'r',label = '$[x,y]=[sin(\\theta), cos(\\theta)]$')
        plt.plot(0.1*np.cos(thetavec), 0.1*np.sin(thetavec))
        plt.text(0.13*np.cos(np.pi/4), 0.1*np.sin(np.pi/4),'$\\theta = $'+f'{theta:.2f} radians')
        
        # x length
        plt.plot([0,x], [y,y],'g',label = f'x = {x:.02f}')
        plt.text(x/2, y+0.05,'x')
        # y length
        plt.plot([x,x], [0,y],'b',label = f'y = {y:.02f}')
        plt.text(x+0.05, y/2,'y')
        
        plt.plot([],[],label = '$\\sqrt{x^2+y^2}=$'+f'{np.sqrt(x**2+y**2):.2f}')
        
        # axis stuff
#         plt.axis('off')
        plt.xlim([-1.1,1.1])
        plt.ylim([-1.1,1.1])
        
        plt.legend(loc='center left', bbox_to_anchor= (1.0, 0.5), ncol=1, borderaxespad=0.5, frameon=False)
        plt.plot
        
    interactive_plot = interactive(f, theta=FloatSlider(min=0, max=2*np.pi, step=1e-4, continuous_update=False, description="theta", value=np.pi/4))
    output = interactive_plot.children[-1]
#     output.layout.height = '650px'
    return interactive_plot


def plot_2_euclidian_vectors():
    
    def f(theta, del_theta):
        theta2 = (theta + del_theta) % (2*np.pi)
        
        thetavec = np.linspace(0,theta)
        thetavec2 = np.linspace(0,theta2)
        x = np.cos(theta)
        y = np.sin(theta)
        x2 = np.cos(theta2)
        y2 = np.sin(theta2)
                
        fig, ax = plt.subplots(figsize=(10,10))
        plt.plot([-2,2],[0,0],'k', alpha = 0.1)
        plt.plot([0,0],[-2,2],'k', alpha = 0.1)
        
        
        plt.plot([0,x], [0,y],'r',label = '${A}=[sin(\\theta_1), cos(\\theta_1)]$')
        plt.plot(0.1*np.cos(thetavec), 0.1*np.sin(thetavec), 'r')
        plt.text(0.13*np.cos(np.pi/4), 0.1*np.sin(np.pi/4),'$\\theta_1 = $'+f'{theta:.2f} radians')
        
        plt.plot([0,x2], [0,y2],'b',label = '${B}=[sin(\\theta_2), cos(\\theta_2)]$')
        plt.plot(0.2*np.cos(thetavec2)*(1+0.05*thetavec2), 0.2*np.sin(thetavec2)*(1+0.05*thetavec2), 'b')
        plt.text(0.23*np.cos(np.pi/4), 0.23*np.sin(np.pi/4),'$\\theta_2 = $'+f'{theta2:.2f} radians')
        
        
        plt.plot([],[],label = '${A}\cdot {B} = $'+f'{x*x2+y*y2:.2f}')
        
        # axis stuff
#         plt.axis('off')
        plt.xlim([-1.1,1.1])
        plt.ylim([-1.1,1.1])
        plt.legend(loc='center left', bbox_to_anchor= (1.0, 0.5), ncol=1, borderaxespad=0.5, frameon=False)
        plt.plot
        
    interactive_plot = interactive(f, theta=FloatSlider(min=0, max=2*np.pi, step=1e-4, continuous_update=False, description="theta", value=4), del_theta = FloatSlider(min=0, max=2*np.pi, step=1e-4, continuous_update=False, description="delta theta", value=0.3))
    output = interactive_plot.children[-1]
#     output.layout.height = '650px'
    return interactive_plot