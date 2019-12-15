"""rotation
----------
This script contains helper functions for the rotation notebook.

"""



import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interactive, FloatSlider
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML


plt.style.use('modules/matplotlibrc')

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
        plt.text(0.13*np.cos(np.pi/4), 0.1*np.sin(np.pi/4),'$\\theta = $'+'{:.2f} radians'.format(theta))
        
        # x length
        plt.plot([0,x], [y,y],'g',label = 'x = {:.02f}'.format(x))
        plt.text(x/2, y+0.05,'x')
        # y length
        plt.plot([x,x], [0,y],'b',label = 'y = {:.02f}'.format(y))
        plt.text(x+0.05, y/2,'y')
        
        plt.plot([],[],label = '$\\sqrt{x^2+y^2}=$'+'{:.2f}'.format(np.sqrt(x**2+y**2)))
        
        # axis stuff
#         plt.axis('off')
        plt.xlim([-1.1,1.1])
        plt.ylim([-1.1,1.1])
        
        plt.legend(loc='center left', bbox_to_anchor= (1, 0.5), ncol=1, borderaxespad=0.5, frameon=False)
        plt.plot
        
    interactive_plot = interactive(f, theta=FloatSlider(min=0, max=2*np.pi, step=1e-4, continuous_update=False, description="$\\theta$", value=np.pi/4))
    output = interactive_plot.children[-1]
#     output.layout.height = '650px'
    return interactive_plot


def animate_plot_1():
    fig, ax = plt.subplots(figsize=(10,10))
    plt.plot([-2,2],[0,0],'k', alpha = 0.1)
    plt.plot([0,0],[-2,2],'k', alpha = 0.1)
    line, = ax.plot([],[],'r', label = '$[x,y]=[sin(\\theta), cos(\\theta)]$')
    xangle, = ax.plot([],[],'r')
    text = plt.text(0.14*np.cos(np.pi/4), 0.14*np.sin(np.pi/4),'')
    
    # x length
    xlength, = plt.plot([], [],'g')
    xtext = plt.text(0, 0,'x')
    # y length
    ylength, = plt.plot([], [],'b')
    ytext = plt.text(0, 0,'y')

    # axis stuff
    plt.axis('off')
    plt.xlim([-1.1,1.1])
    plt.ylim([-1.1,1.1])
    
    def init():
        line.set_data([], [])
        xangle.set_data([],[])
        text.set_text('')
        xlength.set_data([],[])
        xlength.set_label('')
        xtext.set_position([0,0])
        xtext.set_text('')
        ylength.set_data([],[])
        ylength.set_label('')
        ytext.set_position([0,0])
        ytext.set_text('')
        return [line,text,xangle,xlength,xtext,ylength,ytext]
    
    def animate(theta):
        theta = theta *np.pi/50
        thetavec = np.linspace(0,theta)
        x = np.cos(theta)
        y = np.sin(theta)
#         plt.legend(loc='center left', bbox_to_anchor= (1.0, 0.5), ncol=1, borderaxespad=0.5, frameon=False)

        line.set_data([0,x], [0,y])
        xangle.set_data(0.1*np.cos(thetavec), 0.1*np.sin(thetavec))
        text.set_text('$\\theta = $'+'{:.2f} radians'.format(theta))
        xlength.set_data([0,x], [y,y])
        xlength.set_label('x = {:.02f}'.format(x))
        xtext.set_position((x/2, y+0.05))
        xtext.set_text('x = {:.2f}'.format(x))
        ylength.set_data([x,x], [0,y])
        ylength.set_label('y = {:.02f}'.format(y))
        ytext.set_position((x+0.05, y/2))
        ytext.set_text('y = {:.2f}'.format(y))
        return [line,text,xangle,xlength,xtext,ylength,ytext]

    
#     plt.plot([0,x], [0,y],'r',label = '$[x,y]=[sin(\\theta), cos(\\theta)]$')
#     plt.plot(0.1*np.cos(thetavec), 0.1*np.sin(thetavec))
#     plt.text(0.13*np.cos(np.pi/4), 0.1*np.sin(np.pi/4),'$\\theta = $'+'{:.2f} radians'.format(theta))

#     # x length
#     plt.plot([0,x], [y,y],'g',label = 'x = {:.02f}'.format(x))
#     plt.text(x/2, y+0.05,'x')
#     # y length
#     plt.plot([x,x], [0,y],'b',label = 'y = {:.02f}'.format(y))
#     plt.text(x+0.05, y/2,'y')

#     plt.plot([],[],label = '$\\sqrt{x^2+y^2}=$'+'{:.2f}'.format(np.sqrt(x**2+y**2)))

#     # axis stuff
#     plt.axis('off')
#     plt.xlim([-1.1,1.1])
#     plt.ylim([-1.1,1.1])

#     plt.legend(loc='center left', bbox_to_anchor= (1.0, 0.5), ncol=1, borderaxespad=0.5, frameon=False)
    
    
    
    ani = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=100, interval=100, blit=True)
    return HTML(ani.to_jshtml())
    
    

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
        plt.text(0.13*np.cos(np.pi/4), 0.1*np.sin(np.pi/4),'$\\theta_1 = $'+'{:.2f} radians'.format(theta))
        
        plt.plot([0,x2], [0,y2],'b',label = '$B= [sin(\\theta_2), cos(\\theta_2)]$')
        plt.plot(0.2*np.cos(thetavec2)*(1+0.05*thetavec2), 0.2*np.sin(thetavec2)*(1+0.05*thetavec2), 'b')
        plt.text(0.23*np.cos(np.pi/4), 0.23*np.sin(np.pi/4),'$\\theta_2 = $'+'{:.2f} radians'.format(theta2))
        
        
        plt.plot([],[],label = '${A}\cdot B = $'+'{:.2f}'.format(x*x2+y*y2))
        
        # axis stuff
#         plt.axis('off')
        plt.xlim([-1.1,1.1])
        plt.ylim([-1.1,1.1])
        plt.legend(loc='center left', bbox_to_anchor= (1.0, 0.5), ncol=1, borderaxespad=0.5, frameon=False)
        plt.plot
        
    interactive_plot = interactive(f, theta=FloatSlider(min=0, max=2*np.pi, step=1e-4, continuous_update=False, description="$\\theta$", value=4), del_theta = FloatSlider(min=-np.pi, max=np.pi, step=1e-4, continuous_update=False, description="$\\theta_2 - \\theta_1$", value=0.3))
    output = interactive_plot.children[-1]
#     output.layout.height = '650px'
    return interactive_plot