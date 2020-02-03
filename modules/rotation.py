"""rotation
----------
This script contains helper functions for the rotation notebook.


Functions:
-----------

    animate_plot_1()
        Creates an animation of a vector rotating in Euclidian Space.
    
    animate_2_euclidian_vedctors()
        Creates two animations showing two rotating vectors in Euclidian spacetime.
"""



import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interactive, FloatSlider
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import pandas as pd


from modules.lorentz import findnearest
#------------------------------------------------------------ Implemented --------------------------------------------------

def animate_plot_1():
    """Creates an animation of a vector rotating in Euclidian Space."""
    fig, ax = plt.subplots(figsize=(10,10))
    plt.plot([-2,2],[0,0],'k', alpha = 0.1)
    plt.plot([0,0],[-2,2],'k', alpha = 0.1)
    line, = ax.plot([],[],'r', label = '$[x,y]=[sin(\\theta), cos(\\theta)]$')
    xangle, = ax.plot([],[],'r')
    text = plt.text(0.14*np.cos(np.pi/4), 0.14*np.sin(np.pi/4),'', size= 15)
    plt.text(-0.9, 0.95,'L = $\\sqrt{x^2+y^2} = 1$', size= 15)
    
    # x length
    xlength, = plt.plot([], [],'g')
    xtext = plt.text(0, 0,'x', size= 15)
    # y length
    ylength, = plt.plot([], [],'b')
    ytext = plt.text(0, 0,'y', size= 15)

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

    
    ani = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=100, interval=100, blit=True)
    return HTML(ani.to_jshtml())
    
def animate_2_euclidian_vedctors():
    """ Creates two animations showing two rotating vectors in Euclidian spacetime."""
    def init():
        line_1.set_data([], [])
        xangle_1.set_data([],[])
        text_1.set_text('')
        
        line_2.set_data([], [])
        xangle_2.set_data([],[])
        text_2.set_text('')
        return [line_1,text_1,xangle_1, 
                line_2,text_2,xangle_2]
    
    def animate_constant_delta(theta):
        theta = theta *np.pi/50
        del_theta = 0.6
        theta2 = (theta + del_theta) % (2*np.pi)
        thetavec = np.linspace(0,theta)
        thetavec2 = np.linspace(0,theta2)
        x = np.cos(theta)
        y = np.sin(theta)
        x2 = np.cos(theta2)
        y2 = np.sin(theta2)
#         plt.legend(loc='center left', bbox_to_anchor= (1.0, 0.5), ncol=1, borderaxespad=0.5, frameon=False)

        line_1.set_data([0,x], [0,y])
        xangle_1.set_data(0.1*np.cos(thetavec), 0.1*np.sin(thetavec))
        text_1.set_text('$\\theta_1 = $'+'{:.2f} radians'.format(theta))
        
        line_2.set_data([0,x2], [0,y2])
        xangle_2.set_data(0.2*np.cos(thetavec2)*(1+0.05*thetavec2), 0.2*np.sin(thetavec2)*(1+0.05*thetavec2))
        text_2.set_text('$\\theta_2 = $'+'{:.2f} radians'.format(theta2))
        
        dot_text.set_text('$u_1 \cdot u_2 = x_1x_2 + y_1y_2 ='+' ${:.2f}'.format(x*x2 + y*y2))

        
        return [line_1,text_1,xangle_1, 
                line_2,text_2,xangle_2]
    
    def animate_changing_delta(del_theta):
        del_theta = del_theta *np.pi/50
        theta = 1.2
        theta2 = (theta + del_theta) % (2*np.pi)
        thetavec = np.linspace(0,theta)
        thetavec2 = np.linspace(0,theta2)
        x = np.cos(theta)
        y = np.sin(theta)
        x2 = np.cos(theta2)
        y2 = np.sin(theta2)
#         plt.legend(loc='center left', bbox_to_anchor= (1.0, 0.5), ncol=1, borderaxespad=0.5, frameon=False)

        line_1.set_data([0,x], [0,y])
        xangle_1.set_data(0.1*np.cos(thetavec), 0.1*np.sin(thetavec))
        text_1.set_text('$\\theta_1 = $'+'{:.2f} radians'.format(theta))
        
        line_2.set_data([0,x2], [0,y2])
        xangle_2.set_data(0.2*np.cos(thetavec2)*(1+0.05*thetavec2), 0.2*np.sin(thetavec2)*(1+0.05*thetavec2))
        text_2.set_text('$\\theta_2 = $'+'{:.2f} radians'.format(theta2))
        
        
        dot_text.set_text('$u_1 \cdot u_2 = x_1x_2 + y_1y_2 ='+' ${:.2f}'.format(x*x2 + y*y2))
        
        return [line_1,text_1,xangle_1, 
                line_2,text_2,xangle_2]
    
    fig, ax = plt.subplots(figsize=(10,10))
    plt.plot([-2,2],[0,0],'k', alpha = 0.1)
    plt.plot([0,0],[-2,2],'k', alpha = 0.1)
    line_1, = ax.plot([],[],'r', label = '$[x,y]=[sin(\\theta_1), cos(\\theta_1)]$')
    xangle_1, = ax.plot([],[],'r')
    text_1 = plt.text(0.15*np.cos(np.pi/4), 0.05*np.sin(np.pi/4),'', size= 15)
    
    line_2, = ax.plot([],[],'b', label = '$[x,y]=[sin(\\theta_2), cos(\\theta_2)]$')
    xangle_2, = ax.plot([],[],'b')
    text_2 = plt.text(0.23*np.cos(np.pi/4), 0.23*np.sin(np.pi/4),'', size= 15)
    
    dot_text = plt.text(-1.0, 1.0,'', size= 15)
    
    # axis stuff
    plt.axis('off')
    plt.xlim([-1.1,1.1])
    plt.ylim([-1.1,1.1])
    
    ani_constant_delta = animation.FuncAnimation(fig, animate_constant_delta, init_func=init,
                                   frames=100, interval=100, blit=True)
    ani_changing_delta = animation.FuncAnimation(fig, animate_changing_delta, init_func=init,
                                   frames=100, interval=100, blit=True)
    return (HTML(ani_constant_delta.to_jshtml()), HTML(ani_changing_delta.to_jshtml()))


def Minkowski_2_vectors_animate(vec1 = [0,9], vec2 = [0,7], udelta = 0):
    """Creates an animation showing how regularly spaced events move through space for a moving observer with hyperbolae."""
    time=np.linspace(-6,20,100)
    space=np.linspace(-20,20,100)
    line1=np.linspace(-20,20,100)
    line2=np.linspace(20,-20,100)
    line5 = pd.read_hdf('data/lz_line5.hdf', 'line5')
    line6 = pd.read_hdf('data/lz_line6.hdf', 'line6')
    
    def datagen(u=0.75):
        while u > -1:
            u -= 0.05
            yield u
    
    def init():
        l1.set_data(space, line1)
        l2.set_data(space, line2)
        ax.set_xlim(-20,20)
        ax.set_ylim(-2,20)
    values = []
    
    def run(u, udelta, values):
        c = 1
#         u = u*c
#         udelta = udelta*c
        u2 = (u + udelta) / (1 + (u*udelta)/c**2)
#         u2 = u + udelta
        
        x1, t1 = np.dot(vec1, lorentz(u))
        x2, t2 = np.dot(vec2, lorentz(u2))
        
        values.append(x1*x2-t1*t2)
        
        l3.set_data([0,x1],[0,t1])
        l4.set_data([0,x2],[0,t2])
        text.set_text('$u_1$ = {:.2f}c\n$u_2$ = {:.2f}c\n$A\\cdot B$ = {:.2f}'.format(u/c,u2/c, x1*x2-t1*t2))
    
    
    fig, ax = plt.subplots(figsize =(10,7))
    ax.set_xlabel('distance')
    ax.set_ylabel('time')
    
    velocities=np.linspace(-0.999,0.999,2001)
    lines = [np.zeros((len(velocities),2))] * 10
    for j in range(len(lines)):
        for ii in range(len(velocities)):
            vel=velocities[ii]
            gamma=1.0/np.sqrt(1.0-vel*vel)
            lines[j][ii] = np.dot(lorentz(vel),np.array([j,0]))
        plt.plot(lines[j][:,1], lines[j][:,0],linewidth=1,color='black',alpha=0.5)
        
        
    text = plt.text(10,3,'$u$ = {:.2f}'.format(0.1), size = 20)
    l1, = ax.plot([], [], lw=1,color='red')
    l2, = ax.plot([], [], lw=1,color='red')
    l3, = ax.plot([], [], '-o', lw=3, color = 'blue')
    l4, = ax.plot([], [], '-o', lw=3, color = 'green')
    
    ani = animation.FuncAnimation(fig, run, datagen, blit=False, interval=100,
                              repeat=True, init_func=init, fargs = [udelta, values])
    return HTML(ani.to_jshtml())

def lorentz(v):
    """De=fines the Lorentz transformation as a 2x2 matrix."""
    gamma=1.0/np.sqrt(1-v**2)
    return np.array([[gamma,-gamma*v],[-gamma*v,gamma]])

def lorentz2(v):
    """De=fines the Lorentz transformation as a 2x2 matrix."""
    c=3e8
    gamma=1.0/np.sqrt(1-v*v/3e8)
    return np.array([[gamma,-gamma*v],[-gamma*v,gamma]])

#------------------------------------------------------------ WIP ----------------------------------------------------------



#------------------------------------------------------------ Currently Unused ---------------------------------------------

# def plot_euclidian_vector():
    
#     def f(theta):
#         thetavec = np.linspace(0,theta)
#         x = np.cos(theta)
#         y = np.sin(theta)
        
#         fig, ax = plt.subplots(figsize=(10,10))
#         plt.plot([-2,2],[0,0],'k', alpha = 0.1)
#         plt.plot([0,0],[-2,2],'k', alpha = 0.1)
#         plt.plot([0,x], [0,y],'r',label = '$[x,y]=[sin(\\theta), cos(\\theta)]$')
#         plt.plot(0.1*np.cos(thetavec), 0.1*np.sin(thetavec))
#         plt.text(0.13*np.cos(np.pi/4), 0.1*np.sin(np.pi/4),'$\\theta = $'+'{:.2f} radians'.format(theta))
        
#         # x length
#         plt.plot([0,x], [y,y],'g',label = 'x = {:.02f}'.format(x))
#         plt.text(x/2, y+0.05,'x')
#         # y length
#         plt.plot([x,x], [0,y],'b',label = 'y = {:.02f}'.format(y))
#         plt.text(x+0.05, y/2,'y')
        
#         plt.plot([],[],label = '$\\sqrt{x^2+y^2}=$'+'{:.2f}'.format(np.sqrt(x**2+y**2)))
        
#         # axis stuff
# #         plt.axis('off')
#         plt.xlim([-1.1,1.1])
#         plt.ylim([-1.1,1.1])
        
#         plt.legend(loc='center left', bbox_to_anchor= (1, 0.5), ncol=1, borderaxespad=0.5, frameon=False)
#         plt.plot
        
#     interactive_plot = interactive(f, theta=FloatSlider(min=0, max=2*np.pi, step=1e-4, continuous_update=False, description="$\\theta$", value=np.pi/4))
#     output = interactive_plot.children[-1]
# #     output.layout.height = '650px'
#     return interactive_plot


# def plot_2_euclidian_vectors():
    
#     def f(theta, del_theta):
#         theta2 = (theta + del_theta) % (2*np.pi)
        
#         thetavec = np.linspace(0,theta)
#         thetavec2 = np.linspace(0,theta2)
#         x = np.cos(theta)
#         y = np.sin(theta)
#         x2 = np.cos(theta2)
#         y2 = np.sin(theta2)
                
#         fig, ax = plt.subplots(figsize=(10,10))
#         plt.plot([-2,2],[0,0],'k', alpha = 0.1)
#         plt.plot([0,0],[-2,2],'k', alpha = 0.1)
        
        
#         plt.plot([0,x], [0,y],'r',label = '${A}=[sin(\\theta_1), cos(\\theta_1)]$')
#         plt.plot(0.1*np.cos(thetavec), 0.1*np.sin(thetavec), 'r')
#         plt.text(0.13*np.cos(np.pi/4), 0.1*np.sin(np.pi/4),'$\\theta_1 = $'+'{:.2f} radians'.format(theta))
        
#         plt.plot([0,x2], [0,y2],'b',label = '$B= [sin(\\theta_2), cos(\\theta_2)]$')
#         plt.plot(0.2*np.cos(thetavec2)*(1+0.05*thetavec2), 0.2*np.sin(thetavec2)*(1+0.05*thetavec2), 'b')
#         plt.text(0.23*np.cos(np.pi/4), 0.23*np.sin(np.pi/4),'$\\theta_2 = $'+'{:.2f} radians'.format(theta2))
        
        
#         plt.plot([],[],label = '${A}\cdot B = $'+'{:.2f}'.format(x*x2+y*y2))
        
#         # axis stuff
# #         plt.axis('off')
#         plt.xlim([-1.1,1.1])
#         plt.ylim([-1.1,1.1])
#         plt.legend(loc='center left', bbox_to_anchor= (1.0, 0.5), ncol=1, borderaxespad=0.5, frameon=False)
#         plt.plot
        
#     interactive_plot = interactive(f, theta=FloatSlider(min=0, max=2*np.pi, step=1e-4, continuous_update=False, description="$\\theta$", value=4), del_theta = FloatSlider(min=-np.pi, max=np.pi, step=1e-4, continuous_update=False, description="$\\theta_2 - \\theta_1$", value=0.3))
#     output = interactive_plot.children[-1]
# #     output.layout.height = '650px'
#     return interactive_plot
