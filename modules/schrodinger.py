import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import odeint

start_position=-20.0
end_position=20   #set this to be the correct end point.
space_steps=2000
step=(end_position-start_position)/(space_steps-1)
space=np.linspace(start_position,end_position,space_steps)

def plot_potential(psi,psiprime):
    """Plots a potential function and its derivative.
    
    Parameters
    ----------
    psi : list-like, length 2000.
        The potential.
    psiprime : list-like, length 2000.
        The first derivative of the potential.

    Returns
    -------
    no value
    """
    # Create the matplotlib figure and plot functions
    plt.figure()
    plt.plot(space,psi,linewidth=3,label='wavefunction')
    plt.plot(space,psiprime,linewidth=3,label='derivative')
    plt.legend()
    plt.show()
    
def plot_potential_with_scaled(psi,psiprime, scaled_potential):
    """Plots a potential function and its derivative, as well as a scaled potential function.
    
    Parameters
    ----------
    psi : list-like, length 2000.
        The wavefunction.
    psiprime : list-like, length 2000.
        The first derivative of the wavefunction.
    scaled_potential : list-like, length 2000
        A scaled potential.

    Returns
    -------
    no value
    """
    # Create the matplotlib figure and plot functions
    plt.figure()
    plt.plot(space,psi,linewidth=3,label='wavefunction')
    plt.plot(space,psiprime,linewidth=3,label='derivative')
    plt.plot(space,scaled_potential/8+0.5,linewidth=1,label='scaled potential') #show the potential well as well
    plt.legend()
    plt.show()