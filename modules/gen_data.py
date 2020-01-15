"""
This Script is for generating data for the plotting scripts so that everything should be precomputed.
"""

import xarray as xr
import numpy as np
import pandas as pd


def lorentz_transform(v):
    """Defines the Lorentz transformation as a 2x2 matrix"""
    gamma=1.0/np.sqrt(1-v*v)
    return np.array([[gamma,-gamma*v],[-gamma*v,gamma]])

def findnearest(array, value):
    idx = np.abs(array - value).argmin()
    return array[idx]


def lorentz():
    """Constructs the lines needed for the lorentz plotting."""
    # Time and space are the base region for the spacetime plots.
    time=np.linspace(-6,20,100)
    space=np.linspace(-20,20,100)
    np.savetxt("data/lz_time.csv", time, delimiter=",")
    np.savetxt("data/lz_space.csv", space, delimiter=",")
    
    # Line 1 and Line 2 form the light cone.
    line1=np.linspace(-20,20,100)
    line2=np.linspace(20,-20,100)
    np.savetxt("data/lz_line1.csv", line1, delimiter=",")
    np.savetxt("data/lz_line2.csv", line2, delimiter=",")
    
    # Line 3 and Line 4 are reference lines for a flashing lighthouse
    line3=np.zeros(11)
    line4=np.linspace(0,10,11)
    np.savetxt("data/lz_line3.csv", line3, delimiter=",")
    np.savetxt("data/lz_line4.csv", line4, delimiter=",")
    
    U = np.arange(-0.999,0.999,0.001)
    
    line5=np.zeros([len(line3),len(U)])
    line6=np.zeros([len(line3),len(U)])
    
    for jj in range(len(U)):
        u = U[jj]
        for ii in range(len(line3)):
            point=np.array([line4[ii],line3[ii]])  #remember that time is the first element.
            point=np.dot(lorentz_transform(u),point)   #dot does matrix multiplication
            line5[ii,jj]=point[0]
            line6[ii,jj]=point[1]
    
    line5 = pd.DataFrame(line5, index = range(len(line5)), columns = U)
    line5.to_hdf('data/lz_line5.hdf', 'line5')
    line6 = pd.DataFrame(line6, index = range(len(line6)), columns = U)
    line6.to_hdf('data/lz_line6.hdf', 'line6')
    
    
    
    


if __name__ == '__main__':
    lorentz()