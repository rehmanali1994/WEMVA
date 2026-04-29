# Import Timing, Plotting, and Debugging Tools
import time as timer # Timing
import matplotlib.pyplot as plt # Plotting
import pdb # Debugging

# Import Libraries to Implement MATLAB-Equivalent Functions
import numpy as np # NumPy
from scipy.interpolate import RegularGridInterpolator # Similar to interp2 in MATLAB
from scipy.signal import convolve2d # Similar to MATLAB Signal Processing Toolbox
from scipy.io import savemat # Saves MAT files (NOT using compression like -v7.3)
import h5py # Need to Load -v7.3 MAT files for Raw Channel Data

# MATLAB-Equivalent Functions
db = lambda value: 20*np.log10(value); 
dispImage = lambda IMG: db(np.abs(IMG)/np.max(np.abs(IMG))); 

# Define Loadmat Function for HDF5 Format ('-v7.3' in MATLAB)
def loadmat_hdf5(filename):
    file = h5py.File(filename,'r')
    out_dict = {}
    for key in file.keys():
        out_dict[key] = np.ndarray.transpose(np.array(file[key])); 
    file.close()
    return out_dict; 

# Python-Equivalent Command for IMAGESC in MATLAB
def imagesc(x, y, img, rng=None, cmap='gray', numticks=(3, 3), aspect='equal', alpha=1.0):
    if rng == None:
        rng = [np.min(img), np.max(img)]; 
    exts = (np.min(x)-np.mean(np.diff(x)), np.max(x)+np.mean(np.diff(x)), \
        np.min(y)-np.mean(np.diff(y)), np.max(y)+np.mean(np.diff(y))); 
    plt.imshow(np.flipud(img), cmap=cmap, extent=exts, vmin=rng[0], vmax=rng[1], aspect=aspect, alpha=alpha); 
    plt.xticks(np.linspace(np.min(x), np.max(x), numticks[0])); 
    plt.yticks(np.linspace(np.min(y), np.max(y), numticks[1])); 
    plt.gca().invert_yaxis(); 