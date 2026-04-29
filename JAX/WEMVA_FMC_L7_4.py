## Import the Core Utilities of this Project
# Includes HelperFunctions.py for Timing, Plotting, and Debugging Tools + NumPy
from ImagingFunctionsFMC import * # Differentiable RTM / WEMVA
from OptimizationFunctions import * # Optimization Utilities

## Define Parameters
dov = 45e-3; # Max Depth [m]    
upsamp_x = 2; # Upsampling in x (assuming dx = pitch) - must be integer
upsamp_z = 1; # Upsampling in z (assuming dz = lambda/2) - can be fractional
Nx0 = 160; # Number of Points Laterally in x
cbfm = 1540; # initial sound speed [m/s]
fTx = 5.5e6; # frequency [Hz]
fBW = 6.0e6; # bandwidth at half maximum [Hz]
pulse_cutoff = 1e-3; # passband cutoff
ord = 100; # Strength of Anti-Aliasing Window 
# Sound Speed Reconstruction Parameters
z_idx_start = 20; # Depth Start Index to Start Minimizing Residuals
depthCorrection = True; # Whether or Not to Perform Depth-Ambiguity Correction
max_iterations = 10; # Number of Iterations of Conjugate Gradient
roundoff_scaling = 1e-15; # Scaling Residuals to Avoid Over/Underflow Errors (e.g., Step Size Formula) 
useLineSearch = True; # use line search (true); linearized step size (false)
max_iterations_golden_section = 10; # maximum iterations of golden-section search
blur_z_FWHM_wvlens = 8; # blurring length over z [in wavelengths]
blur_x_FWHM_wvlens = 8; # blurring length over x [in wavelengths]
layerPrior = 0.99; # Ranging from 0 (no prior) to 1 (fully layered)
# Parameters for Distortion Matrix / Subsurface-Offset Extended RTM Approach
useDistortionMatrixMethod = True; # Use Distortion Matrix Method (True); Use Image Differences (False)
FWHM_SubsurfaceOffset_Wvlens = 4; # Full-Width at Half-Max (FWHM) in Subsurface Offset [in Wavelengths]
dxmax = np.inf; # Maximum Subsurface Offset [m] 
    # can make dxmax small to reduce number of entries in residual (reduce memory storage)
    # but if dxmax is too small, may need more exact line searches (useLineSearch = True)
# B-Mode Image Parameters/Imaging Window
dBrange = 60; # Dynamic Range [decibels]
rng_min = 1400; rng_max = 1700; # Sound Speed Display Range [m/s]

# Load File and Original k-Wave Simulation Grid
load_filename = '../Datasets/SuperficialAbdominalLayersL7-4.mat'; 
dataset = loadmat_hdf5(load_filename); 
rxAptPos = dataset['rxAptPos']
time = dataset['time'][0]
fsr_dataset_fund = dataset['fsr_dataset_fund']; 
del dataset; 
nT, nRx, nTx = fsr_dataset_fund.shape; 
tx_elmts = np.arange(nTx); 
rxdata_h = fsr_dataset_fund[:,:,tx_elmts]; 
del fsr_dataset_fund; 

## Create Simulation, Image Reconstruction, and Sound Speed Grids
lambda_ = cbfm/fTx; # wavelength [m]
pitch = np.mean(np.diff(rxAptPos[:,0])) # number of elements
no_elements = rxAptPos.shape[0] # element spacing [m]
xpos = pitch*np.arange(-(no_elements-1)/2,1+(no_elements-1)/2); # element position [m]
# Grid for Simulation
x = (pitch/upsamp_x)*np.arange(-(upsamp_x*Nx0-1)/2,1+(upsamp_x*Nx0-1)/2); # m
Nu1 = np.round(dov/((lambda_/2)/upsamp_z)); 
z = (np.arange(Nu1))*(lambda_/2)/upsamp_z; # m
xmax = (np.max(np.abs(xpos))+np.max(np.abs(x)))/2; # Lateral Cutoff for Anti-Aliasing [m]
aawin = 1/np.sqrt(1+(x/xmax)**ord); # Anti-Aliasing Window
# Initial Slowness Guess
clayers_const = cbfm*np.ones((z.size-1, x.size)); 
sLayers_const = 1/clayers_const; 

## Frequency-Domain Representation of Transmitted and Received Wavefields
# Transmit Impulse Response in Frequency Domain
nt = time.size; # [s]
fs = 1/np.mean(np.diff(time)); # [Hz] 
f = (fs/2)*np.arange(-1,1,2/nt); # [Hz]
# Only Keep Positive Frequencies within Passband
passband_f_idx = np.argwhere(np.logical_and(f>=fTx-fBW/2, f<=fTx+fBW/2)).flatten(); 
f = f[passband_f_idx]; 
# Get Full-Matrix Capture (FMC) Receive Channel Data in the Frequency Domain
FMCdata_f = np.fft.fftshift(np.fft.fft(rxdata_h, n=nt, axis=0), axes=0);
FMCdata_f = FMCdata_f[passband_f_idx,:,:]; del rxdata_h; 
FMCdata_f = FMCdata_f * np.exp(-1j*2*np.pi*f[:,np.newaxis,np.newaxis]*time[0]); 
# Pulsed-Wave Frequency Response for Each Element at Transducer Surface
apod = np.eye(no_elements); 
elemresp_f_xdcr = np.zeros((x.size, f.size, tx_elmts.size), dtype=np.csingle); 
for tx_idx in np.arange(tx_elmts.size): 
    # Construct Transmit Responses for Each Element
    apod_x = np.zeros(x.size); 
    for xpos_idx in np.arange(xpos.size):
        start_idx = int((xpos_idx+(Nx0-no_elements)/2)*upsamp_x); 
        apod_x[start_idx:start_idx+upsamp_x] = apod[tx_idx,xpos_idx]; 
    elemresp_f_xdcr[:,:,tx_idx] = apod_x[:,np.newaxis]; 

# Blurring Kernels In Terms of Wavelengths
dx, dz = np.mean(np.diff(x)), np.mean(np.diff(z)); # Grid Spacing [m]
blur_z = np.hanning(int(2*blur_z_FWHM_wvlens*lambda_/dz)); # Z - Blurring Over Reconstruction Grid
blur_x = np.hanning(int(2*blur_x_FWHM_wvlens*lambda_/dx)); # X - Blurring Over Reconstruction Grid
blur_z = blur_z / np.sum(blur_z); blur_x = blur_x / np.sum(blur_x); # Normalization

# Preconditioner/Regularization as Modifications (Blurring + Layerization) to the Gradient Image
blur = lambda grad: convolve2d(grad, np.outer(blur_z, blur_x), mode='same', boundary='symm');  
layerize = lambda grad: (1-layerPrior)*grad+layerPrior*np.mean(grad,axis=1)[:,None]; 
precond = lambda img: layerize(blur(img)); # Symmetric and Positive Definite Preconditioner

# Whether or Not to Perform Depth-Ambiguity Correction
if depthCorrection:
    zshifted = jax.jit(lambda sLayers: jnp.cumsum(jnp.hstack((0,jnp.diff(jnp.copy(z))/(cbfm*jnp.mean(sLayers,axis=1))))))
else:
    zshifted = jax.jit(lambda sLayers: jnp.copy(z))

## Residual Functions for Aberration Correction + Slowness Estimation 
# Distortion Matrix / Subsurface-Offset Extended RTM Approach
resDistortionMatrix = createResidualFunction_SubsurfaceOffset(x, FWHM_SubsurfaceOffset_Wvlens*lambda_, dxmax); 
# Direct Differences Between Migrated Images from Consecutive Receivers - Using Lag-One Optimal TX Apodization
resImageDifferences = createResidualFunction_ImageDifferences(optApod(no_elements,1)); 
# Select Residual Function
if useDistortionMatrixMethod: 
    resFunction = resDistortionMatrix; 
else:
    resFunction = resImageDifferences; 

# Time Gain Compensation for Displaying B-Mode Images
TGC = np.ones_like(z)[1::,np.newaxis]; 

## Create Least-Squares Objective Function for Slowness Reconstruction
# Least-Squares Objective Function for Memory-Efficient Reverse Mode AD
RTM = reverseTimeMigrationV3(x, f, aawin, elemresp_f_xdcr, FMCdata_f, resFunction); 
residual = jax.jit(lambda sLayers: roundoff_scaling*(TGC*RTM(zshifted(sLayers), jnp.copy(sLayers)))[z_idx_start:,:]); 
obj = jax.jit(lambda sLayers: jnp.sum(jnp.abs(residual(sLayers))**2)/2); 
# Equivalent Least-Squares Objective Function for Fast Evaluation of Objective Function / Forward Mode AD
RTM_Fast = reverseTimeMigrationV1(x, f, aawin, elemresp_f_xdcr, FMCdata_f, resFunction); 
residual_fast = jax.jit(lambda sLayers: roundoff_scaling*(TGC*RTM_Fast(zshifted(sLayers), jnp.copy(sLayers)))[z_idx_start:,:]); 
obj_fast = jax.jit(lambda sLayers: jnp.sum(jnp.abs(residual_fast(sLayers))**2)/2); 
# Fast RTM Reconstruction of B-Mode Image + Evaluation of Focusing Criteria
RTM_Image_Recon_Fast = reverseTimeMigrationV1(x, f, aawin, elemresp_f_xdcr, FMCdata_f, imaging); 
img_recon_fast = jax.jit(lambda sLayers: TGC*RTM_Image_Recon_Fast(zshifted(sLayers), jnp.copy(sLayers))); 

# Reconstruct Image Using Constant Sound Speed
start = timer.time(); 
img_recon_const = img_recon_fast(sLayers_const); 
end = timer.time(); 
print(f"B-Mode Image (Estimated Sound Speed) Execution time: {end - start:.6f} seconds")

# Initialize Stuff to be Recorded
cLayers_estim_history = np.zeros((z.size-1, x.size, max_iterations+1), dtype=np.csingle); 
cLayers_estim_history[:,:,0] = 1/sLayers_const; 
img_recon_history = np.zeros((z.size-1, x.size, max_iterations+1), dtype=np.csingle); 
img_recon_history[:,:,0] = img_recon_const; 

# Conjugate Gradient Method (with Exact Line Search Step Size Calculation)
sLayers_estim = jnp.copy(sLayers_const); 
plt.ion(); plt.figure(figsize=(9,9))
for iteration in range(max_iterations):
    print('Iteration '+str(iteration)); 
    # 1) Compute Gradient
    start = timer.time(); 
    grad = jax.grad(obj)(sLayers_estim); 
    end = timer.time(); 
    print(f"Gradient Evaluation: {end - start:.6f} seconds"); 
    # Layered Medium Preconditioning of the Gradient
    precond_grad = precond(grad); # Blurring/Smoothing Over Gradient
    if iteration == 0:
        # 3) Calculate Search Direction
        search_dir = -precond_grad; 
    else:
        # 2) Calculate Momentum (Polak-Ribiere Formula)
        momentum = (np.vdot(grad.flatten(), (precond_grad.flatten()-precond_grad_prev.flatten())) / 
                    np.vdot(grad_prev.flatten(), precond_grad_prev.flatten())); # Polak-Ribiere Formula
        # 3) Calculate Search Direction
        search_dir = momentum*search_dir-precond_grad; 
    # 4) Step Size
    # Initial Step Size Based on Linearized Change in the Residual (Calculated Using JVP)
    start = timer.time(); 
    dres = jax.jvp(residual_fast,(sLayers_estim,),(search_dir,))[1]; 
    stepsize_init = -(np.vdot(grad.flatten(), search_dir.flatten()) / 
                      np.real(np.vdot(dres.flatten(), dres.flatten()))); 
    end = timer.time(); 
    print(f"Linearized Step Size Calculation: {end - start:.6f} seconds")
    print('Linearization-Based Step Size: '+str(stepsize_init)); 
    # Line Search to Determine Optimal Step Size
    if useLineSearch: # Line Search for Step Size Calculation
        start = timer.time(); 
        line_search_objective = lambda stepsize: obj_fast(sLayers_estim+stepsize*search_dir); # Line search objective function
        stepsize = line_search(line_search_objective, stepsize_init, max_iter_golden_section=max_iterations_golden_section); # Line search
        stepsize_ratio = stepsize/stepsize_init; 
        end = timer.time(); 
        print(f"Step Size Calculation: {end - start:.6f} seconds")
        print('Line Search Step Size: '+str(stepsize)); 
        print('Step Size Ratio: '+str(stepsize_ratio));
    else: # Just Use Linearized Step Size Calculation
        stepsize = stepsize_init
    # 5) Update
    sLayers_estim = sLayers_estim + stepsize * search_dir; 
    # Record Previous Gradient
    precond_grad_prev = precond_grad; grad_prev = grad; 
    # Reconstruct Image
    start = timer.time(); 
    img_recon = img_recon_fast(sLayers_estim)
    end = timer.time();  
    print(f"Image Reconstruction: {end - start:.6f} seconds")
    # Plot Image
    plt.clf(); 
    plt.subplot2grid((2, 2), (0, 0)); 
    imagesc(x, z[1::], dispImage(img_recon_const), rng=[-dBrange,0], numticks=(0, 0), cmap='gray'); 
    plt.title('Initial Aberrated B-Mode Image'); 
    plt.xlabel('x [m]'); plt.ylabel('z [m]'); 
    plt.subplot2grid((2, 2), (0, 1)); 
    imagesc(x, z[1::], dispImage(img_recon), rng=[-dBrange,0], numticks=(0, 0), cmap='gray'); 
    plt.title('Corrected Image at Iteration '+str(iteration)); 
    plt.xlabel('x [m]'); plt.ylabel('z [m]'); 
    # Plot Constant Sound Speed
    clayers_estim = 1/sLayers_estim; # sound speed [m/s]
    plt.subplot2grid((2, 2), (1, 0)); 
    imagesc(x, z[1::], clayers_estim, rng=[rng_min, rng_max], numticks=(0, 0), cmap='cubehelix'); 
    plt.xlabel('x [m]'); plt.ylabel('z [m]'); plt.title('Estimated Sound Speed [m/s]'); 
    plt.colorbar();
    # Plot Estimated Sound Speed - Based on Estimated Aberration Delays
    plt.subplot2grid((2, 2), (1, 1)); 
    imagesc(x, z[1::], clayers_estim, rng=[rng_min, rng_max], numticks=(0, 0), cmap='cubehelix'); 
    imagesc(x, z[1::], dispImage(img_recon), rng=[-dBrange,0], numticks=(0, 0), cmap='gray', alpha = 0.4); 
    plt.xlabel('x [m]'); plt.ylabel('z [m]'); plt.title('Sound Speed + Corrected B-Mode ');     
    plt.show(); plt.pause(1); 
    # Save Data / Figures
    plt.savefig('../Figures/SuperficialLayers/Recon'+str(iteration)+'.png'); 
    # Stuff to be Recorded
    cLayers_estim_history[:,:,iteration+1] = clayers_estim; 
    img_recon_history[:,:,iteration+1] = img_recon; 

# Save Final MAT File with Full Image Reconstruction History
data_to_save = {
    'cLayers_estim_history': cLayers_estim_history,
    'img_recon_history': img_recon_history,
    'x': x, 'z': z[1::]
}
savemat('../Figures/SuperficialLayers/FullHistory.mat', data_to_save); 
