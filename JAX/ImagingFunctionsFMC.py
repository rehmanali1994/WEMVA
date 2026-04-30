# Import Timing, Plotting, and Debugging Tools + NumPy
from HelperFunctions import * 

# Import JAX-Related Stuff
import jax # Import JAX Before Importing Any Submodules
import jax.numpy as jnp

# COMPLETE LIST OF HARD-CODED VALUES HERE
BATCH_SIZE = 16; # V2: Checkpoint 1 out of every BATCH_SIZE sample in depth (reverse-mode AD over batches in depth)
NUM_FREQ_CHUNKS = 8; # V3: Setting this equal to 1 is like using the V2 function without chunking over frequencies
# ANY PARAMETER IN V2 AFFECTS V3 BECAUSE V3 BUILDS ON V2

# Autodiff-Friendly Version of Absolute Value-Squared (jnp.abs(z)**2)
abs_sq = lambda z: jnp.real(z*jnp.conj(z)); 

# RTM Imaging Condition
def imaging(elemresp_f, FMCdata_f):
    '''IMAGING Constructs multistatic synthetic aperture image using element responses and FMC data
    img = imaging(elemresp_f, FMCdata_f)
        elemresp_f = X x F x E array of frequency responses at each location for each element at a given depth
            X is the dimension for x-location; F is the frequency dimension; E is the element dimension
        FMCdata_f = F x Rx x TX array of full-matrix capture (FMC) data
        img = X x 1 image of focused and summed multistatic synthetic aperture image'''
    # Define Einstein Summation for Total TX-RX Focusing
    return jnp.einsum('xfi,xfj,fij->x', elemresp_f, elemresp_f, FMCdata_f); 

# Create Objective Function for Aberration Correction with Simple TX Focusing + Lag-One Alignment on RX
def createResidualFunction_ImageDifferences(apodTX):
    '''Returns function that produces residuals at each depth (based on differences between receivers)
    residual = createResidualFunction_ImageDifferences(apodTX)
        img_diff = residual(elemresp_f, FMCdata_f)
            elemresp_f = X x F x E array of frequency responses at each location for each element at a given depth
                X is the dimension for x-location; F is the frequency dimension; E is the element dimension
            FMCdata_f = F x Rx x TX array of full-matrix capture (FMC) data
            img_diff = X x (Rx-1) image of TX focused multistatic synthetic aperture image'''
    
    # Define Layerwise Focused TX Imaging
    def focusedTransmitImaging(elemresp_f, FMCdata_f): 
        '''FOCUSEDTRANSMITIMAGING Constructs transmit-focused image using element responses and FMC data
        img = focusedTransmitImaging(elemresp_f, FMCdata_f)
            elemresp_f = X x F x E array of frequency responses at each location for each element at a given depth
                X is the dimension for x-location; F is the frequency dimension; E is the element dimension
            FMCdata_f = F x Rx x TX array of full-matrix capture (FMC) data
            img = X x Rx image of TX focused multistatic synthetic aperture image'''
        # Einstein Summation at Each Image Point + Residuals at Each Image Point
        return jnp.einsum('xfi,xfj,fij,j->xi', elemresp_f, elemresp_f, FMCdata_f, apodTX); 

    # Residuals for Lag-One Differential Alignment on RX
    def residual(elemresp_f, FMCdata_f):
        '''RESIDUAL Constructs difference between receive elements for transmit-focused images
        img_diff = residual(elemresp_f, FMCdata_f)
            elemresp_f = X x F x E array of frequency responses at each location for each element at a given depth
                X is the dimension for x-location; F is the frequency dimension; E is the element dimension
            FMCdata_f = F x Rx x TX array of full-matrix capture (FMC) data
            img_diff = X x (Rx-1) image of TX focused multistatic synthetic aperture image'''
        return jnp.diff(focusedTransmitImaging(elemresp_f, FMCdata_f), axis=1); 

    # Return Residual Function
    return residual; 

# Create Objective Function for Aberration Correction Using Distortion Matrix / Subsurface Offset
def createResidualFunction_SubsurfaceOffset(x, FWHM, dxmax):
    '''Returns function that produces residuals at each depth (based on penalizing subsurface offset)
    residual = createResidualFunction_SubsurfaceOffset(apodTX)
        weighted_distortion_matrix = residual(elemresp_f, FMCdata_f)
            elemresp_f = X x F x E array of frequency responses at each location for each element at a given depth
                X is the dimension for x-location; F is the frequency dimension; E is the element dimension
            FMCdata_f = F x Rx x TX array of full-matrix capture (FMC) data
            weighted_distortion_matrix = X x X weighted distortion matrix
                weighting used to penalize large subsurface offsets '''
    
    # Define Loss as Function of Distance / Subsurface Offset / Space-Lag
    loss_function_distance = lambda xdiff: np.tanh(xdiff/FWHM); 
    # Calculate Distance Dependent Weights for Distortion Matrix
    distance = np.abs(np.subtract.outer(x, x)); 
    periodic_distance = np.minimum(distance, (x[-1]-2*x[0]+x[1]) - distance); 
    loss_function_distortion_matrix = loss_function_distance(periodic_distance); 
    loss_function_window = (periodic_distance <= dxmax); 

    # RTM Imaging Condition for Distortion Matrix
    def distortionMatrix(elemresp_f, FMCdata_f):
        '''DISTORTIONMATRIX Constructs distortion matrix using element responses and FMC data
        distortion_matrix = distortionMatrix(elemresp_f, FMCdata_f)
            elemresp_f = X x F x E array of frequency responses at each location for each element at a given depth
                X is the dimension for x-location; F is the frequency dimension; E is the element dimension
            FMCdata_f = F x Rx x TX array of full-matrix capture (FMC) data
            distortion_matrix = X x X multistatic synthetic aperture focused distortion matrix'''
        # Define Einstein Summation for Distortion Matrix
        return jnp.einsum('afi,bfj,fij->ab', elemresp_f, elemresp_f, FMCdata_f); 

    # Residuals for Least-Squares Distortion Matrix Loss Function
    def distortionMatrixResidual(elemresp_f, FMCdata_f):
        '''DISTORTIONMATRIXRESIDUAL Constructs residuals to drive image to zero subsurface offset
        weighted_distortion_matrix = residual(elemresp_f, FMCdata_f)
            elemresp_f = X x F x E array of frequency responses at each location for each element at a given depth
                X is the dimension for x-location; F is the frequency dimension; E is the element dimension
            FMCdata_f = F x Rx x TX array of full-matrix capture (FMC) data
            weighted_distortion_matrix = X x X weighted distortion matrix
                weighting used to penalize large subsurface offsets'''
        # Define Residual for Loss Function
        return (loss_function_distortion_matrix*distortionMatrix(elemresp_f, FMCdata_f))[loss_function_window]; 

    # Return Residual Function
    return distortionMatrixResidual; 



# Optimal Apodization for Maximum Short Lag Spatial Coherence
def optApod(Nelem, Nlags, tol=np.finfo(float).eps):
    '''OPTAPOD Find apodization that maximizes short lag autocorrelation
    apod = optApod(Nelem,Nlags,tol,print)
        Nelem = number of elements in aperture
        Nlags = number of lags in optimization
        tol = (optional) numerical tolerance for optimization (default: tol = eps)'''
    
    # Laplacian with Homogeneous BCs
    LagNDiff = np.diag(2*Nlags*np.ones(Nelem));
    for lag in np.arange(1,Nlags+1):
        LagNDiff = LagNDiff - \
            np.diag(np.ones(Nelem-lag),k=lag) - \
            np.diag(np.ones(Nelem-lag),k=-lag);

    # Solve Ball-Exclusion Constrained Optimization Problem
    apod = np.random.randn(Nelem); # Initial Apodization
    prev_lambda = 0; curr_lambda = 1; # Lagrange Multiplier
    while (np.abs(np.linalg.norm(apod)-1) > tol) and (np.abs((curr_lambda-prev_lambda)/curr_lambda) > tol):
        prev_lambda = curr_lambda; # Update Previous Lagrange Multiplier
        apod = np.abs(apod)/np.linalg.norm(apod); # Project Onto Nonnegative Norm-2 Ball
        LagNDiffConstrained = np.vstack((np.hstack((LagNDiff, apod[:,np.newaxis])), np.hstack((apod, 0))));
        sol = np.linalg.solve(LagNDiffConstrained, np.hstack((np.zeros(Nelem), 1)));
        apod = sol[:Nelem]; # Extract Optimal Apodization
        curr_lambda = sol[-1] # New Lagrange Multiplier
    return apod	



# Return JAX-Efficient Function for Depth-Wise Downward Propagation Using Fourier Split-Step Method
def propagator(x, f, aawin):
    '''prop_fun = propagator(x, f, aawin)
    
    PROPAGATOR - Angular Spectrum Propagation of TX/RX Signals into the Medium
    
    This function return a JAX JIT'ed function (prop_fun) that 
    propagates transmit and receive wavefields at from one
    depth to another depth using the angular spectrum method
    
    INPUTS:
    x                  - 1 x X vector of x-grid positions for wavefield
    f                  - 1 x F vector of pulse frequencies in spectrum
    aawin              - 1 x X vector of lateral taper to prevent wraparound
    
    OUTPUT:
    prop_fun           - JAX JIT'ed function (prop_fun) that propagates
                         transmit and receive wavefields at from one
                         depth to another depth using the angular spectrum method
    
    USAGE:    
    wvfield_z2_f = prop_fun(z1, z2, s, wvfield_z1_f)
    
    INPUTS:
    z1                 - depth of input TX and RX wavefields
    z2                 - depth of output TX and RX wavefields
    s                  - 1 x X vector of slowness [s/m] between z1 and z2
    wvfield_z1_f       - X x F x N array of input wavefields at z1
    
    OUTPUT:
    wvfield_z2_f       - X x F x N array of output wavefields at z2 '''

    # Static Arguments Used by Propagation Function
    # Forward and Inverse Fourier Transforms with Anti-Aliasing Windows
    ft = lambda sig: jnp.fft.fft(aawin[:,None,None]*sig, axis=0); 
    ift = lambda sig: aawin[:,None,None]*jnp.fft.ifft(sig, axis=0); 
    # FFT Axis for Lateral Spatial Frequency
    dx = np.mean(np.diff(x)); nx = x.size; # Spatial Grid
    kx = np.mod(np.arange(nx)/(dx*nx)+1/(2*dx),1/dx)-1/(2*dx); dkx = 1/(dx*nx); 
    F, Kx = np.meshgrid(f*(1+1j*np.finfo(np.single).eps),kx); # Create Grid in f-kx
    # Added a small imaginary part to f to stabilize differentiation around kz = 0
    # Equivalent to adding small imaginary part to slowness (small attenuation)

    # Propagation Function
    def prop_fun(z1, z2, s, wvfield_z1_f):
        '''wvfield_z2_f = prop_fun(z1, z2, s, wvfield_z1_f)

        PROP_FUN - Angular Spectrum Propagation of TX/RX Signals into the Medium
        This function propagates transmit and receive wavefields at from one
        depth to another depth using the angular spectrum method
        
        INPUTS:
        z1                 - depth of input TX and RX wavefields
        z2                 - depth of output TX and RX wavefields
        s                  - 1 x X vector of slowness [s/m] between z1 and z2
        wvfield_z1_f       - X x F x N array of input wavefields at z1
        
        OUTPUT:
        wvfield_z2_f       - X x F x N array of output wavefields at z2 '''

        # Mean Slowness and Lateral Variation
        smean = jnp.mean(s); # Mean Slowness [s/m]
        ds = (s - smean)[:,jnp.newaxis]; # Lateral Variation in Slowness [s/m] Along x
        # Continuous Wave Response By Downward Angular Spectrum
        Kz2 = (F*smean)**2-Kx**2; # Axial Spatial Frequency - Squared
        Kz = jnp.sqrt(Kz2); # Axial Spatial Frequency 
        H = jnp.exp(1j*2*np.pi*Kz*(z2-z1)); # Propagation Filter
        dH = jnp.exp(1j*2*np.pi*F*ds*(z2-z1)); # Phase Screen
        # Apply Propagation Filter
        wvfield_z2_f = dH[:,:,None]*ift(H[:,:,None]*ft(wvfield_z1_f)); 
        return wvfield_z2_f; 
        
    # Return JAX-JIT'ed Propagation Function
    return jax.jit(prop_fun); 



## Image Reconstructipn by Reverse-Time Migration
# V1: Only meant for rapid image reconstruction & rapid evaluation of loss function
# NOT meant for use with jax.grad or jax.vjp due to high storage/rematerialization costs
def reverseTimeMigrationV1(x, f, aawin, elemresp_f_xdcr, FMCdata_f, output_func):
    '''RTM_func = reverseTimeMigration(x, f, aawin, elemresp_f_xdcr, FMCdata_f, output_func)
    
    REVERSETIMEMIGRATION - Return JAX JIT'ed Functor for Reverse-Time Migration (RTM)
    
    INPUTS:
    x                  - 1 x X vector of x-grid positions for wavefield
    f                  - 1 x F vector of pulse frequencies in spectrum
    aawin              - 1 x X vector of lateral taper to prevent wraparound
    elemresp_f_xdcr    - X x F x N array of input TX wavefields at transducer surface
    FMCdata_f          - F x N x N array of input full-matrix capture (FMC) data
    output_func        - functor applied at each depth: takes elemresp_f and FMCdata_f as input; 
                         outputs 1 x X image or single objective function value
    
    OUTPUT:
    RTM_func           - JAX JIT'ed function (RTM_func) that performs RTM

    USAGE:    
    img = RTM_func(z, sLayers)

    INPUTS:
    z                  - 1 x Z vector of depths for TX and RX wavefields (1st sample is transducer plane - usually z = 0)
    sLayers            - (Z - 1) x X vector of slowness [s/m] between depths in z
    
    OUTPUT:
    img                - Z x X array RTM image (or Z x 1 array of objective function values) '''

    # Verify the Number of Elements
    Nelem = elemresp_f_xdcr.shape[2]; 
    assert((FMCdata_f.shape[1] == Nelem) and (FMCdata_f.shape[2] == Nelem)), \
        'Number of sources must equal to number of RTM images'; 

    # Create Propagation Function
    prop = propagator(x, f, aawin); 

    # Checkpoint the Objective Function Output
    output_func_checkpointed = jax.checkpoint(lambda elemresp_f: output_func(elemresp_f, FMCdata_f)); 

    # Functor for Reverse Time Migration (RTM)
    def RTM_func(z, sLayers):
        '''img = RTM_func(z, sLayers)

        RTM_FUNC - Reconstructs Reverse-Time Migrated (RTM) Images Based on Given Sound Speed Map
        
        INPUTS:
        z                  - 1 x Z vector of depths for TX and RX wavefields (1st sample is transducer plane - usually z = 0)
        sLayers            - (Z - 1) x X vector of slowness [s/m] between depths in z
        
        OUTPUT:
        img                - Z x X array RTM image output (or Z x 1 array of objective function values) '''

        # Body of Propagation Loop
        def prop_scan_body(elemresp_f, z_idx):
            # Propagate Signals in Depth
            elemresp_f = prop(z[z_idx], z[z_idx+1], sLayers[z_idx,:], elemresp_f); 
            # Compute Image at this Depth + Pack Outputs
            return elemresp_f, output_func_checkpointed(elemresp_f).flatten(); 

        # Propagate Ultrasound Signals in Depth + Imaging 
        elemresp_f, img = jax.lax.scan(prop_scan_body, elemresp_f_xdcr, jnp.arange(z.size-1)); 
        return img.reshape(z.size-1, -1); 

    # Return JAX-Efficient Propagation Function
    return jax.jit(RTM_func); 



##############################################################################################################################
######                                                                                                                  ######
######-------------------------------------------POTENTIAL FUTURE WORK FOR V2-------------------------------------------######
######                                                                                                                  ######
###### Idea 1: Implement a custom_vjp with manual checkpointing for RTM_func - reverse scan - backwards in depth        ######
######                                                                                                                  ######
###### Idea 2: Within the custum_vjp, write a [potentially regularized] inverse for prop_fun (created by propagator)    ######
######         Eliminates the need for full forward rematerialization over batches from fixed checkpoints               ######
######         Instead, elemresp_f is inverse propagated upward during reverse-mode (no more rematerialization cost)    ######
######         However, the inaccuracy of the inversion may cause errors to accumulate in elemresp_f going upward       ######
######         This may necessitate checkpoints (at outputs instead of inputs) to refresh the accuracy of elemresp_f    ######
######         These *output checkpoints* only incur storage costs - no rematerialization cost/overhead to go backward  ######  
######         Storage-rematerialization tradeoff becomes a storage-accuracy tradeoff instead                           ######
######                                                                                                                  ######
##############################################################################################################################    



## Image Reconstructipn by Reverse-Time Migration - With Checkpointing Over Depth Batches (V2)
# APPLY CHECKPOINTING TO EACH BATCH (INNER SCAN) -- limits rematerialization cost to individual batches
#   Issue 1: Single batch (i.e., no batching) requires rematerializing full computation on reverse-mode AD
# LOOP THROUGH EACH BATCH (OUTER SCAN) -- only need to store the inputs/outputs of each batch
#   Issue 2: Z batches (i.e., each batch is 1 iteration) requires storing all intermediates
# Dilemma: Storage/rematerialization tradeoff (Issues 1 and 2 represent opposite extremes)
#   Solution: Balance the storage/rematerialization tradeoff by picking the right batch size
#   BIGGEST DECREASE IN MEMORY COST GOING FROM V1 TO V2 (BUT V2 IS SLOWER THAN V1) 
def reverseTimeMigrationV2(x, f, aawin, elemresp_f_xdcr, FMCdata_f, output_func):
    '''RTM_func = reverseTimeMigration(x, f, aawin, elemresp_f_xdcr, FMCdata_f, output_func)
    
    REVERSETIMEMIGRATION - Return JAX JIT'ed Functor for Reverse-Time Migration (RTM)
    
    INPUTS:
    x                  - 1 x X vector of x-grid positions for wavefield
    f                  - 1 x F vector of pulse frequencies in spectrum
    aawin              - 1 x X vector of lateral taper to prevent wraparound
    elemresp_f_xdcr    - X x F x N array of input TX wavefields at transducer surface
    FMCdata_f          - F x N x N array of input full-matrix capture (FMC) data
    output_func        - functor applied at each depth: takes elemresp_f and FMCdata_f as input; 
                         outputs 1 x X image or single objective function value

    OUTPUT:
    RTM_func           - JAX JIT'ed function (RTM_func) that performs RTM

    USAGE:    
    img = RTM_func(z, sLayers)

    INPUTS:
    z                  - 1 x Z vector of depths for TX and RX wavefields (1st sample is transducer plane - usually z = 0)
    sLayers            - (Z - 1) x X vector of slowness [s/m] between depths in z
    
    OUTPUT:
    img                - Z x 1 array of objective function values '''

    # Verify the Number of Elements
    Nelem = elemresp_f_xdcr.shape[2]; 
    assert((FMCdata_f.shape[1] == Nelem) and (FMCdata_f.shape[2] == Nelem)), \
        'Number of sources must equal to number of RTM images'; 

    # Create Propagation Function
    prop = propagator(x, f, aawin); 

    # Set Batch Size for Checkpointing
    batch_range = jnp.arange(BATCH_SIZE); # Static-Length Array

    # Checkpoint the Objective Function Output
    output_func_checkpointed = jax.checkpoint(lambda elemresp_f: output_func(elemresp_f, FMCdata_f)); 

    # Functor for Reverse Time Migration (RTM)
    def RTM_func(z, sLayers):
        '''img = RTM_func(z, sLayers)

        RTM_FUNC - Reconstructs Reverse-Time Migrated (RTM) Images Based on Given Sound Speed Map
        
        INPUTS:
        z                  - 1 x Z vector of depths for TX and RX wavefields (1st sample is transducer plane - usually z = 0)
        sLayers            - (Z - 1) x X vector of slowness [s/m] between depths in z
        
        OUTPUT:
        img                - Z x 1 array of objective function values '''
        
        # Body of Propagation Loop
        def prop_scan_body(elemresp_f, z_idx):
            # Propagate Signals in Depth
            elemresp_f = prop(z[z_idx], z[z_idx+1], sLayers[z_idx,:], elemresp_f); 
            # Compute Image at this Depth + Pack Outputs
            return elemresp_f, output_func_checkpointed(elemresp_f).flatten();  
        def prop_scan(elemresp_f, z_idx): # formatted for jax.lax.scan
            # whether z_idx is within jnp.arange(z.size-1)
            cond = z_idx < (z.size - 1); 
            # only propagate if cond == True
            def true_fun(elemresp_f):  
                return prop_scan_body(elemresp_f, z_idx)
            # otherwise pass-through option (if z_idx >= z.size)
            def false_fun(elemresp_f): 
                return elemresp_f, false_fun_output; 
            # return propagation formatted for scan
            return jax.lax.cond(cond, true_fun, false_fun, elemresp_f)
            
        # Starting and Ending of Each Batch
        NUM_BATCHES = (z.size-1)//BATCH_SIZE + 1; 

        # Loop over batches
        def batch_prop(elemresp_f, batch_idx):
            z_indices = batch_idx*BATCH_SIZE + batch_range; 
            return jax.lax.scan(prop_scan, elemresp_f, z_indices); 

        # Propagate Ultrasound Signals in Depth + Imaging
        false_fun_output = jnp.zeros_like(output_func_checkpointed(elemresp_f_xdcr).flatten()); 
        elemresp_f, img = jax.lax.scan(jax.checkpoint(batch_prop), elemresp_f_xdcr, jnp.arange(NUM_BATCHES)); 
        return img.reshape(NUM_BATCHES*BATCH_SIZE, -1)[:z.size-1:, :]; #img.flatten()[:z.size-1:]; 

    # Return JAX-Efficient Propagation Function
    return jax.jit(RTM_func); 



## Image Reconstructipn by Reverse-Time Migration 
# Checkpointing Over Depth Batches + Chunking Over Frequency (V3)
# V3 provides additional decrease in memory cost on top of V2 by grouping frequency bins into chunks and checkpointing those chunks.
# WARNING: DO NOT USE IF output_func DOES NOT return an output that adds linearly add over frequency
def reverseTimeMigrationV3(x, f, aawin, elemresp_f_xdcr, FMCdata_f, output_func):
    '''RTM_func = reverseTimeMigration(x, f, aawin, elemresp_f_xdcr, FMCdata_f, output_func)
    
    REVERSETIMEMIGRATION - Return JAX JIT'ed Functor for Reverse-Time Migration (RTM)
    
    INPUTS:
    x                  - 1 x X vector of x-grid positions for wavefield
    f                  - 1 x F vector of pulse frequencies in spectrum
    aawin              - 1 x X vector of lateral taper to prevent wraparound
    elemresp_f_xdcr    - X x F x N array of input TX wavefields at transducer surface
    FMCdata_f          - F x N x N array of input full-matrix capture (FMC) data
    output_func        - functor applied at each depth: takes elemresp_f and FMCdata_f as input; 
                         outputs 1 x X image or single objective function value

    OUTPUT:
    RTM_func           - JAX JIT'ed function (RTM_func) that performs RTM

    USAGE:    
    img = RTM_func(z, sLayers)

    INPUTS:
    z                  - 1 x Z vector of depths for TX and RX wavefields (1st sample is transducer plane - usually z = 0)
    sLayers            - (Z - 1) x X vector of slowness [s/m] between depths in z
    
    OUTPUT:
    img                - Z x 1 array of objective function values '''

    # Verify the Number of Elements
    Nelem = elemresp_f_xdcr.shape[2]; 
    assert((FMCdata_f.shape[1] == Nelem) and (FMCdata_f.shape[2] == Nelem)), \
        'Number of sources must equal to number of RTM images'; 

    # Setup Chunking Over Frequencies
    RTM_functor_list = []; 
    for chunk_idx in np.arange(NUM_FREQ_CHUNKS):
        # Frequencies in Chunk
        f_idx = np.arange(chunk_idx, f.size, NUM_FREQ_CHUNKS); 
        # Append Checkpointed Functor to List
        RTM_functor_list.append(jax.checkpoint(reverseTimeMigrationV2(x, f[f_idx], aawin, 
            elemresp_f_xdcr[:,f_idx,:], FMCdata_f[f_idx,:,:], output_func))); 

    # Functor for Reverse Time Migration (RTM)
    RTM_func = lambda z, sLayers: jnp.sum(jnp.stack([RTM_functor(z, sLayers) for RTM_functor in RTM_functor_list]), axis=0); 
    return jax.jit(RTM_func); # Return JAX-Efficient Propagation Function
