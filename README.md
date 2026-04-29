# WEMVA
JAX Implementation of Wave-Equation Migration Velocity Analysis (WEMVA) for Medical Pulse-Echo Ultrasound Imaging

Wave-equation migration velocity analysis (WEMVA) is a framework for the diffraction tomography of wave-velocity based on imaged reflections.  WEMVA originated in seismic imaging, where it was used to reconstruct the wave velocity profile that optimally focuses the imaged reflectors via reverse-time migration (RTM).  This can roughly be understood as a type of full-waveform inversion (FWI) that acts in the image domain to minimize the misalignment in the imaged reflectors as a function of the reflection angle.  We present two forms of WEMVA (one based on inter-transmit differences; the other based on subsurface offset).  This work uses WEMVA to perform sound speed estimation and aberration correction in medical ultrasound imaging based on a multistatic synthetic aperture setup.

The prior IMPACT method (see https://github.com/rehmanali1994/IMPACT) uses aberration delays (or time shifts) measured between images from single-element transmissions to reconstruct the spatial profile of sound speed in the medium.  Although the forward model was based on RTM, ray tomography was used to reconstruct the sound speed profile.  The first variant of WEMVA builds on top of IMPACT by directly differentiating through the RTM operator to perform an adjoint-state diffraction tomography of sound speed via RTM.  The direct differences between images from single-element transmits is minimized without measuring any intermediate time shifts (or aberration delays).  The second-variant of WEMVA recognizes the limitations of the first variant of WEMVA and utlizes the subsurface-offset extension of RTM to reconstruct the sound speed profile.  This second variant of WEMVA aims to drive imaged content towards zero subsurface offset.

The reconstruction of sound speed from limited-angle pulse-echo ultrasound data is an extremely difficult inverse problem of key diagnostic relevance to medical ultrasound imaging. The primary motivation of this open-source work is to demonstrate the principles behind WEMVA in a more transparent manner so that other researchers can easily reproduce and improve upon it. The sample data and algorithms provided in this respository were used in following work:

> Ali, R.; Mitcham, T.; Doyley, M.; Duric, N.; Dahl, J.  "Wave-Equation Migration Velocity Analysis for Multistatic Synthetic Aperture Ultrasound". IEEE Transactions on Ultrasonics. IN REVIEW.

If you use the algorithms and/or datasets provided in this repository for your own research work, please cite the above paper.

You can reference a static version of this code by its DOI number: ADD ZENODO DOI HERE

# Experimental Datasets

**Please download the sample data (SuperficialAbdominalLayersL7-4.mat; Rat10_Acq3.mat; Rat11_Acq2.mat; PhantomVSX2.mat; PhantomVSX4_1.mat; PhantomVSX4_2.mat) under the [WEMVA/releases](https://github.com/rehmanali1994/WEMVA/releases) tab for this repository, and place that data in the [Datasets](https://github.com/rehmanali1994/IMPACT/tree/main/Datasets/) folder.  Additionally, this repository uses sample data (RatAbdomenL12-3v.mat; PhantomL12-5-50mm.mat) previously released under [IMPACT/releases](https://github.com/rehmanali1994/IMPACT/releases) tab, which should also be placed in the [Datasets](https://github.com/rehmanali1994/IMPACT/tree/main/Datasets/) folder.**

The following JAX/Python scripts correspond to each dataset:
1) SuperficialAbdominalLayersL7-4.mat - [JAX/WEMVA_FMC_L7_4.py](https://github.com/rehmanali1994/WEMVA/blob/main/JAX/WEMVA_FMC_L7_4.py) - This dataset was acquired from the abdomen of a healthy human volunteer under a University of Rochester RSRB-approved protocol using an L7-4 probe.
2) RatAbdomenL12-3v.mat; Rat10_Acq3.mat; Rat11_Acq2.mat - [JAX/WEMVA_FMC_L12_3v.py](https://github.com/rehmanali1994/WEMVA/blob/main/JAX/WEMVA_FMC_L12_3v.py) - These datasets were obtained from the abdomen of obese Zucker rats under a Stanford-approved IACUC protocol using an L12-3v probe.
3) PhantomL12-5-50mm.mat; PhantomVSX2.mat; PhantomVSX4_1.mat; PhantomVSX4_2.mat - [JAX/WEMVA_FMC_L12_5_50mm.py](https://github.com/rehmanali1994/WEMVA/blob/main/JAX/WEMVA_FMC_L12_5_50mm.py) - These datasets was obtained using an L12-5 50mm probe from phantoms with high sound speed alcohol inclusions to induce aberrations in the image.

# k-Wave Simulations

In our past works, we included [k-Wave](http://www.k-wave.org/) simulated datasets (e.g., AbdominalMap3.mat and AbdominalMap4.mat under [IMPACT/releases](https://github.com/rehmanali1994/IMPACT/releases)). This time, rather than provide the datasets themselves, we provide the code to run the k-Wave simulations that generates those datasets. We do this to conserve space and provide codes to simulate data using [k-Wave](http://www.k-wave.org/). Each [k-Wave](http://www.k-wave.org/) simulation involves a 3-step process:

1) [kWaveSims/GenKWaveSimInfo.m](https://github.com/rehmanali1994/WEMVA/blob/main/kWaveSims/GenKWaveSimInfo.m) creates a MAT file that is stored in the [kWaveSims/sim_info](https://github.com/rehmanali1994/WEMVA/tree/main/kWaveSims/sim_info) folder.  This MAT file contains all the information (medium, transducer geometry, and pulse excitation) needed to simulate multistatic synthetic aperture channel data for each of six abdominal maps (`map_number` can be any integer ranging from 1-6).
2) After generating the MAT file in [kWaveSims/sim_info](https://github.com/rehmanali1994/WEMVA/tree/main/kWaveSims/sim_info), we run the actual [k-Wave](http://www.k-wave.org/) simulation using [kWaveSims/GenRFDataSingleTxKWave.m](https://github.com/rehmanali1994/WEMVA/blob/main/kWaveSims/GenRFDataSingleTxKWave.m). The `map_number` parameter corresponding to the simulation case must be specified. [kWaveSims/GenRFDataSingleTxKWave.m](https://github.com/rehmanali1994/WEMVA/blob/main/kWaveSims/GenRFDataSingleTxKWave.m) loops through each single-element transmit. The simulated data for each transmit is then stored in MAT files in the [kWaveSims/scratch](https://github.com/rehmanali1994/WEMVA/tree/main/kWaveSims/scratch) folder.
3) Lastly, [AssembleFullSynthDataKWave.m](https://github.com/rehmanali1994/WEMVA/blob/main/kWaveSims/AssembleFullSynthDataKWave.m) assembles the simulated data from each indivdual transmit/MAT-file in the [kWaveSims/scratch](https://github.com/rehmanali1994/WEMVA/tree/main/kWaveSims/scratch) folder into a single MAT file `AbdominalMap(1|2|3|4|5|6).mat` containing the full multistatic synthetic aperture dataset in the [kWaveSims/datasets](https://github.com/rehmanali1994/WEMVA/tree/main/kWaveSims/datasets) folder.

The JAX/Python script corresponding to these k-Wave datasets is [JAX/WEMVA_FMC_kWave.py](https://github.com/rehmanali1994/WEMVA/blob/main/JAX/WEMVA_FMC_kWave.py).  Note that [JAX/WEMVA_FMC_kWave.py](https://github.com/rehmanali1994/WEMVA/blob/main/JAX/WEMVA_FMC_kWave.py) remains compatible with the previously released k-Wave datasets (AbdominalMap3.mat and AbdominalMap4.mat) under [IMPACT/releases](https://github.com/rehmanali1994/IMPACT/releases). 


# Code

The key functions/classes used in the waveform inversion scripts ([MultiFrequencyWaveformInvKCI.m](https://github.com/rehmanali1994/WaveformInversionUST/blob/main/MultiFrequencyWaveformInvKCI.m); [MultiFrequencyWaveformInvVSX.m](https://github.com/rehmanali1994/WaveformInversionUST/blob/main/MultiFrequencyWaveformInvVSX.m); [MultiFrequencyWaveformInvkWave.m](https://github.com/rehmanali1994/WaveformInversionUST/blob/main/MultiFrequencyWaveformInvkWave.m)) are: 
1) [HelmholtzSolver.m](https://github.com/rehmanali1994/WaveformInversionUST/blob/main/Functions/HelmholtzSolver.m) - Implements the Helmholtz equation solver as a class. For a given set of medium properties, the HelmholtzSolver forms the discretized system of equations that needs to be solved either on CPU or GPU. If an NVIDIA GPU is available, a block LU factorization is performed and stored in memory for subsequent solves using this factorization.
2) [stencilOptParams.m](https://github.com/rehmanali1994/WaveformInversionUST/blob/main/Functions/stencilOptParams.m) - Helper function called by [HelmholtzSolver.m](https://github.com/rehmanali1994/WaveformInversionUST/blob/main/Functions/HelmholtzSolver.m) to generate the stencil used to discretize the Helmholtz equation.
3) [decompBlockLU.cu](https://github.com/rehmanali1994/WaveformInversionUST/blob/main/Functions/decompBlockLU.cu) - This is the MEX CUDA code called by [HelmholtzSolver.m](https://github.com/rehmanali1994/WaveformInversionUST/blob/main/Functions/HelmholtzSolver.m) to perform the block LU factorization. Must be compiled in MATLAB using `mexcuda -lcusolver decompBlockLU.cu` using the cuSOLVER option.
4) [applyBlockLU.cu](https://github.com/rehmanali1994/WaveformInversionUST/blob/main/Functions/applyBlockLU.cu) - This is the MEX CUDA code called by [HelmholtzSolver.m](https://github.com/rehmanali1994/WaveformInversionUST/blob/main/Functions/HelmholtzSolver.m) to apply the block LU factorization computed by [decompBlockLU.cu](https://github.com/rehmanali1994/WaveformInversionUST/blob/main/Functions/decompBlockLU.cu) to set of given sources (or adjoint sources). Must be compiled in MATLAB using `mexcuda -lcublas applyBlockLU.cu` using the cuBLAS option.
5) [ringingRemovalFilt.m](https://github.com/rehmanali1994/WaveformInversionUST/blob/main/Functions/ringingRemovalFilt.m) - Helper function called during waveform inversion scripts ([MultiFrequencyWaveformInvKCI.m](https://github.com/rehmanali1994/WaveformInversionUST/blob/main/MultiFrequencyWaveformInvKCI.m); [MultiFrequencyWaveformInvVSX.m](https://github.com/rehmanali1994/WaveformInversionUST/blob/main/MultiFrequencyWaveformInvVSX.m); [MultiFrequencyWaveformInvkWave.m](https://github.com/rehmanali1994/WaveformInversionUST/blob/main/MultiFrequencyWaveformInvkWave.m)) to remove ringing artifacts in the images.

These codes ran successfully with an NVIDIA GeForce RTX 3060 GPU (12 GB of GPU RAM) on a CPU with 40 GB of RAM in both MATLAB 2021b and 2022b. We therefore recommend running this code on a CPU with at least 32 GB of RAM and a GPU with at least 12 GB of RAM.

# Sample Results
Each waveform inversion script ([MultiFrequencyWaveformInvkWave.m](https://github.com/rehmanali1994/WaveformInversionUST/blob/main/MultiFrequencyWaveformInvkWave.m); [MultiFrequencyWaveformInvKCI.m](https://github.com/rehmanali1994/WaveformInversionUST/blob/main/MultiFrequencyWaveformInvKCI.m); [MultiFrequencyWaveformInvVSX.m](https://github.com/rehmanali1994/WaveformInversionUST/blob/main/MultiFrequencyWaveformInvVSX.m)) saves the results at each iteration to a MAT file in the [Results](https://github.com/rehmanali1994/WaveformInversionUST/tree/main/Results) folder. The results stored in these MAT files can later be visualized using the [viewSavedResults.m](https://github.com/rehmanali1994/WaveformInversionUST/blob/main/viewSavedResults.m)) script. 

1) BenignCyst.mat:

![](https://github.com/rehmanali1994/WaveformInversionUST/blob/main/Results/BenignCyst.gif)

2) Malignancy.mat

![](https://github.com/rehmanali1994/WaveformInversionUST/blob/main/Results/Malignancy.gif)

3) VSX_YezitronixPhantom1.mat

![](https://github.com/rehmanali1994/WaveformInversionUST/blob/main/Results/VSX_YezitronixPhantom1.gif)

4) VSX_YezitronixPhantom2.mat

![](https://github.com/rehmanali1994/WaveformInversionUST/blob/main/Results/VSX_YezitronixPhantom2.gif)

5) kWave_BreastCT.gif

![](https://github.com/rehmanali1994/WaveformInversionUST/blob/main/Results/kWave_BreastCT.gif)

6) kWave_BreastMRI.gif

![](https://github.com/rehmanali1994/WaveformInversionUST/blob/main/Results/kWave_BreastMRI.gif)

