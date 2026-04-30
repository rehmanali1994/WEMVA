# WEMVA
JAX Implementation of Wave-Equation Migration Velocity Analysis (WEMVA) for Medical Pulse-Echo Ultrasound

Wave-equation migration velocity analysis (WEMVA) is a framework for the diffraction tomography of wave-velocity based on imaged reflections.  WEMVA originated in seismic imaging, where it was used to reconstruct the wave velocity profile that optimally focuses the imaged reflectors via reverse-time migration (RTM).  This can roughly be understood as a type of full-waveform inversion (FWI) that acts in the image domain to minimize the misalignment in the imaged reflectors as a function of the reflection angle.  We present two forms of WEMVA (one based on inter-transmit differences; the other based on subsurface offset).  This work uses WEMVA to perform sound speed estimation and aberration correction in medical ultrasound imaging based on a multistatic synthetic aperture setup.

The prior [IMPACT](https://github.com/rehmanali1994/IMPACT) method uses aberration delays (or time shifts) measured between images from single-element transmissions to reconstruct the spatial profile of sound speed in the medium.  Although the forward model was based on RTM, ray tomography was used to reconstruct the sound speed profile.  The first variant of WEMVA builds on top of IMPACT by directly differentiating through the RTM operator to perform an adjoint-state diffraction tomography of sound speed via RTM.  The direct differences between images from single-element transmits is minimized without measuring any intermediate time shifts (or aberration delays).  The second-variant of WEMVA recognizes the limitations of the first variant of WEMVA and utlizes the subsurface-offset extension of RTM to reconstruct the sound speed profile.  This second variant of WEMVA aims to drive imaged content towards zero subsurface offset.

The reconstruction of sound speed from limited-angle pulse-echo ultrasound data is an extremely difficult inverse problem of key diagnostic relevance to medical ultrasound imaging. The primary motivation of this open-source work is to demonstrate the principles behind WEMVA in a more transparent manner so that other researchers can easily reproduce and improve upon it. The sample data and algorithms provided in this respository were used in following work:

> Ali, R.; Mitcham, T.; Doyley, M.; Duric, N.; Dahl, J.  "Wave-Equation Migration Velocity Analysis for Multistatic Synthetic Aperture Ultrasound". IEEE Transactions on Ultrasonics. IN REVIEW.

If you use the algorithms and/or datasets provided in this repository for your own research work, please cite the above paper.

You can reference a static version of this code by its DOI number: ADD ZENODO DOI HERE

# Experimental Datasets

**Please download the sample data (`SuperficialAbdominalLayersL7-4.mat`; `Rat10_Acq3.mat`; `Rat11_Acq2.mat`; `PhantomVSX2.mat`; `PhantomVSX4_1.mat`; `PhantomVSX4_2.mat`) under the [WEMVA/releases](https://github.com/rehmanali1994/WEMVA/releases) tab for this repository, and place that data in the [Datasets](https://github.com/rehmanali1994/IMPACT/tree/main/Datasets/) folder.  Additionally, this repository uses sample data (`RatAbdomenL12-3v.mat`; `PhantomL12-5-50mm.mat`) previously released under [IMPACT/releases](https://github.com/rehmanali1994/IMPACT/releases) tab, which should also be placed in the [Datasets](https://github.com/rehmanali1994/IMPACT/tree/main/Datasets/) folder.**

The following JAX/Python scripts correspond to each dataset:
1) [JAX/WEMVA_FMC_L7_4.py](https://github.com/rehmanali1994/WEMVA/blob/main/JAX/WEMVA_FMC_L7_4.py) - `SuperficialAbdominalLayersL7-4.mat` - This dataset was acquired from the abdomen of a healthy human volunteer under a University of Rochester RSRB-approved protocol using an L7-4 probe.
2) [JAX/WEMVA_FMC_L12_3v.py](https://github.com/rehmanali1994/WEMVA/blob/main/JAX/WEMVA_FMC_L12_3v.py) - `RatAbdomenL12-3v.mat`; `Rat10_Acq3.mat`; `Rat11_Acq2.mat` - These datasets were obtained from the abdomen of obese Zucker rats under a Stanford-approved IACUC protocol using an L12-3v probe.
3) [JAX/WEMVA_FMC_L12_5_50mm.py](https://github.com/rehmanali1994/WEMVA/blob/main/JAX/WEMVA_FMC_L12_5_50mm.py) - `PhantomL12-5-50mm.mat`; `PhantomVSX2.mat`; `PhantomVSX4_1.mat`; `PhantomVSX4_2.mat` - These datasets was obtained using an L12-5 50mm probe from phantoms with high sound speed alcohol inclusions to induce aberrations in the image.

# k-Wave Simulations

In our past works, we included [k-Wave](http://www.k-wave.org/) simulated datasets (e.g., `AbdominalMap3.mat` and `AbdominalMap4.mat` under [IMPACT/releases](https://github.com/rehmanali1994/IMPACT/releases)). This time, rather than provide the datasets themselves, we provide the code to run the k-Wave simulations that generates those datasets. We do this to conserve space and provide codes to simulate data using [k-Wave](http://www.k-wave.org/). Each [k-Wave](http://www.k-wave.org/) simulation involves a 3-step process:

1) [kWaveSims/GenKWaveSimInfo.m](https://github.com/rehmanali1994/WEMVA/blob/main/kWaveSims/GenKWaveSimInfo.m) creates a MAT file that is stored in the [kWaveSims/sim_info](https://github.com/rehmanali1994/WEMVA/tree/main/kWaveSims/sim_info) folder.  This MAT file contains all the information (medium, transducer geometry, and pulse excitation) needed to simulate multistatic synthetic aperture channel data for each of six abdominal maps (`map_number` can be any integer ranging from 1-6).
2) After generating the MAT file in [kWaveSims/sim_info](https://github.com/rehmanali1994/WEMVA/tree/main/kWaveSims/sim_info), we run the actual [k-Wave](http://www.k-wave.org/) simulation using [kWaveSims/GenRFDataSingleTxKWave.m](https://github.com/rehmanali1994/WEMVA/blob/main/kWaveSims/GenRFDataSingleTxKWave.m). The `map_number` parameter corresponding to the simulation case must be specified. [kWaveSims/GenRFDataSingleTxKWave.m](https://github.com/rehmanali1994/WEMVA/blob/main/kWaveSims/GenRFDataSingleTxKWave.m) loops through each single-element transmit. The simulated data for each transmit is then stored in MAT files in the [kWaveSims/scratch](https://github.com/rehmanali1994/WEMVA/tree/main/kWaveSims/scratch) folder.
3) Lastly, [AssembleFullSynthDataKWave.m](https://github.com/rehmanali1994/WEMVA/blob/main/kWaveSims/AssembleFullSynthDataKWave.m) assembles the simulated data from each indivdual transmit/MAT-file in the [kWaveSims/scratch](https://github.com/rehmanali1994/WEMVA/tree/main/kWaveSims/scratch) folder into a single MAT file `AbdominalMap(1|2|3|4|5|6).mat` containing the full multistatic synthetic aperture dataset in the [kWaveSims/datasets](https://github.com/rehmanali1994/WEMVA/tree/main/kWaveSims/datasets) folder.

The JAX/Python script corresponding to these k-Wave datasets is [JAX/WEMVA_FMC_kWave.py](https://github.com/rehmanali1994/WEMVA/blob/main/JAX/WEMVA_FMC_kWave.py).  Note that [JAX/WEMVA_FMC_kWave.py](https://github.com/rehmanali1994/WEMVA/blob/main/JAX/WEMVA_FMC_kWave.py) is also compatible with the previously released k-Wave datasets (`AbdominalMap3.mat` and `AbdominalMap4.mat`) under [IMPACT/releases](https://github.com/rehmanali1994/IMPACT/releases). 

# Code

The key functions/classes used in the WEMVA scripts ([JAX/WEMVA_FMC_kWave.py](https://github.com/rehmanali1994/WEMVA/blob/main/JAX/WEMVA_FMC_kWave.py); [JAX/WEMVA_FMC_L12_5_50mm.py](https://github.com/rehmanali1994/WEMVA/blob/main/JAX/WEMVA_FMC_L12_5_50mm.py); [JAX/WEMVA_FMC_L12_3v.py](https://github.com/rehmanali1994/WEMVA/blob/main/JAX/WEMVA_FMC_L12_3v.py); [JAX/WEMVA_FMC_L7_4.py](https://github.com/rehmanali1994/WEMVA/blob/main/JAX/WEMVA_FMC_L7_4.py)) are implemented in: 
1) [JAX/HelperFunctions.py](https://github.com/rehmanali1994/WEMVA/blob/main/JAX/HelperFunctions.py) - Implements Python-equivalent implementations of `imagesc`, `db`, and a method for loading MAT files saved using the `-v7.3` option in MATLAB, as well as general plotting utilities.
2) [JAX/OptimizationFunctions.py](https://github.com/rehmanali1994/WEMVA/blob/main/JAX/OptimizationFunctions.py) - Implements line searches (bracketing + golden-section search)
3) [JAX/ImagingFunctionsFMC.py](https://github.com/rehmanali1994/WEMVA/blob/main/JAX/ImagingFunctionsFMC.py) - This is where both variants of WEMVA are implemented.  The RTM used in WEMVA is based on the Fourier split-step method used in our prior works ([IMPACT](https://github.com/rehmanali1994/IMPACT); [WaveformInversionUST](https://github.com/rehmanali1994/WaveformInversionUST)).  A checkpointing strategy is used to optimize the storage-rematerialization tradeoff and minimize the memory cost of the reverse-mode automatic differentiation (equivalent to the adjoint-state method). See the documentation in [JAX/ImagingFunctionsFMC.py](https://github.com/rehmanali1994/WEMVA/blob/main/JAX/ImagingFunctionsFMC.py) for more details. 

REVISE THIS.  These codes ran successfully with an NVIDIA RTX PRO 6000 GPU (96 GB of GPU RAM) on a CPU with 128 GB of RAM and an NVIDIA GeForce RTX 4090 GPU (24 GB of GPU RAM) on a CPU with 64 GB of RAM in JAX[cuda12].  We therefore recommend running each script on a CPU with at least 32 GB of RAM and a GPU with at least 24 GB RAM.  

# Sample Results
Each WEMVA scripts ([JAX/WEMVA_FMC_kWave.py](https://github.com/rehmanali1994/WEMVA/blob/main/JAX/WEMVA_FMC_kWave.py); [JAX/WEMVA_FMC_L12_5_50mm.py](https://github.com/rehmanali1994/WEMVA/blob/main/JAX/WEMVA_FMC_L12_5_50mm.py); [JAX/WEMVA_FMC_L12_3v.py](https://github.com/rehmanali1994/WEMVA/blob/main/JAX/WEMVA_FMC_L12_3v.py); [JAX/WEMVA_FMC_L7_4.py](https://github.com/rehmanali1994/WEMVA/blob/main/JAX/WEMVA_FMC_L7_4.py)) saves the results at each iteration to individual `.png` files and all iterations to a single  `FullHistory.mat` file in a corresponding subdirectory of the [Figures](https://github.com/rehmanali1994/WEMVA/tree/main/Figures) folder. The results stored in each `FullHistory.mat` file can later be visualized using MATLAB. The following GIFs of the results were generated using the `.png` files for the subsurface-offset variant of WEMVA.

## `SuperficialAbdominalLayersL7-4.mat`:

![](https://github.com/rehmanali1994/WEMVA/blob/main/Figures/SuperficialLayers/SuperficialLayers.gif)

## Rat Datasets:

### `Rat10_Acq3.mat`
![](https://github.com/rehmanali1994/WEMVA/blob/main/Figures/RatAbdomen/Rat10_Acq3/Rat10_Acq3.gif)
### `Rat11_Acq2.mat`
![](https://github.com/rehmanali1994/WEMVA/blob/main/Figures/RatAbdomen/Rat11_Acq2/Rat11_Acq2.gif)
### `RatAbdomenL12-3v.mat`
![](https://github.com/rehmanali1994/WEMVA/blob/main/Figures/RatAbdomen/RatAbdomenL12-3v/RatAbdomenL12-3v.gif)

## Phantom Datasets:

### `PhantomL12-5-50mm.mat`
![](https://github.com/rehmanali1994/WEMVA/blob/main/Figures/Phantom/PhantomL12-5-50mm/PhantomL12-5-50mm.gif)
### `PhantomVSX2.mat`
![](https://github.com/rehmanali1994/WEMVA/blob/main/Figures/Phantom/PhantomVSX2/PhantomVSX2.gif)
### `PhantomVSX4_1.mat`
![](https://github.com/rehmanali1994/WEMVA/blob/main/Figures/Phantom/PhantomVSX4_1/PhantomVSX4_1.gif)
### `PhantomVSX4_2.mat`
![](https://github.com/rehmanali1994/WEMVA/blob/main/Figures/Phantom/PhantomVSX4_2/PhantomVSX4_2.gif)

## `AbdominalMap(1|2|3|4|5|6).mat`:

### `AbdominalMap1.mat`:
![](https://github.com/rehmanali1994/WEMVA/blob/main/Figures/AbdominalMaps/Map1/Map1.gif)
### `AbdominalMap2.mat`:
![](https://github.com/rehmanali1994/WEMVA/blob/main/Figures/AbdominalMaps/Map2/Map2.gif)
### `AbdominalMap3.mat`:
![](https://github.com/rehmanali1994/WEMVA/blob/main/Figures/AbdominalMaps/Map3/Map3.gif)
### `AbdominalMap4.mat`:
![](https://github.com/rehmanali1994/WEMVA/blob/main/Figures/AbdominalMaps/Map4/Map4.gif)
### `AbdominalMap5.mat`:
![](https://github.com/rehmanali1994/WEMVA/blob/main/Figures/AbdominalMaps/Map5/Map5.gif)
### `AbdominalMap6.mat`:
![](https://github.com/rehmanali1994/WEMVA/blob/main/Figures/AbdominalMaps/Map6/Map6.gif)
