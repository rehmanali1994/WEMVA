# WEMVA
JAX Implementation of Wave-Equation Migration Velocity Analysis (WEMVA) for Medical Pulse-Echo Ultrasound

Wave-equation migration velocity analysis (WEMVA) is an image-domain diffraction tomography that estimates wave velocity by improving the focusing of reflectors in reverse-time migration (RTM). Originally developed for seismic imaging, WEMVA can be viewed as a form of full-waveform inversion that minimizes misalignment across reflection angles. In this work, we adapt WEMVA for sound-speed estimation and aberration correction in multistatic synthetic aperture ultrasound using two approaches: one based on mimimizing inter-transmit differences and another based on subsurface offset.

This work builds on the prior [IMPACT](https://github.com/rehmanali1994/IMPACT) method, which estimated sound speed from aberration delays between single-element transmit images using ray tomography. Our first WEMVA variant instead differentiates directly through the RTM operator, enabling adjoint-state diffraction tomography based on minimizing the direct differences between partial images without measuring intermediate delays. The second variant addresses limitations of the first by using the subsurface-offset extension of RTM to driving imaged energy toward zero subsurface offset.

Recovering sound speed from limited-angle pulse-echo ultrasound data is a challenging inverse problem with major importance for medical ultrasound imaging. This open-source project aims to provide a transparent implementation of WEMVA so researchers can reproduce, understand, and extend these methods. The included datasets and algorithms were used in the following work:

> Ali, R.; Mitcham, T.; Doyley, M.; Duric, N.; Dahl, J.  "Wave-Equation Migration Velocity Analysis for Multistatic Synthetic Aperture Ultrasound". IEEE Transactions on Ultrasonics. IN REVIEW.

If you use the algorithms and/or datasets provided in this repository for your own research work, please cite the above paper.

You can reference a static version of this code by its DOI number: [![DOI](https://zenodo.org/badge/1224779196.svg)](https://doi.org/10.5281/zenodo.19906865)

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

The following codes implement the key functions/classes used in the WEMVA scripts ([JAX/WEMVA_FMC_kWave.py](https://github.com/rehmanali1994/WEMVA/blob/main/JAX/WEMVA_FMC_kWave.py); [JAX/WEMVA_FMC_L12_5_50mm.py](https://github.com/rehmanali1994/WEMVA/blob/main/JAX/WEMVA_FMC_L12_5_50mm.py); [JAX/WEMVA_FMC_L12_3v.py](https://github.com/rehmanali1994/WEMVA/blob/main/JAX/WEMVA_FMC_L12_3v.py); [JAX/WEMVA_FMC_L7_4.py](https://github.com/rehmanali1994/WEMVA/blob/main/JAX/WEMVA_FMC_L7_4.py)): 
1) [JAX/HelperFunctions.py](https://github.com/rehmanali1994/WEMVA/blob/main/JAX/HelperFunctions.py) - Implements Python-equivalent implementations of `imagesc`, `db`, a method for loading MAT files saved using the `-v7.3` option in MATLAB, and general plotting utilities.
2) [JAX/OptimizationFunctions.py](https://github.com/rehmanali1994/WEMVA/blob/main/JAX/OptimizationFunctions.py) - Implements line searches (bracketing + golden-section search)
3) [JAX/ImagingFunctionsFMC.py](https://github.com/rehmanali1994/WEMVA/blob/main/JAX/ImagingFunctionsFMC.py) - This is where both variants of WEMVA are implemented.  The RTM used in WEMVA is based on the Fourier split-step method used in our prior works ([IMPACT](https://github.com/rehmanali1994/IMPACT); [WaveformInversionUST](https://github.com/rehmanali1994/WaveformInversionUST)).  A checkpointing strategy is used to optimize the storage-rematerialization tradeoff and minimize the memory cost of the reverse-mode automatic differentiation (equivalent to the adjoint-state method).  See the documentation in [JAX/ImagingFunctionsFMC.py](https://github.com/rehmanali1994/WEMVA/blob/main/JAX/ImagingFunctionsFMC.py) for more details.  Both `BATCH_SIZE` and `NUM_FREQ_CHUNKS` (defined in defined near the top of [JAX/ImagingFunctionsFMC.py](https://github.com/rehmanali1994/WEMVA/blob/main/JAX/ImagingFunctionsFMC.py)) can be adjusted for the specific imaging grid and the memory constraints of the GPU.  A good rule of thumb for `BATCH_SIZE` is that it should roughly be square root of the number of depths in the imaging grid. 

These codes ran successfully with an NVIDIA RTX PRO 6000 GPU (96 GB of GPU RAM) and an NVIDIA GeForce RTX 4090 GPU (24 GB of GPU RAM) using JAX[cuda12]. In the specific case of running [JAX/WEMVA_FMC_L12_5_50mm.py](https://github.com/rehmanali1994/WEMVA/blob/main/JAX/WEMVA_FMC_L12_5_50mm.py) on the NVIDIA GeForce RTX 4090 GPU (24 GB of GPU RAM), the `NUM_FREQ_CHUNKS` was increased from 4 to 8 fit the WEMVA model in GPU memory.  

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
