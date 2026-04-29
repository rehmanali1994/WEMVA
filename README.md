# WEMVA
Wave-Equation Migration Velocity Analysis (WEMVA) for Medical Pulse-Echo Ultrasound Imaging

Wave-equation migration velocity analysis (WEMVA) is a framework for the diffraction tomography of wave-velocity based on imaged reflections.  WEMVA originated in seismic imaging, where it was used to reconstruct the wave velocity profile that optimally focuses the imaged reflectors via reverse-time migration (RTM).  This can roughly be understood as a type of full-waveform inversion (FWI) that acts in the image domain to minimize the misalignment in the imaged reflectors as a function of the reflection angle.  We present two forms of WEMVA (one based on inter-transmit differences; the other based on subsurface offset).  This work uses WEMVA to perform sound speed estimation and aberration correction in medical ultrasound imaging based on a multistatic synthetic aperture setup.

The prior IMPACT method (see https://github.com/rehmanali1994/IMPACT) uses aberration delays (or time shifts) measured between images from single-element transmissions to reconstruct the spatial profile of sound speed in the medium.  Although the forward model was based on RTM, ray tomography was used to reconstruct the sound speed profile.  The first variant of WEMVA builds on top of IMPACT by directly differentiating through the RTM operator to perform an adjoint-state diffraction tomography of sound speed via RTM.  The direct differences between images from single-element transmits is minimized without measuring any intermediate time shifts (or aberration delays).  The second-variant of WEMVA recognizes the limitations of the first variant of WEMVA and utlizes the subsurface-offset extension of RTM to reconstruct the sound speed profile.  This second variant of WEMVA aims to drive imaged content towards zero subsurface offset.

The reconstruction of sound speed from limited-angle pulse-echo ultrasound data is an extremely difficult inverse problem of key diagnostic relevance to medical ultrasound imaging. The primary motivation of this open-source work is to demonstrate the principles behind WEMVA in a more transparent manner so that other researchers can easily reproduce and improve upon it. The sample data and algorithms provided in this respository were used in following work:

> Ali, R.; Mitcham, T.; Doyley, M.; Duric, N.; Dahl, J.  "Wave-Equation Migration Velocity Analysis for Multistatic Synthetic Aperture Ultrasound". IEEE Transactions on Ultrasonics. IN REVIEW.

If you use the algorithms and/or datasets provided in this repository for your own research work, please cite the above paper.

You can reference a static version of this code by its DOI number: ADD ZENODO DOI HERE

# Code and Sample Datasets

**Please download the sample data (SuperficialAbdominalLayersL7-4.mat; Rat10_Acq3.mat; Rat11_Acq2.mat; PhantomVSX2.mat; PhantomVSX4_1.mat; PhantomVSX4_2.mat) under the [releases](https://github.com/rehmanali1994/IMPACT/releases) tab for this repository, and place that data in the [Datasets](https://github.com/rehmanali1994/IMPACT/tree/main/Datasets/) folder.**

The following scripts correspond to each dataset:
1) AbdominalMap3.mat and AbdominalMap4.mat - [AberrationTomographyShotGatherMigKWave.m](https://github.com/rehmanali1994/IMPACT/blob/main/MATLAB/AberrationTomographyShotGatherMigKWave.m)) and [AberrationTomographyShotGatherMigKWave.py](https://github.com/rehmanali1994/IMPACT/blob/main/Python/AberrationTomographyShotGatherMigKWave.py)) - These datasets were simulated in k-Wave with a known ground-truth sound speed profile.
2) RatAbdomenL12-3v.mat - [AberrationTomographyShotGatherMigL12_3v.m](https://github.com/rehmanali1994/IMPACT/blob/main/MATLAB/AberrationTomographyShotGatherMigL12_3v.m)) and [AberrationTomographyShotGatherMigL12_3v.py](https://github.com/rehmanali1994/IMPACT/blob/main/Python/AberrationTomographyShotGatherMigL12_3v.py)) - This dataset was obtained from the abdomen of an obese Zucker rat under a Stanford-approved IACUC protocol using an L12-3v probe.
3) PhantomL12-5-50mm.mat - [AberrationTomographyShotGatherMigL12_5_50mm.m](https://github.com/rehmanali1994/IMPACT/blob/main/MATLAB/AberrationTomographyShotGatherMigL12_5_50mm.m)) and [AberrationTomographyShotGatherMigL12_5_50mm.py](https://github.com/rehmanali1994/IMPACT/blob/main/Python/AberrationTomographyShotGatherMigL12_5_50mm.py)) - This dataset was obtained using an L12-5 50mm probe from a phantom with three high sound speed alcohol inclusions to induce aberrations in the image.

The following key MATLAB functions (implemented in Python within [functions.py](https://github.com/rehmanali1994/IMPACT/blob/main/Python/functions.py)) used in these scripts are: 
1) [line_pixel_intersection.m](https://github.com/rehmanali1994/IMPACT/tree/main/MATLAB/functions/line_pixel_intersection.m) - Computation of path length over each pixel in sound speed map for travel-time tomography
2) [optApod.m](https://github.com/rehmanali1994/IMPACT/tree/main/MATLAB/functions/optApod.m) - Computes the optimal apodization that maximizes the short-lag spatial coherence. If using this script, please cite the following work:
> Ali, R.; Duric, N.; Dahl, J. "Optimal Transmit Apodization for the Maximization of Lag-One Coherence with Applications to Aberration Delay Estimation". Ultrasonics. 2023 Apr 23, p.107010.
3) [propagate.m](https://github.com/rehmanali1994/IMPACT/tree/main/MATLAB/functions/propagate.m) - The Fourier split-step angular spectrum method used in the wavefield correlation technique
4) [spray.m](https://github.com/rehmanali1994/IMPACT/tree/main/MATLAB/functions/spray.m) - Adjoint of masking operation used to select imaging points in the medium

Overall, we strongly recommend and prefer the MATLAB code over the Python code because of its faster run times. We also recommend running this code on a system with a large amount of RAM (at least 16 GB, but 32-64 GB is ideal).
