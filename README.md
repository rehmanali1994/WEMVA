# WEMVA
Wave-Equation Migration Velocity Analysis (WEMVA) for Medical Pulse-Echo Ultrasound Imaging

Wave-equation migration velocity analysis (WEMVA) is a framework for the diffraction tomography of wave-velocity based on imaged reflections.  WEMVA originated in seismic imaging, where it was used to reconstruct the wave velocity profile that optimally focuses the imaged reflectors via reverse-time migration (RTM).  This can roughly be understood as a type of full-waveform inversion (FWI) that acts in the image domain to minimize the misalignment in the imaged reflectors as a function of the reflection angle.  We present two forms of WEMVA (one based on inter-transmit differences; the other based on subsurface offset).  This work uses WEMVA to perform sound speed estimation and aberration correction in medical ultrasound imaging based on a multistatic synthetic aperture setup.

The prior IMPACT method (see https://github.com/rehmanali1994/IMPACT) uses aberration delays (or time shifts) measured between images from single-element transmissions to reconstruct the spatial profile of sound speed in the medium.  Although the forward model was based on RTM, ray tomography was used to reconstruct the sound speed profile.  The first variant of WEMVA builds on top of IMPACT by directly differentiating through the RTM operator to perform an adjoint-state diffraction tomography of sound speed via RTM.  The direct differences between images from single-element transmits is minimized without measuring any intermediate time shifts (or aberration delays).  The second-variant of WEMVA recognizes the limitations of the first variant of WEMVA and utlizes the subsurface-offset extension of RTM to reconstruct the sound speed profile.  This second variant of WEMVA aims to drive imaged content towards zero subsurface offset.

The reconstruction of sound speed from limited-angle pulse-echo ultrasound data is an extremely difficult inverse problem of key diagnostic relevance to medical ultrasound imaging. The primary motivation of this open-source work is to demonstrate the principles behind WEMVA in a more transparent manner so that other researchers can easily reproduce our work and improve upon it. The sample data and algorithms provided in this respository were used in following work:

> Ali, R.; Mitcham, T.; Doyley, M.; Duric, N.; Dahl, J.  "Wave-Equation Migration Velocity Analysis for Multistatic Synthetic Aperture Ultrasound". IEEE Transactions on Ultrasonics. IN REVIEW.

If you use the algorithms and/or datasets provided in this repository for your own research work, please cite the above paper.

You can reference a static version of this code by its DOI number: ADD ZENODO DOI HERE

