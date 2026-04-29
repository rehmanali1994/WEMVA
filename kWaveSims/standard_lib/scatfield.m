function [cextend,seedUsed] = scatfield(num_scat,scat_size,scat_scale,c0,omega0,dx,ax,ncycles,dX,dZ,nXextend,nZ)

% Resolution cell size
res_cell = rescell2d(c0,omega0,dx,ax,ncycles,dX,dZ);

% Scatter sample size
scat_size_samp = round(scat_size/dX);

% Scatterer field
[cextend, seedUsed] = generate_c_scat(c0, scat_scale, scat_size_samp, num_scat, res_cell, nXextend, nZ);

