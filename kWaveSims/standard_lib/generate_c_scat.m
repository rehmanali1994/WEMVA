% GENERATE_C_SCAT - generate scatterer grid
%
% [SLICEX] = GENERATE_C_SCAT(C0,CSCALE,SCAT_SIZE,NUM_SCAT,RES_CELL,NX,NZ)
% generates a 2-D scatterer grid for the 2-D Westerveldt simulation program.
% The scatterer grid hold the speed of sound values for the scatterers.
% SLICEX is the scatterer grid, C0 is the base speed of sound, CSCALE is the
% maximum percent variation in the scatterer impedance, SCAT_SIZE is the
% scatterer size in number of samples, NUM_SCAT is the number of scatterers
% per resolution cell, RES_CELL is the area of the resolution cell in
% samples, NX is the number of lateral samples in the imaging domain, and NZ
% is the number of axial samples in the imaging domain.
%
% [SLICEX,SEED] = GENERATE_C_SCAT(...) performs the same as above but
% returns the randomizer seed used so that the exact same grid can be
% reproduced.

% Modification History
% ----------------------
%
% 090507, jjd - created
% 011413, nms25- modify scat_grid

function [slicex,seed] = generate_c_scat(c0,cscale,scat_size_samp,num_scat,res_cell,nX,nZ)


% General Parameters
rand('state',sum(100*clock)) % Initialize randomizer
seed = rand('state'); % Save seed of randomizer

if (scat_size_samp > 1)
  % Scatterer is larger than one grid point
  
  % Parameters
  scat_density = num_scat/res_cell; % Scatterers per resolution cell
  num_scat_X = round(nX/scat_size_samp); % Max num scatterers in X
  num_scat_Z = round(nZ/scat_size_samp); % Max num scatterers in Z
  
  % Scattering values for the maximum number of scatterers
  max_scat_grid = rand(1,num_scat_X*num_scat_Z);
  
  % Scattering density = cumulative sum of scatterer PDF
  idx = find(max_scat_grid > scat_density);
  max_scat_grid(idx) = 0;
  
  % Reshape vector into grid of scatterer values, normalized by scattering
  % density to achieve uniform PDF over CSCALE range
  max_scat_grid= reshape(max_scat_grid,num_scat_X,num_scat_Z)/scat_density;
  
  % Scatter shape
  scatterer = biellipse(scat_size_samp/2,scat_size_samp/2);
  
  % Map scatterer values onto full-sized grid, using scatterer shape as a
  % mask to achieve shape and position of scatters on full-sized grid.
  scat_grid= zeros(nX,nZ);
  for j=1:num_scat_X
    for k=1:num_scat_Z
      for jj=1:scat_size_samp
	for kk=1:scat_size_samp
	  scat_grid((j-1)*scat_size_samp+jj,(k-1)*scat_size_samp+kk) = max_scat_grid(j,k).*scatterer(jj,kk);
	end
      end
    end
  end
  % Modification-1/14/2013 (nms25)--> Adjust Scatter Grid Size
  % if size(scat_grid)~=[nX,nZ]
      % Check nXextend
      if size(scat_grid,1)<nX
          scat_grid(end+1:nX,:)=0;
      elseif size(scat_grid,1)>nX
          scat_grid=scat_grid(1:nX,:);
      end
      % Check nZ
      if size(scat_grid,2)<nZ
          scat_grid(:,end+1:nZ)=0;
      elseif size(scat_grid,2)>nZ
          scat_grid=scat_grid(:,1:nZ);
      end
  % end
  % End Modification
  
  % Assign speed of sound values to scattering points
  scat_grid = scat_grid*c0*cscale;
  scat_grid = scat_grid + ones(nX,nZ)*c0;
  
  slicex = scat_grid;
  
else
  % Scatterer size is a single grid point

  % Parameters
  num_res_cell = round(nX*nZ/res_cell); % Number of resolution cells in image
  n = num_res_cell*num_scat; % Number of scatterers in image
  
  % Positions and Amplitude
  xpos = round(rand(n,1)*nX); % Randomized x-positions
  zpos = round(rand(n,1)*nZ); % Randomized z-positions
  amp = rand(n,1); % Randomized Amplitude
  
  % Fix and remove any 0-indicies
  p = find(xpos > 0);
  xpos = xpos(p);
  zpos = zpos(p);
  amp = amp(p);
  p = find(zpos > 0);
  xpos = xpos(p);
  zpos = zpos(p);
  amp = amp(p);
  
  % Assign positions and amplitude to velocity map
  slicex = zeros(nX,nZ);
  for xx = 1:length(amp)
    slicex(xpos(xx),zpos(xx)) = amp(xx);
  end
  
  % Scale velocities and add base velocity
  slicex = slicex*c0*cscale;
  slicex = slicex+ones(nX,nZ)*c0;
end

return
