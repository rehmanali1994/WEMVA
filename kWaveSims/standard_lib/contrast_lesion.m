% CONTRAST_LESION - create a contrast lesion for the fullwave sim
%
% C = CONTRAST_LESION(C,C0,NX,NZ,DX,DZ,POS,RADIUS,AMP) creates a contrast
% lesion with contrast AMP for every lesion described in POS and RADIUS.
% Each row of POS contains the (x,z) coordinates for a lesion, and each row
% of RADIUS contains the corresponding x- and z-axis radii.  C is the speed
% of sound scatterer field, C0 is the default speed of sound, NX is the
% number of lateral samples, NZ is the number of depth samples, DX is the
% lateral grid spacing, and DZ is the axial grid spacing.

function [c] = contrast_lesion(c,c0,nX,nZ,dX,dZ,pos,radius,amp)

% Convert amplitude from dB
amp = 10^(amp/20);

% Grid positions
xaxis = (0:nX-1)*dX;
xaxis = xaxis-mean(xaxis);
zaxis = (0:nZ-1)*dZ;
[z,x] = meshgrid(zaxis,xaxis);

p = find((((x-pos(1))/radius(1)).^2 + ((z-pos(2))/radius(2)).^2) <= 1);
c(p) = (c(p)-c0)*amp+c0;

%
%for j=1:nX
%  for k=1:nZ
%    if(((xaxis(j)-pos(1))/radius(1))^2 + ((zaxis(k)-pos(2))/radius(2))^2 ...
%       < 1)
%      if (c(j,k) ~= c0)
%	c(j,k) = (c(j,k)-c0)*amp+c0;
%      end
%    end
%  end
%end
