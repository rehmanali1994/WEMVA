function [Pic_vec] = ic_spherical(nY,nT,dY,dT,tMin,nYducer,omega0,dy,c0,ncycles,mdropoff)

pulse = @(x) exp(-(x*omega0/(ncycles*pi))^(2*mdropoff))*sin(x*omega0);
Pic_vec = zeros(1,nY*nT);
pi*ncycles/omega0;
%tMin = -nT*dT/2;
for t=0:nT-1
  tau = t*dT + tMin;
  if(tau<2*pi*ncycles/omega0)
    for j=0:nY-1
      idy = j*dY-nY*dY/2;
      if(abs(idy)<=nYducer/2*dY)
	Pic_vec(j*nT+t+1) = pulse(tau+sqrt(idy.^2+dy^2)/c0-dy/c0);
      end
    end
  end
end
