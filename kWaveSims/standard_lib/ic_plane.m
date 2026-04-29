% IC_PLANE - create plane wave pulse initial condition for 2d simulation
%
% PIC_VEC =
% IC_PLANE(NX,NT,DT,NXDUCER,OMEGA0,NCYCLES,DURATION)
% generates a plane wave pulse as an initial condition for the 2d
% Westerveldt simulation.

function [Pic_vec] = ic_plane(nX,nT,dT,nXducer,tMin,omega0,ncycles,mdropoff)

% Pulse shape/function
pulse = @(x) exp(-(x*omega0/(ncycles*pi))^(2*mdropoff))*sin(x*omega0);

% initial condition matrix
Pic_vec = zeros(nXducer,nT);

% Pulse length [samples]
len = round(2*pi*ncycles/omega0/dT);

% Create plane wave pulse
for t=0:len*3-1
  tau = t*dT-len*2*dT;
  Pic_vec(:,t+1) = repmat(pulse(tau),nXducer,1);
end

% Append zeros to outer edge of matrix since no transmission occurs
% outside of the transducer area
d = round((nX-nXducer)/2);
Pic_vec = cat(1,zeros(d,nT),Pic_vec,zeros(d,nT));

% Set values less than tol to 0
tol = 1e-8;
p = find(abs(Pic_vec) < tol);
Pic_vec(p) = 0;


