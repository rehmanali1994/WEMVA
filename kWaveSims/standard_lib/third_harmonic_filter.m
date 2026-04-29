function [aline1 aline2 aline3] = third_harmonic_filter(aline,dT,omega0,nwidth)

fs = 1/(dT);
fsize = length(aline);
f = fs*(0:fsize-1)/fsize;
Faline = fft(aline,fsize)/fsize;
%figure, plot(f,Faline)

idx = find(f>=omega0/2/pi); idx=idx(1);
%nwidth = 30;
aline1 = zeros(size(Faline));
tmp = zeros(size(Faline));
tmp(idx-nwidth+1:idx+nwidth) = Faline(idx-nwidth+1:idx+nwidth).*hanning(2*nwidth);
aline1 = ifft(tmp,'symmetric')*fsize;
% figure(30), plot(aline1)

idx = find(f>=2*omega0/2/pi); idx=idx(1);
%nwidth = 30;
aline2 = zeros(size(Faline));
tmp = zeros(size(Faline));
tmp(idx-nwidth+1:idx+nwidth) = Faline(idx-nwidth+1:idx+nwidth).*hanning(2*nwidth);
aline2 = ifft(tmp,'symmetric')*fsize;
% figure(31), plot(aline2)

idx = find(f>=3*omega0/2/pi); idx=idx(1);
%nwidth = 30;
aline3 = zeros(size(Faline));
tmp = zeros(size(Faline));
tmp(idx-nwidth+1:idx+nwidth) = Faline(idx-nwidth+1:idx+nwidth).*hanning(2*nwidth);
aline3 = ifft(tmp,'symmetric')*fsize;
% figure(31), plot(aline2)
