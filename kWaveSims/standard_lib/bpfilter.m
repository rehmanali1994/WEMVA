% BPFILTER - bandpass filter data
%
% [OUT] = BPFILTER(IN,FC1,FC2,FS,NUM_TAPS,FLAG) bandpass filters the data
% IN with the cutoff frequencies FC1 and FC2 and returns the filtered
% data in OUT.  FS is the axial sampling frequency.   FC1, FC2, and FS
% are specified in Hz. NUM_TAPS is the number of desired taps in the FIR
% filter, and should be even.  FLAG specifies whether to use a FIR or IIR
% filter, or both.  See the following table for the relationship between
% FLAG and OUT. 
%
%      FLAG     OUT
%
%        1      FIR filtered data
%        2      IIR filtered data
%
% Example:
% out = bpfilter(in,3e6,10e6,40e6,50,1);

% by BJG 11/14/99
% updated 8/6/03 by jjd to increase speed and make more general

function out = bpfilter(in,fc1,fc2,fs,num_taps,flag)

% Convert analog to digital frequencies
wn = 2*[fc1 fc2]/fs;

% Zero pad the input to obtain the proper output
%pad_size = round(taps/2);  % pad_size is half the # of taps
%pad = zeros(pad_size,size(in,2));
%in = cat(1,in,pad);

% Filter the data
if (flag == 1)
  % FIR filter
  % Zero pad the input to obtain the proper output
  pad_size = round(num_taps/2);    % pad_size is half the # of taps
  pad = zeros(pad_size,size(in,2));
  in = cat(1,in,pad);
  b_fir = fir1(num_taps, wn,hanning(num_taps+1));
  a_fir = 1;
  out_fir = filter(b_fir,a_fir,in);
elseif (flag == 2)
  % IIR filter
  [b_iir a_iir] = butter(num_taps,wn);
  [row col] = size(in);
  for jj = 1:col
    zi = filtic(b_iir,a_iir,zeros(1,num_taps),in(1,jj)*ones(1,num_taps));
    out_iir(:,jj) = filter(b_iir,a_iir,in(:,jj),zi);
  end
else
  error('Flag not valid.')
end

% Show response of filters
showresponse = 0;
if (showresponse == 1)
  if (flag == 1)
    [h_fir f_fir] = freqz(b_fir,a_fir,100,fs);
    figure
    plot(f_fir, abs(h_fir))
    title('FIR')
  else
    [h_iir f_iir] = freqz(b_iir,a_iir,100,fs);
    figure
    plot(f_iir, abs(h_iir))
    title('IIR')
  end
end

% Show Power Spectral Density of input and output data
showpsd = 0;
if (showpsd == 1)
  if (flag == 1) % Show only for FIR filter
    for k=1:size(in,2)
      [p_in(:,k) ff_in] = psd(in(:,k),[],fs);
      [p_out_fir(:,k) ff_fir] = psd(out_fir(:,k),[],fs);
    end
    pp_in = mean(p_in');
    pp_in_log = 20*log10(pp_in);
    figure
    plot(ff_in,pp_in_log)
    pp_out_fir = mean(p_out_fir');
    pp_out_fir_log = 20*log10(pp_out_fir);
    figure
    plot(ff_fir,pp_out_fir_log)
    figure
    colormap(gray)
    imagesc(in)
    figure
    colormap(gray)
    imagesc(out_fir)
  end
end

% designate output data
if (flag == 1)
  out = out_fir(pad_size+1:end,:);
else
  out = out_iir;
end

return


