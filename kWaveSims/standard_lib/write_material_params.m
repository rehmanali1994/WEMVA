function [] = write_material_params(nlines,outdir,cextend,Nextend,Aextend,rhoextend,pIc_basic,nY,nZ,nT,dY,dZ,dT,c0,N0,A0,gamma,modY,modZ,modT,dline_samp,ppe,nXducer,num_elements)

half = round(nlines/2);

physical_ele_size = (nXducer/num_elements); % [samples]
apod_profile_indices = (0:nXducer-1)'; % the position of each virtual element of apod profile in samples


for m=1:nlines
  m
  apod_profile = zeros(nXducer,1);
%  idc_on = zeros(nXducer,1);
%  idc_on = (m-1)*ppe+1:min(m*ppe,size(apod_profile,1));
  apod_profile( ((apod_profile_indices >= (m-1)*physical_ele_size) & (apod_profile_indices< physical_ele_size*m)) ) = 1;
  
  d = round((nY-nXducer)/2);
  apod_profile2 = cat(1,zeros(d,nT),repmat(apod_profile,[1 nT]),zeros(d,nT));

  % apply apodization profile
  pIc = pIc_basic .* apod_profile2';  

  % c = cextend(1+dline_samp*(m-1):nY+dline_samp*(m-1),:);
  % N = Nextend(1+dline_samp*(m-1):nY+dline_samp*(m-1),:);
  % A = Aextend(1+dline_samp*(m-1):nY+dline_samp*(m-1),:);
  % rho = rhoextend(1+dline_samp*(m-1):nY+dline_samp*(m-1),:);
  c = cextend(1+dline_samp*(m-1):nY+dline_samp*(m-1),:);
  N = Nextend(1+dline_samp*(m-1):nY+dline_samp*(m-1),:);
  A = Aextend(1+dline_samp*(m-1):nY+dline_samp*(m-1),:);
  rho = rhoextend(1+dline_samp*(m-1):nY+dline_samp*(m-1),:);
  fid=fopen([outdir num2str(m-1) '/c.dat'],'wb'); fwrite(fid,c','float'); fclose(fid);
  fid = fopen([outdir num2str(m-1) '/pIc.dat'],'wb'); fwrite(fid,pIc','float'); fclose(fid);
  fid=fopen([outdir num2str(m-1) '/A.dat'],'wb'); fwrite(fid,A','float'); fclose(fid);
  fid=fopen([outdir num2str(m-1) '/N.dat'],'wb'); fwrite(fid,N','float'); fclose(fid);
  fid=fopen([outdir num2str(m-1) '/rho.dat'],'wb'); fwrite(fid,rho','float'); fclose(fid);
  writevabs2d ([outdir num2str(m-1) '/vabs2d.m'],nY,nZ,nT,dY,dZ,dT,c0,N0,A0,gamma,modY,modZ,modT,[outdir num2str(m-1) '/']);
end
