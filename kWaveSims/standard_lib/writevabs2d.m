function [] = writevabs2d (fname,nY,nZ,nT,dY,dZ,dT,c0,N,A,gamma,modY,modZ,modT,outdir);

fid = fopen(fname,'wt');
fprintf(fid,'outdir = ''%s''\n',outdir);
fprintf(fid,'nY = %d\n',nY);
fprintf(fid,'nZ = %d\n',nZ);
fprintf(fid,'nT = %d\n',nT);
fprintf(fid,'dY = %e\n',dY);
fprintf(fid,'dZ = %e\n',dZ);
fprintf(fid,'dT = %e\n',dT);
fprintf(fid,'c0 = %e\n',c0);
fprintf(fid,'N = %e\n',N);
fprintf(fid,'A = %e\n',A);
fprintf(fid,'gamma = %e\n',gamma);
fprintf(fid,'modY = %d\n',modY);
fprintf(fid,'modZ = %d\n',modZ);
fprintf(fid,'modT = %d\n',modT);
fclose(fid);
