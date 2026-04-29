function [c rho A N rho0 c0] = imagemap_params(folder, nXextend,dX,nZ,dZ,tissue,fat,muscle,connective,imset)

switch imset
 case 1
  img = imread([folder '/r102gh.tif'],'TIFF');
  % img = imread('/data/mjakovlj/data/mast/r102gh.tif','TIFF');
 case 2
  img = imread([folder '/r120de.tif'],'TIFF');
  % img = imread('/data/mjakovlj/data/mast/r120de.tif','TIFF');
 case 3
  img = imread([folder '/r87de.tif'],'TIFF');
  % img = imread('/data/mjakovlj/data/mast/r87de.tif','TIFF');
 case 4
  img = imread([folder '/r77ba.tif'],'TIFF');
  % img = imread('/data/mjakovlj/data/mast/r77ba.tif','TIFF');
 case 5
  img = imread([folder '/r120fe.tif'],'TIFF');
  % img = imread('/data/mjakovlj/data/mast/r120fe.tif','TIFF');
 case 6
  img = imread([folder '/r75hi.tif'],'TIFF');
  % img = imread('/data/mjakovlj/data/mast/r75hi.tif','TIFF');
 otherwise
  img = imread([folder '/r102gh.tif'],'TIFF');
  % img = imread('/data/mjakovlj/data/mast/r102gh.tif','TIFF');
end

% Get relevant imaging area
switch imset
 case 4
  img = img(:,1:round(dX/(1e-3/11.81102)*nXextend))';
 otherwise
  img = img(:,650-round(dX/(1e-3/11.81102)*nXextend/2):650+round(dX/(1e-3/11.81102)*nXextend/2))';
end
% Move surface of abdominal layer to edge if necessary
g = min(find(img == 0));
mx = size(img,1);
zz = floor(g/mx)+1;
tmp = 250*ones(size(img));
tmp(:,1:end-zz+1) = img(:,zz:end);
img = tmp;

yaxisimg = (0:size(img,1)-1)*1e-3/11.81102;
zaxisimg = (0:size(img,2)-1)*1e-3/11.81102;
[X Y] = meshgrid(yaxisimg,zaxisimg);
[XI YI] = meshgrid(0:dX:yaxisimg(end),0:dZ:zaxisimg(end));
img2 = interp2(X,Y,double(img'),XI,YI,'nearest')';

img3 = ones(nXextend,nZ)*250;
img3(1:size(img2,1),1:size(img2,2)) = img2;
img3 = img3(1:nXextend,:);

c = zeros(size(img3));
alpha = zeros(size(img3));
rho = zeros(size(img3));
beta = zeros(size(img3));

% water/liver
[row col] = find(img3==250);
for i=1:length(row)
  c(row(i),col(i)) = tissue.c0;
  rho(row(i),col(i)) = tissue.rho0;
  alpha(row(i),col(i)) = tissue.alpha;
  beta(row(i),col(i)) = tissue.beta;
end
% fat
[row col] = find(img3==200);
for i=1:length(row)
  c(row(i),col(i)) = fat.c0;
  rho(row(i),col(i)) = fat.rho0;
  alpha(row(i),col(i)) = fat.alpha;
  beta(row(i),col(i)) = fat.beta;
end

% muscle
[row col] = find(img3==138);
for i=1:length(row)
  c(row(i),col(i)) = muscle.c0;
  rho(row(i),col(i)) = muscle.rho0;
  alpha(row(i),col(i)) = muscle.alpha;
  beta(row(i),col(i)) = muscle.beta;
end

% connective tissue
[row col] = find(img3==0);
for i=1:length(row)
  c(row(i),col(i)) = connective.c0;
  rho(row(i),col(i)) = connective.rho0;
  alpha(row(i),col(i)) = connective.alpha;
  beta(row(i),col(i)) = connective.beta;
end

rho0 = 1000;
c0 = 1540;
N = -beta/(rho0*c0^4);
%A = 2*c0^2*(alpha)/omega0^2/c0^4;
A = alpha;
