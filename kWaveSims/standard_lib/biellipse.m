% BIELLIPSE - Generate Binary Image of Ellipse
%
% IM = BIELLIPSE(A,B) generates a binary image of an ellipse
% where A specifies the radius on the horizontal axis and B
% specifies the radius on the vertical axis.

function [im] = biellipse(a,b);

yrange = -floor(a):floor(a);
xrange = -floor(b):floor(b);

% create the elliptical field
for x = xrange;
  for y = yrange;
    ellipse(x+floor(b)+1,y+floor(a)+1) = ((x.^2)/(b^2))+((y.^2)/(a^2));
  end;
end;

% ellipse exist for value <= 1
im = le(ellipse,1);
im = double(im);

% convert matrix to binary image
im = mat2gray(im,[0,1]);
im = im2bw(im);
return;



