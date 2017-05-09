clc;
clear;
r = 512;
c = 512;
padX = 3;
padY = 3;
blockX = 16;
blockY = 16;
file = fopen('../textFiles/varianceInput.txt','w');

NBx = ceil( ( c - padX ) /  (blockX + padX) );
NBy = ceil( ( r - padY ) /  (blockY + padY) );

GW = NBx * blockX + (NBx+1) * padX;
GH = NBy * blockY + (NBy+1) * padY;
diffH = GH - r;
diffW = GW - c;
H = GH;
W = GW;

mask=zeros(H,W);
patternString = ones(H, W);
for col = 1:blockX+padX:blockX+padX %1:N+P:N+P
    patternString(:,col:col+padX-1) = 0; % (:,col:col+P-1) = 0;
end
for row = 1:blockY+padY:blockY+padY  % row = 1:N+P:N+P
    patternString(row:row+padY-1,:) = 0;
end
for col = blockX+padX+1:blockX+padX:c+diffW % N+P+1:N+P:c+4
    patternString(:,col:col+padX-1) = 0;
end
for row = blockY+padY+1:blockY+padY:r+diffH
    patternString(row:row+padY-1,:) = 0;
end
%image = rand(W,H);
count = 1;
for i=0:W-1
    for j = 0:H-1
        image(i+1,j+1) = randi([-1,1]);
        count = count +1;
    end
end
image = image';
patternString = ~patternString;
tempMaskedImage = mask + patternString;
%tempMaskedImage = logical(tempMaskedImage);
%tempMaskedImage = ~tempMaskedImage;
%imshow(tempMaskedImage);
%figure;
%image = image .* tempMaskedImage;
fprintf(file,'%f\n', image);
%imshow(image);
%figure;

fclose(file);

im1 = image(4:19,4:19);
s1 = sum(sum(im1));
a1 = s1/256;
var1 = (im1 - a1).^2;
v1 = sum(sum(var1))/(blockX*blockY)
im2 = image(23:38,4:19);
s2 = sum(sum(im2));
a2 = s2/256;
var2 = (im2 - a2).^2;
v2 = sum(sum(var2))/(blockX*blockY)
im3 = image(4:19,23:38);
s3 = sum(sum(im3));
a3 = s3/256;
var3 = (im3 - a3).^2;
v3 = sum(sum(var3))/(blockX*blockY)
im4 = image(23:38,23:38);
s4 = sum(sum(im4));
a4 = s4/256;
var4 = (im4 - a4).^2;
v4 = sum(sum(var4))/(blockX*blockY)




