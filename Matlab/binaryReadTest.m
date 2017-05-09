clear
clc
r = 512;
c = 512;
padX = 3;
padY = 3;
blockX = 16;
blockY = 16;
totalFrame = 2;

NBx = ceil( ( c - padX ) /  (blockX + padX) );
NBy = ceil( ( r - padY ) /  (blockY + padY) );

GW = NBx * blockX + (NBx+1) * padX;
GH = NBy * blockY + (NBy+1) * padY;
diffH = GH - r;
diffW = GW - c;
H = GH;
W = GW;
path = '../textFiles/Pattern/';
patternString = '';
dirName = '';

intPercent = 40;
frame = 2;
patternString = [num2str(GH) 'by' num2str(GW)]; %516by516_30

dirName = [num2str(H) 'by' num2str(W) '_' num2str(intPercent) '/Result/'];
lightDir = [path num2str(H) 'by' num2str(W) '_' num2str(intPercent) '/Result/lighting/'];
gtLightDir = [path num2str(H) 'by' num2str(W) '_' num2str(100) '/Result/lighting/groundTruth/'];
%cubicDir = [path num2str(H) 'by' num2str(W) '_' num2str(intPercent) '/Result/tricubic/'];

dirName = strcat(path,dirName);
lightingFile = strcat(dirName,'lighting/');
gtLightingFile = gtLightDir

rgbFile = ['rgb_' num2str(frame) '.bin'];

lRGB = strcat(lightingFile, rgbFile)

lRGB = fopen(lRGB, 'r');
lrgb = fread(lRGB, 'float32');
[row col plane] = size(lrgb);
lBinRed = lrgb(1:3:row);
lBinGreen = lrgb(2:3:row);
lBinBlue = lrgb(3:3:row);

lImageR = reshape(lBinRed, [H W]);
lImageG = reshape(lBinGreen, [H W]);
lImageB = reshape(lBinBlue, [H W]);
lightImage = cat(3, lImageR, lImageG, lImageB);
%lightImage = uint8(lightImage);
imshow(lightImage, []);
title('Reconstructed image');
figure;

gtLightRGB = strcat(gtLightingFile,rgbFile)

gtLightRGB = fopen(gtLightRGB,'r');


gtLightRGB = fread(gtLightRGB, 'float32');
gtlBinRed = gtLightRGB(1:3:row);
gtlBinGreen = gtLightRGB(2:3:row);
gtlBinBlue = gtLightRGB(3:3:row);

gtlImageR = reshape(gtlBinRed, [H W]);
gtlImageG = reshape(gtlBinGreen, [H W]);
gtlImageB = reshape(gtlBinBlue, [H W]);
GTlightImage = cat(3, lImageR, lImageG, lImageB);
%GTlightImage = uint8(lightImage);
imshow(GTlightImage, []);
title('Ground Truth image');
%figure;
lightSub = abs(GTlightImage - lightImage);
%imshow(lightSub, []);
psnrLightImage = 20 * log10(255) - 10*log10(sum(sum(lightSub.^2))/(H*W));
