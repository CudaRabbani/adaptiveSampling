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

intPercent = 30;
frame = 2;
patternString = [num2str(GH) 'by' num2str(GW)]; %516by516_30

dirName = [num2str(H) 'by' num2str(W) '_' num2str(intPercent) '/Result/'];
lightDir = [path num2str(H) 'by' num2str(W) '_' num2str(intPercent) '/Result/lighting/'];
cubicDir = [path num2str(H) 'by' num2str(W) '_' num2str(intPercent) '/Result/tricubic/'];

dirName = strcat(path,dirName);
lightingFile = strcat(dirName,'lighting/');
gtLightingFile = strcat(lightDir,'groundTruth/');


redFile = ['red_' num2str(frame) '.txt'];
greenFile = ['green_' num2str(frame) '.txt'];
blueFile = ['blue_' num2str(frame) '.txt'];


lRed = strcat(lightingFile, redFile)
lGreen = strcat(lightingFile, greenFile)
lBlue = strcat(lightingFile, blueFile)

lRed = fopen(lRed, 'r');
lGreen = fopen(lGreen, 'r');
lBlue = fopen(lBlue, 'r');

lightRed = fscanf(lRed, '%f');
lightGreen = fscanf(lGreen, '%f');
lightBlue = fscanf(lBlue, '%f');

lImageR = reshape(lightRed, [H W]);
lImageG = reshape(lightGreen, [H W]);
lImageB = reshape(lightBlue, [H W]);
lightImage = cat(3, lImageR, lImageG, lImageB);
%lightImage = uint8(lightImage);
imshow(lightImage, []);
figure;

gtLightRed = strcat(gtLightingFile, redFile);
gtLightGreen = strcat(gtLightingFile, greenFile);
gtLightBlue = strcat(gtLightingFile, blueFile);

gtLRed = fopen(gtLightRed, 'r');
gtLGreen = fopen(gtLightGreen, 'r');
gtLBlue = fopen(gtLightBlue, 'r');

glightRed = fscanf(gtLRed, '%f');
glightGreen = fscanf(gtLGreen, '%f');
glightBlue = fscanf(gtLBlue, '%f');

gtlImageR = reshape(glightRed, [H W]);
gtlImageG = reshape(glightGreen, [H W]);
gtlImageB = reshape(glightBlue, [H W]);
GTlightImage = cat(3, lImageR, lImageG, lImageB);
%GTlightImage = uint8(lightImage);
imshow(GTlightImage, []);

figure;
lightSub = abs(GTlightImage - lightImage);
imshow(lightSub, []);
psnrLightImage = 20 * log10(255) - 10*log10(sum(sum(lightSub.^2))/(H*W))
