clear;
clc;
r = 512;
c = 512;
padX = 3;
padY = 3;
blockX = 16;
blockY = 16;
totalFrame = 100;

NBx = ceil( ( c - padX ) /  (blockX + padX) );
NBy = ceil( ( r - padY ) /  (blockY + padY) );

GW = NBx * blockX + (NBx+1) * padX;
GH = NBy * blockY + (NBy+1) * padY;
diffH = GH - r;
diffW = GW - c;
H = GH;
W = GW;
dataSet = 'skull';
dirName = ['../Results/' num2str(H) 'by' num2str(W) '/'];
psnrLinearLightOn = zeros(1,totalFrame);
psnrLinearLightOff = zeros(1,totalFrame);
psnrCubicLightOn = zeros(1,totalFrame);
psnrCubicLightOff = zeros(1,totalFrame);
psnrIsoSurface = zeros(1, totalFrame);
count = 1;

for frame = 1:totalFrame
    
    LinearLightOn = 0;
    LinearLightOff = 0;
    CubicLightOn = 0;
    CubicLightOff = 0;
    isoSurface = 0;
     rgbFile = ['rgb_' num2str(frame) '.bin'];
    
    
    % tri-linear
    gtLinearOn = [dirName 'triLinear/lightingOn/groundTruth/'];
    gtLinearOff = [dirName 'triLinear/lightingOff/groundTruth/'];
    LinearOn = [dirName 'triLinear/lightingOn/reconstructed/'];
    LinearOff = [dirName 'triLinear/lightingOff/reconstructed/'];
    
    triLinearOnFile = strcat(LinearOn, rgbFile);
    triLinearOffFile = strcat(LinearOff, rgbFile);
    gtTriLinearOnFile = strcat(gtLinearOn, rgbFile);
    gtTriLinearOffFile = strcat(gtLinearOff, rgbFile);
    triLinearOnRGB = fopen(triLinearOnFile,'r');
    triLinearOnValue = fread(triLinearOnRGB,'float32');
    gtTriLinearOnRGB = fopen(gtTriLinearOnFile,'r');
    gtTriLinearOnValue = fread(gtTriLinearOnRGB,'float32');
    triLinearOffRGB = fopen(triLinearOffFile,'r');
    triLinearOffValue = fread(triLinearOffRGB,'float32');
    gtTriLinearOffRGB = fopen(gtTriLinearOffFile,'r');
    gtTriLinearOffValue = fread(gtTriLinearOffRGB,'float32');
    [row col plane] = size(triLinearOnValue);
    triLinearOnRed = triLinearOnValue(1:3:row);
    triLinearOnGreen = triLinearOnValue(2:3:row);
    triLinearOnBlue = triLinearOnValue(3:3:row);
    linearOnRed = reshape(triLinearOnRed, [H W]);
    linearOnGreen = reshape(triLinearOnGreen, [H W]);
    linearOnBlue = reshape(triLinearOnBlue, [H W]);
    linearLightOn = cat(3, linearOnRed, linearOnGreen, linearOnBlue);
    
    gtTriLinearOnRed = gtTriLinearOnValue(1:3:row);
    gtTriLinearOnGreen = gtTriLinearOnValue(2:3:row);
    gtTriLinearOnBlue = gtTriLinearOnValue(3:3:row);
    gtlinearOnRed = reshape(gtTriLinearOnRed, [H W]);
    gtlinearOnGreen = reshape(gtTriLinearOnGreen, [H W]);
    gtlinearOnBlue = reshape(gtTriLinearOnBlue, [H W]);
    gtlinearLightOn = cat(3, gtlinearOnRed, gtlinearOnGreen, gtlinearOnBlue);
    
    error = gtlinearLightOn - linearLightOn;
    MSE = sum(sum(sum(error.^2))) / (H * W * 3);
    PSNR = 20*log10(max(max(max(gtlinearLightOn))))-10*log10(MSE);
    LinearLightOn = LinearLightOn + PSNR;
    psnrLinearLightOn(frame) = PSNR;
    PSNR = 0;
    
    triLinearOffRed = triLinearOffValue(1:3:row);
    triLinearOffGreen = triLinearOffValue(2:3:row);
    triLinearOffBlue = triLinearOffValue(3:3:row);
    linearOffRed = reshape(triLinearOffRed, [H W]);
    linearOffGreen = reshape(triLinearOffGreen, [H W]);
    linearOffBlue = reshape(triLinearOffBlue, [H W]);
    linearLightOff = cat(3, linearOffRed, linearOffGreen, linearOffBlue);
    
    gtTriLinearOffRed = gtTriLinearOffValue(1:3:row);
    gtTriLinearOffGreen = gtTriLinearOffValue(2:3:row);
    gtTriLinearOffBlue = gtTriLinearOffValue(3:3:row);
    gtlinearOffRed = reshape(gtTriLinearOffRed, [H W]);
    gtlinearOffGreen = reshape(gtTriLinearOffGreen, [H W]);
    gtlinearOffBlue = reshape(gtTriLinearOffBlue, [H W]);
    gtlinearLightOff = cat(3, gtlinearOffRed, gtlinearOffGreen, gtlinearOffBlue);
    
    error = gtlinearLightOff - linearLightOff;
    MSE = sum(sum(sum(error.^2))) / (H * W * 3);
    PSNR = 20*log10(max(max(max(gtlinearLightOff))))-10*log10(MSE);
    LinearLightOff = LinearLightOff + PSNR;
    psnrLinearLightOff(frame) = PSNR;
    PSNR = 0;
    
    
    % tri-cubic
    gtCubicOn = [dirName 'triCubic/lightingOn/groundTruth/'];
    gtCubicOff = [dirName 'triCubic/lightingOff/groundTruth/'];
    CubicOn = [dirName 'triCubic/lightingOn/reconstructed/'];
    CubicOff = [dirName 'triCubic/lightingOff/reconstructed/'];
    triCubicOnFile = strcat(CubicOn, rgbFile);
    triCubicOffFile = strcat(CubicOff, rgbFile);
    gtTriCubicOnFile = strcat(gtCubicOn, rgbFile);
    gtTriCubicOffFile = strcat(gtCubicOff, rgbFile);
    triCubicOnRGB = fopen(triCubicOnFile,'r');
    triCubicOnValue = fread(triCubicOnRGB,'float32');
    gtTriCubicOnRGB = fopen(gtTriCubicOnFile,'r');
    gtTriCubicOnValue = fread(gtTriCubicOnRGB,'float32');
    triCubicOffRGB = fopen(triCubicOffFile,'r');
    triCubicOffValue = fread(triCubicOffRGB,'float32');
    gtTriCubicOffRGB = fopen(gtTriCubicOffFile,'r');
    gtTriCubicOffValue = fread(gtTriCubicOffRGB,'float32');
    
    [row col plane] = size(triCubicOnValue);
    triCubicOnRed = triCubicOnValue(1:3:row);
    triCubicOnGreen = triCubicOnValue(2:3:row);
    triCubicOnBlue = triCubicOnValue(3:3:row);
    cubicOnRed = reshape(triCubicOnRed, [H W]);
    cubicOnGreen = reshape(triCubicOnGreen, [H W]);
    cubicOnBlue = reshape(triCubicOnBlue, [H W]);
    cubicLightOn = cat(3, cubicOnRed, cubicOnGreen, cubicOnBlue);
    
    gtTriCubicOnRed = gtTriCubicOnValue(1:3:row);
    gtTriCubicOnGreen = gtTriCubicOnValue(2:3:row);
    gtTriCubicOnBlue = gtTriCubicOnValue(3:3:row);
    gtcubicOnRed = reshape(gtTriCubicOnRed, [H W]);
    gtcubicOnGreen = reshape(gtTriCubicOnGreen, [H W]);
    gtcubicOnBlue = reshape(gtTriCubicOnBlue, [H W]);
    gtcubicLightOn = cat(3, gtcubicOnRed, gtcubicOnGreen, gtcubicOnBlue);
    
    error = gtcubicLightOn - cubicLightOn;
    MSE = sum(sum(sum(error.^2))) / (H * W * 3);
    PSNR = 20*log10(max(max(max(gtcubicLightOn))))-10*log10(MSE);
    CubicLightOn = CubicLightOn + PSNR;
    psnrCubicLightOn(frame) = PSNR;
    PSNR = 0;
    
    triCubicOffRed = triCubicOffValue(1:3:row);
    triCubicOffGreen = triCubicOffValue(2:3:row);
    triCubicOffBlue = triCubicOffValue(3:3:row);
    cubicOffRed = reshape(triCubicOffRed, [H W]);
    cubicOffGreen = reshape(triCubicOffGreen, [H W]);
    cubicOffBlue = reshape(triCubicOffBlue, [H W]);
    cubicLightOff = cat(3, cubicOffRed, cubicOffGreen, cubicOffBlue);
    
    gtTriCubicOffRed = gtTriCubicOffValue(1:3:row);
    gtTriCubicOffGreen = gtTriCubicOffValue(2:3:row);
    gtTriCubicOffBlue = gtTriCubicOffValue(3:3:row);
    gtcubicOffRed = reshape(gtTriCubicOffRed, [H W]);
    gtcubicOffGreen = reshape(gtTriCubicOffGreen, [H W]);
    gtcubicOffBlue = reshape(gtTriCubicOffBlue, [H W]);
    gtcubicLightOff = cat(3, gtcubicOffRed, gtcubicOffGreen, gtcubicOffBlue);
    
    error = gtcubicLightOff - cubicLightOff;
    MSE = sum(sum(sum(error.^2))) / (H * W * 3);
    PSNR = 20*log10(max(max(max(gtcubicLightOff))))-10*log10(MSE);
    CubicLightOff = CubicLightOff + PSNR;
    psnrCubicLightOff(frame) = PSNR;
    PSNR = 0;
    
    % iso-surface
    gtIsoSurface = [dirName 'isoSurface/groundTruth/'];
    IsoSurface = [dirName 'isoSurface/reconstructed/'];
    
   
    
    %tri-linear
   
    %tri-cubic
    
    %iso-surface
    isoSurfaceFile = strcat(IsoSurface, rgbFile);
    gtIsoSurfaceFile = strcat(gtIsoSurface, rgbFile);
    
    %file-open and reading
    
    

    
    isoSurfaceRGB = fopen(isoSurfaceFile,'r');
    triCubicOnValue = fread(isoSurfaceRGB,'float32');
    gtIsoSurfaceRGB = fopen(gtIsoSurfaceFile,'r');
    triCubicOnValue = fread(gtIsoSurfaceRGB,'float32');
    % Tri-Linear PSNR---------------------------------------->>>>>>
    
    
    % Tri-Cubic PSNR---------------------------------------->>>>>>
    
    
    
    
    fclose('all');
end
x = 1:totalFrame;
yLightingOn = psnrLinearLightOn(x);
yLightingOff = psnrLinearLightOff(x);

plot(x, yLightingOn,'-o', x, yLightingOff,'-*');
legend('Tri-Linear Lighting on', 'Tri-Linear Lighting off');
grid on
grid minor
axis equal square
title('PSNR for Tri-linear Lighting On Vs Lighting off');
xlabel('Frame No');
ylabel('PSNR');
%hold on
figure;

yCubicLightOn = psnrCubicLightOn(x);
yCubicLightOff = psnrCubicLightOff(x);
plot(x, yCubicLightOn,'-o', x, yCubicLightOff,'-*');
legend('Tri-Cubic Lighting on', 'Tri-cubic Lighting off');
grid on
grid minor
axis equal square
title('PSNR for Tri-Cubic Lighting On Vs Lighting off');
xlabel('Frame No');
ylabel('PSNR');
%{
x = 1:count-1;
yLight = psnrRatioLight(x);
yCubic = psnrRatioCubic(x);
figure;
plot(x,yLight,'-o',x,yCubic,'-o','LineWidth',2);
grid on
grid minor
legend('Tri-linear','Tri-cubic');
axis equal square
p = {'30'; '40'; '50'; '60'; '70'; '80'; '90'};
set(gca, 'XTick',[1:count-1],'XTickLabel', p)
title('PSNR for Tri-linear and Tri-cubic');
xlabel('percentage of using pixels');
ylabel('PSNR');
saveas(gcf,name);
%}
%{
path = '../textFiles/Pattern/';
patternString = '';
dirName = '';
gtLightDir = [path num2str(H) 'by' num2str(W) '_' num2str(100) '/Result/lighting/groundTruth/'];
lightDir = [path num2str(H) 'by' num2str(W) '_' num2str(50) '/Result/lighting/'];
frame = 2;
rgbFile = ['rgb_' num2str(frame) '.bin'];

lightRGB = strcat(lightDir,rgbFile)
lightingRGB = fopen(lightRGB, 'r');
lightingRGB = fread(lightingRGB, 'float32');
[row col plane] = size(lightingRGB);
lBinRed = lightingRGB(1:3:row);
lBinGreen = lightingRGB(2:3:row);
lBinBlue = lightingRGB(3:3:row);

lBinRed = reshape(lBinRed, [H W]);
lBinGreen = reshape(lBinGreen, [H W]);
lBinBlue = reshape(lBinBlue, [H W]);

lightImage = cat(3, lBinRed, lBinGreen, lBinBlue);
imshow(lightImage, []);
title('image');
figure;

gtLightRGB = strcat(gtLightDir,rgbFile)
gtLightRGB = fopen(gtLightRGB,'r');
gtLightRGB = fread(gtLightRGB, 'float32');
[row col plabe] = size(gtLightRGB);
gtlBinRed = gtLightRGB(1:3:row);
gtlBinGreen = gtLightRGB(2:3:row);
gtlBinBlue = gtLightRGB(3:3:row);

gtlImageR = reshape(gtlBinRed, [H W]);
gtlImageG = reshape(gtlBinGreen, [H W]);
gtlImageB = reshape(gtlBinBlue, [H W]);
GTlightImage = cat(3, gtlImageR, gtlImageG, gtlImageB);
%GTlightImage = uint8(lightImage);
imshow(GTlightImage, []);
title('Ground Truth image');
%}