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
percentageSet = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]; %, 0.5, 0.6, 0.7, 0.8, 0.9 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
[m n] = size(percentageSet);
psnrRatio = zeros(1,totalFrame+1);
count = 1;

name = '../resultImages/psnrLighting.png';
for i=1:n
        psnrLight = 0;
        frameCounter = 1;
    for frame = 1:totalFrame%0:totalFrame-1
        path = '../textFiles/Pattern/';
        patternString = '';
        dirName = '';
        intPercent = percentageSet(i) * 100;
        gtLightDir = [path num2str(H) 'by' num2str(W) '_' num2str(100) '/Result/lighting/groundTruth/'];
        lightDir = [path num2str(H) 'by' num2str(W) '_' num2str(intPercent) '/Result/lighting/'];
        rgbFile = ['rgb_' num2str(frame) '.bin'];
        
        lightRGB = strcat(lightDir,rgbFile);
        lightingRGB = fopen(lightRGB, 'r');
        lightingRGB = fread(lightingRGB, 'float32');
        %indices = find(isnan(lightingRGB));
        %lightingRGB(isnan(lightingRGB)) = 0;
        [row col plane] = size(lightingRGB);
        lBinRed = lightingRGB(1:3:row);
        lBinGreen = lightingRGB(2:3:row);
        lBinBlue = lightingRGB(3:3:row);      
        lRed = reshape(lBinRed, [H W]);
        lGreen = reshape(lBinGreen, [H W]);
        lBlue = reshape(lBinBlue, [H W]);
        
        lightImage = cat(3, lRed, lGreen, lBlue);
%        u = repmat(uint8(255.*lightImage),[1 1 3]);

%         imshow(lightImage, []);
%         title('Reconstucted');
%         figure;
        
        gtLightRGB = strcat(gtLightDir,rgbFile);
        gtLightRGB = fopen(gtLightRGB,'r');
        gtLightRGB = fread(gtLightRGB, 'float32');
%        gtLightRGB(isnan(gtLightRGB)) = 0;
        %gtLightRGB(indices) = 0;
        [row col plabe] = size(gtLightRGB);
        gtlBinRed = gtLightRGB(1:3:row);
        gtlBinGreen = gtLightRGB(2:3:row);
        gtlBinBlue = gtLightRGB(3:3:row);       
        gtlImageR = reshape(gtlBinRed, [H W]);
        gtlImageG = reshape(gtlBinGreen, [H W]);
        gtlImageB = reshape(gtlBinBlue, [H W]);
        
        GTlightImage = cat(3, gtlImageR, gtlImageG, gtlImageB);
%         imshow(GTlightImage,[]);
%         title('Ground Truth');
%         figure;
%{
        subLight = abs(lightImage - GTlightImage);
         psnrLighting = 20 * log10(255) - 10*log10(sum(sum(subLight.^2))/(H*W*3));
         psnr = (psnrLighting(:,:,1)+psnrLighting(:,:,2)+psnrLighting(:,:,3))/3;
         psnrLight = psnrLight+((psnrLighting(:,:,1)+psnrLighting(:,:,2)+psnrLighting(:,:,3))/3);
%}

        error = GTlightImage - lightImage;
        MSE = sum(sum(sum(error.^2))) / (H * W * 3);
        
        PSNR = 20*log10(max(max(max(GTlightImage))))-10*log10(MSE);
        psnrLight = psnrLight + PSNR;
        fclose('all');
        frameCounter = frameCounter + 1;
    end
    psnrRatio(i) = psnrLight/(frameCounter);
    count = count + 1;
end

x = 1:count-1;
y = psnrRatio(x);
figure;
f = plot(x,y,'-o','LineWidth',2);
axis equal square
p = {'10';'20';'30'; '40'; '50'; '60'; '70'; '80'; '90'};
set(gca, 'XTick',[1:count-1],'XTickLabel', p)
title('PSNR for Lighting');
xlabel('percentage of using pixels');
ylabel('PSNR');
grid minor
saveas(gcf,name);

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