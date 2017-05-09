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
percentageSet = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]; %, 0.5, 0.6, 0.7, 0.8, 0.9
[m n] = size(percentageSet);
psnrRatioLight = zeros(1,totalFrame);
psnrRatioCubic = zeros(1,totalFrame);
count = 1;
name = '../resultImages/psnrLightingCubic.png';
for i=1:n
        psnrLight = 0;
        psnrCubic = 0;
        frameCounter = 1;
    for frame = 10:10:totalFrame
        path = '../textFiles/Pattern/';
        patternString = '';
        dirName = '';
        intPercent = percentageSet(i) * 100;
        gtLightDir = [path num2str(H) 'by' num2str(W) '_' num2str(100) '/Result/lighting/groundTruth/'];
        lightDir = [path num2str(H) 'by' num2str(W) '_' num2str(intPercent) '/Result/lighting/'];
        
        gtCubicDir = [path num2str(H) 'by' num2str(W) '_' num2str(100) '/Result/tricubic/groundTruth/'];
        CubicDir = [path num2str(H) 'by' num2str(W) '_' num2str(intPercent) '/Result/tricubic/'];
        
        rgbFile = ['rgb_' num2str(frame) '.bin'];
        
        lightRGB = strcat(lightDir,rgbFile);
        cubicRGB = strcat(CubicDir,rgbFile);
        
        lightingRGB = fopen(lightRGB, 'r');
        lightingRGB = fread(lightingRGB, 'float32');
%         indices = find(isnan(lightingRGB));
%         lightingRGB(isnan(lightingRGB)) = 0;
        [row col plane] = size(lightingRGB);
        lBinRed = lightingRGB(1:3:row);
        lBinGreen = lightingRGB(2:3:row);
        lBinBlue = lightingRGB(3:3:row);      
        lRed = reshape(lBinRed, [H W]);
        lGreen = reshape(lBinGreen, [H W]);
        lBlue = reshape(lBinBlue, [H W]);
        
        lightImage = cat(3, lRed, lGreen, lBlue);
%         imshow(lightImage, []);
%         title('Reconstucted');
%         figure;
        triCubicRGB = fopen(cubicRGB, 'r');
        triCubic = fread(triCubicRGB, 'float32');
%         indices = find(isnan(triCubic));
%         triCubic(isnan(triCubic)) = 0;
        [row col plane] = size(triCubic);
        tBinRed = triCubic(1:3:row);
        tBinGreen = triCubic(2:3:row);
        tBinBlue = triCubic(3:3:row);      
        lRed = reshape(tBinRed, [H W]);
        lGreen = reshape(tBinGreen, [H W]);
        lBlue = reshape(tBinBlue, [H W]);
        
        cubicImage = cat(3, lRed, lGreen, lBlue);

        
        gtLightRGB = strcat(gtLightDir,rgbFile);
        gtLightRGB = fopen(gtLightRGB,'r');
        gtLightRGB = fread(gtLightRGB, 'float32');
%        gtLightRGB(isnan(gtLightRGB)) = 0;
%        gtLightRGB(indices) = 0;
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
        gtCubicRGB = strcat(gtCubicDir,rgbFile);
        gtCubicRGB = fopen(gtCubicRGB,'r');
        gtCubicRGB = fread(gtCubicRGB, 'float32');
%        gtLightRGB(isnan(gtLightRGB)) = 0;
%        gtCubicRGB(indices) = 0;
        [row col plabe] = size(gtCubicRGB);
        gtCBinRed = gtCubicRGB(1:3:row);
        gtCBinGreen = gtCubicRGB(2:3:row);
        gtCBinBlue = gtCubicRGB(3:3:row);       
        gtCImageR = reshape(gtCBinRed, [H W]);
        gtCImageG = reshape(gtCBinGreen, [H W]);
        gtCImageB = reshape(gtCBinBlue, [H W]);
        
        GTCubicImage = cat(3, gtCImageR, gtCImageG, gtCImageB);
        
        %{
        subLight = abs(lightImage - GTlightImage);
         psnrLighting = 20 * log10(255) - 10*log10(sum(sum(subLight.^2))/(H*W));
         psnrLight = psnrLight+((psnrLighting(:,:,1)+psnrLighting(:,:,2)+psnrLighting(:,:,3))/3);
        %}
        error = GTlightImage - lightImage;
        MSE = sum(sum(sum(error.^2))) / (H * W * 3);        
        PSNR = 20*log10(max(max(max(GTlightImage))))-10*log10(MSE);
        psnrLight = psnrLight + PSNR;
   %{      
         subCubic = abs(cubicImage - GTCubicImage);
         psnrTriCubic = 20 * log10(255) - 10*log10(sum(sum(subCubic.^2))/(H*W));
         psnrCubic = psnrCubic+((psnrTriCubic(:,:,1)+psnrTriCubic(:,:,2)+psnrTriCubic(:,:,3))/3);
     %}    
         error = GTCubicImage - cubicImage;
        MSE = sum(sum(sum(error.^2))) / (H * W * 3);        
        PSNR = 20*log10(max(max(max(GTCubicImage))))-10*log10(MSE);
        psnrCubic = psnrCubic+ + PSNR;
          
         
          frameCounter = frameCounter +1;
        fclose('all');
    end
    psnrRatioLight(i) = psnrLight/frameCounter;
    psnrRatioCubic(i) = psnrCubic/frameCounter;
    count = count + 1;
end

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