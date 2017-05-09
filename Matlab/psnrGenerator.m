r = 512;
c = 512;
padX = 3;
padY = 3;
blockX = 16;
blockY = 16;
totalFrame = 5;

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
psnrRatio = zeros(1,totalFrame+1);
count = 1;

for i =1:n
    psnrLight = 0;
    psnrCubic = 0;
    for frame = 1:totalFrame
    path = '../textFiles/Pattern/';
    patternString = '';
    dirName = '';
    percentage = percentageSet(i);
    intPercent = percentage * 100;
    patternString = [num2str(GH) 'by' num2str(GW)]; %516by516_30
    
    dirName = [num2str(H) 'by' num2str(W) '_' num2str(intPercent) '/Result/'];
    lightDir = [path num2str(H) 'by' num2str(W) '_' num2str(intPercent) '/Result/lighting/'];
    cubicDir = [path num2str(H) 'by' num2str(W) '_' num2str(intPercent) '/Result/tricubic/'];
    
    dirName = strcat(path,dirName);
    lightingFile = strcat(dirName,'lighting/');
    gtLightingFile = strcat(lightDir,'groundTruth/');
    tricubicFile = strcat(dirName,'tricubic/');
    gtCubicFile = strcat(cubicDir,'groundTruth/');
    
    redFile = ['red_' num2str(frame) '.txt'];
    greenFile = ['green_' num2str(frame) '.txt'];
    blueFile = ['blue_' num2str(frame) '.txt'];
 
    
    lRed = strcat(lightingFile, redFile);
    lGreen = strcat(lightingFile, greenFile);
    lBlue = strcat(lightingFile, blueFile);
    
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
    lightImage = uint8(lightImage);
    
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
    GTlightImage = uint8(lightImage);
    
    
    
    cRed = strcat(tricubicFile, redFile);
    cGreen = strcat(tricubicFile, greenFile);
    cBlue = strcat(tricubicFile, blueFile);
    
    cRed = fopen(cRed, 'r');
    cGreen = fopen(cGreen, 'r');
    cBlue = fopen(cBlue, 'r');
    
    cubicRed = fscanf(cRed, '%f');
    cubicGreen = fscanf(cGreen, '%f');
    cubicBlue = fscanf(cBlue, '%f');
    
    cImageR = reshape(cubicRed, [H W]);
    cImageG = reshape(cubicGreen, [H W]);
    cImageB = reshape(cubicBlue, [H W]);
    cubicImage = cat(3, cImageR, cImageG, cImageB);
    cubicImage = uint8(cubicImage);
    
    
    gtCRed = strcat(gtCubicFile, redFile);
    gtCGreen = strcat(gtCubicFile, greenFile);
    gtCBlue = strcat(gtCubicFile, blueFile);
    
    GtcRed = fopen(gtCRed, 'r');
    GtcGreen = fopen(gtCGreen, 'r');
    GtcBlue = fopen(gtCBlue, 'r');
    
    GTcubicRed = fscanf(GtcRed, '%f');
    GTcubicGreen = fscanf(GtcGreen, '%f');
    GTcubicBlue = fscanf(GtcBlue, '%f');
    
    GtcImageR = reshape(GTcubicRed, [H W]);
    GtcImageG = reshape(GTcubicGreen, [H W]);
    GtcImageB = reshape(GTcubicBlue, [H W]);
    gtCubicImage = cat(3, GtcImageR, GtcImageG, GtcImageB);
    gtCubicImage = uint8(gtCubicImage);
    
    
    
    lightSub = abs(GTlightImage - lightImage);
    cubicSub = abs(gtCubicImage - cubicImage);
    psnrLightImage = 20 * log10(255) - 10*log10(sum(sum(lightSub.^2))/(H*W))
    psnrCubicImage = 20 * log10(255) - 10*log10(sum(sum(cubicSub.^2))/(H*W));
    
    psnrLight = psnrLight + ((psnrLightImage(:,:,1)+psnrLightImage(:,:,2)+psnrLightImage(:,:,3))/3);
    psnrCubic = psnrCubic + ((psnrCubicImage(:,:,1)+psnrCubicImage(:,:,2)+psnrCubicImage(:,:,3))/3);
    
    
    fclose('all');
    end
    psnrRatioLight(count) = psnrLight/totalFrame;
    psnrRatioCubic(count) = psnrCubic/totalFrame; 
    count = count +1;
       
end

psnrRatioLight;
psnrRatioCubic;

x = 1:count-1;
c = 1:count-1;
yLight = psnrRatioLight(c);
yCubic = psnrRatioCubic(c);
figure;
%Xtick([30 40 50 60 70 80 90])
%xticklabels({'30', '40', '50', '60', '70', '80','90'})

plot(x,yLight, '-o')
p = {'30'; '40'; '50'; '60'; '70'; '80'; '90'};
set(gca, 'XTick',[1:count-1],'XTickLabel', p)
%set(gca, 'XTickLabel',['30'; '40'; '50'; '60'; '70'; '80'; '90'])
title('PSNR for Lighting');
xlabel('percentage of missing pixels');
ylabel('PSNR');
grid minor
figure;
plot(x,yCubic, '-*');
set(gca, 'XTick',[1:count-1],'XTickLabel', p)
grid minor
title('PSNR for Tri-Cubic interpolation');
xlabel('percentage of using pixels');
ylabel('PSNR');


