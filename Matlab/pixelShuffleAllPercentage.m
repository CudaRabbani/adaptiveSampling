clc;
clear;
r = 512;
c = 512;
padX = 3;
padY = 3;
blockX = 16;
blockY = 16;

NBx = ceil( ( c - padX ) /  (blockX + padX) );
NBy = ceil( ( r - padY ) /  (blockY + padY) );

GW = NBx * blockX + (NBx+1) * padX;
GH = NBy * blockY + (NBy+1) * padY;
diffH = GH - r;
diffW = GW - c;
H = GH
W = GW
percentageSet = [0.1]; % , 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
[m n] = size(percentageSet);
%s = strcat(num2str(r),'by', num2str(c))

for i =1:n
    path = '../textFiles/Pattern/';
    patternString = '';
    dirName = '';
    percentage = percentageSet(i);
    intPercent = percentage * 100;
    patternString = [num2str(GH) 'by' num2str(GW)]; %516by516_30
    
    dirName = [num2str(H) 'by' num2str(W) '_' num2str(intPercent)];
    dirName = strcat(path,dirName);
    dirName = char(dirName)
    mkdir(dirName);
    
    timerDirLight = [num2str(H) 'by' num2str(W) '_' num2str(intPercent) '/Result/lighting/timing'];
    timerDirLight = strcat(path, timerDirLight);
    timerDirLight = char(timerDirLight);
    mkdir(timerDirLight);
    
    timerDirIsoSurface = [num2str(H) 'by' num2str(W) '_' num2str(intPercent) '/Result/isoSurface/timing'];
    timerDirIsoSurface = strcat(path, timerDirIsoSurface);
    timerDirIsoSurface = char(timerDirIsoSurface);
    mkdir(timerDirIsoSurface);
    
    timerDirTriCubic = [num2str(H) 'by' num2str(W) '_' num2str(intPercent) '/Result/tricubic/timing'];
    timerDirTriCubic = strcat(path, timerDirTriCubic);
    timerDirTriCubic = char(timerDirTriCubic);
    mkdir(timerDirTriCubic);
    
    resultDir = [num2str(H) 'by' num2str(W) '_' num2str(intPercent) '/Result'];
    resultDir= strcat(path, resultDir);
    resultDir = char(resultDir);
    mkdir(resultDir);
    
      
    triCubic = [num2str(H) 'by' num2str(W) '_' num2str(intPercent) '/Result/tricubic'];
    triCubic = strcat(path, triCubic);
    triCubic = char(triCubic);
    mkdir(triCubic);
    
    gtCubic = [num2str(H) 'by' num2str(W) '_' num2str(intPercent) '/Result/tricubic/groundTruth'];
    gtCubic = strcat(path, gtCubic);
    gtCubic = char(gtCubic);
    mkdir(gtCubic);
    
    lighTing = [num2str(H) 'by' num2str(W) '_' num2str(intPercent) '/Result/lighting'];
    lighTing = strcat(path, lighTing);
    lighTing = char(lighTing);
    mkdir(lighTing);
    
    gtLight = [num2str(H) 'by' num2str(W) '_' num2str(intPercent) '/Result/lighting/groundTruth'];
    gtLight = strcat(path, gtLight);
    gtLight = char(gtLight);
    mkdir(gtLight);
    
    isoSurface = [num2str(H) 'by' num2str(W) '_' num2str(intPercent) '/Result/isoSurface'];
    isoSurface = strcat(path, isoSurface);
    isoSurface = char(isoSurface);
    mkdir(isoSurface);
    
    gtIsoSurface = [num2str(H) 'by' num2str(W) '_' num2str(intPercent) '/Result/isoSurface/groundTruth'];
    gtIsoSurface = strcat(path, gtIsoSurface);
    gtIsoSurface = char(gtIsoSurface);
    mkdir(gtIsoSurface);
    
    path = strcat(dirName,'/');%path = strcat(path,patternString,'/'); % path = strcat(path,num2str(GH),'/')
    dim = strcat(patternString,'/');
    xString = [patternString 'Xcoord'];
    yString = [patternString 'Ycoord'];
    ext = [{'.txt'}];
    
    
    patternFileName = strcat(path,patternString,ext);
    patternFileName = char(patternFileName)
    xFile = strcat(path, xString, ext);
    xFile = char(xFile);
    yFile = strcat(path,yString, ext);
    yFile = char(yFile);
    patternIdx = [patternString '_ptrnIdx'];
    patternIdx = strcat(path,patternIdx,ext);
    patternIdx = char(patternIdx)
    
    patternInfo = '_patternInfo';
    patternFile =strcat(path,patternString,patternInfo,ext);
    patternFile = char(patternFile)
    
    patternWithHolo = fopen(patternFileName, 'wt');
    patternXcoords = fopen(xFile, 'wt');
    patternYcoords = fopen(yFile, 'wt');
    patternFileId = fopen(patternFile, 'wt');
    patternLinIdx = fopen(patternIdx,'wt');
    
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
    
    
    G2=1278;
    G1=1873;
    inc=abs(G2-G1);
    NUM=H*W*percentage;
    pixCount = int64(NUM)
    x=0;y=0;N=0;
    
    counter = 1;
    while N<NUM
        if and(x<W, y<H)
            mask(sub2ind(size(mask), y+1, x+1))=1;
            N=N+1;
        end
        x=mod(x+inc, G1);
        y=mod(y+inc, G2);
    end
    
    patternString = ~patternString;
    tempMaskedImage = mask + patternString;
    tempMaskedImage = logical(tempMaskedImage);
    
    effectivePixel = sum(tempMaskedImage(:));
    effectivePercentage = effectivePixel/(H * W) * 100;
    Percentage(i) = effectivePercentage;
    xCoords = zeros(effectivePixel,1);
    yCoords = zeros(effectivePixel,1);
    linCoords = zeros(effectivePixel,1);
    info = [GW, GH, percentage*100, effectivePixel];
    fprintf(patternFileId, '%d\n', effectivePixel);
    
%    imshow(tempMaskedImage);
    tempMaskedImage = tempMaskedImage';
%    imshow(tempMaskedImage,[]);
%    figure;
    fprintf(patternWithHolo, '%d\n', tempMaskedImage);
    counter = 1;
    for i = 1:GH
        for j = 1:GW
            if(tempMaskedImage(j,i) == 1)
                local = (i-1)*GW+(j-1);
                linCoords(counter) = local;
                xCoords(counter) = j-1;
                yCoords(counter) = i-1;
                counter = counter + 1;
            end
        end
    end
    counter
    fprintf(patternXcoords, '%d\n', xCoords);
    fprintf(patternYcoords, '%d\n', yCoords);
    fprintf(patternLinIdx, '%d\n', linCoords);
    fclose('all');    
end


Percentage
GH
GW





