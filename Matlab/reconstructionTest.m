clear;
clc;
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
percentageSet = [0.8]; %, 0.5, 0.6, 0.7, 0.8, 0.9 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
[m n] = size(percentageSet);
psnrRatio = zeros(1,totalFrame+1);
count = 1;
for i=1:n
        psnrLight = 0;
    for frame = 1:totalFrame-1
        path = '../textFiles/Reconstruction/';
        patternString = '';
        dirName = '';
        intPercent = percentageSet(i) * 100;
                
        redText = ['redOutRecon_' num2str(frame) '.txt'];
        greenText = ['greenOutRecon_' num2str(frame) '.txt'];
        blueText = ['blueOutRecon_' num2str(frame) '.txt'];
        
        redFile = strcat(path,redText)
        greenFile = strcat(path,greenText)
        blueFile = strcat(path,blueText)
        
        R = fopen(redFile,'r');
        G = fopen(greenFile,'r');
        B =fopen(blueFile,'r');
        
        rVal = fscanf(R,'%f');
        gVal = fscanf(G,'%f');
        bVal = fscanf(B,'%f');
        
        lRed = reshape(rVal, [H W]);
        lGreen = reshape(gVal, [H W]);
        lBlue = reshape(bVal, [H W]);
        
        
        lightImage = cat(3, lRed, lGreen, lBlue);
        imshow(lightImage, []);
        title('Reconstucted');
        figure;
        
        
    end
end