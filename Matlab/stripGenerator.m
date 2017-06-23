clc;
clear;
r = 1024;
c = 1024;
padX = 3;
padY = 3;
blockX = 16;
blockY = 16;
percentage = 0;

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
dirName = [num2str(H) 'by' num2str(W) '_' num2str(percentage)];
dirName = strcat(path,dirName);
dirName = char(dirName)
mkdir(dirName);
file = 'strip.txt';
fileName = strcat(dirName,'/',file); %../textFiles/Pattern/516by516_0/strip.txt
ptrnLinIdx = [dirName '/' num2str(H) 'by' num2str(W) '_ptrnIdx.txt']
xFile = [dirName '/' num2str(H) 'by' num2str(W) 'Xcoord.txt'];
yFile = [dirName '/' num2str(H) 'by' num2str(W) 'Ycoord.txt'];
patternMatrix = [dirName '/' num2str(H) 'by' num2str(W) 'matrix.txt'];
totalPixel = [dirName '/' num2str(H) 'by' num2str(W) '_patternInfo.txt'];

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
patternString = ~patternString;
imshow(patternString);
onPixel = sum(sum(patternString(:,:)))
totalPixel = fopen(totalPixel,'wt');
fprintf(totalPixel,'%d\n',onPixel);
X = fopen(xFile,'wt');
Y = fopen(yFile,'wt');
linearPattern = fopen(ptrnLinIdx,'wt');
matrix = fopen(patternMatrix,'wt');
counter = 1;
for i = 1:GH
    for j = 1:GW
        if(patternString(j,i) == 1)
            local = (i-1)*GW+(j-1);
            linCoords(counter) = local;
            xCoords(counter) = j-1;
            yCoords(counter) = i-1;
            counter = counter + 1;
        end
    end
end
counter
fprintf(matrix, '%d\n',patternString);
fprintf(linearPattern,'%d\n', linCoords);
fprintf(X, '%d\n', xCoords);
fprintf(Y, '%d\n', yCoords);
fclose('all');
