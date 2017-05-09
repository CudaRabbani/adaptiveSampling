percentage = 0.5;
r = 1024;
c = 1024;
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



%s = strcat(num2str(r),'by', num2str(c))
path = '../textFiles/Pattern/';
patternString = [num2str(GH) 'by' num2str(GW) ]; %516by516

dirName = [num2str(H) 'by' num2str(W)];
dirName = strcat(path,dirName);
dirName = char(dirName);
mkdir(dirName);

path = strcat(path,patternString,'/'); % path = strcat(path,num2str(GH),'/')
dim = strcat(patternString,'/');
xString = [patternString 'Xcoord'];
yString = [patternString 'Ycoord'];
ext = [{'.txt'}];


patternFileName = strcat(path,patternString,ext);
patternFileName = char(patternFileName);
xFile = strcat(path, xString, ext);
xFile = char(xFile);
yFile = strcat(path,yString, ext);
yFile = char(yFile);
patternIdx = [patternString '_ptrnIdx'];
patternIdx = strcat(path,patternIdx,ext);
patternIdx = char(patternIdx);

patternInfo = '_patternInfo';
patternFile =strcat(path,patternString,patternInfo,ext);
patternFile = char(patternFile);

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




% imshow(patternString);
% title('Only 0 and 1');
%figure;
% Pixel Shuffling ------------------------------>>>   
% 0, 1, 1, 1, 2, 3, 4, 6, 9, 13, 19, 28, 41, 60, 88, 129, 189, 277, 406, 595, 872, 1278, 1873, 2745, 4023, 5896

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
xCoords = zeros(effectivePixel,1);
yCoords = zeros(effectivePixel,1);
linCoords = zeros(effectivePixel,1);
info = [GW, GH, percentage*100, effectivePixel];
fprintf(patternFileId, '%d\n', effectivePixel);

imshow(tempMaskedImage);
tempMaskedImage = tempMaskedImage';
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
