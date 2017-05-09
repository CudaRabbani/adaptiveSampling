clc;
clear;
percentage = 0.3;
r = 1080;
c = 1920;
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
mask=zeros(H,W);
% imshow(mask);
% figure;
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
G2=2745;
G1=4023;
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
N
patternString = ~patternString;
tempMaskedImage = mask + patternString;
tempMaskedImage = logical(tempMaskedImage);
imshow(mask);
