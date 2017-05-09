clear;
img = imread('../Data/lena_512.jpg');
[r, c, p] = size(img);

padX = 3;
padY = 3;
blockX = 16;
blockY = 16;
NBx = ceil( ( c - padX ) /  (blockX + padX) );
NBy = ceil( ( r - padY ) /  (blockY + padY) );

GW = NBx * blockX + (NBx+1) * padX;
GH = NBy * blockY + (NBy+1) * padY;

newHight = GH
newWidth = GW

redVol=fopen('../textFiles/redOutVol.txt', 'r');
greenVol=fopen('../textFiles/greenOutVol.txt', 'r');
blueVol=fopen('../textFiles/blueOutVol.txt', 'r');


red=fopen('../textFiles/redOutRecon.txt', 'r');
green=fopen('../textFiles/greenOutRecon.txt', 'r');
blue=fopen('../textFiles/blueOutRecon.txt', 'r');

% red=fopen('../textFiles/redStY.txt', 'r');
% green=fopen('../textFiles/greenStY.txt', 'r');
% blue=fopen('../textFiles/blueStY.txt', 'r');


red = fscanf(red,'%f');
green = fscanf(green,'%f');
blue = fscanf(blue,'%f');

redVol = fscanf(redVol,'%f');
greenVol = fscanf(greenVol,'%f');
blueVol = fscanf(blueVol,'%f');

imgRed = reshape(red, [newHight newWidth]);
%R = imgRed(1:r,1:c);
imgGreen = reshape(green, [newHight newWidth]);
%G = imgGreen(1:r,1:c);
imgBlue = reshape(blue, [newHight newWidth]);
%B = imgBlue(1:r,1:c);
image = cat(3, imgRed, imgGreen, imgBlue);
%image = cat(3,R,G,B);

volRed = reshape(redVol, [newHight newWidth]);
volGreen = reshape(greenVol, [newHight newWidth]);
volBlue = reshape(blueVol, [newHight newWidth]);
vol = cat(3, volRed, volGreen, volBlue);
imshow(vol, [0 255]);
title('Volume Output');
figure;


imshow(image, [0 255]);
%imwrite(image, '../Data/output.jpg');
title('Reconstructed Output');







%{
data = ones(512,512);
x = 0;
for i = 1:512
    for j = 1:512
        data(i,j) = x;
        x = x + 1;
    end
end

bIdx = 5;
bIdy = 5;
TILE_W = 16;
PAD = 3;

Ix = bIdx * TILE_W + ( bIdx + 1 ) * PAD;
Iy = bIdy * TILE_W + ( bIdy + 1 ) * PAD;
corner_y = bIdy * TILE_W + (bIdy + 1) * PAD;
corner_x = bIdx * TILE_W + (bIdx + 1) * PAD;
row = 1;
col = 1;
for i=corner_y-PAD : corner_y+PAD+TILE_W
    for j = corner_x-PAD:corner_x+PAD+TILE_W
        holo(row,col) = data(i * 512 + j);
        col = col + 1;
    end
    row = row + 1;
end
%}