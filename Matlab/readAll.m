clear;
% /home/reza/cuda-workspace/sharedMemoryProject/Data/lena_256.jpg
% //  0, 1, 1, 1, 2, 3, 4, 6, 9, 13, 19, 28, 41, 60, 88, 129, 189, 277, 406, 595, 872, 1278, 1873, 2745, 4023, 5896
s = pwd
pixel = 0.5;
img = imread('../Data/lena_512.jpg');
fileID1=fopen('../textFiles/Pattern.txt','wt');
styFile = fopen('../textFiles/StY.txt', 'wt');
redSty = fopen('../textFiles/redStY.txt', 'wt');
greenSty = fopen('../textFiles/greenStY.txt', 'wt');
blueSty = fopen('../textFiles/blueStY.txt', 'wt');
patternWithHolo = fopen('../textFiles/finalPattern.txt', 'wt');
patternXcoords = fopen('../textFiles/patternXcoords.txt', 'wt');
patternYcoords = fopen('../textFiles/patternYcoords.txt', 'wt');

imgFile = fopen('../textFiles/image.txt','wt');
redFile = fopen('../textFiles/redImage.txt','wt');
greenFile = fopen('../textFiles/greenImage.txt','wt');
blueFile = fopen('../textFiles/blueImage.txt','wt');

RED = double(img(:,:,1));
GREEN = double(img(:,:,2));
BLUE = double(img(:,:,3));
%reverseHolo = fopen('../textFiles/revStY.txt','wt');
[r, c, p]=size(img);

padX = 3;
padY = 3;
blockX = 16;
blockY = 16;
NBx = ceil( ( c - padX ) /  (blockX + padX) )
NBy = ceil( ( r - padY ) /  (blockY + padY) );

GW = NBx * blockX + (NBx+1) * padX;
GH = NBy * blockY + (NBy+1) * padY;
% H=r+4;
% W=c+4;
diffH = GH - r;
diffW = GW - c;
H = GH;
W = GW;

GH
GW
mask=zeros(H,W);
pattern = ones(r, c);
grayImage = double(rgb2gray(img));
grayImage = grayImage./255;
redImage = RED ./ 255;
greenImage = GREEN ./ 255;
blueImage = BLUE ./ 255;
paddedGray = zeros(H,W);
paddedRed = zeros(H,W);
paddedGreen = zeros(H,W);
paddedBlue = zeros(H, W);

paddedGray(1:r, 1:c) = grayImage(1:r, 1:c);
paddedRed(1:r, 1:c) = redImage(1:r, 1:c);
paddedGreen(1:r, 1:c) = greenImage(1:r, 1:c);
paddedBlue(1:r, 1:c) = blueImage(1:r, 1:c);

% N = 16;
% P = 3;
for col = 1:blockX+padX:blockX+padX %1:N+P:N+P
    pattern(:,col:col+padX-1) = 0; % (:,col:col+P-1) = 0;
end
for row = 1:blockY+padY:blockY+padY  % row = 1:N+P:N+P
    pattern(row:row+padY-1,:) = 0;
end
for col = blockX+padX+1:blockX+padX:c+diffW % N+P+1:N+P:c+4
    pattern(:,col:col+padX-1) = 0;
end
for row = blockY+padY+1:blockY+padY:r+diffH
    pattern(row:row+padY-1,:) = 0;
end


% Pixel Shuffling ------------------------------>>>   
% 0, 1, 1, 1, 2, 3, 4, 6, 9, 13, 19, 28, 41, 60, 88, 129, 189, 277, 406, 595, 872, 1278, 1873, 2745, 4023, 5896
G2=595;
G1=872;
inc=abs(G2-G1);

NUM=H*W*pixel;

pixCount = int64(NUM)
x=0;y=0;N=0;
xCoords = zeros(pixCount,1);
yCoords = zeros(pixCount,1);
counter = 1;
while N<NUM
    if and(x<W, y<H)
        mask(sub2ind(size(mask), y+1, x+1))=1;
        xCoords(counter) = x+1;
        yCoords(counter) = y+1;
        counter = counter+1;
        N=N+1;
    end
    x=mod(x+inc, G1);
    y=mod(y+inc, G2);
end
pattern = ~pattern;
%tempGrayImage = grayImage .* pattern;
tempGrayImage = paddedGray .* pattern;
fprintf(patternXcoords, '%d\n', xCoords);
fprintf(patternYcoords, '%d\n', yCoords);
fprintf(imgFile,'%f',grayImage);
fprintf(redFile,'%f',redImage);
fprintf(greenFile,'%f',greenImage);
fprintf(blueFile,'%f',blueImage);


imshow(grayImage, [0, 1]);
figure;
tempMaskedImage = mask + pattern;
tempMaskedImage = logical(tempMaskedImage);

%holoImage = grayImage .* tempMaskedImage;
holoImage = paddedGray .* tempMaskedImage;
holoRed = paddedRed .* tempMaskedImage;
holoGreen = paddedGreen .* tempMaskedImage;
holoBlue = paddedBlue .* tempMaskedImage;
%imshow(holoImage);
%title('Holo Image');
%figure;

fprintf(patternWithHolo, '%d\n', tempMaskedImage);
fprintf(styFile, '%f\n', holoImage);
fprintf(redSty, '%f\n', holoRed);
fprintf(greenSty, '%f\n', holoGreen);
fprintf(blueSty, '%f\n', holoBlue);
%revHolo = imcomplement(tempMaskedImage);
%revHolo = revHolo .* grayImage;
%revHolo = revHolo .* paddedGray;
% imshow(revHolo, [0 1]);
% figure;
% fprintf(reverseHolo, '%f\n', revHolo);


s = sprintf('Mask using %.2f pixel', pixel * 100);
fprintf(fileID1, '%d\n', mask);
imshow(tempMaskedImage);
title (s);
figure;
imshow(mask);
title('mask');
fclose('all');


