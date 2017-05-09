clc;
clear;
H = 516;
W = 516;
path = '../textFiles/Pattern/';
lightDir = [path num2str(H) 'by' num2str(W) '_' num2str(30)]
lightDir = strcat(lightDir,'/Result/lighting/rgb_5.bin')

file = fopen(lightDir,'r');
img = fread(file,'float32');

ind = find(isnan(img))

% [r c] = size(img);
% imgR = img(1:3:r);
% imgG = img(2:3:r);
% imgB = img(3:3:r);
% 
% R = reshape(imgR, [H W]);
% G = reshape(imgG, [H W]);
% B = reshape(imgB, [H W]);
% finalImage = cat(3, R, G, B);
% imshow(finalImage, []);
% fclose(file);