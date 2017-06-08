clc;
clear;
%{
w = 13;
h = 13;
vals = 0:(13*13)-1;
vals = reshape(vals, [13 13]);
vals = vals';
bdim_x = 2;
bdim_y = 2;
pad = 3;
stripe = w * (bdim_y+pad);

bid_x = 0;
bid_y = 0;
tid_x = 1;
tid_y = 1;

halo = bid_x * stripe + pad * w + (tid_y * w) + (bid_x + 1) * pad + bid_x * bdim_x + tid_x
%}

fileID= fopen('../textFiles/tmepAddress.txt','r');
d = fscanf(fileID, '%d');
d = reshape(d, [516, 516]);
d = d';
imshow(d);
