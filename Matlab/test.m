X = fopen('../textFiles/Pattern/516by516/516by516Xcoord.txt','r');
Y = fopen('../textFiles/Pattern/516by516/516by516Ycoord.txt','r');

xCoord = fscanf(X,'%d');
yCoord = fscanf(Y,'%d');
pix = 247533;
r = 516;
c = 516;
%size(xCoord)
image = zeros(r,c);
for i=1:pix
    x = xCoord(i)+1;
    y = yCoord(i)+1;
    image(y,x) = 1;
end
n = sum(sum(image))
imshow(image);
