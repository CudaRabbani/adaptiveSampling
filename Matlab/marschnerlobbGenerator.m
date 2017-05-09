clc;
clear;
fileName = fopen('../Data/synthetic.raw','w');
x = 1;
y = 1;
z = 1;
alpha = 0.25;

r = 0.5;
%vol = zeros(x*y*z);

counter = 1;
xDimCounter = 1;
for i = -x:1/255:x
    yDimCounter = 1;
    for j = -y:1/255:y
        zdimCounter = 1;
        for k= -z:1/255:z
%             xVal = (i/(x - 1) - 0.5);
%             yVal = (j/(y - 1) - 0.5);
%             zVal = (k/(z - 1) - 0.5);
%             vol(counter) = ((1 - sin((pi*zVal)/2) + alpha * (1 + rho_r(sqrt(xVal*xVal + yVal*yVal))))/(2 * ( 1 + alpha)));
            vol(counter) = sin(i*i + j*j + k*k);
            counter = counter + 1;
            zdimCounter = zdimCounter + 1;
        end
        yDimCounter = yDimCounter + 1;
    end
    xDimCounter = xDimCounter + 1;
end
xDimCounter
yDimCounter
zdimCounter
vol = uint8(vol);
%vol = vol.*255;
fwrite(fileName,vol,'double');
fclose(fileName);