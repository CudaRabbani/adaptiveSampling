clear;
clc;
r = 512;
c = 512;
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
H = GH;
W = GW;
percentageSet = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
[m n] = size(percentageSet);

figureName = '../resultImages/fpsComparison.png';

for i =1:n
    path = '../textFiles/Pattern/';
    percentage = percentageSet(i);
    intPercent = percentage * 100;
    patternString = '';
    dirName = '';
    patternString = [num2str(GH) 'by' num2str(GW)]; %516by516_30   
    dirName = [num2str(H) 'by' num2str(W) '_' num2str(intPercent) '/Result/'];
    linDir = strcat(path,dirName);
    cubDir = strcat(path,dirName);
    
    linFile = 'lighting/timing/timer.txt';
    cubicFile = 'tricubic/timing/timer.txt';
    
    linFile = strcat(linDir, linFile);
    cubicFile = strcat(cubDir, cubicFile);
    
%    timerFile = fopen(dirName,'r');
%    timingInfo = fscanf(timerFile, '%f');
    
    linTimer = fopen(linFile,'r');
    linearTime = fscanf(linTimer,'%f');
    cubTimer = fopen(cubicFile,'r');
    cubicTime = fscanf(cubTimer,'%f');
      
    LineartotalFrame(i) = linearTime(1);
    LinearvolumeTime(i) = linearTime(2);
    LinearreconstructionTime(i) = linearTime(3);
    LineartotalTime(i) = LinearvolumeTime(i)+LinearreconstructionTime(i);
    LinearFPS(i) = linearTime(6);
    
    CubictotalFrame(i) = cubicTime(1);
    CubicvolumeTime(i) = cubicTime(2);
    CubicreconstructionTime(i) = cubicTime(3);
    CubictotalTime(i) = CubicvolumeTime(i)+CubicreconstructionTime(i);
    CubicFPS(i) = cubicTime(6);
    
end

x = 1:n;
yLinear = LinearFPS(x);
yCubic = CubicFPS(x);
plot(x,yLinear, '-o',x,yCubic, '-o', 'LineWidth',2);
ylim([10 100]);
grid on
grid minor
axis equal square
p = {'30'; '40'; '50'; '60'; '70'; '80'; '90'};
set(gca,'DefaultTextFontSize',18)
set(gca, 'XTick',[1:n],'XTickLabel', p, 'FontSize',18)
title('Frame Per Second Comparison');

xlabel('percentage of using pixels');
ylabel('Frame Per Second');
legend('Tri-linear','Tri-cubic');
saveas(gcf,figureName);

%{
x = 1:n;
yVol= volumeTime(x);
plot(x,yVol, '-o', 'LineWidth',2);
ylim([0 40]);
axis equal square
p = {'30'; '40'; '50'; '60'; '70'; '80'; '90'};
set(gca, 'XTick',[1:n],'XTickLabel', p)
title('Volume Rendering Time');
xlabel('percentage of using pixels');
ylabel('Time in ms');
grid on
grid minor
saveas(gcf,volName);
figure;

yRecon = reconstructionTime(x);
plot(x,yRecon, '-o', 'LineWidth',2);
grid on
grid minor
ylim([0 40]);
axis equal square
p = {'30'; '40'; '50'; '60'; '70'; '80'; '90'};
set(gca, 'XTick',[1:n],'XTickLabel', p)
title('Reconstruction Time');
xlabel('percentage of using pixels');
ylabel('Time in ms');
saveas(gcf,reconName);
figure;

yFPS = FPS(x);
plot(x,yFPS, '-o', 'LineWidth',2);
grid on
grid minor
ylim([0 40]);
axis equal square
p = {'30'; '40'; '50'; '60'; '70'; '80'; '90'};
set(gca, 'XTick',[1:n],'XTickLabel', p)
title('FPS Calculation');
xlabel('percentage of using pixels');
ylabel('Time in ms');
saveas(gcf,fpsName);
figure;

yTime = totalTime(x);
plot(x,yVol,'r-o', x,yRecon,'g-o', x,yTime,'b-o', 'LineWidth',2);
grid on
grid minor
ylim([0 50]);
axis equal square
p = {'30'; '40'; '50'; '60'; '70'; '80'; '90'};
set(gca, 'XTick',[1:n],'XTickLabel', p)
title('Total Timing');
xlabel('percentage of using pixels');
ylabel('Time in ms');
legend('Volume Rendering Time','Reconstruction Time','Total Time')
saveas(gcf,timingName);
%}