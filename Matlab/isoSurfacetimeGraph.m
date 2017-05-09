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
volName = '../resultImages/isoSurface/volFigure.png';
reconName = '../resultImages/isoSurface/reconFigure.png';
fpsName = '../resultImages/isoSurface/fpsFigure.png';
timingName = '../resultImages/isoSurface/timingFigure.png';

for i =1:n
    path = '../textFiles/Pattern/';
    percentage = percentageSet(i);
    intPercent = percentage * 100;
    patternString = '';
    dirName = '';
    patternString = [num2str(GH) 'by' num2str(GW)]; %516by516_30   
    dirName = [num2str(H) 'by' num2str(W) '_' num2str(intPercent) '/Result/isoSurface/timing/timer.txt'];
    dirName = strcat(path,dirName);
    timerFile = fopen(dirName,'r');
    timingInfo = fscanf(timerFile, '%f');
    totalFrame(i) = timingInfo(1);
    volumeTime(i) = timingInfo(2);
    reconstructionTime(i) = timingInfo(3);
    totalTime(i) = volumeTime(i)+reconstructionTime(i);
    FPS(i) = timingInfo(6);   
end

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
%ylim([0 40]);
axis equal square
p = {'30'; '40'; '50'; '60'; '70'; '80'; '90'};
set(gca, 'XTick',[1:n],'XTickLabel', p)
title('FPS Calculation');
xlabel('percentage of using pixels');
ylabel('FPS');
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