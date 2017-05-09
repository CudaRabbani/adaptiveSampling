% window size generator

Width = [512, 1024, 1024, 1280, 1366, 2048, 3840, 4096]
Height = [512, 768, 1024, 800, 768, 1080, 2160, 2160]


padX = 3;
padY = 3;
blockX = 16;
blockY = 16;

path = '../textFiles/Pattern/';

for i = 1:size(Height,2)
    NBx = ceil( ( Width(1,i) - padX ) /  (blockX + padX) )
    NBy = ceil( ( Height(1,i) - padY ) /  (blockY + padY) );

    GW = NBx * blockX + (NBx+1) * padX;
    GH = NBy * blockY + (NBy+1) * padY;
    H = GH;
    W = GW;
    dirName = [num2str(H) 'by' num2str(W)];
    dirName = strcat(path,dirName);
    dirName = char(dirName);
    mkdir(dirName);    
end
    
    