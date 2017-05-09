percentageSet = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]; %, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
[m n] = size(percentageSet);
W = 16;
H = 16;
mask = zeros(H,W);

path = '../blockPattern/';



for i=1:n
    G2=19;
    G1=28;
    inc=abs(G2-G1);
    percentage = percentageSet(i)*100;
    fileName = num2str(percentage);
    fileName = [fileName,'.bin'];
    fileName = [path fileName];
    fileName = char(fileName);
    fileID = fopen(fileName,'w');
    NUM=H*W*percentageSet(i);
    pixCount = int64(NUM);
    x=0;y=0;N=0;
    % 0, 1, 1, 1, 2, 3, 4, 6, 9, 13, 19, 28, 41, 60, 88, 129, 189, 277, 406, 595, 872, 1278, 1873, 2745, 4023, 5896
    
    counter = 1;
    while N<NUM
        if and(x<W, y<H)
            mask(sub2ind(size(mask), y+1, x+1))=1;
            N=N+1;
        end
        x=mod(x+inc, G1);
        y=mod(y+inc, G2);
    end
    mask = mask';
    fwrite(fileID,mask,'uint8');
    fclose(fileID);
    
%     figure;
%     imshow(mask);
%     title(percentage);
    
    
    
end