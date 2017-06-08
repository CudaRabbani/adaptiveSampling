percentageSet = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]; %, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
[m n] = size(percentageSet);
W = 16;
H = 16;
mask = zeros(H,W);

path = '../blockPattern/';
G2=19;
G1=28;
    inc=abs(G2-G1);
maskTen = zeros(H,W);
x=0;y=0;N=0;
while N<32
    if and(x<W, y<H)
        maskTen(sub2ind(size(maskTen), y+1, x+1))=1;
        N=N+1;
    end
    x=mod(x+inc, G1);
    y=mod(y+inc, G2);
end
s = sum(maskTen(:))
maskTen = maskTen';
fileID= fopen('../blockPattern/10.txt','wt');
fprintf(fileID,'%d\n', maskTen);
for i=2:n
%     G2=19;
%     G1=28;
    inc=abs(G2-G1);
    percentage = percentageSet(i)*100;
    fileName = num2str(percentage);
    fileName = [fileName,'.txt'];
    fileName = [path fileName];
    fileName = char(fileName);
    fileID = fopen(fileName,'wt');
    NUM=H*W*percentageSet(i) - 25;
    pixCount = int64(NUM);
    x=0;y=0;N=0;
    % 0, 1, 1, 1, 2, 3, 4, 6, 9, 13, 19, 28, 41, 60, 88, 129, 189, 277, 406, 595, 872, 1278, 1873, 2745, 4023, 5896
    
    counter = 1;
    while N<NUM
        if and(x<W, y<H)
            if(maskTen(sub2ind(size(mask), y+1, x+1)) ~=1)
                mask(sub2ind(size(mask), y+1, x+1))=1;
                N=N+1;
            end
        end
        x=mod(x+inc, G1);
        y=mod(y+inc, G2);
    end
    mask = mask';
    fprintf(fileID,'%d\n', mask);
    fclose(fileID);
    
%     figure;
     imshow(mask);
     pause;
     hold on;
%     title(percentage);
    
    
    
end