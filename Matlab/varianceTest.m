a = [1 2 3; 4 5 6; 7 8 9];
a = [1 1 1; 1 1 1; 1 1 1];
m = mean(a(:));
v = var(a(:))

varMat = a - m;

varMat = varMat.^2;

s = sum(varMat(:))
v = s/9