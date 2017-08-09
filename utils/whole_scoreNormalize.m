function output = whole_scoreNormalize(input)
%COLUMNNORMALIZE Summary of this function goes here
%
% Normalization for Math
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n = size(input,1);
output = zeros(n,1);



[I,J] = find(input  <= 300);
output(I,1) = 0.1;

[I,J] = find(input > 300 & input <= 800);
output(I,1) = 0.2;

[I,J] = find(input > 800 & input <= 1200);
output(I,1) = 0.3;

[I,J] = find(input > 1200 & input <= 2000);
output(I,1) = 0.4;

[I,J] = find(input > 2000 & input <= 3000);
output(I,1) = 0.5;

[I,J] = find(input > 3000 & input <= 5000);
output(I,1) = 0.6;

[I,J] = find(input > 5000 & input <= 8000);
output(I,1) = 0.7;

[I,J] = find(input > 8000 & input <= 15000);
output(I,1) = 0.8;

[I,J] = find(input > 15000 & input <= 50000);
output(I,1) = 0.9;

[I,J] = find(input > 50000);
output(I,1) = 1;
end

