function output = part_scoreNormalize(input)
%COLUMNNORMALIZE Summary of this function goes here
%
% Normalization for Math
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n = size(input,1);
output = zeros(n,1);



[I,J] = find(input  <= 1);
output(I,1) = 0.1;

[I,J] = find(input > 1 & input <= 12);
output(I,1) = 0.2;

[I,J] = find(input > 12 & input <= 100);
output(I,1) = 0.3;

[I,J] = find(input > 100 & input <= 1000);
output(I,1) = 0.4;

[I,J] = find(input > 1000 & input <= 5000);
output(I,1) = 0.5;

[I,J] = find(input > 5000 & input <= 50000);
output(I,1) = 0.6;

[I,J] = find(input > 50000 & input <= 100000);
output(I,1) = 0.7;

[I,J] = find(input > 100000 & input <= 400000);
output(I,1) = 0.8;

[I,J] = find(input > 400000 & input <= 500000);
output(I,1) = 0.9;

[I,J] = find(input > 500000);
output(I,1) = 1;
end

