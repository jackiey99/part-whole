function [normalizedX, mindata, maxdata] = minmaxNormalize(X,minX, maxX)
    if nargin == 1 % construct min-max
        mindata = min(X);
        maxdata = max(X);
        normalizedX = bsxfun(@rdivide, bsxfun(@minus, X, mindata), maxdata - mindata);
    end
    
    if nargin == 3 % apply min-max
        normalizedX = bsxfun(@rdivide, bsxfun(@minus, X, minX), maxX - minX);
    end
    
end