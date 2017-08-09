function t = robust_grad(e, name, k)

if strcmp(name, 'huber')
    t = e;
   
    ind2 = (abs(t) > k);
    t(ind2) = k * sign(t(ind2));
end

if strcmp(name, 'bisquare')
    t = e;
    ind1 = (abs(t) <= k);
    t(ind1) = t(ind1) .* (1 - (t(ind1)/k).^2).^2;
    ind2 = (abs(t) > k);
    t(ind2) = 0;
end

end