function t = robust_function(e, name, k)

if strcmp(name, 'huber')
    t = e;
    ind1 = (abs(t) <= k);
    t(ind1) = 0.5 * (t(ind1).^2);
    ind2 = (abs(t) > k);
    t(ind2) = k * abs(t(ind2)) - 0.5 * k * k;
end

if strcmp(name, 'bisquare')
    t = e;
    ind1 = (abs(t) <= k);
    t(ind1) = (k*k/6) * (1 - (1 - (t(ind1)/k ).^2).^3);
    ind2 = (abs(t) > k);
    t(ind2) = k*k/6;
end

end