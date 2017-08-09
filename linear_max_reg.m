function [w1, w2, J_history] = linear_max_reg(F1, y1, F2, y2, mapping, opt)
% This function uses the maximum part-whole relationship 
% Input --
%  F1, y1: whole features/target
%  F2, y2: part features/target
%  mapping: whole to part mapping
%  opt: various parameters
% Output --
%  w1, w2: parameters for predicting whole/part
%  J_history: objective function values

    rng(2);

    beta = opt.('beta');
    gamma = opt.('gamma');
    tau = opt.('tau');
    tol = opt.('tol');
    max_iter = opt.('max_iter');
    verbose = opt.('verbose');
    
    [m, d1] = size(F1);
    [n, d2] = size(F2);
    
    % initialize w1, w2
    w1 = -1/sqrt(m) + (2/sqrt(m))*rand(d1,1);
    w2 = -1/sqrt(n) + (2/sqrt(n))*rand(d2, 1);
    
    J_history = [];
    J_history(1) = compute_obj(F1, y1, F2, y2, mapping, beta, gamma, w1, w2);
    
    for iter = 1:max_iter 
        % print out information
        if verbose == 1 && mod(iter,100) == 0
            info = ['The ', num2str(iter), '-th iteration, obj val:', num2str(J_history(iter))];
            disp(info);
        end
        
        % fast implementation with vectorization
        pred_y1 = F1 * w1;
        w1_g1 = (1/m) * F1' * (pred_y1 - y1) + gamma * w1;
        pred_y2 = F2 * w2;
        
        t1 = exp(pred_y2);
        r1 = pred_y1 - log(mapping * t1);
        w1_g2 = (beta/m) * F1' * r1;
        
        grad_w1 = w1_g1 + w1_g2;
        
        w2_g1 = (1/n) * F2' * (pred_y2 - y2) + gamma * w2;
        t2 = bsxfun(@times, F2, t1);
        trans_F2 = bsxfun(@rdivide, mapping*t2, mapping * t1);
        w2_g2 = (beta/m) * trans_F2' * r1;
        
        grad_w2 = w2_g1 - w2_g2;
        
        % update w1 and w2
        w1 = w1 - tau * grad_w1;
        w2 = w2 - tau * grad_w2;
        
        % compute the objective value
        J_history(iter + 1) = compute_obj(F1, y1, F2, y2, mapping, beta, gamma, w1, w2);

        if (abs(J_history(iter + 1) - J_history(iter)) < tol) || isnan(J_history(iter + 1))
            break;
        end
        
    end
    
    function val = compute_obj(F1, y1, F2, y2, mapping, beta, gamma, w1, w2)
        [m, d1] = size(F1);
        [n, d2] = size(F2);
        v1 = (1/(2*m)) * sum((F1 * w1 - y1).^2) + (gamma/2) * sum(w1.^2);
        v2 = (1/(2*n)) * sum((F2 * w2 - y2).^2) + (gamma/2) * sum(w2.^2);

        e = F1 * w1 - log(mapping * exp(F2 * w2));
        v3 = (beta/(2*m)) * (sum(e.^2));
        val = v1 + v2 + v3;
    