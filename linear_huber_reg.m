function [w1, w2, J_history] = linear_huber_reg(F1, y1, F2, y2, mapping, opt)
 % This function uses the robust part-whole relationship w/huber estimator
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
    huber_k = opt.('huber_k');
    tau = opt.('tau');
    tol = opt.('tol');
    max_iter = opt.('max_iter');
    verbose = opt.('verbose');
    
    
    alpha = 1;
    
    [m, d1] = size(F1);
    [n, d2] = size(F2);
    % initialize w1, w2, and part_weights
    w1 = -1/sqrt(m ) + (2/sqrt(m))*rand(d1,1);
    w2 = -1/sqrt(n) + (2/sqrt(n))*rand(d2, 1);
    
    part_weights = mapping;
    
    J_history = [];
    J_history(1) = compute_obj(F1, y1, F2, y2, part_weights, beta, gamma, w1, w2, alpha, huber_k);

    for iter = 1:max_iter 
        % print out information
        if verbose == 1 && mod(iter,100) == 0
            info = ['The ', num2str(iter), '-th iteration, obj val:', num2str(J_history(iter))];
            disp(info);
        end
        
        % fast implementation with vectorization
        pred_y1 = F1 * w1;
        e1 = pred_y1 - y1;
        w1_g1 = alpha * (1/m) * F1' * (e1) + gamma * w1;
        pred_y2 = F2 * w2;
        
        t1 = part_weights * pred_y2;
        r1 = pred_y1 - t1;
        rho1 = robust_grad(r1, 'huber', huber_k);
        w1_g2 = (beta/m) * F1' * (rho1);
        
        grad_w1 = w1_g1 + w1_g2;
        
        e2 = pred_y2 - y2;
        w2_g1 = alpha * (1/n) * F2' * (e2) + gamma * w2;
        
        t2 = part_weights * F2;
        w2_g2 = (beta/m) * t2' * (rho1);
        
        grad_w2 = w2_g1 - w2_g2;
        
        % t3 = bsxfun(@times, mapping, -pred_y2');
        t3 = mapping * spdiags(-pred_y2, 0, n, n);
        % grad_part_weights = (beta/m) * bsxfun(@times, r1, t3);
        grad_part_weights = (beta/m) * spdiags(rho1, 0, m,m) * t3;
        
        
        % update w1 and w2 and part_weights
        w1 = w1 - tau * grad_w1;
        w2 = w2 - tau * grad_w2;
        part_weights = part_weights - tau * grad_part_weights;
        
        % compute the objective value
        J_history(iter + 1) = compute_obj(F1, y1, F2, y2, part_weights, beta, gamma, w1, w2, alpha, huber_k);
        if abs(J_history(iter + 1) - J_history(iter)) < tol
            break;
        end
        
    end
    
    function val = compute_obj(F1, y1, F2, y2, part_weights, beta, gamma, w1, w2, alpha, huber_k)
        [m, d1] = size(F1);
        [n, d2] = size(F2);
        e1 = F1 * w1 - y1;
        e2 = F2 * w2 - y2;
        
        
        v1 = alpha * (1/(2*m)) * sum( (e1.^2 )) + (gamma/2) * sum(w1.^2);
        v2 = alpha * (1/(2*n)) * sum( (e2.^2 )) + (gamma/2) * sum(w2.^2);

        e = F1 * w1 - part_weights * (F2 * w2);
        t = robust_function(e, 'huber', huber_k);
        
        v3 = (beta/(2*m)) * (sum(t));
        val = v1 + v2 + v3;
    