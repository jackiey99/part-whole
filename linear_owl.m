function [w1, w2, J_history] = linear_owl(F1, y1, F2, y2, mapping, opt)
 % This function uses the ordered sparse part-whole relationship
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
    lambda = opt.('lambda');
    tau = opt.('tau');
    tol = opt.('tol');
    max_iter = opt.('max_iter');
    verbose = opt.('verbose');
    
    
    alpha = 1;
    
    
    [m, d1] = size(F1);
    [n, d2] = size(F2);
    
    % initialize w1, w2, and part_weights
    w1 = -1/sqrt(m * 1) + (2/sqrt(m * 1))*rand(d1,1);
    w2 = -1/sqrt(n * 1) + (2/sqrt(n * 1))*rand(d2, 1);
    
    part_weights = mapping;
    W = create_weights(part_weights, 1, 1);
    
    J_history = [];
    J_history(1) = compute_obj(F1, y1, F2, y2, part_weights, beta, gamma, w1, w2, alpha, lambda,W);
   
    for iter = 1:max_iter 
        % print out information
        if verbose == 1 && mod(iter,100) == 0
            info = ['The ', num2str(iter), '-th iteration, obj val:', num2str(J_history(iter))];
            disp(info);
        end
        
        % fast implementation with vectorization
        pred_y1 = F1 * w1;
        w1_g1 = alpha * (1/m) * F1' * (pred_y1 - y1) + gamma * w1;
        pred_y2 = F2 * w2;
        
        t1 = part_weights * pred_y2;
        r1 = pred_y1 - t1;
        w1_g2 = (beta/m) * F1' * r1;
        
        grad_w1 = w1_g1 + w1_g2;
        
        w2_g1 = alpha * (1/n) * F2' * (pred_y2 - y2) + gamma * w2;
        
        t2 = part_weights * F2;
        w2_g2 = (beta/m) * t2' * r1;
        
        grad_w2 = w2_g1 - w2_g2;
        
        % t3 = bsxfun(@times, mapping, -pred_y2');
        t3 = mapping * spdiags(-pred_y2, 0, n, n);
        % grad_part_weights = (beta/m) * bsxfun(@times, r1, t3);
        grad_part_weights =  spdiags(r1, 0, m,m) * t3;
        
        
        % update w1 and w2 and part_weights
        w1 = w1 - tau * grad_w1;
        w2 = w2 - tau * grad_w2;
        part_weights = part_weights - tau * grad_part_weights;
        
        thres = tau * lambda ;
        
        part_weights = prox_owl(part_weights, thres * W);
        
        % compute the objective value
        J_history(iter + 1) = compute_obj(F1, y1, F2, y2, part_weights, beta, gamma, w1, w2, alpha, lambda, W);
        if abs(J_history(iter + 1) - J_history(iter)) < tol
            break;
        end
        
    end
end
    
function val = compute_obj(F1, y1, F2, y2, part_weights, beta, gamma, w1, w2, alpha, lambda, W)
    [m, d1] = size(F1);
    [n, d2] = size(F2);
    v1 = alpha * (1/(2*m)) * sum((F1 * w1 - y1).^2) + (gamma/2) * sum(w1.^2);
    v2 = alpha * (1/(2*n)) * sum((F2 * w2 - y2).^2) + (gamma/2) * sum(w2.^2);

    e = F1 * w1 - part_weights * (F2 * w2);
    owl = lambda * sum( sort(abs(part_weights), 2, 'descend').*W, 2);
    v3 = (beta/(2*m)) * (sum(e.^2) + sum(owl));
    val = v1 + v2 + v3;
end

function Sn = prox_owl(S, W)
Sn = S;
[B, I] = sort(abs(S), 2, 'descend');
 A = B - W;
 % project onto Km
 [m, n] = size(W);

 for i =1:m
     a = A(i, :);
     w = W(i,:) >0;
     y = a(w);
     
     nz = sum(w);
     x = nz:-1:1;
     p = lsqisotonic(x, full(y));
     p = max(0, p);
     ind = I(i, :);
     Sn(i, ind(w)) = p;
 end
Sn = sign(S).*Sn;
end

function W = create_weights(S, lam1, lam2)
    
    [m, n] = size(S);
    [B, ~] = sort(abs(S), 2, 'descend');
    [I, J] = find(B);
    nnz_rows = sum(S>0, 2);
    R = nnz_rows(I);
    V = lam1 + lam2 * (R - J);
    W = sparse(I, J, V, m, n);
end