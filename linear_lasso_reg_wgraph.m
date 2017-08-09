function [w1, w2, J_history, part_weights] = linear_lasso_reg_wgraph(F1, y1, F2, y2, mapping, graph, opt)
% This function uses the sparse part-whole relationship w/ part-part network
% Input --
%  F1, y1: whole features/target
%  F2, y2: part features/target
%  mapping: whole to part mapping
%  graph: part-part network 
%  opt: various parameters
% Output --
%  w1, w2: parameters for predicting whole/part
%  J_history: objective function values
%  part_weights: part's contribution to the whole


    rng(2);
    
    beta = opt.('beta');
    gamma = opt.('gamma');
    lambda = opt.('lambda');
    tau = opt.('tau');
    tol = opt.('tol');
    max_iter = opt.('max_iter');
    verbose = opt.('verbose');
    alpha_graph = opt.('alpha_graph');
    
    
    alpha = 1;
    
    
    [m, d1] = size(F1);
    [n, d2] = size(F2);
    
    % initialize w1, w2, and part_weights
    w1 = -1/sqrt(m * 1) + (2/sqrt(m * 1))*rand(d1,1);
    w2 = -1/sqrt(n * 1) + (2/sqrt(n * 1))*rand(d2, 1);
    
    part_weights = mapping;
    
    degs = sum(graph, 2);
    D = sparse(1:n, 1:n, degs, n, n);
    L = D - graph;
    
    
    
    degs(degs == 0) = eps;
    % calculate D^(-1/2)
    D = spdiags(1./(degs.^0.5), 0, size(D, 1), size(D, 2));
    
    % calculate normalized Laplacian
    L = D * L * D;
    
 
    J_history = [];
    J_history(1) = compute_obj(F1, y1, F2, y2, part_weights, beta, gamma, w1, w2, alpha, lambda);
   
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
        
        w2_graph = (alpha_graph/n) * F2'*(L * pred_y2);
        
        grad_w2 = w2_g1 - w2_g2 + w2_graph;
        
        t3 = mapping * spdiags(-pred_y2, 0, n, n);
        grad_part_weights = (beta/m) * spdiags(r1, 0, m,m) * t3;
        
%         grad_part_contributions = part_con(part_weights, graph, mapping);
        
        
        % update w1 and w2 and part_weights
        w1 = w1 - tau * grad_w1;
        w2 = w2 - tau * grad_w2;
%         part_weights = part_weights - tau * (grad_part_weights +  grad_part_contributions);
         part_weights = part_weights - tau * (1 * grad_part_weights);
        thres = tau * lambda / (beta/2 *m);
        % part_weights = sign(part_weights) .* max(0, abs(part_weights) - thres);
        [I, J, V] = find(part_weights);
        shrinkV = sign(V).* max(0, abs(V) - thres);
        part_weights = sparse(I, J, shrinkV, size(part_weights,1), size(part_weights,2));
        
        % compute the objective value
        J_history(iter + 1) = compute_obj(F1, y1, F2, y2, part_weights, beta, gamma, w1, w2, alpha, lambda);
        if abs(J_history(iter + 1) - J_history(iter)) < tol
            break;
        end
        
    end
end
    
    function val = compute_obj(F1, y1, F2, y2, part_weights, beta, gamma, w1, w2, alpha, lambda)
        [m, d1] = size(F1);
        [n, d2] = size(F2);
        v1 = alpha * (1/(2*m)) * sum((F1 * w1 - y1).^2) + (gamma/2) * sum(w1.^2);
        v2 = alpha * (1/(2*n)) * sum((F2 * w2 - y2).^2) + (gamma/2) * sum(w2.^2);

        e = F1 * w1 - part_weights * (F2 * w2);
        lasso = lambda * sum(abs(part_weights), 2);
        v3 = (beta/(2*m)) * (sum(e.^2) + sum(lasso));
        val = v1 + v2 + v3;
    end

    function grad_part_contributions = part_con(part_weights, graph, mapping)
        grad_part_contributions = part_weights;
        [m, n] = size(part_weights);
        for i = 1:m
            pw = part_weights(i, :);
            parts = mapping(i, :) > 0;
            a = pw(parts);
            sub_graph = graph(parts, parts);
            nn = size(sub_graph, 1);
            degs = sum(sub_graph, 2);
            D = sparse(1:nn, 1:nn, degs, nn, nn);
            L = D - sub_graph;
      
            degs(degs == 0) = eps;
            % calculate D^(-1/2)
            D = spdiags(1./(degs.^0.5), 0, size(D, 1), size(D, 2));
    
            % calculate normalized Laplacian
            L = D * L * D;
            new_a = L * a';
            grad_part_contributions(i, parts) = new_a';
        end
    end
    