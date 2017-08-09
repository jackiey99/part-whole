% load utils
addpath('utils/');

tr = 0.5;

train_rate = tr;
test_rate = 0.1;

% train the joint model using average aggregation
data_split = preprocess_data(train_rate, test_rate, 'min-max');
whole_train_fea = data_split.('whole_train_fea');
whole_train_target = data_split.('whole_train_target');
part_train_fea = data_split.('part_train_fea');
part_train_target = data_split.('part_train_target');
train_mapping = data_split.('train_mapping');
train_graph = data_split.('train_graph');

% train the joint model using aggregation
% the parameters need to be tuned for each part-whole relationship
opt.('beta') = 0.6;
opt.('gamma') =0.01;
opt.('lambda') = 1;
opt.('alpha_graph') = 3;
opt.('tau') = 0.1;
opt.('tol') = 10^-5;
opt.('max_iter') = 10000;
opt.('verbose') = 1;

% w/o part-part graph
% [w_whole, w_part, J_history, part_weights] = linear_lasso_reg(whole_train_fea, whole_train_target, ...
%                                                 part_train_fea, part_train_target, train_mapping, opt);


% w/ part-part graph
[w_whole, w_part, J_history, part_weights] = linear_lasso_reg_wgraph(whole_train_fea, whole_train_target, ...
                                                part_train_fea, part_train_target, train_mapping, train_graph, opt);


% evaluate the models
[rmse_whole, rmse_part, rmse_total] = eval_models(w_whole, w_part, data_split);



      