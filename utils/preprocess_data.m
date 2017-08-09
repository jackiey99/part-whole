function data_split = preprocess_data(train_rate, test_rate, nor_method)
%% preprocess the math data, read the mat files, split training and testing, feature normalization
%%
    % read the data
    data_path = '../data/';
    load([data_path, 'movie_data.mat']);
    
    part_fea = part_feature_mat;
    part_target = part_scoreNormalize(part_target);
    whole_fea = whole_feature_mat;
    whole_target = whole_scoreNormalize(whole_target);
    
    m = size(whole_fea, 1);
    n = size(part_fea, 1);

    mapping = spdiags(1./sum(mapping, 2), 0, m, m) * mapping;
    network = actor_actor;
    
    %mapping =  bsxfun(@times, M, 1./(sum(M, 2)));
    
   
    if strcmp(nor_method, 'min-max')
        % split the training and testing for whole and part
        n_whole = length(whole_target);
        train_num = floor(n_whole * train_rate);
        test_point = floor(n_whole * (1 - test_rate));
        whole_train_ind = 1:train_num;
        whole_test_ind = test_point + 1 : n_whole;
        
        % whole training
        whole_train_fea = whole_fea(whole_train_ind,:);
        whole_fea_ind = (max(whole_train_fea) - min(whole_train_fea)) ~= 0; % remove possible columns with same value
        whole_train_fea = whole_train_fea(:, whole_fea_ind);
        [whole_train_fea, whole_min, whole_max] = minmaxNormalize(whole_train_fea); % min-max normalization
        whole_train_fea = [whole_train_fea, ones(size(whole_train_fea,1), 1)];
        whole_train_target = whole_target(whole_train_ind);
        
        % whole testing
        whole_test_fea = whole_fea(whole_test_ind, :);
        whole_test_fea = whole_test_fea(:,whole_fea_ind);
        whole_test_fea = minmaxNormalize(whole_test_fea, whole_min, whole_max);
        whole_test_fea = [whole_test_fea, ones(size(whole_test_fea, 1), 1)];
        whole_test_target = whole_target(whole_test_ind);
        
        n_part = length(part_target);
        part_train_ind = find(sum(mapping(whole_train_ind, :), 1));
        part_test_ind = find(sum(mapping(whole_test_ind, :), 1));
        
        % part training
        part_train_fea = part_fea(part_train_ind, :);
        part_fea_ind = (max(part_train_fea) - min(part_train_fea)) ~= 0;
        part_train_fea = part_train_fea(:, part_fea_ind);
        [part_train_fea, part_min, part_max] = minmaxNormalize(part_train_fea);
        part_train_fea = [part_train_fea, ones(size(part_train_fea, 1), 1)];
        part_train_target = part_target(part_train_ind);
        
        % part testing
        part_test_fea = part_fea(part_test_ind, :);
        part_test_fea = part_test_fea(:, part_fea_ind);
        part_test_fea = minmaxNormalize(part_test_fea, part_min, part_max);
        part_test_fea = [part_test_fea, ones(size(part_test_fea, 1), 1)];
        part_test_target = part_target(part_test_ind);
    end
    
    if strcmp(nor_method, 'std')
        % split the training and testing for whole and part
        n_whole = length(whole_target);
        train_num = floor(n_whole * train_rate);
        test_point = floor(n_whole * (1 - test_rate));
        whole_train_ind = 1:train_num;
        whole_test_ind = test_point + 1 : n_whole;
        
        % whole training
        whole_train_fea = whole_fea(whole_train_ind,:);
        whole_fea_ind = (max(whole_train_fea) - min(whole_train_fea)) ~= 0; % remove possible columns with same value
        whole_train_fea = whole_train_fea(:, whole_fea_ind);
        [whole_train_fea, whole_mean, whole_std] = stdNormalize(whole_train_fea); % standarization
        whole_train_fea = [whole_train_fea, ones(size(whole_train_fea,1), 1)];
        whole_train_target = whole_target(whole_train_ind);
        
        % whole testing
        whole_test_fea = whole_fea(whole_test_ind, :);
        whole_test_fea = whole_test_fea(:,whole_fea_ind);
        whole_test_fea = stdNormalize(whole_test_fea, whole_mean, whole_std);
        whole_test_fea = [whole_test_fea, ones(size(whole_test_fea, 1), 1)];
        whole_test_target = whole_target(whole_test_ind);
        
        n_part = length(part_target);
        part_train_ind = find(sum(mapping(whole_train_ind, :), 1));
        part_test_ind = find(sum(mapping(whole_test_ind, :), 1));
        
        % part training
        part_train_fea = part_fea(part_train_ind, :);
        part_fea_ind = (max(part_train_fea) - min(part_train_fea)) ~= 0;
        part_train_fea = part_train_fea(:, part_fea_ind);
        [part_train_fea, part_mean, part_std] = stdNormalize(part_train_fea);
        part_train_fea = [part_train_fea, ones(size(part_train_fea, 1), 1)];
        part_train_target = part_target(part_train_ind);
        
        % part testing
        part_test_fea = part_fea(part_test_ind, :);
        part_test_fea = part_test_fea(:, part_fea_ind);
        part_test_fea = stdNormalize(part_test_fea, part_mean, part_std);
         part_test_fea = [part_test_fea, ones(size(part_test_fea, 1), 1)];
        part_test_target = part_target(part_test_ind);
    end
    
    
    % training mapping
    train_mapping = mapping(whole_train_ind, part_train_ind);
    train_graph = network(part_train_ind, part_train_ind);

    
    % output
    data_split.('whole_train_fea') = whole_train_fea;
    data_split.('whole_train_target') = whole_train_target;
    data_split.('whole_test_fea') = whole_test_fea;
    data_split.('whole_test_target') = whole_test_target;
    
    data_split.('part_train_fea') = part_train_fea;
    data_split.('part_train_target') = part_train_target;
    data_split.('part_test_fea') = part_test_fea;
    data_split.('part_test_target') = part_test_target;
    
    data_split.('train_mapping') = train_mapping;
    data_split.('train_graph') = train_graph;

end