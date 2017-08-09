function [rmse_whole, rmse_part, rmse_total] = eval_models(w_whole, w_part, data_split)
    seed = RandStream('mt19937ar','Seed',0);

    whole_test_fea = data_split.('whole_test_fea');
    whole_test_target = data_split.('whole_test_target');
    part_test_fea = data_split.('part_test_fea');
    part_test_target = data_split.('part_test_target');
    
    % err of whole
    pred_whole = whole_test_fea * w_whole;
    err_whole = pred_whole - whole_test_target;
    rmse_whole = sqrt(mean(err_whole.^2));
    info = ['Test RMSE on whole is:', num2str(rmse_whole)];
    disp(info);
    
    % err of part
    pred_part = part_test_fea * w_part;
    err_part = pred_part - part_test_target;
    rmse_part = sqrt(mean(err_part.^2));
    info = ['Test RMSE on parts is:', num2str(rmse_part)];
    disp(info);
    
    % overall error
    n_test_whole = length(pred_whole);
    n_test_part = length(pred_part);
    rmse_total = sqrt((n_test_whole * (rmse_whole^2) + n_test_part * (rmse_part^2))/(n_test_whole + n_test_part));
    info = ['Total RMSE is:', num2str(rmse_total)];
    disp(info);
    
end