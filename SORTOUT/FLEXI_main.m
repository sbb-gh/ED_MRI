n_train = 10^4
n_valid = n_train/10
n_test = n_train/10
save_dir = '/home/stefano/Desktop/MATLAB/Data/'

%[s_train, s_valid, s_test, p_train, p_valid, p_test, map] = generate_2CM_fexi_data(n_train, n_valid, n_test);
[train_inp, val_inp, test_inp, train_tar, val_tar, test_tar, measurement_to_dependents] = generate_2CM_fexi_data(n_train, n_valid, n_test);

% Can someone simplify the below, please, can you loop over the names?

save_f = strcat(save_dir,"train_inp")
save(save_f, "train_inp")

save_f = strcat(save_dir,"val_inp")
save(save_f, "val_inp")

save_f = strcat(save_dir,"test_inp")
save(save_f, "test_inp")

save_f = strcat(save_dir,"train_tar")
save(save_f, "train_tar")

save_f = strcat(save_dir,"val_tar")
save(save_f, "val_tar")

save_f = strcat(save_dir,"test_tar")
save(save_f, "test_tar")

save_f = strcat(save_dir,"measurement_to_dependents")
save(save_f, "measurement_to_dependents")