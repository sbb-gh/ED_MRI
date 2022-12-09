%==========================================================================
% 
% Generate MRI signals using the 2 compartment FEXI model
%
% Use: [s_train, s_valid, s_test, p_train, p_valid, p_test, map] = generate_2CM_fexi_data(n_train, n_valid, n_test)   
%
% Inputs    - n_train:  int, number of training samples
%           - n_valid:  int, number of validation samples
%           - n_test:   int, number of testing samples
%
% Output: 	- s_train:  single, signals for training dataset, [n_train, n_acq]
%           - s_valid:  single, signals for validation dataset, [n_valid, n_acq]
%           - s_test:   single, signals for testing dataset, [n_test, n_acq]
%           - p_train:  single, model parameters for training dataset, [n_train, 4]
%           - p_valid:  single, model parameters for validation dataset, [n_train, 4]
%           - p_test:   single, model parameters for testing dataset, [n_train, 4]
%           - map:      container map, indicates dependent acquisitions for each b==0
%
% Author: Elizabeth Powell, 02/12/2022
%
%==========================================================================

function [s_train, s_valid, s_test, p_train, p_valid, p_test, map] = generate_2CM_fexi_data(n_train, n_valid, n_test)   
    %% User defined parameters
    rng(1)
    
    % acquisition parameters
    bf_i = (0:50:1000)*1e6;                 % filter block b-values [m^2/s]
    b_i = (0:50:1000)*1e6;                  % measurement block b-values [m^2/s]
    tm_i = .02:.02:.42;                     % mixing times [s]
    acq_params = combvec(bf_i,b_i,tm_i)';   % all combinations of bf, b, tm
    n_acq = size(acq_params,1);             % total number of acquisitions
    % combinations separated into individual parameters
    bf = acq_params(:,1);
    b = acq_params(:,1);
    tm = acq_params(:,1);
    
    % model parameter ranges for uniform dist (training & validation)
    da_u = [.5, 3]*1e-9;    % diffusivity, compartment A, [m^2/s]
    db_u = [3, 30]*1e-9;    % diffusivity, compartment B, [m^2/s]
    faeq_u = [.9, 1];       % signal fraction range, compartment A at equilibrium
    k_u = [.1, 6];          % exchange rate range, [s-1]
    
    % model parameter [mean, SD] for normal dist (testing)
    da_n = [1, .1]*1e-9;    % diffusivity, compartment A, [m^2/s]
    db_n = [10, 3]*1e-9;    % diffusivity, compartment B, [m^2/s]
    faeq_n = [.95, .1];     % signal fraction, compartment A at equilibrium
    k_n = [3, 1];           % exchange rate, [s-1]
        
    %% Set up model
    
    % signal fractions immediately after filter block
    fa0 = @(faeq,da,db,bf) faeq*exp(-bf*da) ./ (faeq*exp(-bf*da) + (1-faeq)*exp(-bf*db));
    % signal fractions after mixing time
    fatm = @(faeq,fa0,k,tm) faeq + (fa0 - faeq).*exp(-k*tm);
    % generate signals
    s = @(fatm,da,db,b) ( (1-fatm).*exp(-b*db) + fatm.*exp(-b*da) );
    
    %% Generate training data
    
    fprintf('Generating training data... ')
    
    % generate model parameters from uniform distribution between pre-defined ranges
    da_train = da_u(1) + (da_u(2)-da_u(1)).*rand(n_train,1);
    db_train = db_u(1) + (db_u(2)-db_u(1)).*rand(n_train,1);
    faeq_train = faeq_u(1) + (faeq_u(2)-faeq_u(1)).*rand(n_train,1);
    k_train = k_u(1) + (k_u(2)-k_u(1)).*rand(n_train,1);
    
    s_train = zeros(n_train,n_acq,'single');
    for i = 1:n_train
        s_train(i,:) = s(fatm(faeq_train(i),fa0(faeq_train(i),da_train(i),db_train(i),bf),k_train(i),tm),da_train(i),db_train(i),b);
    end
    
    fprintf('Done\n')
    
    %% Generate validation data
    
    fprintf('Generating validation data... ')
    
    % generate model parameters from uniform distribution between pre-defined ranges
    da_valid = da_u(1) + (da_u(2)-da_u(1)).*rand(n_valid,1);
    db_valid = db_u(1) + (db_u(2)-db_u(1)).*rand(n_valid,1);
    faeq_valid = faeq_u(1) + (faeq_u(2)-faeq_u(1)).*rand(n_valid,1);
    k_valid = k_u(1) + (k_u(2)-k_u(1)).*rand(n_valid,1);
    
    s_valid = zeros(n_valid,n_acq,'single');
    for i = 1:n_valid
        s_valid(i,:) = s(fatm(faeq_valid(i),fa0(faeq_valid(i),da_valid(i),db_valid(i),bf),k_valid(i),tm),da_valid(i),db_valid(i),b);
    end
    
    fprintf('Done\n')
    
    %% Generate testing data
    
    fprintf('Generating testing data... ')
    
    % generate model parameters from normal distribution 
    da_test = da_n(1) + da_n(2).*randn(n_test,1);
    db_test = db_n(1) + db_n(2).*randn(n_test,1);
    faeq_test = faeq_n(1) + faeq_n(2).*randn(n_test,1);
    k_test = k_n(1) + k_n(2).*randn(n_test,1);
    
    s_test = zeros(n_test,n_acq,'single');
    for i = 1:n_test
        s_test(i,:) = s(fatm(faeq_test(i),fa0(faeq_test(i),da_test(i),db_test(i),bf),k_test(i),tm),da_test(i),db_test(i),b);
    end
    
    fprintf('Done\n')
    
    %% Concatenate and normalise model parameters to return

    p_train = single([da_train, db_train, faeq_train, k_train]) .* [1e9,1e9,1,1];
    p_valid = single([da_valid, db_valid, faeq_valid, k_valid]) .* [1e9,1e9,1,1];
    p_test = single([da_test, db_test, faeq_test, k_test]) .* [1e9,1e9,1,1];

    %% Create container map / hash table of dependent acquisitions 
    % ensures valid b=0 is available for data normalisation

    % keys index each measurement (i.e. combination of bf,b,tm)
    keys = string(1:n_acq);

    % get unique combinations of bf, tm
    [uniq,idx,~] = unique(acq_params(:,[1,3]),'rows','stable');
    
    % get index of b==0 for each unique combination of bf, tm


    % assign values to keys (only combinations b==0 will have values)
    values = cell(1,n_acq);
    for i = 1:length(idx) 
        key = find(all(acq_params(:,[1,3])==uniq(i,:),2) & acq_params(:,2)==0);
        vals = find(all(acq_params(:,[1,3])==uniq(i,:),2) & acq_params(:,2)>0);
        values{key} = vals;
    end

    map = containers.Map(keys,values);

end