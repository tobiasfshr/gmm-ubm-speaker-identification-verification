close all;
clc;
p = parameters();
[currentpath, name, ext] = fileparts(mfilename('fullpath')); %get script folder

%% initialisation
% gather training data
train_data = cell(1,length(p.core_set_indices));
disp('Gathering training data...');
if exist(strcat(currentpath, '\train_data.mat'), 'file')
    train_data = getfield(load(strcat(currentpath, '\train_data.mat')), 'train_data');
else
    for i = 1:length(p.core_set_indices)
        train_data{i} = getYohoData(p.yoho_path, p.core_set_indices(i), 'ENROLL');
        fprintf('%f/%f\n', i, length(p.core_set_indices));
    end
    cd(currentpath); %switch to script folder
    save('train_data.mat', 'train_data') %save data to shorten runtime
end    
disp('done.');

%gathering test data
test_data = cell(1,length(p.core_set_indices));
disp('Gathering test data...');
if exist(strcat(currentpath, '\test_data.mat'), 'file')
    test_data = getfield(load(strcat(currentpath, '\test_data.mat')), 'test_data');
else
    for i = 1:length(p.core_set_indices)
        test_data{i} = getYohoData(p.yoho_path, p.core_set_indices(i), 'VERIFY');
        fprintf('%f/%f\n', i, length(p.core_set_indices));
    end
    cd(currentpath); %switch to script folder
    save('test_data.mat', 'test_data') %save data to shorten runtime
end   
disp('done.');

%gathering UBM training data
ubm_data = cell(1,length(p.background_set_indices));
disp('Gathering UBM data...');
if exist(strcat(currentpath, '\ubm_data.mat'), 'file')
    ubm_data = getfield(load(strcat(currentpath, '\ubm_data.mat')), 'ubm_data');
else
    for i = 1:length(p.background_set_indices)
        ubm_data{i} = getYohoData(p.yoho_path, p.background_set_indices(i), 'ENROLL');
        fprintf('%f/%f\n', i, length(p.background_set_indices));
    end
    cd(currentpath); %switch to script folder
    save('ubm_data.mat', 'ubm_data') %save data to shorten runtime
end   
disp('done.');

%gathering impostor data
impostor_data = cell(1,length(p.background_set_indices));
disp('Gathering impostor data...');
if exist(strcat(currentpath, '\impostor_data.mat'), 'file')
    impostor_data = getfield(load(strcat(currentpath, '\impostor_data.mat')), 'impostor_data');
else
    for i = 1:length(p.impostor_set_indices)
        impostor_data{i} = getYohoData(p.yoho_path, p.impostor_set_indices(i), 'VERIFY');
        fprintf('%f/%f\n', i, length(p.impostor_set_indices));
    end
    cd(currentpath); %switch to script folder
    save('impostor_data.mat', 'impostor_data') %save data to shorten runtime
end   
disp('done.');

%% creating GMM, UBM and adapted GMM-UBM
%creating from-scratch GMM for core set speakers
gauss_mix_models = cell(1, length(p.core_set_indices));
disp('Creating gaussian mixture models...');
if exist(strcat(currentpath, '\gaussian_mix_models_', num2str(p.K), '.mat'), 'file')
    gauss_mix_models = getfield(load(strcat(currentpath, '\gaussian_mix_models_', num2str(p.K), '.mat')), 'gauss_mix_models');
else
    for i = 1:length(p.core_set_indices)
        [weights, means, covariances] = estGaussMixEM(train_data{i}(1:3400, :), p.K, p.n_iter, p.epsilon);
        
        gauss_mix_models{i}.w = weights;
        gauss_mix_models{i}.mu = means;
        gauss_mix_models{i}.sigma = covariances;
        fprintf('%f/%f\n', i, length(p.core_set_indices));
    end
    cd(currentpath); %switch to script folder
    save(strcat('gaussian_mix_models_', num2str(p.K), '.mat'), 'gauss_mix_models') %save data to shorten runtime
end   
disp('done.');

%plot GMM performance (accumulated loglikelihoods for test data) against number of gaussians
disp('Creating GMM performance vs number of gaussians plot...');
if exist(strcat(currentpath, '\gmm_likelihoods.mat'), 'file')
    logLikelihood = getfield(load(strcat(currentpath, '\gmm_likelihoods.mat')), 'logLikelihood');
else
    for k = 1:64
        [weights, means, covariances] = estGaussMixEM(train_data{1}, k, p.n_iter, p.epsilon);
        logLikelihood(k) = getLogLikelihood(means, weights, covariances, test_data{1});
    end
    cd(currentpath); %switch to script folder
    save('gmm_likelihoods.mat', 'logLikelihood') %save data to shorten runtime
end
disp('done.');
figure;
plot(1:length(logLikelihood), logLikelihood)
hold on;


%creating universal background model based on UBM dataset
disp('Creating universal background model...');
if exist(strcat(currentpath, '\ubm_', num2str(p.K), '.mat'), 'file')
    ubm = getfield(load(strcat(currentpath, '\ubm_', num2str(p.K), '.mat')), 'ubm');
else
    ubm_data_merged = []; %merge all ubm speakers to one 2D vector of MFCCs
    for i = 1:length(p.background_set_indices)
        ubm_data_merged = [ubm_data_merged; ubm_data{i}(1:3400, :)];
    end

    [weights, means, covariances] = estGaussMixEM(ubm_data_merged, p.K, p.n_iter, p.epsilon);
        
    ubm.w = weights;
    ubm.mu = means;
    ubm.sigma = covariances;
    
    cd(currentpath); %switch to script folder
    save(strcat('ubm_', num2str(p.K), '.mat'), 'ubm') %save data to shorten runtime
end   
disp('done.');

%adapting UBM to build GMM UBM speaker models using MAP
disp('Adapting UBM to build speaker models with MAP estimation...');
gmm_ubm_models = cell(1, length(p.core_set_indices));
if exist(strcat(currentpath, '\gmm_ubm_', num2str(p.K), '.mat'), 'file')
    gmm_ubm_models = getfield(load(strcat(currentpath, '\gmm_ubm_', num2str(p.K), '.mat')), 'gmm_ubm_models');
else
    for i = 1:length(p.core_set_indices)
        gmm_ubm_models{i} = mapAdapt(ubm, train_data{i}, p.r); %create GMM UBM models for core set speakers using Netlab Toolbox (MAP Adaptation)
        fprintf('%f/%f\n', i, length(p.core_set_indices));
    end
    cd(currentpath); %switch to script folder
    save(strcat('gmm_ubm_', num2str(p.K), '.mat'), 'gmm_ubm_models') %save data to shorten runtime
end   
disp('done.');

%% speaker identification
%creating confusion matrices for both plain GMM and GMM UBM models to
%evaluate performance
disp('Creating confusion matrices...');
confusion_matrix_gmm = zeros(length(p.core_set_indices), length(p.core_set_indices));
confusion_matrix_gmm_ubm = zeros(length(p.core_set_indices), length(p.core_set_indices));

if exist(strcat(currentpath, '\confusion_matrix_gmm.mat'), 'file') && exist(strcat(currentpath, '\confusion_matrix_gmm_ubm.mat'), 'file')
    confusion_matrix_gmm = getfield(load('confusion_matrix_gmm.mat'), 'confusion_matrix_gmm');
    confusion_matrix_gmm_ubm = getfield(load('confusion_matrix_gmm_ubm.mat'), 'confusion_matrix_gmm_ubm');
else
    for i = 1:length(p.core_set_indices)
        %create 40 chunks out of test data (40 utterances)
        current_data = createChunks(test_data{i}, 40);
        
        for j = 1:length(current_data)
            likelihood_gmm = zeros(1, length(p.core_set_indices));
            likelihood_gmm_ubm = zeros(1, length(p.core_set_indices));
        
            for k = 1:length(p.core_set_indices)                           
                %compute loglikelihoods for all speaker models
                likelihood_gmm(k) = getLogLikelihood(gauss_mix_models{k}.mu, gauss_mix_models{k}.w, gauss_mix_models{k}.sigma, convert_2d(current_data(j, :, :)));
                likelihood_gmm_ubm(k) = getLogLikelihood(gmm_ubm_models{k}.mu, gmm_ubm_models{k}.w, gmm_ubm_models{k}.sigma, convert_2d(current_data(j, :, :)));
            end
            %check which model has highest loglikelihood and assign to speaker
            %in confusion matrix
            [~, argmax] = max(likelihood_gmm);
            confusion_matrix_gmm(i, argmax) = confusion_matrix_gmm(i, argmax) + 1;
        
            %same for ubm_gmm_models
            [argvalue, argmax] = max(likelihood_gmm_ubm);
            confusion_matrix_gmm_ubm(i, argmax) = confusion_matrix_gmm_ubm(i, argmax) + 1;
        end
        fprintf('%f/%f rows\n', i, length(p.core_set_indices));
    end  
    cd(currentpath); %switch to script folder
    save('confusion_matrix_gmm.mat', 'confusion_matrix_gmm');
    save('confusion_matrix_gmm_ubm.mat', 'confusion_matrix_gmm_ubm');
end
disp('done.');

disp('Confusion matrices:');
confusion_matrix_gmm
%output confusion matrices
figure;
imshow(confusion_matrix_gmm, [], 'InitialMagnification', 1600);
colorbar;
axis on;
hold on;

confusion_matrix_gmm_ubm
figure;
imshow(confusion_matrix_gmm_ubm, [], 'InitialMagnification', 1600);
colorbar;
axis on;

%% speaker verification system
%set thresholds with training data and ubm data so that both FAR and
%FRR are minimized over the datasets
disp('setting thresholds for speaker verification...');
if (not(isfield(gmm_ubm_models{1}, 'threshold'))) %if thresholds are not already computed..
    for i = 1:length(p.core_set_indices)
        %create 96 chunks out of training data (96 utterances)
        current_data = createChunks(train_data{i}, 96);
        
        %compute LLR for each utterance
        LLR_min_own = 100000; %init very high
        for j = 1:length(current_data)
            LLR = getLLR(gmm_ubm_models{i}, ubm, convert_2d(current_data(j, :, :)));
            if (LLR < LLR_min_own)
                LLR_min_own = LLR;
            end
        end
        
        LLR_max_ubm = -100000; %init very low
        for k = 1:length(ubm_data)
            %create 96 chunks out of ubm data (96 utterances)
            current_data = createChunks(ubm_data{k}, 96);
            
            %compute LLR for each utterance
            for j = 1:length(current_data)
                LLR = getLLR(gmm_ubm_models{i}, ubm, convert_2d(current_data(j, :, :)));
                if (LLR > LLR_max_ubm)
                    LLR_max_ubm = LLR;
                end
            end
        end
        
        %set threshold according to LLRs as avg to balance between FAR and FRR
        gmm_ubm_models{i}.threshold = (LLR_min_own + LLR_max_ubm) / 2;
        
    end
    cd(currentpath); %switch to script folder
    save(strcat('gmm_ubm_', num2str(p.K), '.mat'), 'gmm_ubm_models') %save data to shorten runtime
end
disp('done.');



% test the GMM UBM system with impostor trials
disp('Conducting impostor trials and creating confusion matrices...');
impostor_confusion_matrices = cell(1, length(p.core_set_indices));
if exist(strcat(currentpath, '\impostor_confusion_matrices.mat'), 'file') && exist(strcat(currentpath, '\impostor_confusion_matrices.mat'), 'file')
    impostor_confusion_matrices = getfield(load('impostor_confusion_matrices.mat'), 'impostor_confusion_matrices');
else
    for i = 1:length(p.core_set_indices)
        %create 40 chunks out of test data (40 utterances)
        current_data = createChunks(test_data{i}, 40);
        
        %initialise confusion matrix
        impostor_confusion_matrices{i} = zeros(2, 2);
        
        %compute LLR for each utterance
        for j = 1:length(current_data)
            LLR_avg_own = getLLR(gmm_ubm_models{i}, ubm, convert_2d(current_data(j, :, :)));
            %decide if utterance is accepted by verification system
            if (LLR_avg_own > gmm_ubm_models{i}.threshold)
                %accept (1)
                impostor_confusion_matrices{i}(1, 1) = impostor_confusion_matrices{i}(1, 1) + 1;
            else
                %reject (2)
                impostor_confusion_matrices{i}(1, 2) = impostor_confusion_matrices{i}(1, 2) + 1;
            end
        end
        
        for k = 1:length(impostor_data)
            %create 40 chunks out of test data (40 utterances)
            current_data = createChunks(impostor_data{k}, 40);
            
            %compute LLR for each utterance
            for j = 1:length(current_data)
                LLR_avg_impostor = getLLR(gmm_ubm_models{i}, ubm, convert_2d(current_data(j, :, :)));
                %decide if utterance is accepted by verification system
                if (LLR_avg_impostor > gmm_ubm_models{i}.threshold)
                    %accept (1)
                    impostor_confusion_matrices{i}(2, 1) = impostor_confusion_matrices{i}(2, 1) + 1;
                else
                    %reject (2)
                    impostor_confusion_matrices{i}(2, 2) = impostor_confusion_matrices{i}(2, 2) + 1;
                end
            end
        end
        
    end
    cd(currentpath); %switch to script folder
    save(strcat('impostor_confusion_matrices.mat'), 'impostor_confusion_matrices') %save data to shorten runtime
end
disp('done.');

disp('FAR, FRR and confusion matrices for impostor trials:');
for i = 1:length(impostor_confusion_matrices)
    FRR(i) = impostor_confusion_matrices{i}(1,2) / (impostor_confusion_matrices{i}(1,2) + impostor_confusion_matrices{i}(1,1));
    FAR(i) = impostor_confusion_matrices{i}(2,1) / (impostor_confusion_matrices{i}(2,1) + impostor_confusion_matrices{i}(2,2));
    FAR(i)
    FRR(i)
    impostor_confusion_matrices{i}
end    
avg_FAR = sum(FAR) / 10
avg_FRR = sum(FRR) / 10

        



