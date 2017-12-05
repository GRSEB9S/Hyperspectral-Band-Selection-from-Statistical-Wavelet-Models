function [ nhmc_success_rate ] = svm_test_GMM(nStates,nSelBand,cost,gamma,cviter)

% Input
% nStates - number of Gaussian states
% nSelBand - number of bands to keep
% cost - parameter of SVM
% gamma - parameter of RBF kernel
% cviter - fold number in a 5-fold cross validation setup

addpath(genpath('./libsvm-3.18/'))

% SVM parameter setup
libsvm_options = ['-c ',num2str(cost),' -g ',num2str(gamma)];

% Load both training and test data
load(strcat('./fullIndianPine_20151215_training_cv',num2str(cviter),'.mat'))
load(strcat('./fullIndianPine_20151215_test_cv',num2str(cviter),'.mat'))

% Create output directory
indir = strcat('./Equal State Prior Probability Analysis/',...
    num2str(nStates),'States/GMM/');
outdir = strcat(indir,num2str(nSelBand),' Bands/');

% Load selected bands
load(strcat(outdir,'selBandKeep_cv',num2str(cviter),'.mat'))

% Extract selected bands from both training and test data
train_data = min_data_training_cv(selBandKeep,:);
test_data = min_data_test_cv(selBandKeep,:);

% SVM model training
model = svmtrain(correctlabels_training_cv',train_data',libsvm_options);
compminind_nhmc = svmpredict(correctlabels_test_cv',test_data',model);
compminind_nhmc = compminind_nhmc';
compminind_nhmc = double(compminind_nhmc);
labels = [compminind_nhmc; double(compminind_nhmc-correctlabels_test_cv == 0)];
nhmc_success_rate = sum(compminind_nhmc-correctlabels_test_cv == 0)*100/length(compminind_nhmc)
    
rmpath(genpath('./libsvm-3.18/'))

end