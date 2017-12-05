function [ ] = demo_nhmc_GMM(nStates,nSelBand,cviter)

% Input
% nStates - number of Gaussian states
% nSelBand - number of bands to keep
% cviter - fold number in a 5-fold cross validation setup

t1 = clock;

% Wavelet transform
wvlt(cviter,'db1')

% Generate NHMC state label matrix for each hyperspectral spectrum.
% The step is:
% 1. spectrum -> 
% 2. wavelet coefficient matrix -> 
% 3. NHMC model parameters -> 
% 4. NHMC state label matrix.
% Here is an example using GMM with 5 Gaussian components.
nhmclabel(nStates,cviter)

% Band selection
band_selection_GMM(nStates,nSelBand,cviter)

% Classification using support vector machine (SVM)
indir = strcat('./Equal State Prior Probability Analysis/',...
    num2str(nStates),'States/GMM/');
outdir = strcat(indir,num2str(nSelBand),' Bands/');
if ~exist(outdir,'dir')
    mkdir(outdir)
end
if ~exist(strcat(outdir,'max_accuracy_cv',num2str(cviter),'.mat'),'file')
    max_accuracy = svm_test_whole_GMM(nStates,nSelBand,cviter);
else
    load(strcat(outdir,'max_accuracy_cv',num2str(cviter),'.mat'))
end

% Time Consumption Measurement
timecspt = etime(clock,t1)/3600;
save(strcat(outdir,'timecspt.mat'),'timecspt')
disp(['The whole process takes ',num2str(timecspt),' hours.'])

end