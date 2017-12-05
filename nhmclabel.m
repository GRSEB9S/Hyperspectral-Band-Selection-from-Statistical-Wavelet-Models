%   classification: performs hyperspectral signagure classification based 
%   on NHMC models.
%
%   BEGIN COPYRIGHT NOTICE
%
%   NHMC code -- (c) 2013-2014 Siwei Feng
%
%    This code is provided as is, with no guarantees except that 
%    bugs are almost surely present.  Published reports of research 
%    using this code (or a modified version) should cite the 
%    article that describes the algorithm: 
%
%      [NHMC] S. Feng, M. F. Duarte, Wavelet-Based Non-Homogeneous
%      Hidden Markov Chain Model Features for Hyperspectral
%      Signature Classification, 2014.
%
%    Comments and bug reports are welcome.  Email to siwei@engin.umass.edu. 
%    I would also appreciate hearing about how you used this code, 
%    improvements that you have made to it, or translations into other
%    languages.    
%
%    You are free to modify, extend or distribute this code, as long 
%    as this copyright notice is included whole and unchanged. 
%
%    END COPYRIGHT NOTICE
%
%    Usage: [] = nhmclabel(nStates)
%    nStates - number of Gaussian states
%    cviter - fold number in a 5-fold cross validation setup

function [  ] = nhmclabel(nStates,cviter)

t1 = clock;

outdir1 = './Equal State Prior Probability Analysis/';
if ~exist(outdir1,'dir')
    mkdir(outdir1)
end

outdir_GMM = strcat(outdir1,num2str(nStates),'States/GMM/');
if ~exist(outdir_GMM,'dir')
    mkdir(outdir_GMM)
end

load(strcat('./fullIndianPine_20151215_training_cv',num2str(cviter),'.mat'))

% Compile cpp codes
mex AfinalHSI.cpp
mex Bviterbi.cpp

load(strcat(outdir1,'min_wvlt_train_cv',num2str(cviter),'.mat'))

% Train NHMC parameters based on GMM
disp('Parameter training started...')
tStart = clock;
fname_optimal_model_GMM = strcat(outdir_GMM,...
    'optimal_model_GMM_cv',...
    num2str(cviter),'.mat');
if ~exist(fname_optimal_model_GMM,'file')
    dimA = [size(min_wvlt_train), nStates];
    [transitions_GMM,initprobs_GMM,gaussmu,gaussSigma,stateprobs_GMM] = ...
        AfinalHSI(min_wvlt_train, dimA);
    save(fname_optimal_model_GMM,'transitions_GMM','initprobs_GMM',...
        'gaussmu','gaussSigma','stateprobs_GMM')
else
    load(fname_optimal_model_GMM)
end
disp(['Finished, the process takes ',num2str(etime(clock,tStart)/3600),...
    ' hours.'])

% Compute state labels for training data based on GMM
disp('Label computing for training data started...')
tStart = clock;
fname_optimal_labels_training_GMM = strcat(outdir_GMM,...
    'optimal_labels_training_GMM_cv',num2str(cviter),'.mat');
if ~exist(fname_optimal_labels_training_GMM,'file')
    dimA = [size(min_wvlt_train), nStates];
    qp1_training_GMM = Bviterbi(min_wvlt_train,dimA,transitions_GMM,...
        initprobs_GMM,gaussmu,gaussSigma);
    save(fname_optimal_labels_training_GMM,'qp1_training_GMM');
else
    load(fname_optimal_labels_training_GMM)
end
disp(['Finished, the process takes ',num2str(etime(clock,tStart)/3600),...
    ' hours.'])

disp(['The whole process takes ',num2str(etime(clock,t1)/3600),' hours.'])

end