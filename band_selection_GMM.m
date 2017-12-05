function [  ] = band_selection_GMM(nStates,nSelBand,cviter)

% Input
% nStates - number of Gaussian states
% nSelBand - number of bands to keep
% cviter - fold number in a 5-fold cross validation setup

% Load state label matrices generated in previous step.
indir = strcat('./Equal State Prior Probability Analysis/',...
    num2str(nStates),'States/GMM/');
load(strcat(indir,'optimal_labels_training_GMM_cv',num2str(cviter),'.mat'))

% Create output directory
outdir = strcat(indir,num2str(nSelBand),' Bands/');
if ~exist(outdir,'dir')
    mkdir(outdir)
end

% Load training data.
load(strcat('./fullIndianPine_20151215_training_cv',num2str(cviter),'.mat'))

% Calculate the number of samples in each class in the training dataset.
table_correctlabels_training_cv = tabulate(correctlabels_training_cv);

% Calculate the state label matrix for each class.
correctlabels_training_cv_cum = cumsum(table_correctlabels_training_cv(:,2));
[x,y,z] = size(qp1_training_GMM);
qp1_sum_cls = zeros(x,y,nClass);
for i = 1:nClass
    temp = sum(qp1_training_GMM(:,:,correctlabels_training_cv_cum(i)-...
        table_correctlabels_training_cv(i,2)+1:correctlabels_training_cv_cum(i)),3);
    qp1_sum_cls(:,:,i) = temp/table_correctlabels_training_cv(i,2);
end

% Calculate the matrix (qp1_corr) which measures the similarity of each
% class pair in each spectral band
qp1_corr = zeros(nClass,nClass,nBand);
for i = 1:nBand
    for j = 1:nClass
        for k = 1:nClass
            temp1 = squeeze(qp1_sum_cls(:,i,j));
            temp2 = squeeze(qp1_sum_cls(:,i,k));
            qp1_corr(j,k,i) = dot(temp1,temp2)/(norm(temp1,2)*norm(temp2,2));
        end
    end
end
save(strcat(indir,'qp1_corr_cv',num2str(cviter),'_GMM.mat'),'qp1_corr')

% Use the matrix "qp1_corr" to select bands.
qp1_corr_temp = sum(squeeze(sum(qp1_corr,1)),1)/(nClass^2);
singu = find(isnan(qp1_corr_temp)==1);
for i = singu
    qp1_temp = squeeze(qp1_corr(:,:,i));
    qp1_temp_num = double(~isnan(qp1_temp));
    qp1_temp(find(qp1_temp_num==0)) = 0;
    if sum(qp1_temp_num(:)) == 0
        qp1_corr_temp(i) = 1;
    else
        qp1_corr_temp(i) = sum(qp1_temp(:))/sum(qp1_temp_num(:));
    end
end
[val,ind] = sort(qp1_corr_temp);
selBandKeep = ind(1:nSelBand);
selBandKeep_corrcoef = val(1:nSelBand);
save(strcat(outdir,'selBandKeep_cv',num2str(cviter),'.mat'),'selBandKeep',...
    'selBandKeep_corrcoef')

end