function [  ] = wvlt(cviter,wvltType)

% Input
% cviter - fold number in a 5-fold cross validation setup
% wvltType - mother wavelet type

% Create output directory
outdir1 = './Equal State Prior Probability Analysis/';
if ~exist(outdir1,'dir')
    mkdir(outdir1)
end

% Wavelet transform
disp('Wavelet transform started...')
tStart = clock;
fname_min_wvlt_train = strcat(outdir1,'min_wvlt_train_cv',num2str(cviter),...
    '.mat');
if ~exist(fname_min_wvlt_train,'file')
    fname = strcat('./fullIndianPine_20151215_training_cv',num2str(cviter),'.mat');
    load(fname)
    wf = wvltType;
    wflag = 3;
    prc_cnd = 'exact'; % 90,80,non,exact
    scalelabel = 'linear';
    nrm = 'on'; % {'on','off'}
    scales = 2:10;
    min_data_train = min_data_training_cv;
    min_data_train = min_data_train./repmat(max(min_data_train,[],1),...
        [size(min_data_train,1),1]);
    min_wvlt_train = createwvltcoeffsWHISPERSToolbox([wavelengths...
        min_data_train],scales,wf,wflag);
    if strcmp(prc_cnd,'90')
        l = prctile(abs(min_wvlt_train(:)),90);
        min_wvlt_train(min_wvlt_train>l) = l;
        min_wvlt_train(min_wvlt_train<-l) = -l;
    elseif strcmp(prc_cnd,'80')
        l = prctile(abs(min_wvlt_train(:)),80);
        min_wvlt_train(min_wvlt_train>l) = l;
        min_wvlt_train(min_wvlt_train<-l) = -l;
    elseif strcmp(prc_cnd,'exact')
        boundaryleft = ceil((scales-1)/2);
        boundaryright = floor((scales-1)/2);
        for i=1:size(scales,2)
            % Remove edge effect
            min_wvlt_train(i,1:boundaryleft(i),:) = 0;
            min_wvlt_train(i,end-boundaryright(i)+1:end,:)=0;
        end
    end
    for i = 1:size(min_wvlt_train,3)
        min_wvlt_train(:,:,i) = flipud(min_wvlt_train(:,:,i));
    end
    save(fname_min_wvlt_train,'min_wvlt_train')
else
    load(fname_min_wvlt_train)
end

end