function [ max_accuracy ] = svm_test_whole_GMM(nStates,nSelBand,cviter)

% Input
% nStates - number of Gaussian states
% nSelBand - number of bands to keep
% cviter - fold number in a 5-fold cross validation setup

t1 = clock;

% Parameter setup for RBF kernel based SVM
cost = 2.^(1:3);
gamma = 2.^(1:3);

accuracy = zeros(length(cost),length(gamma));

for i = 1:length(cost)
    for j = 1:length(gamma)
        success_rate = svm_test_GMM(nStates,nSelBand,cost(i),gamma(j),cviter);
        accuracy(i,j) = success_rate;
    end
end

indir = strcat('./Equal State Prior Probability Analysis/',...
    num2str(nStates),'States/GMM/');
outdir = strcat(indir,num2str(nSelBand),' Bands/');
if ~exist(outdir,'dir')
    mkdir(outdir)
end
save(strcat(outdir,'accuracy_cv',num2str(cviter),'.mat'),'accuracy')
max_accuracy = max(accuracy(:))
save(strcat(outdir,'max_accuracy_cv',num2str(cviter),'.mat'),'max_accuracy')

x = log2(gamma);
y = log2(cost);
set(0,'DefaultFigureVisible','off')
figure,
imagesc(x,y,accuracy)
xlabel('\gamma')
ylabel('C')
imname = strcat('accuracy_cv',num2str(cviter),'.png');
save([outdir imname])

disp(['The process takes ',num2str(etime(clock,t1)/3600),' hours.'])

end