function wc = createwvltcoeffsWHISPERSToolbox(x,scales,wf,mode)

dwtmode('per','nodisp')
%dwtmode('asym','nodisp')

% get rid of nans, track them
nanmap = double(isnan(x));
x(isnan(x)) = 0;

% l = max(max(scales),size(x,1));
% wc = zeros(length(scales),size(x,1),size(x,2)-1);
% wcnan = wc;
% shift1 = round((l-size(x,1))/2);
% shift2 = l-shift1-size(x,1);
%
% for i=2:size(x,2),
%     tmp = cwt([ones(shift1,1)*x(1,i); x(:,i);ones(shift2,1)*x(end,i)],scales,wf);
%     wc(:,:,i-1) = tmp(:,shift1+(1:size(x,1)));
%     tmp = cwt([ones(shift1,1)*nanmap(1,i); nanmap(:,i);ones(shift2,1)*nanmap(end,i)],scales,wf);
%     wcnan(:,:,i-1) = tmp(:,shift1+(1:size(x,1)));
%     % wc(:,:,i-1) = tmp;
% end


switch mode
    case 1,
        wc = zeros(length(scales),size(x,1),size(x,2)-1);
    case 2,
        tmp = cwtft(x(:,1),'wavelet',wf,'scales',scales);
        wc = zeros(length(tmp.scales),size(x,1),size(x,2)-1);
    case 3,
        wc = zeros(length(scales),size(x,1),size(x,2)-1);
end
wcnan = wc;

L = ceil(log2(size(x,1)));
shift1 = round((2^L-size(x,1))/2);
shift2 = 2^L-shift1-size(x,1);

for i=2:size(x,2),
    switch mode
        case 1,
            tmp = cwt([ones(shift1,1)*x(1,i); x(:,i);ones(shift2,1)*x(end,i)],scales,wf);
            wc(:,:,i-1) = tmp(:,shift1+(1:size(x,1)));
            tmp = cwt([ones(shift1,1)*nanmap(1,i); nanmap(:,i);ones(shift2,1)*nanmap(end,i)],scales,wf);
            wcnan(:,:,i-1) = tmp(:,shift1+(1:size(x,1)));
            %             wc(:,:,i-1) = cwt(x(:,i),scales,wf);
            %             wcnan(:,:,i-1) = cwt(nanmap(:,i),scales,wf);
        case 2,
            tmp = cwtft(x(:,i),'wavelet',wf,'scales',scales);
            wc(:,:,i-1) = tmp.cfs;
            tmp = cwtft(nanmap(:,i),'wavelet',wf,'scales',scales);
            wcnan(:,:,i-1) = tmp.cfs;
        case 3,
            wc(:,:,i-1) = cwt(x(:,i),scales,wf);
            wcnan(:,:,i-1) = cwt(nanmap(:,i),scales,wf);
    end
end

% patch nans back in
wcnanmap = (wcnan ~= 0);
wc(wcnanmap) = nan;
