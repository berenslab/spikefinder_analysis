function err = predval(x)

global X Y

ncell = size(X,2);

pspk = pred(x);     % predict spikes for all cells

cc = zeros(1,ncell);

for i = 1:ncell   
    nsamp = find(isnan(Y(:,i)) | isnan(X(:,i)) | isnan(pspk(:,i)),1)-1;
    if(isempty(nsamp))
        nsamp = size(Y,1);
    end
    
    xx = resample(Y(1:nsamp,i),1,4);    % binning as done by Spikefinder - it may be slightly different
    yy = resample(pspk(1:nsamp,i),1,4);    
   
    cc(i) = corr(xx,yy,'type','pearson');

end

err = -mean(cc);    % mean correlation coefficient across all cells in the dataset


