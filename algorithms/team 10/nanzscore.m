function y = nanzscore(x)

y = (x-ones(size(x,1),1)*nanmean(x))./(ones(size(x,1),1)*nanstd(x));
