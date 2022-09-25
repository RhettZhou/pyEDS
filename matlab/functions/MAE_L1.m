function y = MAE_L1(truth, est)

y = sum(abs(truth(:)-est(:)))/sum(abs(truth(:)));
% y = mean(abs(truth(:)-est(:))./abs(truth(:)+1));