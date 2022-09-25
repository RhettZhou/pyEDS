function [U,S,V] = svd_new(X,axes)

% This SVD implementation came from a user reply on:
% http://stackoverflow.com/questions/12698433/matlab-how-to-compute-pca-on-a-huge-data-set
% The solution is the reply "Solution based on Eigen Decomposition" by user
% "petrichor", answered on Oct 3, 2012 at 10:36. 

if size(X,1) == 0
    warning('off')
end

[V,D] = eig(X'*X);
V = V(:,(end-axes+1):end);
S = sqrt(D((end-axes+1):end,(end-axes+1):end));
U = X*V*S^(-1); % Note that S is a diagonal matrix

if size(X,1) == 0
    warning('on')
end