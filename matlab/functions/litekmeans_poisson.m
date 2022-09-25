function label = litekmeans_poisson(X, k,label) 
n = size(X,2); 
last = 0;

if (nargin < 3)
    label = ceil(k*rand(1,n));  % random initialization    
end

while any(label ~= last)
    %[~,~,label] = unique(label);   % remove empty clusters
    E = sparse(1:n,label,1,n,k,n);  % transform label into indicator matrix.
    center = X*(E*spdiags(1./sum(E,1)',0,k,k));    % compute center of each cluster
    last = label;
%    [~,label] = max(bsxfun(@minus,center'*X,0.5*sum(center.^2,1)')); % assign samples to the nearest centers
    [~,label] = max(bsxfun(@minus,log(center)'*X,sum(center,1)')); % assign samples to the nearest centers
end
