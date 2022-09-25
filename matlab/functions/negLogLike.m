function L = negLogLike(Y,Lambda)

Lambda = max(Lambda,eps);

L = sum(Lambda(:) - Y(:).*log(Lambda(:)));
