% Do a Poisson singular value decomposition with a constant column.
%   x: matrix to be analyzed (2d patches)
%   l: number of factors to find (incl the constant column) (param.nb_axes)
%   iters: computation limit (param.nb_iterations)
%   startu, startv: starting point for optimization (determined from
%   denoise_clusters_black_cube)
% Returns u (with l columns) and v (with l rows) such that (x - exp(u*v))
% is small.  First column of u is constrained to be constant (which
% implies that the first element of each column of v is a bias term which
% will be used to make sure that exp(u*v) has the same column sums that
% x does).

function [u, v, lastexpy] = newp1svd_joe_poisson_SPIRAL(x, l, iters, startu, startv,epsilon_stop,epsilon_cond,func_tau) % x is ima_patchs_vect(indexes{k},:). l is num of axes. 

[n, d] = size(x); % n is number of patches assigned to cluster k. d is w^2*w_3d

tau=func_tau({d,n});

u = startu;
v = startv;

if 0
   norma=sqrt(sum(v(1,:).^2,2));
   v(1,:)=norma*ones(1,d)/sqrt(d);
end


cond_mat=epsilon_cond*eye(l);
product_uv=u*v;
lastexpy=exp(product_uv/2);
curnorm = sum(sum(lastexpy.^2)) ;

L_tot=curnorm-sum(sum(x.*product_uv));


xinit = zeros(l,1);
miniter = 2;
maxiter = 20;


for iter = 1:iters
   
   vprime=v';
   for i = 1:n
      
      % tau=func_tau({10^(-15)*log(+sum(x(i,:)))});
      %u(i,:) = SPIRALTAPBregman(x(i,:)',vprime,v,xinit,tau,miniter,maxiter);
      u(i,:) = SPIRALTAPBregman(x(i,:)',vprime,v,u(i,:)',tau,miniter,maxiter);
   end
   
   lastexpy2 = exp(u*v/2);
   lasterru= u'*(x-lastexpy2.^2);
   
   for j = 1:d
      
      int_v=bsxfun(@times,u,lastexpy2(:,j));
      covar = (int_v'*int_v);
      covar=covar+cond_mat;
      v(:,j) =v(:,j)+ covar \ (lasterru(:,j));
      %      v(:,j) =v(:,j)+ epsilon_cond*lasterru(:,j);
      
   end
   
   
   product_uv=u*v;
   lastexpy2=exp(product_uv/2);
   
   
   lastexp_square=lastexpy2.^2;
   curnorm = sum(sum(lastexp_square)) ;
   
   
   L_tot2=curnorm-sum(sum(x.*product_uv));
   
   ratio=abs((L_tot-L_tot2)/L_tot);
   %verbose or not
        %fprintf('%3d:  %g \n', iter,ratio);
   
   if ( (ratio < epsilon_stop))
      break;
   end
   
   L_tot=(L_tot2);
   lastexpy=lastexpy2;
   
   
end
