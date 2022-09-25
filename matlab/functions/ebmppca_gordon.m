function patchs_ppca = ebmppca_gordon(patchs,startu,startv,func_factorization) % patchs = ima_patchs_vect(indexes{k},:)

    if size(patchs, 3) > 1 % true if there is a z dimension
        isimage = 1; 
        [M,N,P] = size(patchs); % size of the patchs
        MN = M*N;
        patchs = reshape(patchs, MN, P); % reshapes into a 2d matrix
    else
        isimage = 0;
        [MN,P] = size(patchs); % size of the 2d patch. MN is the number of patches at the kth cluster. P is patch_width*patch_width*patch_3d
    end
    
[u v ] =func_factorization({patchs,startu,startv}); % size of u: number of clusters at "k" by number of axes. size of v: number of axes by number of clusters at "k"        
patchs_ppca.coefs = u;
patchs_ppca.dicos{1}.axis = v;

