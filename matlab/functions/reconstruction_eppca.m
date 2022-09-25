function patchs = reconstruction_eppca(patchs_pca)

coefs = patchs_pca.coefs; % output from func_factorization
[nb_axis,P]=size(patchs_pca.dicos{1}.axis); % output from func_factorization

    if size(coefs, 3) > 1 % if the z dimension is greater than 1
        isimage = 1; 
        [M,N,R] = size(coefs); 
        MN = M*N;
        coefs=reshape(coefs,[M*N, R]); % reshapes the 3d matrix to a 2d matrix
       % iddico = reshape(patchs_pca.iddico, MN, 1);
    else
        if nb_axis==1 % if the number of axes is 1
        
            [M,N]=size(coefs);   
            MN=M*N;
            coefs=reshape(coefs,[M*N, 1]); % reshape 2d matrix into 1d matrix
        else
            isimage = 0;
            [MN,R] = size(coefs); % MN are the number of patches at cluster k. R is the number of axes        
     %       iddico = patchs_pca.iddico;
    
        end
    end
    %P = size(patchs_pca.dicos{1}.axis, 1);
    L = length(patchs_pca.dicos);
    patchs = zeros(MN, P); %  MN are the number of patches at cluster k. P is w^2*w_3d

    
    for i = 1:L
        

         patchs =exp(coefs*patchs_pca.dicos{i}.axis);
         %patchs =patchs_pca.dicos{i}.varaxis;
         
    end
   
    clear coefs;
    %clear iddico;

    if nb_axis==1
        
        patchs = reshape(patchs,M,N,P);
     %   size(patchs)
    else
        if isimage
            patchs = reshape(patchs, M, N, P);
     %               size(patchs)

        end
    end % does nothing (in this case)?