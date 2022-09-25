function ima_fil=NL_PCA_cube_3d(ima_nse_poiss,~,w,w_3d,nb_axis,...
                       nb_clusters,SPIRALTAP,parallel,func_thresholding,func_recontruction,...
                       func_denoising_patches,func_clustering,func_clustering_old)

%% Initilization 

[M,N,P] = size(ima_nse_poiss); 
% note: M, N, P are the X, Y, Z dimensions of original image. w is the
% patch width and w_3d is the patch depth.

ima_fil = zeros(M,N,P); % ima_fil will be the reconstructed image

%% Clustering

sum_z = sum(ima_nse_poiss,3); % sum poisson image along z dimension

ima_patchs = spatial_patchization(sum_z,w); 
% Stores the patches of 'sum_z' in the form of a 3D matrix of size (M-w+1)
% by (N-w+1) by w*w. The first 2 dimensions indicate the location of the
% patch. The 3rd dimension indicates the size of the patch (i.e. w*w). (The
% vectorized patches extend along the Z dimension of ima_patchs.)

if nb_clusters > 1 
    
    IDX = ceil(nb_clusters*rand(1,(M-w+1)*(N-w+1))); 
    % IDX is a horizontal vector of length (M-w+1)*(N-w+1). This is the
    % vector that randomly classifies each patch in 'sum_z' to a cluster.

    [~,IDX_int] = func_clustering_old({ima_patchs,IDX});
    % IDX_int is a horizontal vector of length (M-w+1)*(N-w+1). This is the
    % vector that classifies each patch in 'sum_z' to a cluster.
    
else 
    
    IDX_int = ones(size((M-w+1)*(N-w+1))); 
    % If there is only one cluster, each patch in 'sum_z' is assigned a
    % value of 1.
    
end

IDX_int = repmat(IDX_int,1,(P-w_3d+1));
% IDX_int is repeated (P-w_3d+1) times to create a horizontal vector of
% size (M-w+1)*(N-w+1)*(P-w_3d+1). This is done because it creates an
% initial index for ALL of the patches in the cube. Otherwise, IDX_int only
% selects an index for the first spectral band. This line ensures that an
% index is selected for all the spectral bands - i.e. the entire cube.

%% Patching the cube
    
ima_patchs = new_spatial_patchization_cube_3d(ima_nse_poiss,w,w_3d); 
% This creates a 4D matrix of size (M-w+1) by (N-w+1) by (P-w_3d+1) by
% (w*w*w_3d). The first 3 coordinates represent the location of the
% patches. The 4th coordinate indicates how many elements are in the patch.

ima_patchs_vect = reshape(ima_patchs,[(M-w+1)*(N-w+1)*(P-w_3d+1),w*w*w_3d]); 
% Generates a 2D matrix of patches by reshaping ima_patchs. The size is
% (M-w+1)*(N-w+1)*(P-w_3d+1) by (w*w*w_3d). Each horizontal vector is the
% vectorized patch of 'ima_nse_poiss.' Thus, the first dimension
% represents how many patches there are [(M-w+1)*(N-w+1)*(P-w_3d+1)]. The
% second dimension represents how many elements there are in each patch
% (w*w*w_3d).

clear ima_patchs

%% Denoising the patches

ima_fil_int = denoise_clusters_black_cube(ima_patchs_vect,func_denoising_patches,func_thresholding,func_recontruction,IDX_int,w,nb_axis,nb_clusters,M,N,w*w*w_3d,SPIRALTAP,parallel);   
% Outputs 2D matrix of size (M-w+1)*(N-w+1)*(P-w_3d+1) by (w*w*w_3d). The
% patches are represented by horizontal vectors. Thus, the first dimension
% represents how many patches there are. The second dimension is how many
% elements are in each cubic patch (i.e. w*w*w_3d).

clear ima_patchs_vect

%% Reprojection 

ima_fil(:,:,:) = new_reprojection_UWA_cube_3d(ima_fil_int,w,w_3d,M,N,P);
% takes the 2D matrix ima_fil_int and reconstructs a 3D matrix of size M by
% N by P.

%% Normalization

ones_matrix = ones((M-w+1)*(N-w+1)*(P-w_3d+1),w*w*(w_3d)); 
% creates a matrix of ones. (M-w+1)*(N-w+1)*(P-w_3d+1) is the number of
% patches in the cube. w*w*(w_3d) are the number of elements in each
% patch.

normalization_cube = new_reprojection_UWA_cube_3d(ones_matrix,w,w_3d,M,N,P); 
% Size: M by N by P. This is a matrix that will "normalize" ima_fil. Note:
% it is not "reconstructing" the original image; rather, it's
% "reconstructing" a matrix of ones.

clear ones_matrix

ima_fil = ima_fil./normalization_cube; 
% Normalizes ima_fil. Otherwise, ima_fil would have edge artifacts.