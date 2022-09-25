function [ima_fil,IDX_fil] = denoise_poisson_kmeans_poisson_PCA_l1_4d_cube_3d(...
    ima_nse_poiss,param) % inputs noisy image and parameters

% Set up pparallel pool if option on
if param.parallel ~= 0
    poolobj = parpool;
end



func_thresholding = @(ima_ppca) no_thresholding(ima_ppca);

func_clustering = @(X) clustering_litekmeans_smaller_poisson_4d(X{1},param.nb_clusters,X{2});

func_clustering_old = @(X) old_clustering_litekmeans_smaller_poisson_4d(X{1},param.nb_clusters,X{2});

func_recontruction = @(X)reconstruction_eppca(X);

func_factorization = @(X)newp1svd_joe_poisson_SPIRAL(X{1},param.nb_axis,param.nb_iterations,X{2},X{3},param.eps_stop,param.epsilon_cond,param.func_tau);

func_denoising_patches = @(X)...
    ebmppca_gordon(X{1},X{2},X{3},func_factorization);

ima_fil = NL_PCA_cube_3d(ima_nse_poiss,ima_nse_poiss,param.Patch_width,param.Patch_width_3d,param.nb_axis,...
    param.nb_clusters,param.SPIRALTAP,param.parallel,func_thresholding,func_recontruction,...
    func_denoising_patches,func_clustering,func_clustering_old);



% Turn off parallel pool 
if param.parallel ~= 0
    delete(poolobj);
end
