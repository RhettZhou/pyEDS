
function ima_fil=NLPCA(patch_width,fileName)

if ~exist('patch_width','var')
    patch_width = 24;
end

if ~exist('fileName','var')
    [fileName, pathName] = uigetfile('*.*','Select a file');
    fileName = [pathName fileName];
end

addpath('functions')


%% Image generation
% TODO: replace the variable in readNPY with the path to the data saved by
% the python notebook.
%ima_pn = readNPY('../FCGT_AAp_5bands.npy'); Rhett 20210725
ima_pn = readNPY(fileName);
ima_nse_poiss = double(ima_pn);
ima_nse_poiss = permute(ima_nse_poiss, [2,3,1]);

[dim1,dim2,dim3]=size(ima_nse_poiss);
 
%% Parameters:

% TODO: replace the first parameter, param.Patch_width with the size of
% repeating unit in px. For example, if the data comes from a single
% crystal, this would be the px size of a single unit cell. If you are not
% sure what number to use, then keep it at 24.

param.Patch_width = patch_width;  % Default: 24
param.Patch_width_3d = dim3;
param.nb_axis = 15; 
param.nb_clusters = 8;
param.eps_stop=1e-1; %loop stoping criterion
param.epsilon_cond=1e-3; %condition number for Hessian inversion
param.double_iteration=0;%1 or 2 pass of the whole algorithm
param.nb_iterations=4;
param.bandwith_smooth=2;
param.sub_factor=2;
param.big_cluster1=1;% special case for the biggest cluster 1st pass
param.big_cluster2=1;% special case for the biggest cluster 2nd pass
param.cste=70;
param.func_tau=@(X) lasso_tau(X{1},X{2},param.cste);
param.parallel = 1; % 0/1 determines if parallelization is used
param.SPIRALTAP = 0; % 0/1 determines if Newton's method is used (0 is recommended)

%% computation
tic
ima_fil=denoise_poisson_kmeans_poisson_PCA_l1_4d_cube_3d(ima_nse_poiss,param);
toc

% TODO: replace the filename in save with the name you want.
if param.Patch_width == 24
    par = '';
else
    par = ['_' int2str(param.Patch_width)];
end

saveFileName = [fileName(1:end-4) par '.mat'];
save(saveFileName, 'ima_fil');

%% Save as 16-bit tif file after normalization
% TODO: replace the filename in imwrite with the name you want.
% temp = ima_fil(:,:,3) + ima_fil(:,:,2) + ima_fil(:,:,1);
% temp = temp ./ max(temp(:));
% temp = im2uint16(temp);
% imwrite(temp, 'FCGT_AAp_combined_NLPCAdenoised.tiff');
% 
% temp = ima_fil(:,:,1);
% temp = temp ./ max(temp(:));
% temp = im2uint16(temp);
% imwrite(temp, 'FCGT_AAp_Pt_Ka_NLPCAdenoised.tiff');
% % 
% temp = ima_fil(:,:,2);
% temp = temp ./ max(temp(:));
% temp = im2uint16(temp);
% imwrite(temp, 'FCGT_AAp_Pt_La_NLPCAdenoised.tiff');
% % 
% temp = ima_fil(:,:,3);
% temp = temp ./ max(temp(:));
% temp = im2uint16(temp);
% imwrite(temp, 'FCGT_AAp_Au_La_NLPCAdenoised.tiff');

% temp = ima_fil(:,:,4);
% temp = temp ./ max(temp(:));
% temp = im2uint16(temp);
% imwrite(temp, 'FCGT_AAp_Cu_Ka_NLPCAdenoised.tiff');
% 
% temp = ima_fil(:,:,5);
% temp = temp ./ max(temp(:));
% temp = im2uint16(temp);
% imwrite(temp, 'FCGT_AAp_Ge_Ka_NLPCAdenoised.tiff');
   
end

