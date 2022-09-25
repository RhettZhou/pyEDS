function [ima_patchs_vect,IDX] = old_clustering_litekmeans_smaller_poisson_4d(ima_patchs,nb_clusters,label) 

[m2,n2,w2] = size(ima_patchs);
ima_patchs_vect = reshape(ima_patchs,[(m2)*(n2),w2]);
IDX = litekmeans_poisson(ima_patchs_vect',nb_clusters); 

