function final_estimate=denoise_clusters_black_cube(ima_patchs_vect,func_denoising_patches,func_thresholding,func_recontruction,IDX,w,nb_axis,nb_clusters,M,N,num_patch_ele,SPIRALTAP,parallel)

% Initialization

final_estimate = zeros(size(ima_patchs_vect)); % size: (M-w+1)*(N-w+1)*(P-w_3d+1) by (w*w*w_3d)
indexes = cell(nb_clusters,1); %  holds the locations of patches belonging to a specific cluster
size_cluster = zeros(nb_clusters,1);





% Filling "indexes"

switch parallel
    case 1 % Use parallelization
        parfor i = 1:nb_clusters
            indexes{i} = find(IDX == i); % finds the location in IDX for a specific cluster number
            size_cluster(i) = size((indexes{i}),2); % number of patches in each cluster
        end
        
    case 0 % No parallelization
        for i = 1:nb_clusters
            indexes{i} = find(IDX == i); % finds the location in IDX for a specific cluster number
            size_cluster(i) = size((indexes{i}),2); % number of patches in each cluster
        end
end





% Finding Cluster Size

best = 1;
max_number = 0;

for j = 1:nb_clusters
    if size_cluster(j) > max_number % finds which cluster has the most number of patches
        best = j;
        max_number = size_cluster(j);
    end
end





% Denoising Clusters

switch parallel
    
    case 1 % -------------------- USE PARALLELIZATION --------------------
        
        temp_clusters = cell(1,nb_clusters);
        
        for k = 1:(nb_clusters)
            % this finds all the patches in a specific cluster and inserts
            % it into 'temp_clusters'
            temp_clusters(1,k) = {ima_patchs_vect(indexes{k},:)};
        end
        
        temp_final_est = cell(1,nb_clusters);
        ima_ppca_cluster = cell(1,nb_clusters);
        ima_patchs_fil_cluster = cell(1,nb_clusters);
        
        parfor p = 1:(nb_clusters)
            
            [utmp1,stmp1,vtmp1] = svd_new((temp_clusters{1,p}),nb_axis); % est 1
            est1 = utmp1*stmp1*vtmp1';
            est1 = est1.*(est1>0); % sets negative elements of est1 to 0
            
            [utmp2,stmp2,vtmp2] = svd_new(log((temp_clusters{1,p})+1),nb_axis); % est 2
            est2 = exp(utmp2*stmp2*vtmp2');
            
            [utmp3,stmp3,vtmp3] = svd_new(log(est1+1),nb_axis); % est 3
            est3 = exp(utmp3*stmp3*vtmp3');
            
            negLogLike1 = negLogLike((temp_clusters{1,p}),est1);
            negLogLike2 = negLogLike((temp_clusters{1,p}),est2);
            negLogLike3 = negLogLike((temp_clusters{1,p}),est3);
            [~,bestEst] = min([negLogLike1,negLogLike2,negLogLike3]); % compares est1, est2, est3
            
            switch bestEst % fills final_estimate with the best estimate for those values
                case 1
                    temp_final_est(1,p) = {est1};
                case 2
                    temp_final_est(1,p) = {est2};
                case 3
                    temp_final_est(1,p) = {est3};
            end
            
            if SPIRALTAP % Determines if SPIRALTAP is used
                
                if negLogLike2 < negLogLike3
                    utmp = utmp2;
                    stmp = stmp2;
                    vtmp = vtmp2;
                else
                    utmp = utmp3;
                    stmp = stmp3;
                    vtmp = vtmp3;
                end
                
                startv = vtmp'; % finds starting matrices
                startu = utmp*stmp;
                
                ima_ppca_cluster{1,p} = func_denoising_patches({temp_clusters{1,p};startu;startv}); % est 4
                ima_patchs_fil_cluster{1,p} = func_recontruction(ima_ppca_cluster{1,p});
                
                if negLogLike(temp_clusters{1,p},ima_patchs_fil_cluster{1,p}) < negLogLike(temp_clusters{1,p},temp_final_est{1,p})
                    % compares the best of est(1,2,3) with est(4)
                    temp_final_est(1,p) = {ima_patchs_fil_cluster{1,p}};
                    disp(strcat('Cluster  ',num2str(p),': Newton iteration did improve subspace estimate'))
                else
                    disp(strcat('Cluster  ',num2str(p),': Newton iteration did not improve subspace estimate'))
                end
                
                ima_ppca_cluster{1,p} = []; % clears the cell at cluster k
                ima_patchs_fil_cluster{1,p} = []; % clears the cell at cluster k
            end
        end
        
        for q = 1:nb_clusters % fills 'final_estimate' with the denoised clusters
            final_estimate(indexes{q},:) = temp_final_est{1,q};
        end
        
        
        
        
        
    case 0 % -------------------- NO PARALLELIZATION --------------------
        
        ima_ppca_cluster = cell(1,nb_clusters);
        ima_patchs_fil_cluster = cell(1,nb_clusters);
        
        for k = 1:(nb_clusters)
            
            temp_clusters = ima_patchs_vect(indexes{k},:); % takes the patches of the kth cluster of ima_patchs_vect
            
            [utmp1,stmp1,vtmp1] = svd_new(temp_clusters,nb_axis); % est 1
            est1 = utmp1*stmp1*vtmp1';
            est1 = est1.*(est1>0); % sets negative elements of est1 to 0
            
            [utmp2,stmp2,vtmp2] = svd_new(log(temp_clusters+1),nb_axis); % est 2
            est2 = exp(utmp2*stmp2*vtmp2');
            
            [utmp3,stmp3,vtmp3] = svd_new(log(est1+1),nb_axis); % est 3
            est3 = exp(utmp3*stmp3*vtmp3');
            
            negLogLike1 = negLogLike(temp_clusters,est1);
            negLogLike2 = negLogLike(temp_clusters,est2);
            negLogLike3 = negLogLike(temp_clusters,est3);
            [~,bestEst] = min([negLogLike1,negLogLike2,negLogLike3]); % compares est1, est2, est3
            
            switch bestEst % fills final_estimate with the best estimate for those values
                case 1
                    final_estimate(indexes{k},:) = est1;
                case 2
                    final_estimate(indexes{k},:) = est2;
                case 3
                    final_estimate(indexes{k},:) = est3;
            end
            
            if SPIRALTAP % Determines if SPIRALTAP is used
                
                if negLogLike2 < negLogLike3
                    utmp = utmp2;
                    stmp = stmp2;
                    vtmp = vtmp2;
                else
                    utmp = utmp3;
                    stmp = stmp3;
                    vtmp = vtmp3;
                end
                
                startv = vtmp'; % finds starting matrices
                startu = utmp*stmp;
                
                ima_ppca_cluster{k} = func_denoising_patches({ima_patchs_vect(indexes{k},:);startu;startv}); % est 4
                ima_patchs_fil_cluster{k} = func_recontruction(ima_ppca_cluster{k});
                
                if negLogLike(temp_clusters,ima_patchs_fil_cluster{k}) < negLogLike(temp_clusters,final_estimate(indexes{k},:))
                    % compares the best of est(1,2,3) with est(4)
                    final_estimate(indexes{k},:) = ima_patchs_fil_cluster{k};
                    disp(strcat('Cluster ',num2str(k),': Newton iteration did improve subspace estimate'))
                else
                    disp(strcat('Cluster ',num2str(k),': Newton iteration did not improve subspace estimate'))
                end
                
                ima_ppca_cluster{k} = []; % clears the cell at cluster k
                ima_patchs_fil_cluster{k} = []; % clears the cell at cluster k
            end
        end
end
