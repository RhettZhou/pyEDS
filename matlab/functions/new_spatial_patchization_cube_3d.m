function ima_patchs = new_spatial_patchization_cube_3d(image,w,w_3d)

%% Initialization
	
[M,N,P] = size(image); % size of imput image

ima_patchs = zeros(M-w+1,N-w+1,P-w_3d+1,w*w*w_3d); % matrix that contains the patches

delta = (0:w-1)-1; 
% Vector of length w. This defines the vector that will traverse the X and
% Y dimensions of 'image'

delta_3d = (0:w_3d-1)-1;
% Vector of length w_3d. This defines the vector that will traverse the Z
% dimension of 'image'

%% Patchization

for k = 1:P-w_3d+1 % Traverses Z dimension last
    
    zrange = mod(delta_3d+k,P)+1; 
    
    for j = 1:N-w+1 % Traverses Y dimension second

        yrange = mod(delta+j,N)+1;  
        
        for i = 1:M-w+1 % Traverses X dimension first
                
            xrange = mod(delta+i,M)+1;
        
            B = image(xrange, yrange, zrange); 
            % B is a cubic patch of 'image' defined by 'xrange', 'yrange',
            % and 'zrange.'
            
            ima_patchs(i,j,k,:) = B(:);
            % B is converted to a vector; then it's stored in ima_patchs.
            % The first three dimensions - (M-w+1), (N-w+1), and (P-w_3d+1)
            % represent the spatial location of the patch. The last
            % dimension (w*w*w_3d) represents the number of elements in the
            % cubic patch.
            
        end
         
    end
    
end
    

    
