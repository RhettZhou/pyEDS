function ima_patchs = spatial_patchization(image,w)

%% Initialization    
    
[M,N] = size(image); % size of imput image

ima_patchs = zeros(M-w+1,N-w+1,w*w); % matrix to be filled with the patches of 'image'
delta = (0:(w-1))-1; % vector of length w

%% Patchization

for j = 1:N-w+1 % Vector that traverses 'image' in the Y direction
	
	yrange = mod(delta+j,N)+1;
	
	for i = 1:M-w+1 % Vector that traverses 'image' in the X direction
		
		xrange = mod(delta+i,M)+1;
		
		B = image(xrange, yrange);
		% B is a square patch of 'image' defined by 'xrange' and 'yrange'
		
		ima_patchs(i,j,:) = B(:);
		% B is converted to a vector; then it's stored in ima_patchs. The
		% first two dimensions (M-w+1) and (N-w+1) represent the spatial
		% location of the patch. The last dimension (w*w) represents the
		% number of elements in each patch.
		
	end
	
end
