function ima = new_reprojection_UWA_cube_3d(ima_patchs,w,w_3d,M,N,P)

%% Initialization

ima = zeros(M,N,P); % ima is the reconstructed image

delta=(-1)+(1:w)-1;
% vector of length w. This will be the vector that traverses the X and Y
% dimensions when reconstructing the cube.

delta_3d = (-1)+(1:w_3d)-1;
% vector of length w_3d. This will be the vector that traverses the Z
% dimension when reconstructing the cube.

patch = 1;

%% Reconstruction

for k = 1:P-w_3d+1 % traverses Z dimension last

	zrange = mod(delta_3d+k,P)+1;

	for j = 1:N-w+1 % traverses Y dimension second

		yrange = mod(delta+j,N)+1;

		for i = 1:M-w+1 % traverses X dimension first

			xrange = mod(delta+i,M)+1;

			ima(xrange, yrange, zrange) = ima(xrange, yrange, zrange) + reshape(ima_patchs(patch,:),[w,w,w_3d]);
			% This first takes the patch from ima_patchs, which is a
			% horizontal vector of length w*w*w_3d. Then it reshapes it
			% into a cube of size w by w by w_3d. Then it places that cube
			% in 'ima' at a specified range.

			patch = patch + 1;

		end

	end

end