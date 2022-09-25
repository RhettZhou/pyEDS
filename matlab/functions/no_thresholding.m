function ima_ppca = no_thresholding(ima_ppca)

     coefs = ima_ppca.coefs;
%     if size(coefs, 3) > 1
%         isimage = 1;
%         [M,N,P] = size(coefs);
%         MN = M*N;
%         coefs = reshape(coefs, MN, P);
%     else
%         isimage = 0;
%         [MN,P] = size(coefs);
%     end

    coefs = coefs;
%     
%     if isimage
%         ima_ppca.coefs = reshape(coefs, M, N, P);
%     else
%         ima_ppca.coefs = reshape(coefs, MN, P);
%     end
