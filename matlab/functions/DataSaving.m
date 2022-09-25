temp = ima_fil(:,:,5) + ima_fil(:,:,1);
temp = temp ./ max(temp(:));
temp = im2uint16(temp);
imwrite(temp, 'FCGT_AA_Ge_combined_NLPCAdenoised.tiff');

temp = ima_fil(:,:,1);
temp = temp ./ max(temp(:));
temp = im2uint16(temp);
imwrite(temp, 'FCGT_AA_Ge_La_NLPCAdenoised.tiff');

temp = ima_fil(:,:,2);
temp = temp ./ max(temp(:));
temp = im2uint16(temp);
imwrite(temp, 'FCGT_AA_Te_La_NLPCAdenoised.tiff');

temp = ima_fil(:,:,3);
temp = temp ./ max(temp(:));
temp = im2uint16(temp);
imwrite(temp, 'FCGT_AA_Fe_Ka_NLPCAdenoised.tiff');

temp = ima_fil(:,:,4);
temp = temp ./ max(temp(:));
temp = im2uint16(temp);
imwrite(temp, 'FCGT_AA_Co_Ka_NLPCAdenoised.tiff');

temp = ima_fil(:,:,5);
temp = temp ./ max(temp(:));
temp = im2uint16(temp);
imwrite(temp, 'FCGT_AA_Ge_Ka_NLPCAdenoised.tiff');