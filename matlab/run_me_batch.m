crop = NaN;
% crop = [2,62,4,124];                                                     % change here
eds_bin = 9;                                                               % change here
energy_low = 0.4;                                                          % change here
energy_high = 9.4;                                                         % change here
analysis_path = '//mufs4/x.zhou/TEM/nrr/STO';                              % change here
data_path = '//mufs4/x.zhou/TEM/nrr/STO/EDS/20220208_1847.emd';            % change here
signal = 'HAADF';
group = 'lambda';                                                          % change here
para = {'lambda_20','lambda_100','lambda_400','lambda_600'};               % change here
name = 'Denoise.npy';
[~,folder_name,~] = fileparts(data_path);
corrected_spectrum_name = ['EDS-',num2str(energy_low),'-',num2str(energy_high),'-bin',num2str(eds_bin)];
if ~isnan(crop)
    crop_prefix = ['L',num2str(crop(1)),'-R',num2str(crop(2)),'-T',num2str(crop(3)),'-B',num2str(crop(4))];
end
for i = 1:length(para)
    if isnan(crop)
        f = fullfile(analysis_path,folder_name,signal,group,para{i},corrected_spectrum_name,name);
    else
        f = fullfile(analysis_path,folder_name,signal,group,para{i},corrected_spectrum_name,crop_prefix,name);
    end
    NLPCA(24,f);
end
