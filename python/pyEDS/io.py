import hyperspy.api as hs
import os
import numpy as np
import dask.array as da
import cv2
import matplotlib
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import scipy.io as sio
from tkinter import filedialog
import bz2
import struct
import shutil
import matplotlib as mpl
from shutil import copyfile

import scipy.misc as smisc
import re as regexp
import math
import scipy
import time
from scipy import optimize
from scipy import ndimage as ndi
from skimage.feature import peak_local_max

from pylab import *
import copy

from pyEDS.matchseries import MatchSeries


# 1.0 Before NRR calculation
#################################################################################################
def load_emd_lazy(data_path=None):   # Load the original emd data, lazy mode
    if data_path == None:
        data_path = filedialog.askopenfilename(filetypes=[("Velox file", "*.emd")])
    data_raw  = hs.load(data_path,lazy=True, sum_frames=False, load_SI_image_stack=True)
    numbDim = len(data_raw)  # How many signals in the data cube
    for i in range(1,numbDim):                         
        if len(data_raw[i].data.shape) >= 2:            ## Rhett changed here 20220401
            data_raw[i].axes_manager[0].offset = 0
            data_raw[i].axes_manager[1].offset = 0
    file_name = os.path.split(data_path)[1]  # Get the name of the analysed file 
    analysis_path = os.path.split(os.path.split(data_path)[0])[0]
    return data_raw, file_name[:-4], data_path, analysis_path

def load_emd(data_path):       # Load the original emd data as normal
    data_raw  = hs.load(data_path)
    numbDim = len(data_raw)  # How many signals in the data cube
    for i in range(1,numbDim):
        if len(data_raw[i].data.shape) >= 2:            ## Rhett changed here 20220401
            data_raw[i].axes_manager[0].offset = 0
            data_raw[i].axes_manager[1].offset = 0
    return data_raw[-1]

def plot_signal(signal,data):
    plt.ion()
    for i in data:
        if signal == i.metadata.General.title:
            if i.axes_manager.navigation_dimension == 0:
                i.plot()
            elif i.axes_manager.navigation_dimension == 1:
                i.integrate1D(0).plot()
            elif i.axes_manager.navigation_dimension > 1:
                pass
            return i

def sort_band(band_list):
    bandlist = sorted(band_list, key = lambda x:x[1])
    return band_list

def resize_data(data,rSize=256):
    xDim = data[-1].axes_manager.navigation_shape[0]
    yDim = data[-1].axes_manager.navigation_shape[1]
    print('Original x dimension: ' + str(xDim))
    print('Original y dimension: ' + str(yDim))
    numbFrames = data[-1].axes_manager.navigation_shape[2]
    numbDim = len(data)  # How many signals in the data cube
    numbChannel = data[0].axes_manager.shape[0]  # Length of the energy spectrum
    for i in range(1,numbDim):
        if data[i].axes_manager.navigation_dimension == 0:
            if len(data[i].data.shape) >= 2:        ## Rhett changed here 20220401
                if yDim != rSize:
                    fake = hs.stack([data[i].T, hs.signals.Signal2D(np.zeros((rSize - yDim,xDim))).T], \
                            axis= 1, lasy = True).T
                else:
                    fake = hs.stack([data[i].T],axis= 1, lasy = True).T
                if xDim != rSize:
                    fake = hs.stack([fake.T,hs.signals.Signal2D(np.zeros((rSize,rSize - xDim))).T], \
                                    #axis= 0, lasy = True, signal = data[i].T).T
                                    axis= 0, lasy = True).T
                fake.axes_manager = data[i].axes_manager
                fake.axes_manager[-2].size = rSize
                fake.axes_manager[-1].size = rSize
                fake.metadata = data[i].metadata
                data[i] = fake
        elif data[i].axes_manager.navigation_dimension == 1:
            if yDim != rSize:
                fake = hs.stack([data[i].T, hs.signals.Signal2D(np.zeros((numbFrames,rSize - yDim,xDim))).T], \
                        axis= 1, lasy = True).T
            else:
                fake = hs.stack([data[i].T],axis= 1, lasy = True).T
            if xDim != rSize:
                fake = hs.stack([fake.T,hs.signals.Signal2D(np.zeros((numbFrames,rSize,rSize - xDim))).T], \
                                #axis= 0, lasy = True, signal = data[i].T).T
                                axis= 0, lasy = True).T
            fake.axes_manager = data[i].axes_manager
            fake.axes_manager[-2].size = rSize
            fake.axes_manager[-1].size = rSize
            fake.metadata = data[i].metadata
            data[i] = fake
        elif len(data[i].data.shape) >= 2:          ## Rhett changed here 20220401
            if yDim != rSize:
                fake = hs.stack([data[i], hs.signals.EDSTEMSpectrum(da.zeros((numbFrames,rSize - yDim,xDim,numbChannel))).as_lazy()], \
                                axis= 1, lasy = True)
            else:
                fake = hs.stack([data[i]], axis= 1, lasy = True)
            if xDim != rSize:
                fake = hs.stack([fake,hs.signals.EDSTEMSpectrum(da.zeros((numbFrames,rSize,rSize - xDim,numbChannel))).as_lazy()], \
                                axis= 0, lasy = True)
            fake.axes_manager = data[i].axes_manager
            fake.axes_manager[0].size = rSize
            fake.axes_manager[1].size = rSize
            fake.metadata = data[i].metadata
            data[i] = fake
        print(data[i])
    return data, xDim, yDim

def resize_data_HAADF(data,rSize=1024):
    xDim = data[-1].axes_manager.signal_shape[0]
    yDim = data[-1].axes_manager.signal_shape[1]
    print('Original x dimension: ' + str(xDim))
    print('Original y dimension: ' + str(yDim))
    numbFrames = data[-1].axes_manager.navigation_shape[0]
    numbDim = len(data)  # How many signals in the data cube
    print('numbDim: ' + str(numbDim))
    for i in range(1,numbDim):     
        if len(data[i].data.shape) >= 2:
            if yDim != rSize:
                fake = hs.stack([data[i].T, hs.signals.Signal2D(np.zeros((numbFrames,rSize - yDim,xDim))).T], \
                        axis= 1, lasy = True).T
            else:
                fake = hs.stack([data[i].T],axis= 1, lasy = True).T
            if xDim != rSize:
                fake = hs.stack([fake.T,hs.signals.Signal2D(np.zeros((numbFrames,rSize,rSize - xDim))).T], \
                                #axis= 0, lasy = True, signal = data[i].T).T
                                axis= 0, lasy = True).T
            fake.axes_manager = data[i].axes_manager
            fake.axes_manager[-2].size = rSize
            fake.axes_manager[-1].size = rSize
            fake.metadata = data[i].metadata
            data[i] = fake
            print(data[i])
    return data, xDim, yDim


def analysis_path(data_name,signal,analysis_path=None):         # Select a folder for analysis
    if analysis_path == None:
        analysis_path = filedialog.askdirectory()
    name = analysis_path + '/' + data_name + '/' + signal
    return name

def nrr_setup(data_signal,analysis_path,set_lambda,skip_frames):
    calculation = MatchSeries(data_signal,analysis_path)
    calculation.run()                                          ## This is only for setting up. Do not run yet
    calculation.configuration["lambda"] = set_lambda
    calculation.configuration["templateSkipNums"] = skip_frames
    return calculation

def generate_batch(base,group,para,nrr_system,system_path_1='',system_path_2='',num_jobs_persub=1):
    if os.path.exists(os.path.join(base,group)) == False:
            os.mkdir(os.path.join(base,group))
    for i in range(len(para)):            # make folders
        path = str(para[i]) + '/'
        if os.path.exists(os.path.join(base,group,path)) == False:
            os.mkdir(os.path.join(base,group,path))
        
    default_file = os.path.join(base, 'config.par')   # make *.par files
    if nrr_system == 'win_linux':
        base_update = base.replace(system_path_1,system_path_2) + '/'
        module_location = system_path_2 + 'privatemodules/opt/nrr/matchSeries '
    else:
        base_update = base + '/'
        module_location = system_path_1 + 'privatemodules/opt/nrr/matchSeries '

    with open(default_file) as f:         
        datafile = f.readlines()
        for i in range(len(para)):
            para_name = para[i].split('_')[0]
            para_value = para[i].split('_')[1]
            par_file = os.path.join(base, group, str(para[i]), 'config.par')
            print(par_file)
            f2 = open(par_file,"w+")
            for j in range(len(datafile)):
                temp = datafile[j].split(' ')
                if temp[0] == 'templateNamePattern':
                    f2.write('templateNamePattern '+ base_update + temp[1])
                elif temp[0] == 'saveDirectory':
                    f2.write('saveDirectory '+ base_update + group + '/' + str(para[i]) + '/output/\n')
                elif temp[0] == para_name:
                    f2.write(para_name + ' '+ para_value + '\n')
                else:
                    f2.write(datafile[j])
            f2.close()
    
    njobs_per_submission = num_jobs_persub
    ifile = 0
    isubmission = 0
    #######Change for different linux system ##################################################################################################
    for i in range(len(para)):
        path = str(para[i]) + '/'
        if ifile == 0:
            submission_file = os.path.join(base, group + '_' + str(isubmission) + '.sh')
            print(submission_file)
            f2 = open(submission_file,"w+")
            f2.write('#!/bin/sh\n\n')
            f2.write('#SBATCH --job-name=Test_' + str(isubmission) + '\n')
            f2.write('#SBATCH --error=Test_' + str(isubmission) + '.err\n')
            f2.write('#SBATCH --output=Test_' + str(isubmission) + '.out\n')
            f2.write('#SBATCH --time=500:00:00\n')                              # Rhett changed here 20220925
            f2.write('#SBATCH --nodes=1\n')
            f2.write('#SBATCH --ntasks=2\n')
            f2.write('##SBATCH --export=ALL\n')
            f2.write('echo "Started on:"\ndate\n\n')
            isubmission = isubmission + 1
        path_temp = base_update + group + '/' + path
        f2.write(module_location + \
                path_temp + 'config.par\n')
        f2.write('rm -r ' + path_temp + 'output/stage1 ' + path_temp + 'output/stage2\n\n')
        ifile = ifile + 1
    ##########################################################################################################################################
        if ifile == njobs_per_submission:
            ifile = 0
            f2.write('echo "Finished on:"\ndate\n')
            f2.close()
            # After the whole file writing, replace Windows style \r\n with Unix style \n
            fileContents = open(submission_file,"r").read()
            f2 = open(submission_file,"w", newline="\n")
            f2.write(fileContents)
            f2.close()

def run_win(calculation,analysis_path,group,para):
    path = str(para) + '/'
    calculation.path = os.path.join(analysis_path,group,path)
    calculation.run_win() 


# 2.0 Quantify NRR (Optional)
#############################################################################
def LoadImage(path):
#     path = '/srv/home/chenyu/NRR/STO_NRR/ParameterTest/Lambda/100/HAADF_NRR/stage3/average.q2bz'
    # read header in read text mode
    fid = bz2.open(path, mode='rt',encoding = "ISO-8859-1")
    next(fid)    # magic number P9, skip
#     print(fid.readline())   # description line
    fid.readline()    # do not print, but read the description line
    size = fid.readline()   # image size in width, height
    width = int(size.split(' ')[0])
    height = int(size.split(' ')[1][:-1])
#     print(width, height)
#     print(fid.readline())   # max?
    # each number is a 8 byte double format

    # read data part in read binary mode
    img = np.zeros((height,width))
    fid = bz2.open(path, mode='rb')
    for _ in range(4):
        next(fid)
    for icol in range(width):
        for irow in range(height):
            read_bytes = fid.read(8)
            img[irow,icol] = struct.unpack('d',read_bytes)[0]

#     plt.imshow(img)
#     plt.colorbar()
    return img.T  #  I changed here.... 20210813

def LoadNumSamples(path):
    # crop image accroding to numsamples
#     path = '/srv/home/chenyu/NRR/STO_NRR/ParameterTest/Lambda/100/HAADF_NRR/stage3/numSamples.q2bz'
    # read header in read text mode
    fid = bz2.open(path, mode='rt',encoding = "ISO-8859-1")
    next(fid)    # magic number P9, skip
    # print(fid.readline())   # description line
    fid.readline()
    size = fid.readline()   # image size in width, height
    width = int(size.split(' ')[0])
    height = int(size.split(' ')[1][:-1])
    # print(width, height)
    # print(fid.readline())   # max?
    # each number is a 8 byte double format

    # read data part in read binary mode
    nSamples = np.zeros((height,width))
    fid = bz2.open(path, mode='rb')
    for _ in range(4):
        next(fid)
    for icol in range(width):
        for irow in range(height):
            read_bytes = fid.read(8)
            nSamples[irow,icol] = struct.unpack('d',read_bytes)[0]

    nSamples = np.heaviside(nSamples-np.amax(nSamples),1)

#     plt.imshow(nSamples)
#     plt.colorbar()
    return nSamples.T #  I changed here.... 20210813


def gaussianx2(xdata_tuple,bg,height, center_x, center_y, width_x, width_y):

    (x, y) = xdata_tuple
    width_x = float(width_x)
    width_y = float(width_y)
    g = height*exp(
                 -(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)+bg
    return g.ravel()

def fitGaussianx2(data, guess, bounds):
    x = np.linspace(0, data.shape[1]-1, data.shape[1])
    y = np.linspace(0, data.shape[0]-1, data.shape[0])
    x, y = np.meshgrid(x, y)
    xdata_tuple = (x,y)
    popt, pcov = optimize.curve_fit(gaussianx2, xdata_tuple, data.ravel(), p0=guess,bounds=bounds,method='trf',verbose=0,maxfev=100000)
    return popt

def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = [int(w/2), int(h/2)]
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

def FindStd(array_x,array_y,test_pixel=3,tolorence=0.5,max_factor=3):
    
    array_x1_copy = copy.deepcopy(array_x)
    array_y1_copy = copy.deepcopy(array_y)
    y_sep = []
    while array_x1_copy.size != 0:
        x_cor = array_x1_copy[0]
        index = np.where((array_x1_copy<x_cor+test_pixel) & (array_x1_copy>x_cor-test_pixel))
        array_y_sub = array_y1_copy[index]
        array_y_sub = np.sort(array_y_sub)
        for i in range(array_y_sub.shape[0]-1):
            spacing = array_y_sub[i+1] - array_y_sub[i]
            y_sep.append(spacing)
        array_x1_copy = np.delete(array_x1_copy,index)
        array_y1_copy = np.delete(array_y1_copy,index)
    y_mean = np.mean(np.asarray(y_sep))

    array_x1_copy = copy.deepcopy(array_x)
    array_y1_copy = copy.deepcopy(array_y)
    x_sep = []
    while array_x1_copy.size != 0:
        y_cor = array_y1_copy[0]
        index = np.where((array_y1_copy<y_cor+test_pixel) & (array_y1_copy>y_cor-test_pixel))
        array_x_sub = array_x1_copy[index]
        array_x_sub = np.sort(array_x_sub)
        for i in range(array_x_sub.shape[0]-1):
            spacing = array_x_sub[i+1] - array_x_sub[i]
            x_sep.append(array_x_sub[i+1] - array_x_sub[i])
        array_x1_copy = np.delete(array_x1_copy,index)
        array_y1_copy = np.delete(array_y1_copy,index)
    x_mean = np.mean(np.asarray(x_sep))

    x_gap = x_mean*tolorence
    y_gap = y_mean*tolorence

    array_x1_copy = copy.deepcopy(array_x)
    array_y1_copy = copy.deepcopy(array_y)
    y_sep = []
    while array_x1_copy.size != 0:
        x_cor = array_x1_copy[0]
        index = np.where((array_x1_copy<x_cor+x_gap) & (array_x1_copy>x_cor-x_gap))
        array_y_sub = array_y1_copy[index]
        array_y_sub = np.sort(array_y_sub)
        for i in range(array_y_sub.shape[0]-1):
            spacing = array_y_sub[i+1] - array_y_sub[i]
            if (spacing > y_gap/max_factor) & (spacing < y_gap*max_factor):
                y_sep.append(spacing)
        array_x1_copy = np.delete(array_x1_copy,index)
        array_y1_copy = np.delete(array_y1_copy,index)
    y_std = np.std(np.asarray(y_sep))
    
    array_x1_copy = copy.deepcopy(array_x)
    array_y1_copy = copy.deepcopy(array_y)
    x_sep = []
    while array_x1_copy.size != 0:
        y_cor = array_y1_copy[0]
        index = np.where((array_y1_copy<y_cor+y_gap) & (array_y1_copy>y_cor-y_gap))
        array_x_sub = array_x1_copy[index]
        array_x_sub = np.sort(array_x_sub)
        for i in range(array_x_sub.shape[0]-1):
            spacing = array_x_sub[i+1] - array_x_sub[i]
            if (spacing > x_gap/max_factor) & (spacing < x_gap*max_factor):
                x_sep.append(array_x_sub[i+1] - array_x_sub[i])
        array_x1_copy = np.delete(array_x1_copy,index)
        array_y1_copy = np.delete(array_y1_copy,index)
    x_std = np.std(np.asarray(x_sep))
    x_mean = np.mean(np.asarray(x_sep))
    y_mean = np.mean(np.asarray(y_sep))
    
    return x_std, y_std, x_mean, y_mean


def IntegratedInt(average_file,nSamples_file,save,xDim,yDim,crop,plot,min_distance=3,test_pixel=3,tolorence=0.5,max_factor=3):
    # average_file = '//mufs4/x.zhou/TEM/nrr/Fe2NiAl/Fe2NiAl_1645/HAADF/lambda/lambda_800/output/stage3/average.q2bz'
    # nSamples_file = '//mufs4/x.zhou/TEM/nrr/Fe2NiAl/Fe2NiAl_1645/HAADF/lambda/lambda_800/output/stage3/numSamples.q2bz'
    # save: save the output image
    # min_distance: pixel, the minimum distance to find peaks
    # test_pixel: pixel, the first round to estimate x, y spacing
    # tolorence: ratio, the maximum ratio to search for the spots in the same line or column.
    # max_factor: ratio, spacing*tolorence*max_factor, spacing*tolorence/max_factor, define the
    #             maximum and minimum to find next spot in the same line or column

    average = LoadImage(average_file)
    nSamples = LoadNumSamples(nSamples_file)
    if xDim is None and yDim is None:
        xDim = average.shape[1]
        yDim = average.shape[0]
    if crop is None:
        average = average[:yDim,:xDim]
        nSamples = nSamples[:yDim,:xDim]
        save_fig_position = save + '_position.tiff'
        save_fig = save + '_fit.tiff'
        save_fig_raw = save + '_fit_raw.tiff'
        save_txt = save + '_co.txt'
        crop_prefix = ''
        save = save_fig
    else:
        average = average[crop[2]:crop[3],crop[0]:crop[1]]
        nSamples = nSamples[crop[2]:crop[3],crop[0]:crop[1]]
        crop_prefix = "L" + str(crop[0]) + "-R" + str(crop[1]) + "-T" + str(crop[2]) + "-B" + str(crop[3])
        file_prefix = os.path.split(save)[1]
        save = os.path.join(os.path.split(save)[0],crop_prefix)
        try:
            os.stat(save)
        except:
            os.mkdir(save)
        save_fig_position = os.path.join(save, file_prefix + '_position.tiff')
        save_fig = os.path.join(save, file_prefix + '_fit.tiff')
        save_fig_raw = os.path.join(save, file_prefix + '_fit_raw.tiff')        
        save_txt = os.path.join(save, file_prefix + '_co.txt')

    coordinates = peak_local_max(average, min_distance=min_distance)

    x_max = average.shape[1]
    y_max = average.shape[0]

    peaks_x_fit = []
    peaks_y_fit = []

    ps = min_distance  #  Patch size must be Even

    for i in range(coordinates.shape[0]):

        x = int(coordinates[i][1])
        y = int(coordinates[i][0])
        if nSamples[y,x] == 0:
            continue

        if x >=ps+1 and y >=ps+1 and x <=x_max-ps-1 and y <=y_max-ps-1:
            patch = average[y-ps:y+ps+1,x-ps:x+ps+1]
            patch = patch - amin(patch)
            bg = 0

            # parameters are in the order of : background, height, cx, cy, wx, wy
            guess = (bg,patch[ps,ps]-bg,ps, ps, ps/2, ps/2)
            bounds = ([0,0,0,0,1,1],[np.inf,np.inf,2*(ps-1),2*(ps-1),2*(ps-1),2*(ps-1)])
            param = fitGaussianx2(patch, guess, bounds)

            peaks_x_fit.append(param[2]+x-ps)
            peaks_y_fit.append(param[3]+y-ps)

    a = np.asarray([peaks_x_fit])
    b = np.asanyarray([peaks_y_fit])
    co_txt = np.concatenate((a.T,b.T),axis = 1)
    np.savetxt(save_txt,co_txt,fmt='%.8f')
    
    xprec, yprec, xmean, ymean = FindStd(np.asarray(peaks_x_fit),np.asarray(peaks_y_fit),test_pixel,tolorence,max_factor)

    int_list = []
    for i in range(len(peaks_x_fit)):
        mask = create_circular_mask(average.shape[0], average.shape[1], center=(peaks_x_fit[i],peaks_y_fit[i]), radius=min_distance)
        int_list.append(np.sum(mask * average))

    if xDim <= yDim:
        DimMax = yDim
    else:
        DimMax = xDim    

    pad_inch_set = DimMax/8192
    if pad_inch_set < 0.01:
        pad_inch_set = 0.01
        
    plt.ioff()
    plt.figure(figsize=(2*math.ceil(xDim/128), 2*math.ceil(yDim/128)))
    average_position = copy.deepcopy(average)
    average_position[average_position>0] = 0
    plt.imshow(average_position) 
    plt.xticks([])
    plt.yticks([])
    plt.scatter(peaks_x_fit, peaks_y_fit,s=min_distance,c='r',alpha=0.4)
    plt.savefig(save_fig_position, bbox_inches='tight', transparent=True, pad_inches=pad_inch_set)

    if plot == 0:
        plt.ioff()
    else:
        plt.ion()
    plt.figure(figsize=(2*math.ceil(xDim/128), 2*math.ceil(yDim/128)))
    plt.imshow(average)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(save_fig_raw, bbox_inches='tight', transparent=True, pad_inches=pad_inch_set)
    plt.scatter(peaks_x_fit, peaks_y_fit,s=min_distance,c='r',alpha=0.4)
    plt.savefig(save_fig, bbox_inches='tight', transparent=True, pad_inches=pad_inch_set)
   
    # if crop is None:
    #     pass
    # else:
    #     plt.close()
    
    intensity_mean  = np.asarray(int_list).mean()
    intensity_std = np.asarray(int_list).std()/np.sqrt(i+1)
    
    return xmean, ymean, xprec,yprec,intensity_mean,intensity_std,crop_prefix

def nrr_compare(base_path,group,para,data_signal,min_distance=5,xDim=None,yDim=None,crop=None,plot=0):
    pix_size = 1000*data_signal.axes_manager.signal_axes[0].scale
    int_mean_list = []
    int_std_list = []
    para_list = []
    xprec_list = []
    yprec_list = []
    xmean_list = []
    ymean_list = []
    xlabel_name = para[0].split('_')[0]
    for i in range(len(para)):
        path = os.path.join(base_path,group,para[i],'output','stage3')
        average_file = os.path.join(path, 'average.q2bz')
        print(average_file)
        nsamples_file = os.path.join(path, 'numSamples.q2bz')
        save = os.path.join(base_path,group,para[i])
        xmean, ymean,xprec,yprec,mean,std,crop_prefix = IntegratedInt(average_file,nsamples_file,save,xDim,yDim,crop,plot,min_distance)
        xmean_list.append(xmean)
        ymean_list.append(ymean)
        int_mean_list.append(mean)
        int_std_list.append(std)
        para_list.append(int(regexp.findall("\d+", para[i])[0]))
        xprec_list.append(xprec)
        yprec_list.append(yprec)
    
    xmean_list = [xmean_list[i]*pix_size for i in range(len(xmean_list))]
    ymean_list = [ymean_list[i]*pix_size for i in range(len(ymean_list))]
    xprec_list = [xprec_list[i]*pix_size for i in range(len(xprec_list))]
    yprec_list = [yprec_list[i]*pix_size for i in range(len(yprec_list))]

    if crop_prefix == '':
        x_y_save = os.path.join(base_path,group,'Unit_length.tiff')
        int_save = os.path.join(base_path,group,'Intensity.tiff')
        prec_save = os.path.join(base_path,group,'Precision.tiff')
    else:
        x_y_save = os.path.join(base_path,group,crop_prefix,'Unit_length.tiff')
        int_save = os.path.join(base_path,group,crop_prefix,'Intensity.tiff')
        prec_save = os.path.join(base_path,group,crop_prefix,'Precision.tiff')

    plt.ioff()
    fig, ax = plt.subplots(1, 1)
    plt.errorbar(para_list,int_mean_list,int_std_list,marker='s', mfc='blue', ms=14)
    plt.tick_params(direction='in')
    plt.yticks([])
    ax.set_xscale('log')
    ax.grid('on')
    plt.xlabel(xlabel_name,fontsize=14)
    plt.ylabel('Avearge intensity', fontsize=14)
    plt.savefig(int_save)

    fig, ax = plt.subplots(1, 1)
    plt.plot(para_list,xprec_list,'.',markersize=14,c='blue',label='Fast scan')
    plt.plot(para_list,yprec_list,'*',markersize=14,c='red',label='Slow scan')
    plt.legend(fontsize=14, loc='upper right')
    plt.tick_params(direction='in')
    ax.set_xscale('log')
    ax.grid('on')
    plt.xlabel(xlabel_name,fontsize=14)
    plt.ylabel('Precision (pm)', fontsize=14)
    plt.savefig(prec_save)

    fig, ax = plt.subplots(1, 1)
    plt.plot(para_list,xmean_list,'.',markersize=14,c='blue',label='Fast scan')
    plt.plot(para_list,ymean_list,'*',markersize=14,c='red',label='Slow scan')
    plt.legend(fontsize=14, loc='upper right')
    plt.tick_params(direction='in')
    ax.set_xscale('log')
    ax.grid('on')
    plt.xlabel(xlabel_name,fontsize=14)
    plt.ylabel('Unit length (pm)', fontsize=14)
    plt.savefig(x_y_save)


# 3.0 After NRR calculation
#########################################################################################

def load_deform(analysis_path,group,para,para_index):
    subfolder = group + '/' + para[para_index]
    loaded = MatchSeries.load(analysis_path,subfolder)
    return subfolder, loaded


def format_raw_spectrum(eds_bin,energy_low,energy_high,analysis_path,raw_spectrum_load,band_list,xDim=None,yDim=None,crop=None):
    specmap = raw_spectrum_load
    lines = len(band_list)
    specmap.metadata.Sample.elements = []
    lines_list = []
    for i in range(lines):
        lines_list.append(band_list[i][0])
    specmap.add_lines(lines_list)
    offset = specmap.axes_manager[-1].offset
    scale = specmap.axes_manager[-1].scale
    low = int((energy_low - offset)/scale)
    high = int((energy_high - offset)/scale)
    rbsm = specmap.isig[low:high]
    rbsm = rbsm.rebin(scale=(1, 1, eds_bin))

    subfolder = os.path.join(analysis_path,'without_deform')
    if os.path.exists(subfolder) == False:
        os.mkdir(subfolder)
    raw_spectrum_name = "EDS-" + str(energy_low) + "-" + str(energy_high) + "-bin" + str(eds_bin)
    raw_spectrum_path = subfolder + '/' + raw_spectrum_name
    if os.path.exists(raw_spectrum_path) == False:
        os.mkdir(raw_spectrum_path)

    if xDim is None and yDim is None:
        xDim = specmap.axes_manager.navigation_shape[0]
        yDim = specmap.axes_manager.navigation_shape[1]
    if crop is None:
        rbsm = rbsm.inav[:xDim,:yDim]
    else:
        rbsm = rbsm.inav[crop[0]:crop[1],crop[2]:crop[3]]
        crop_prefix = "L" + str(crop[0]) + "-R" + str(crop[1]) + "-T" + str(crop[2]) + "-B" + str(crop[3])
        raw_spectrum_path = os.path.join(raw_spectrum_path,crop_prefix)
        try:
            os.stat(raw_spectrum_path)
        except:
            os.mkdir(raw_spectrum_path)
    
    # fake.axes_manager = specmap.axes_manager
    # fake.axes_manager[-2].size = rSize
    # fake.axes_manager[-1].size = rSize
    # fake.metadata = data[i].metadata
    # data[i] = fake

    raw_name = raw_spectrum_path + "/raw_spectrum.hspy"
    rbsm.save(raw_name)
    return specmap,rbsm,raw_name

def apply_deform_to_spectrum(eds_bin,energy_low,energy_high,analysis_path,subfolder,data,loaded,band_list,xDim=None,yDim=None,crop=None):
    specmap = data[-1]
    lines = len(band_list)
    specmap.metadata.Sample.elements = []
    lines_list = []
    for i in range(lines):
        lines_list.append(band_list[i][0])
    specmap.add_lines(lines_list)
    offset = specmap.axes_manager[-1].offset
    scale = specmap.axes_manager[-1].scale
    low = int((energy_low - offset)/scale)
    high = int((energy_high - offset)/scale)
    rbsm = specmap.isig[low:high]
    rbsm = rbsm.rebin(scale=(1, 1, 1, eds_bin))
    corrected_spectrum = loaded.apply_deformations_to_spectra(rbsm, sum_frames=True) 
    corrected_spectrum.compute()
    corrected_spectrum_name = "EDS-" + str(energy_low) + "-" + str(energy_high) + "-bin" + str(eds_bin)
    corrected_spectrum_path = os.path.join(analysis_path, subfolder, corrected_spectrum_name)
    try:
        os.stat(corrected_spectrum_path)
    except:
        os.mkdir(corrected_spectrum_path)

    if xDim is None and yDim is None:
        xDim = specmap.axes_manager.navigation_shape[0]
        yDim = specmap.axes_manager.navigation_shape[1]
    if crop is None:
        corrected_spectrum = corrected_spectrum.inav[:xDim,:yDim]
    else:
        corrected_spectrum = corrected_spectrum.inav[crop[0]:crop[1],crop[2]:crop[3]]
        crop_prefix = "L" + str(crop[0]) + "-R" + str(crop[1]) + "-T" + str(crop[2]) + "-B" + str(crop[3])
        corrected_spectrum_path = os.path.join(corrected_spectrum_path,crop_prefix)
        try:
            os.stat(corrected_spectrum_path)
        except:
            os.mkdir(corrected_spectrum_path)
        
    corrected_name = corrected_spectrum_path + "/corrected_spectrum.hspy"
    corrected_spectrum.save(corrected_name)
    return corrected_spectrum,corrected_name

def load_corrected_spectrum(corrected_name=None):
    if corrected_name == None:
        corrected_name = filedialog.askopenfilename(filetypes=[("Hyperspy file", "*.hspy")])           # Open the .hdf5 file for analysis
    corrected_spectrum = hs.load(corrected_name)
    return corrected_spectrum,corrected_name

def plot_intergrated_spectrum(spectrum,eds_bin=None,energy_low=None,plot=1):
    # Plot the integrated spectrum and use it to determine the start and end energy for each EDS band
   
    data_noisy = spectrum
    data_noisy = np.asarray(data_noisy)

    if eds_bin == None:
        eds_bin = 1
    if energy_low == None:
        energy_low = spectrum.axes_manager[-1].offset

    offset = 1000*energy_low   # energy offset in eV
    e_channel = 1000*spectrum.axes_manager[-1].scale  # energy in eV for each channel   ## Rhett changed here 20220520
   
    nchannel = data_noisy.shape[2]
    energy = offset + np.linspace(0, nchannel * e_channel, nchannel, endpoint = False)
    temp = np.sum(np.sum(data_noisy, axis = 0), axis = 0)
    # print(energy,temp)
    if plot == 1:
        fig, ax = plt.subplots(1,1, figsize = (9,6))
        ax.plot(energy / 1000, temp)
        ax.set_xlabel('Energy (keV)',fontsize = 16)
        ax.set_ylabel('Integrated counts', fontsize = 16)
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
    return energy, temp
    # TODO: on the plot, figure out the peaks that should be used, and the integration range for each peak.

def generate_map_list(band_list,spectrum,eds_bin,energy_low):
    # Integrate spectrum into elemental bands. Each band in the format of name, start energy, and end energy in keV
    data_noisy = spectrum
    data_noisy = np.asarray(data_noisy)
    offset = 1000*energy_low   # energy offset in eV
    e_channel = 1000*spectrum.axes_manager[-1].scale   # energy in eV for each channel    ## Rhett changed here 20220520
    #band_list = sorted(band_list, key = lambda x:x[1])
    map_list = []
    for element in band_list:
        name = element[0]
        low = element[1]
        high = element[2]
        low_channel = int((low * 1000 - offset) // e_channel)
        high_channel = int((high * 1000 - offset) // e_channel)
        element_map = np.sum(data_noisy[:,:,low_channel:high_channel+1], axis = 2)  # 20210816 Rhett Changed here, +1
        map_list.append(element_map)
    return map_list

# def map_list_auto(spectrum,line_width=[1.3,1.8]):                #  Not used...because we could not control the intergrated windows for each element.
#     spectrum.plot(integration_windows='auto')
#     bw = spectrum.estimate_background_windows(line_width=line_width)
#     map_signal = spectrum.get_lines_intensity(background_windows=bw)
#     map_list = []
#     for i in range(len(map_signal)):
#         map_list.append(map_signal[i].data)
#     return map_list

def save_map(map_list,folder_path):
    # Save the nosiy maps into a single 3D data cube in numpy format
    data_band = np.asarray(map_list)
    save_path = os.path.join(folder_path,'Denoise.npy')
    # TODO: rename the file with a proper name
    np.save(save_path, data_band)

def plot_eds_elemental_map(band_list,eds_bin,energy_low,spectrum,spectrum_name):
    energy, temp = plot_intergrated_spectrum(spectrum,eds_bin,energy_low,plot=0)
    map_list = generate_map_list(band_list,spectrum,eds_bin,energy_low)

    # Plot the selected EDS bands in the integrated spectrum, and the noisy elemental maps
    offset = 1000*energy_low   # energy offset in eV
    e_channel = 1000*spectrum.axes_manager[-1].scale   # energy in eV for each channel    ## Rhett changed here 20220520
    if len(band_list) % 3 == 0:
        extra_row = 0
    else:
        extra_row = 1
    nrow = 1 + len(band_list) // 3 + extra_row
    fig = plt.figure(figsize=(8, nrow * 3))
    grid = plt.GridSpec(nrow, 3, hspace=0.5, wspace=0)
    spectrum_ax = fig.add_subplot(grid[0:1, :])

    spectrum_ax.plot(energy / 1000, temp)
    idx = 1
    for element in band_list:
        name = element[0]
        low = element[1]
        high = element[2]
        low_channel = int((low * 1000 - offset) // e_channel)
        high_channel = int((high * 1000 - offset) // e_channel)
        section = np.arange(low_channel, high_channel+1, 1)  # 20210816 Rhett Changed here, +1
        # print(energy, temp, section,low, high, low_channel, high_channel)
        spectrum_ax.fill_between(energy[section] / 1000, temp[section], label = name, color = 'C' + str(idx + 1))
        idx += 1

    plt.legend(fontsize = 12)
    spectrum_ax.set_xlabel('Energy (keV)',fontsize = 16)
    spectrum_ax.set_ylabel('Integrated counts', fontsize = 16)
    spectrum_ax.tick_params(axis='x', labelsize=16)
    spectrum_ax.tick_params(axis='y', labelsize=16)
    spectrum_ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

    for i in range(len(band_list)):
        if len(band_list) == 2 and i == 1:
            map_ax = fig.add_subplot(grid[1 + i // 3, i % 3 + 1])
        else:
            map_ax = fig.add_subplot(grid[1 + i // 3, i % 3])
        map_ax.imshow(map_list[i])
        map_ax.set_xticks([])
        map_ax.set_yticks([])
        map_ax.set_title(band_list[i][0], color = 'C' + str(i + 2), fontsize = 16)

    folder_path = os.path.split(spectrum_name)[0] 
    save_path = folder_path + '/' + 'Spectrum_RawMap.tiff'
    plt.savefig(save_path)
    plt.show()
    save_map(map_list,folder_path)

def generate_NPY_group(base,group,para,band_list,eds_bin,energy_low,energy_high,crop):
    corrected_spectrum_name = "EDS-" + str(energy_low) + "-" + str(energy_high) + "-bin" + str(eds_bin)
    if crop is not None:
        crop_prefix = "L" + str(crop[0]) + "-R" + str(crop[1]) + "-T" + str(crop[2]) + "-B" + str(crop[3])
    else:
        crop_prefix = ''
    for i in range(len(para)):            # make folders
        corrected_name = os.path.join(base, group, str(para[i]),corrected_spectrum_name,crop_prefix,'corrected_spectrum.hspy')
        corrected_spectrum,spectrum_name = load_corrected_spectrum(corrected_name)
        plot_eds_elemental_map(band_list,eds_bin,energy_low,corrected_spectrum,spectrum_name)

# 4.0 After NLPCA
#######################################################################################

def load_NLPCA(denoised_name=None):
    # Load the denoised elemental maps from .mat file and comare to noisy data
    # TODO: replace the current denoised file name with the actual denoised file name generated by Matlab.
    if denoised_name == None:
        denoised_name = filedialog.askopenfilename(filetypes=[("Matlab file", "*.mat")])           # Open the .hdf5 file for analysis
    denoised = sio.loadmat(denoised_name)
    denoised = denoised['ima_fil']
    return denoised, denoised_name

def scale_range(input, min, max):
    input += -(np.min(input))
    input /= np.max(input) / (max - min)
    input += min
    return input 

# def plot_denoised(denoised,denoised_name):
#     # Create RGB image, currently hard coded as 
#     # TODO: replace the layer index in map_list with the index that correspond to the EDS peak .

#     # R: Al, layer 0+4, 
#     # G: Fe, layer 1,
#     # B: Ni, layer 2.

#     channel_R = scale_range(denoised[:,:,0].astype('float'), 0, 1)
#     channel_G = scale_range(denoised[:,:,1].astype('float'), 0, 1)
#     channel_B = scale_range(denoised[:,:,2].astype('float'), 0, 1)
#     denoised_stack = np.dstack((channel_R, channel_G, channel_B))

#     plt.figure(figsize=(5,4))

#     plt.imshow(denoised_stack, cmap = 'gray')
#     plt.xticks([])
#     plt.yticks([])
#     plt.title("denoised")

#     # TODO: replace the label with the element name for the three channels.
#     legend_data = [[[255,0,0],"Al"],
#                 [[0,255,0],"Fe"],
#                 [[0,0,255],"Ni"]]  

#     handles = [Rectangle((0,0),1,1, color = tuple((v / 255 for v in c))) for c,n in legend_data]
#     labels = [n for c,n in legend_data]

#     plt.legend(handles,labels, ncol=1, bbox_to_anchor = (1.30, 1), loc = 'upper right')

#     save_path = os.path.split(denoised_name)[0] + '/' + 'Overlay.tiff'
#     plt.savefig(save_path)

def plot_denoised_alpha(denoised,colorlist,band_list,denoised_name,overlay_list,rescale=None,label=0):
    # Create RGB image, currently hard coded as 
    # TODO: replace the layer index in map_list with the index that correspond to the EDS peak .

    elements = denoised.shape[2]
    img_size = max(denoised.shape[0],denoised.shape[1])
    img_inten_list = []

    if rescale == 'rescale':
        fig = plt.figure('Denoised: Overlay & Rescaled',frameon=False)
        title = 'Denoised: Overlay & Rescaled'
    else:
        fig = plt.figure('Denoised: Overlay',frameon=False)
        title = 'Denoised: Overlay'
    if denoised.shape[0] <= denoised.shape[1]:
        width = 6
        height = (denoised.shape[0]/denoised.shape[1])*width
    else:
        height = 6
        width = (denoised.shape[1]/denoised.shape[0])*height

    fig.set_size_inches(width, height)
    plt.axis("off")
    for i in range(elements):
        if i in overlay_list:
            if rescale == 'rescale':
                img = scale_range(denoised[:,:,i].astype('float'), 0, 1)
            else:
                img = denoised[:,:,i]
            img_inten_list.append(img.max())
    img_inten_list = img_inten_list/np.linalg.norm(img_inten_list)
    k = 0
    if img_size <= 64:
        x_a = 6 
        x_b = 10
    elif img_size > 64 and img_size <= 128:
        x_a = 12 
        x_b = 20
    elif img_size > 128:
        x_a = 24 
        x_b = 40

    for i in range(elements):
        if i in overlay_list:
            if rescale == 'rescale':
                img = scale_range(denoised[:,:,i].astype('float'), 0, 1)
            else:
                img = denoised[:,:,i]
            zvals = np.ones(img.shape)
            cmap1 = mpl.colors.LinearSegmentedColormap.from_list('my_cmap',[colorlist[i],colorlist[i]],1)
            plt.imshow(zvals, cmap=cmap1, alpha=img_inten_list[k]*(img/(img.max())), interpolation='bilinear')
            if label == 1:
                plt.text(x_a + x_b*k, x_a, band_list[i][0], ha='center', va='center', bbox={'facecolor': colorlist[i], \
                        'pad': 8, 'alpha': 0.5, 'edgecolor':'none'})
            k += 1
    plt.show()
    subname = os.path.split(denoised_name)[1][:-4]
    if rescale == 'rescale':
        save_path = os.path.split(denoised_name)[0] + '/' + subname + '_Overlay_Rescaled.tiff'
    else:
        save_path = os.path.split(denoised_name)[0] + '/' + subname + '_Overlay.tiff'
    plt.savefig(save_path)

def plot_element_denoised_alpha(denoised,colorlist,band_list,denoised_name,step,stage='stage3'):
    
    lines = len(band_list)

    if lines <= 3:
        per_row = lines
        nrow = 1
        fig = plt.figure(figsize=(lines*6,6))
    elif lines == 4:
        per_row = 2
        nrow = 2
        fig = plt.figure(figsize=(12, 12))
    elif lines == 5 or lines == 6 or lines == 9:
        per_row = 3
        nrow = 2
        fig = plt.figure(figsize=(8,12))
    else:
        per_row = 4 
        if lines % 4 == 0:
            extra_row = 0
        else:
            extra_row = 1
        nrow = lines // 4 + extra_row
        fig = plt.figure(figsize=(2*nrow, 8))
    
    grid = plt.GridSpec(nrow, per_row, hspace=0.5, wspace=0)

    for i in range(len(band_list)):
        map_ax = fig.add_subplot(grid[i // per_row, i % per_row])
        img = scale_range(denoised[:,:,i].astype('float'), 0, 1)
        zvals = np.ones(img.shape)
        cmap1 = mpl.colors.LinearSegmentedColormap.from_list('my_cmap',[colorlist[i],colorlist[i]],1)
        map_ax.imshow(zvals, cmap=cmap1,alpha=img/(img.max()),interpolation='bilinear')
        map_ax.set_xticks([])
        map_ax.set_yticks([])
        map_ax.set_title(band_list[i][0], color = colorlist[i], fontsize = 16)
    
    subname = os.path.split(denoised_name)[1][:-4]
    denoised_path = os.path.split(denoised_name)[0]
    save_path = denoised_path + '/' + subname + '.tiff'
    plt.savefig(save_path)

    deform_path = os.path.dirname(os.path.dirname(denoised_name))
    deform_path = os.path.join(deform_path,'output',stage)
    if os.path.exists(deform_path) == True:
        export_deform(step=step,folder=deform_path)
        copy_a_file(deform_path,denoised_path,'reduceDef.tiff')
        copy_a_file(deform_path,denoised_path,'average.png')


def copy_a_file(source,destination,name):
    source = os.path.join(source,name)
    copy = os.path.join(destination,name)
    copyfile(source,copy)

# --------------------------------------------------------------------------------------------- #
def Load_bz2_Image(path=None):
    # read header in read text mode
    if path == None:
        path = filedialog.askopenfilename(filetypes=[("Output file", "*.bz2")])           # Open the .q2bz file for analysis
    
    fid = bz2.open(path, mode='rt',encoding = "ISO-8859-1")
    next(fid)    # magic number P9, skip
    fid.readline()    # do not print, but read the description line
    size = fid.readline()   # image size in width, height
    width = int(size.split(' ')[0])
    height = int(size.split(' ')[1][:-1])
    # each number is a 8 byte double format
    # read data part in read binary mode
    img = np.zeros((height,width))
    fid = bz2.open(path, mode='rb')
    for _ in range(4):
        next(fid)
    for icol in range(width):
        for irow in range(height):
            read_bytes = fid.read(8)
            img[irow,icol] = struct.unpack('d',read_bytes)[0]
    return img.T   #  I changed here.... 20210813

def export_deform(step=8,frames=None,level=None,output_type='images',folder=None):
    if folder == None:
        folder = filedialog.askdirectory()
        
    if frames == None and level == None:
        name_x = os.path.join(folder,'reduceDef_0.dat.bz2')
        name_y = os.path.join(folder,'reduceDef_1.dat.bz2')
        save_name = os.path.join(folder,'reduceDef.tiff')
        save_deform(step,name_x,name_y,save_name)
    else:
        if os.path.exists(os.path.join(folder,output_type)) == False:
            os.mkdir(os.path.join(folder,output_type))
        for frame in frames:
            folder_frame = os.path.join(folder,str(frame) + '-r')
            name_x = os.path.join(folder_frame,'deformation_%02d' %level + '_0.dat.bz2')
            name_y = os.path.join(folder_frame,'deformation_%02d' %level + '_1.dat.bz2')
            save_name = os.path.join(folder,output_type,str(frame) + '_deformation_%02d' %level + '.tiff')
            save_deform(step,name_x,name_y,save_name)
        if output_type == 'video':
            export_video(folder)


def save_deform(step,name_x,name_y,save_name):
    x_deform_lambda = Load_bz2_Image(name_x)
    y_deform_lambda = Load_bz2_Image(name_y)
    plt.ioff()
    plt.figure(figsize=(6, 6))
    plt.quiver( x_deform_lambda[::step,::step], y_deform_lambda[::step,::step])
    plt.xticks([])
    plt.yticks([])
    plt.axis('scaled')
    plt.savefig(save_name)
    plt.close()
    plt.ion()
    

def export_video(folder):
    video_name = os.path.join(folder,'video.avi')
    image_folder = os.path.join(folder,'video')
    images = [img for img in os.listdir(image_folder) if img.endswith(".tiff")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_name, 0, 60, (width,height))
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))
    cv2.destroyAllWindows()
    video.release()
    shutil.rmtree(image_folder)


# --------------------------------------------------------------------------------------------- #
def blur(maps,blur=0,G_blur=0):
    if blur != 0:
        maps = cv2.blur(maps,(blur,blur))
    if G_blur != 0:
        maps = cv2.GaussianBlur(maps,(G_blur,G_blur),0)
    return maps


def averageZeroes(a):
    average = np.average(a)
    [h,w] = a.shape
    for i in range(h):
        for j in range(w):
            if a[i,j] == 0:
                if (j == 0 and a[i,1] != 0):
                    a[i,j] = a[i,1]
                elif (j == w-1 and a[i,j-1] != 0):
                    a[i,j] = a[i,j-1]
                elif (j > 0 and j < w-1 and a[i,j-1] != 0 and a[i,j+1] != 0):    
                    a[i,j] = (a[i,j-1] + a [i,j+1])/2
                else:
                    a[i,j] = average
    return a

def rearrange_intensities(intensities,band_list):
    intensities_new = []
    lines = len(band_list)
    for i in range(lines):
        for j in range(lines):
            if intensities[j].metadata.Sample.xray_lines[0] == band_list[i][0]:
                intensities_new.append(intensities[j])
            else:
                pass
    return intensities_new

def resetUnits(spectrum):
    if spectrum.axes_manager.navigation_axes[0].units == 'nm':
        spectrum.axes_manager.navigation_axes[0].scale = 1000*spectrum.axes_manager.navigation_axes[0].scale
        spectrum.axes_manager.navigation_axes[0].offset = 1000*spectrum.axes_manager.navigation_axes[0].offset
        spectrum.axes_manager.navigation_axes[0].units = 'pm'
    if spectrum.axes_manager.navigation_axes[1].units == 'nm':
        spectrum.axes_manager.navigation_axes[1].scale = 1000*spectrum.axes_manager.navigation_axes[1].scale
        spectrum.axes_manager.navigation_axes[1].offset = 1000*spectrum.axes_manager.navigation_axes[1].offset
        spectrum.axes_manager.navigation_axes[1].units = 'pm'
    return spectrum

def EDS_quanty(band_list,denoised_name,denoised=None,pre_blur=0,pre_G_blur=0,post_blur=0,post_G_blur=0,spectrum=None):
    
    base_folder = os.path.dirname(denoised_name)
    if denoised is None:
        file_prefix = 'Without_NLPCA_'
    else:
        file_prefix = os.path.split(denoised_name)[1][:-4] + '_'
    if pre_blur==0 and pre_G_blur==0 and post_blur==0 and post_G_blur==0:
        blur_prefix = 'noBlur_'
    else:
        blur_prefix = str(pre_blur) + '-' + str(pre_G_blur) + '-' + str(post_blur) + '-' + str(post_G_blur) + '_'
    
    if spectrum == None:
        spectrum_name = os.path.join(base_folder,'corrected_spectrum.hspy')
        if os.path.exists(spectrum_name) == False:
            spectrum_name = os.path.join(base_folder,'raw_spectrum.hspy')
        corrected_spectrum,spectrum_name = load_corrected_spectrum(spectrum_name)
        spectrum = corrected_spectrum 

    spectrum = resetUnits(spectrum)

    lines = len(band_list)
    plt.ioff()
    spectrum.plot(integration_windows='auto')
    intensities = spectrum.get_lines_intensity()
    plt.close()
    intensities = rearrange_intensities(intensities,band_list)

    inten = []
    kfactors = []
    for i in range(lines):
        kfactors.append(band_list[i][3])
        if denoised is not None:
            # intensities[i].data = 1000000*denoised[:,:,i]  # 20210822 Rhett changed, for removing zero artifact
            intensities[i].data = denoised[:,:,i]  # 20210822 Rhett changed, for removing zero artifact
        intensities[i].data = blur(intensities[i].data,pre_blur,pre_G_blur)
        inten.append(intensities[i])
    
    atomic_percent = spectrum.quantification(intensities, method='CL',factors=kfactors)

    s_overall = spectrum.sum()
    in_overall = s_overall.get_lines_intensity()
    in_overall = rearrange_intensities(in_overall,band_list)
    atomic_percent_overall = s_overall.quantification(in_overall, method='CL',factors=kfactors)
    for i in range(lines):
        string = atomic_percent_overall[i].metadata.Sample.elements[0] + ': ' + format(atomic_percent_overall[i].data[0],'.1f') + ' %'
        print(string)

    qual = []
    for i in range(lines):
        atomic_percent[i].data = averageZeroes(atomic_percent[i].data)
        atomic_percent[i].data = blur(atomic_percent[i].data,post_blur,post_G_blur)
        qual.append(atomic_percent[i])

    if lines <= 3:
        per_row = lines
        figsize=(lines*6,6)
    elif lines == 4:
        per_row = 2
        figsize=(12, 12)
    elif lines == 5 or lines == 6 or lines == 9:
        per_row = 3
        figsize=(8,12)
    else:
        per_row = 4 
        if lines % 4 == 0:
            extra_row = 0
        else:
            extra_row = 1
        nrow = lines // 4 + extra_row
        fig = plt.figure()
        figsize=(2*nrow, 12)
    
    plt.close()
    
    plt.ion()
    fig = plt.figure(figsize=figsize)
    hs.plot.plot_images(inten, tight_layout=False, axes_decor='off', scalebar= [], per_row = per_row,
                    cmap=['RdYlBu_r','RdYlBu_r','RdYlBu_r','RdYlBu_r','RdYlBu_r','RdYlBu_r'],fig=fig)
    inten_name = file_prefix + blur_prefix + 'intensity.tiff'
    inten_file = os.path.join(base_folder,inten_name)
    plt.savefig(inten_file)

    plt.ion()
    fig = plt.figure(figsize=figsize)
    hs.plot.plot_images(qual, tight_layout=False, axes_decor='off', scalebar= [], per_row = per_row,
                    cmap=['RdYlBu_r','RdYlBu_r','RdYlBu_r','RdYlBu_r','RdYlBu_r','RdYlBu_r'],fig=fig)
    qual_name = file_prefix + blur_prefix + 'quantification.tiff'
    qual_file = os.path.join(base_folder,qual_name)
    plt.savefig(qual_file)

    return atomic_percent_overall,inten,qual


def load_NLPCA_group(analysis_path,group,para,colorlist1,colorlist,band_list,eds_bin,energy_low,energy_high,
                    overlay_list,pre_blur,pre_G_blur,post_blur,post_G_blur,step,crop=None):
    corrected_spectrum_name = "EDS-" + str(energy_low) + "-" + str(energy_high) + "-bin" + str(eds_bin)
    if crop is not None:
        crop_prefix = "L" + str(crop[0]) + "-R" + str(crop[1]) + "-T" + str(crop[2]) + "-B" + str(crop[3])
    else:
        crop_prefix = ''
    for i in range(len(para)):
        corrected_name = os.path.join(analysis_path, group, str(para[i]),corrected_spectrum_name,crop_prefix,'Denoise.mat')
        denoised, denoised_name = load_NLPCA(corrected_name)
        plot_denoised_alpha(denoised,colorlist1,band_list,denoised_name,overlay_list)
        plot_denoised_alpha(denoised,colorlist1,band_list,denoised_name,overlay_list,'rescale')
        plot_element_denoised_alpha(denoised,colorlist,band_list,denoised_name,step)
        atomic_percent_overall,inten,qual = EDS_quanty(band_list,denoised_name,denoised,pre_blur=pre_blur,
                    pre_G_blur=pre_G_blur,post_blur=post_blur,post_G_blur=post_G_blur)
        plt.close('all')

############################################################################################################