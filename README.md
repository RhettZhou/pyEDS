# pyEDS

## 1.0 Introduction
__________________________________________________________________________________________________________________________________________________________________

Data denoising is urgently needed for high resolution electron microscopy spectroscopy analysis, as the raw data will have different types of noise due to spherical and chromatic aberrations, stigma, vibration, and thermal drift. pyEDS provides a systematic data processing approach to produce high-quality energy dispersive X-ray Spectroscopy (EDS) data analysis for material science study. Here we use the non-rigid registration (NRR) to reduce image distortion and non-local principal component analysis (NLPCA) to increase signal-to-noise ratio.

The NRR step is conducted by using Jupyter notebook. In principle, all the data analysis can be done in the windows system. However, the NRR is a sluggish step causing a huge amount of CPU time. Therefore, we specifically prepared the port for Linux job submissions. Please refer to the Jupyter notebook for the details. 
Non-local principal component analysis for increase signal-to-noise ratio is conducted in Matlab.  

Reference

[1] B. Berkels et al., Ultramicroscopy 138, 46 (2014).

[2] A. B. Yankovich et al., Nat. Commun. 5, 4155 (2014).

[3] C. Y. Zhang et al., Microsc. Microanal. 27, 90 (2021).

[4] C. Y. Zhang et al., Microsc. Microanal. 22, 1406 (2016).

[5] Niels Cautaerts, pymatchSeries:10.5281/zenodo.4506873



## 2.0 Installation

__________________________________________________________________________________________________________________________________________________________________

### 2.1 In anaconda

#### 2.1.1 Windows:

-> conda create -n pyEDS python=3.8

-> conda activate base

---- This command is useful for me, otherwise the environment could not be activated successfully. 

-> conda activate pyEDS

-> conda install hyperspy==1.6.2 -c conda-forge

-> conda install pip

-> pip install pyMatchSeries

-> conda install -c conda-forge match-series

-> pip install opencv-python

---- Be careful: you must install the package using the pip in your environment. Sometimes the direct use “pip” will fail because the system cannot use the packages in your base environment. You will get error messages, such as “No module named 'cv2’”.  To solve this issue, you could run the following codes. You could use the following steps to fix the issue.

-> where pip

---- To find the pip in your environment

-> …\anaconda3\envs\pyEDS\Scripts\pip install pyMatchSeries

---- You can find “…” from running the previous code.  

### 2.1.2 Linux:

-> conda create -n pyEDS python=3.8

-> source activate pyEDS

-> conda install hyperspy==1.6.2 -c conda-forge

-> conda install pip

-> pip install pyMatchSeries

-> conda install -c conda-forge match-series

-> pip install opencv-python

__________________________________________________________________________________

Compile match-series (Linux)

Reference: https://github.com/berkels/match-series

-> git clone https://github.com/berkels/match-series

-> cd match-series/quocGCC

---- -DUSE_C++11=0 -> -DUSE_C++11=1

->./goLinux.sh

-> make

-> make test

---- Copy the compile code (matchSeries) to your private modules.

---- From …\software\match-series\quocGCC\projects\electronMicroscopy\

---- To …\privatemodules\opt\nrr\

__________________________________________________________________________________

### 2.2 Matlab

I used Matlab R2018b. There is no special requirement. However, if any package is missing, please add them accordingly. 

The matlab code is from Dr. Chengyu Zhang, Cornell 



## 3.0 Tips for using
____________________________________________________________________________________________________________________________________________________________________
The major steps are in the Jupyter notebook named as “pyEDS.ipynb”
__________________________________________________________________________________
You can run NRR in Linux using the following steps:

-> cd ~/TEM/nrr/STO/20220208_1847/HAADF

-> chmod 755 lambda_0.sh

-> nohup ./lambda_0.sh &>/dev/null &

__________________________________________________________________________________

Open Jupyter notebook in Linux for running

Example 1, Linux CMTI

ssh -x -L 8000:localhost:30000 xuzhou@cmti001.bc.rzg.mpg.de

https://localhost:8000


Example 2, Linux CMTI

source ~/.bashrc (every time)

source activate pyEDS

go to the notebook/pyEDS folder

xvfb-run -a jupyter notebook --no-browser --port=8889 --ip=0.0.0.0

copy link to website

__________________________________________________________________________________

For NLPCA

Change the parameters in ‘run_me_single.m’ or ‘run_me_batch.m’ and run one of these matlab scripts. 

Reference: https://github.com/CY-Zhang/EDSDenoising




