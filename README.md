# PIVlab-line-command-pythonProcessing
for Matlab command line processing and python postprocessing of PIVlab acquired images.


MATLAB processing of images
- You need to get the MATLAB files from PIVlab: use the latest beta version (https://github.com/Shrediquette/PIVlab/archive/refs/heads/main.zip)
    - save the directory on your computer, and add the directory and sub-directory to the MATLAB path    
    - If you have used the Apps from optolution in your MATLAB before, make sure you delete all the files or at least remove all the directories from the MATLAB path.
  
- In the folder "Example_scripts" replace the files by the one provided on this repository

- You can run the MATLAB GUI: PIVlab_GUI.m, then import the images and process them in the GUI, recommanded at least for the first pair to set all the parameters and to use the calibration tool to get the px -> m,
- don't forget to click on clear calibration before exporting the frame if you want process uncalibrated frame (recommanded).
- If you use the GUI, once all frames have been processed and validation applied, go to file -> export: choose MAT file and click on 'all frames', best is to not apply any calibration (go calibration and click on clear calibration to be sure). write down the px->m value and the dt.
- or you can use the scripts you have copied in the 'Example_scripts' directory, it will save all the uncalibrated data in a results file.


 PYTHON processing:
 - copy the python notebook on your computer
 - you will need
   - openCV
   - dpivsoft (pip install dpivsoft) - optional, I use it for the vorticity and divergence calculation but you could build your own routine
   - scipy
   - mpl_toolkits
   - os
   - matplotlib
   - numpy
  
  -  set the path for the directory containing the results .mat file from either the GUI processing or the command line processing.
  -  you need to provide the dt and the scaling factor. You can get the scaling factor by using the MATLAB GUI and the calibration tool if you forgot to save it. For the dt there should be a file 'acquisition_log.txt' in the directory containing all the images. You will find the separation time of the pulse and the fps of the camera.
     

