# Catalog


* [1 Getting started](#1-getting-started)
* [2 Figure Generation for Elsevier](#2-Figure Generation for Elsevier)
* [3 Contacts](#3-contacts)


****


# 1 Getting started

1. Brain Net Viewer

   Here is a detailed tutorial for the software, please go 
   https://github.com/EnzeXu/Brain_View



2. Paraview

   Here is a detailed tutorial for the software, please go 
   https://github.com/EnzeXu/Brain_Surface

   (1) Data Prepare

   Prepare a .txt file with 160 rows and m columns. Elements in one row are split by space. Each element can be an integer or float number.
Example: file data/example.txt. Here m=1 and data type is float.

   (2)Build .vtk file

   In a Windows terminal, cd to this repo's root directory, then

   ```shell
   $ .\bin\Network2DestrieuxSurfaceRendering.exe .\bin\icbm_avg_mid_sym_mc_right_hires.vtk ${YOUR_INPUT_TXT_PATH} ${YOUR_RIGHT_OUTPUT_TXT_PATH_WITHOUT_SUFFIX} -V .\bin\Atlas_Right_Destrieux.txt -R .\bin\RightROI2NodeLookupTable.txt -m ${THE_COLUMN_NUMBER_M}
   $ .\bin\Network2DestrieuxSurfaceRendering.exe .\bin\icbm_avg_mid_sym_mc_left_hires.vtk ${YOUR_INPUT_TXT_PATH} ${YOUR_LEFT_OUTPUT_TXT_PATH_WITHOUT_SUFFIX} -V .\bin\Atlas_Left_Destrieux.txt -R .\bin\LeftROI2NodeLookupTable.txt -m ${THE_COLUMN_NUMBER_M}
   ```

   Example: 
   
   ```shell
   $ .\bin\Network2DestrieuxSurfaceRendering.exe .\bin\icbm_avg_mid_sym_mc_right_hires.vtk .\data\example.txt .\output\example_right -V .\bin\Atlas_Right_Destrieux.txt -R .\bin\RightROI2NodeLookupTable.txt -m 1
   $ .\bin\Network2DestrieuxSurfaceRendering.exe .\bin\icbm_avg_mid_sym_mc_left_hires.vtk .\data\example.txt .\output\example_left -V .\bin\Atlas_Left_Destrieux.txt -R .\bin\LeftROI2NodeLookupTable.txt -m 1
   ```
   
   Then the output .vtk file will be generated in the given output path, like file output/example_right_1.vtk and output/example_left_1.vtk).

   For convenience generating the figure for this project.
   You can see the word file in the original_spatial_txt file. Just copy paste the command line
   into the windows terminal, then you get the .vtk files ready for use.

   (3)Install App ParaView

   Home page: https://www.paraview.org/

   Click "DOWNLOAD"

   Download and install "ParaView-5.11.0-Windows-Python3.9-msvc2017-AMD64.msi"

   (5)Operations in ParaView

   File -> Open ... (choose the .vtk files)

Hint: you can open multiple .vtk files at a time. Click the "eye" button near the .vtk file's name to make it visible (or invisible)

# 2 Figure Generation for Elsevier
Title: A multiscale model to explain the spatiotemporal progression of
amyloid beta and tau pathology in Alzheimer’s disease.


1. Figure 3.

   This graph generates Aβ and p-tau abnormalities in CSF and PET imaging
   with varied secretion rates. 

   run figure3.py to get the graph.

   If you want to regenerate the .npy files used in the code, you can see line 17-20
   to get the corresponding .npy files. Make sure to change the `abnor_graph_name` everytime
   to align with the params' change.


2. Figure 4.
   
   This graph generate the spatial spread of 2 key biomarkers in the Alzheimer's disease.
   We will use the figure4.py to generate the required `.txt` files.

   (1)What it Does:

   For each category (APET, TPET) and for each subject file i = 0–3, the script:
Compares predicted results to ground truth labels. 
Computes the accuracy for each file.
Highlights mismatches in the prediction:
0.00 = low accumulation,
1.00 = high accumulation,
2.00 = False Positive (Predicted 1, Truth 0),
3.00 = False Negative (Predicted 0, Truth 1).
Saves a new version of each prediction file as {category}_{i}_new.txt.

   (2)output:

   `resub/pred/APET/APET_0_new.txt`

   `resub/pred/TPET/TPET_0_new.txt`

   `...`

   (3)How to use:

   Use the file generated in the resubmission.py to create the `.vtk` files
   described above. Hint: the command line word document is inside the folder
   respectively. 
   
   The color file is stored inside the screen_shot file if you need.

3. Figure 5.

   This graph generates the spatial dynamics of neurodegeneration and correlation analysis with
   amyloid and tau.

   Just run figure5.py to get part B-H.

   To get part A, You need to use `BrainNetViewer` in Matlab. Please refer to the first part of
   getting started.

   After running the `BrainNet.m`, in the load file, select `surface.nv` for the first line, and
   `figure5/output/N_{}.node` for the second line. Leave the rest empty and click "ok".
   In the "BrainNet_option", click "Node" in the left tool bar, select "label none" in Label, select "raw" in
   Size on the right of "Value", select "Colormap", "jet", amd "fixed" [0.00, 5.00] in Color.
   After all these, click "apply". Then you can get part A.

   
4. Figure 6.

   This graph generates Vulnerability analysis by increasing key parameters in the multiscale model.
   
   To get the brain graph in each section, you need to use `BrainNetViewer` in Matlab. Please refer to the first part of
   getting started. 

   After running the `BrainNet.m`, in the load file, select `surface.nv` for the first line, and
   `figure6/output/na2T/AD.node` for the second line. Leave the rest empty and click "ok".
   In the "BrainNet_option", click "Node" in the left tool bar, select "Above Threshold" and select "0.5"
   select "0.5" on the right, select "label none" in Label, select "raw" in
   Size on the right of "Value", select "Colormap", "jet", amd "fixed" [0.00, 5.00] in Color.
   After all these, click "apply". Then you can get each brain graph in figure6 by changing the `.node` file 
   each time.



# 3 Contacts


Please email chenm@wfu.edu




