
# MultiscaleAD

This repository contains the code and visualization instructions for the article:

**“A multiscale model to explain the spatiotemporal progression of amyloid beta and tau pathology in Alzheimer’s disease.”**

---

## Catalog

* [1 Getting Started](#1-getting-started)
* [2 Figure Generation for Publication](#2-figure-generation-for-publication)
* [3 Visualization Tools](#3-visualization-tools)
* [4 Citation](#4-citation)
* [5 Contact](#5-contact)
* [6 License](#6-license)


---

## 1. Getting Started

###  Clone Repository

```bash
git clone https://github.com/EnzeXu/MultiscaleAD.git
```

###  Setup Environment (Python 3.7–3.9 recommended)

```bash
python3 -m venv ./venv
source venv/bin/activate
pip install -r requirements.txt
```

To deactivate:

```bash
deactivate
```

---

## 2. Figure Generation for Publication

###  Figure 2 - Biomarkers' Temporal Dynamics


```bash
cd figure2
python figure2.py
# Time String (as folder name): 20250410_101924_466084
# Figure are saved to figure/20250410_101924_466084/
```

###  Figure 3 – CSF & PET Abnormality

- Run `figure3.py` to generate Aβ and p-Tau abnormalities.
```bash
cd figure3
python figure3.py
# Figure are saved to 'figure/Abnormality'
```
- (optional) Regenerate `.npy` files (lines 17–20) as needed.
Use `abnor_graph_name` for naming.

###  Figure 4 – Spatial Biomarker Spread

- Run `figure4.py`:
```bash
cd figure4
python figure4.py
```
  - Compare predicted vs. true APET & TPET accuracy.
  - Label mismatches:
    - `0.00`: true negative
    - `1.00`: true positive
    - `2.00`: false positive
    - `3.00`: false negative
- Files saved to 

   `resub/pred/APET/APET_0_new.txt`

   `resub/pred/TPET/TPET_0_new.txt`

   `...`
- How to use:

   Use the file generated in the figure4.py to create the `.vtk` files
   via [Paraview](#3-visualization-tools). Hint: the command line word document is inside the folder
   respectively. 
   
   The color file is stored inside the `figure4` file if you need.

###  Figure 5 – FDG, Amyloid, and Tau Dynamics

- Run `figure5.py` for Part B–H.
```bash
cd figure5
python figure5.py
```
- To get part A:
    
    You need to use [BrainNetViewer](#3-visualization-tools) in Matlab.
    
    After running the `BrainNet.m`, in the load file, select `surface.nv` for the first line, and
   `figure5/output/N_{}.node` for the second line. Leave the rest empty and click "ok".
   In the "BrainNet_option", click "Node" in the left tool bar, select "label none" in Label, select "raw" in
   Size on the right of "Value", select "Colormap", "jet", amd "fixed" [0.00, 5.00] in Color.
   After all these, click "apply". Then you can get part A.

###  Figure 6 – Vulnerability Analysis

- (Optional) Run `figure6.py` to generate the needed `.node` file again.
```bash
cd figure6
python figure6.py
```
- You need to use [BrainNetViewer](#3-visualization-tools) in Matlab.

    After running the `BrainNet.m`, in the load file, select `surface.nv` for the first line, and
   `figure6/output/{param}/{stage}.node` for the second line. Leave the rest empty and click "ok".
   In the "BrainNet_option", click "Node" in the left tool bar, select "Above Threshold" and select "0.5"
   select "0.5" on the right, select "label none" in Label, select "raw" in
   Size on the right of "Value", select "Colormap", "jet", amd "fixed" [0.00, 5.00] in Color.
   After all these, click "apply". Then you can get each brain graph in figure6 by changing the `.node` file 
   each time.


---

## 3. Visualization Tools

###  BrainNet Viewer (MATLAB)

Here is a detailed tutorial for the software, please go 
   https://github.com/EnzeXu/Brain_View

###  ParaView

   Here is a detailed tutorial for the software, please go 
   https://github.com/EnzeXu/Brain_Surface

   For convenience generating the figure for this project.
   You can see the word file in the directory `figure4`. Just copy paste the command line
   into the windows terminal, then you get the .vtk files ready for use.

---

## 4. Citation

```bibtex
@article{multiscalead2025,
  title={A multiscale model to explain the spatiotemporal progression of amyloid beta and tau pathology in Alzheimer’s disease},
  author={Author Names},
  journal={Journal Name},
  year={2025},
  doi={10.xxxx/yyyyy}
}
```

---

## 5. Contact

📧 chenm@wfu.edu

---

## 6. License

MIT © Project MultiscaleAD Developers
